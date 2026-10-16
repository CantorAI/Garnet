// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using Garnet::MetalContext;
static void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try{
    MetalContext gpu,other;size_t checks=0,admitted=0;
    const std::vector<std::pair<size_t,size_t>> shapes={{257,4096},{1024,1024},{8192,128},{256,4096},{8193,128},{1024,1025},{1024,512},{1024,1020}};
    for(auto shape:shapes){
        const size_t n=shape.first,k=shape.second;std::vector<float> weight(n*k);std::vector<uint16_t> bits(n*k);
        for(size_t i=0;i<bits.size();++i){weight[i]=float(int((i*11+i/k)%19)-9)/64;uint32_t word;std::memcpy(&word,&weight[i],4);bits[i]=uint16_t(word>>16);}
        auto candidate=gpu.UploadBF16(bits.data(),bits.size()*2,n,k);
        // floatPrefill bypasses packing, retaining the original BF16 decode
        // shader for a same-context bit-for-bit control at logical B1/B2.
        auto control=gpu.UploadBF16(bits.data(),bits.size()*2,n,k,true);
        const bool selected=n>256&&n<=8192&&k%4==0&&n*k>=1048576;
        need(candidate.InterleavedBodyReady()==selected&&!control.InterleavedBodyReady(),"Native body policy mismatch");admitted+=selected;
        for(size_t batch:{size_t(1),size_t(2)})for(int pattern=0;pattern<4;++pattern){
            std::vector<float> values(batch*k);
            for(size_t r=0;r<batch;++r)for(size_t j=0;j<k;++j){
                float x=std::sin(float((j*17+r*53)%1301)*.013f)*.2f;
                if(pattern==1)x*=float(j%2?-16:16);
                if(pattern==2)x=j==((r*53+17)%k)?1.0f:0.0f;
                if(pattern==3)x=std::ldexp(x,int(j%9)-4);values[r*k+j]=x;
            }
            auto input=gpu.Upload(values,batch,k),raw=gpu.Allocate(batch,n),out=gpu.Allocate(batch,n);
            gpu.Begin();gpu.EncodeBF16MatVec(input,control,raw);gpu.SubmitAndWait();auto expected=gpu.Download(raw);
            need(gpu.LastInterleavedBodyDispatches()==0,"Raw control used private layout");
            gpu.Begin();gpu.EncodeBF16MatVec(input,candidate,out);bool rejected=false;
            try{gpu.Download(out);}catch(const std::logic_error&){rejected=true;}need(rejected,"Pending native body output read");
            other.Begin();rejected=false;try{other.EncodeBF16MatVec(input,candidate,out);}catch(const std::logic_error&){rejected=true;}
            need(rejected,"Foreign native body output accepted");other.Abort();gpu.SubmitAndWait();auto actual=gpu.Download(out);
            need(gpu.LastInterleavedBodyDispatches()==size_t(selected),"Native body route count mismatch");
            need(actual.size()==expected.size()&&!std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),"Native BF16 body bits differ from raw control");
            double eps=std::ldexp(1.,-23),terms=std::ceil(double(k)/32)+32,gamma=terms*eps/(1-terms*eps);
            for(size_t r=0;r<batch;++r)for(size_t c=0;c<n;++c){
                double sum=0,absolute=0;for(size_t j=0;j<k;++j){double product=double(values[r*k+j])*weight[c*k+j];sum+=product;absolute+=std::abs(product);}
                need(std::isfinite(actual[r*n+c])&&std::abs(double(actual[r*n+c])-sum)<=gamma*absolute+1e-7,"Native body exceeded unchanged FP64 envelope");
            }
            gpu.Begin();gpu.EncodeBF16MatVec(input,candidate,out);gpu.Abort();need(gpu.LastInterleavedBodyDispatches()==0,"Abort retained native body count");
            rejected=false;try{gpu.Download(out);}catch(const std::logic_error&){rejected=true;}need(rejected,"Aborted native body readable");
            auto empty=gpu.Allocate(batch,k);gpu.Begin();rejected=false;
            try{gpu.EncodeBF16MatVec(empty,candidate,out);}catch(const std::logic_error&){rejected=true;}
            need(rejected&&gpu.LastInterleavedBodyDispatches()==0,"Uninitialized native body admitted");gpu.Abort();
            gpu.Begin();gpu.EncodeBF16MatVec(input,candidate,out);gpu.SubmitAndWait();need(gpu.Download(out)==expected,"Native body abort recovery changed values");++checks;
        }
        // Chunk projections always retain the original master/MPS path.
        auto prefill=gpu.Upload(std::vector<float>(3*k,.125f),3,k),out=gpu.Allocate(3,n);
        gpu.Begin();gpu.EncodeBF16MatVec(prefill,candidate,out);gpu.SubmitAndWait();
        need(gpu.LastInterleavedBodyDispatches()==0,"Prefill used decode-only private layout");
    }
    need(checks==64&&admitted==3,"Native body test coverage incomplete");
    std::cout<<"Native body original bits/FP64, policy/tail/fallback, ownership, abort/recovery and prefill checks="<<checks<<" PASS\n";
}catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}}
