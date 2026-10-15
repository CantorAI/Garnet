// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
static void need(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
int main(){try{
    Garnet::MetalContext gpu;size_t checks=0;
    for(const auto shape:std::vector<std::array<size_t,2>>{{7,1},{7,3},{7,17},{7,33},{896,151936},{1024,151936},{1024,248320}}){
        size_t k=shape[0],n=shape[1];std::vector<uint16_t> bits(k*n);
        for(size_t i=0;i<bits.size();++i){float value=float(int(i%17)-8)/128;uint32_t raw;std::memcpy(&raw,&value,4);bits[i]=uint16_t(raw>>16);}
        auto weight=gpu.UploadBF16(bits.data(),bits.size()*2,n,k);
        for(size_t b:{size_t(1),size_t(2)})for(float scale:{1.f,-1.f,.5f,2.f}){
            std::vector<float> values(b*k);for(size_t row=0;row<b;++row)for(size_t i=0;i<k;++i)values[row*k+i]=std::sin(float(i)*.013f)*.3f*scale*(row?-1.f:1.f);
            auto a=gpu.Upload(values,b,k),logits=gpu.Allocate(b,n),debug=gpu.Allocate(b,n),prediction=gpu.Allocate(b,2),reference=gpu.Allocate(b,2);
            gpu.Begin();gpu.EncodeBF16MatVec(a,weight,logits);gpu.EncodeArgmax(logits,reference);gpu.SubmitAndWait();auto native=gpu.Download(logits),expected=gpu.Download(reference);
            gpu.Begin();need(gpu.LastFusedHeadArgmaxDispatches()==0,"Begin did not reset fusion count");need(gpu.EncodeBF16Argmax(a,weight,prediction,&debug),"Admitted shape did not fuse");
            bool rejected=false;try{gpu.Download(prediction);}catch(const std::logic_error&){rejected=true;}need(rejected,"Pending prediction became readable");
            gpu.SubmitAndWait();auto emitted=gpu.Download(debug);need(gpu.LastFusedHeadArgmaxDispatches()==1,"Missing actual fusion count");
            need(std::memcmp(emitted.data(),native.data(),native.size()*4)==0&&gpu.Download(prediction)==expected,"Fused logits or greedy prediction changed");
            std::array<double,17> sums{},absolute{};
            for(size_t phase=0;phase<17;++phase)for(size_t i=0;i<k;++i){double term=double(values[i])*(double(int((phase+i)%17)-8)/128);sums[phase]+=term;absolute[phase]+=std::abs(term);}
            double unit=std::ldexp(1.,-23),rounds=std::ceil(double(k)/32)+32,gamma=rounds*unit/(1-rounds*unit);
            for(size_t row=0;row<b;++row)for(size_t col=0;col<n;++col){auto phase=(col*k)%17;need(std::abs(double(emitted[row*n+col])-sums[phase]*(row?-1:1))<=gamma*absolute[phase]+1e-7,"Fused result exceeds unchanged FP64 envelope");}
            gpu.Begin();need(gpu.EncodeBF16Argmax(a,weight,prediction),"Production emission-off path unavailable");gpu.SubmitAndWait();need(gpu.Download(prediction)==expected,"Production greedy path changed");
            gpu.Begin();need(gpu.EncodeBF16Argmax(a,weight,prediction,&debug),"Abort probe did not encode");gpu.Abort();need(gpu.LastFusedHeadArgmaxDispatches()==0,"Abort did not reset fusion count");
            rejected=false;try{gpu.Download(prediction);}catch(const std::logic_error&){rejected=true;}need(rejected,"Aborted prediction remained readable");
            rejected=false;try{gpu.Download(debug);}catch(const std::logic_error&){rejected=true;}need(rejected,"Aborted full logits remained readable");
            gpu.Begin();need(gpu.EncodeBF16Argmax(a,weight,prediction),"Scratch did not recover after abort");gpu.SubmitAndWait();need(gpu.Download(prediction)==expected,"Greedy scratch recovery changed result");++checks;
        }
    }
    const uint16_t zeros[21]={};auto w=gpu.UploadBF16(zeros,sizeof(zeros),3,7);
    auto a=gpu.Upload(std::vector<float>(14,-0.0f),2,7),o=gpu.Allocate(2,2);
    gpu.Begin();need(gpu.EncodeBF16Argmax(a,w,o),"Tie did not encode");gpu.SubmitAndWait();need(gpu.Download(o)==std::vector<float>({0,0,0,0}),"Lowest-index tie/signed-zero mismatch");
    auto unsupported=gpu.Upload(std::vector<float>(21,1.f),3,7),fallback=gpu.Upload({9,8,7,6,5,4},3,2);
    gpu.Begin();need(!gpu.EncodeBF16Argmax(unsupported,w,fallback),"Batch-three fallback was not preserved");gpu.SubmitAndWait();need(gpu.LastFusedHeadArgmaxDispatches()==0&&gpu.Download(fallback)==std::vector<float>({9,8,7,6,5,4}),"Fallback encoded or changed output");
    auto uninitialized=gpu.Allocate(2,7);gpu.Begin();bool rejected=false;try{gpu.EncodeBF16Argmax(uninitialized,w,o);}catch(const std::logic_error&){rejected=true;}need(rejected,"Uninitialized input accepted");gpu.Abort();
    for(float bad:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){
        auto nonfinite=gpu.Upload(std::vector<float>(14,bad),2,7),raw=gpu.Allocate(2,3),expected=gpu.Allocate(2,2);
        gpu.Begin();gpu.EncodeBF16MatVec(nonfinite,w,raw);gpu.EncodeArgmax(raw,expected);gpu.SubmitAndWait();auto control=gpu.Download(expected);
        gpu.Begin();need(gpu.EncodeBF16Argmax(nonfinite,w,o),"Nonfinite path did not encode");gpu.SubmitAndWait();need(gpu.Download(o)==control&&control[1]==1&&control[3]==1,"Nonfinite detection changed");
    }
    auto wrong=gpu.Allocate(2,3);gpu.Begin();rejected=false;try{gpu.EncodeBF16Argmax(a,w,wrong);}catch(const std::invalid_argument&){rejected=true;}need(rejected,"Invalid prediction layout accepted");gpu.Abort();
    Garnet::MetalContext other;auto shared=gpu.Allocate(2,2);gpu.Begin();need(gpu.EncodeBF16Argmax(a,w,shared),"Ownership setup failed");other.Begin();rejected=false;try{other.EncodeBF16Argmax(a,w,shared);}catch(const std::logic_error&){rejected=true;}need(rejected,"Another command batch stole pending output");other.Abort();gpu.Abort();
    std::cout<<"Fused head full-bit/FP64/production, padded strides, ties, nonfinite, batch fallback, ownership and abort recovery PASS "<<checks<<" numeric cases\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
