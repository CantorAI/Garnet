// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include "screened_head_index.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
static void need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
static uint16_t bf16(float value){uint32_t bits;std::memcpy(&bits,&value,4);return uint16_t(bits>>16);}
int main(){try{
    Garnet::MetalContext gpu;size_t checked=0;
    for(size_t k:{size_t(896),size_t(1024)}){
        const size_t n=8209;std::vector<float> query(k);for(size_t i=0;i<k;++i)query[i]=std::sin(float(i)*.013f)*.3f;
        std::vector<uint16_t> bits(n*k);for(size_t c=0;c<n;++c)for(size_t i=0;i<k;++i)bits[c*k+i]=bf16(std::sin(float((c*31+i)%1709))*.02f);
        for(size_t i=0;i<k;++i){auto value=bf16(std::copysign(.25f,query[i]));bits[8192*k+i]=bits[(n-1)*k+i]=value;bits[2*k+i]=uint16_t(value^0x8000);}
        auto weight=gpu.UploadBF16(bits.data(),bits.size()*2,n,k);need(gpu.PrepareScreenedHead(weight),"Admitted head index not prepared");need(gpu.PrepareScreenedHead(weight),"Repeated prepare changed admission");
        for(size_t batch:{size_t(1),size_t(2)})for(int pattern=0;pattern<8;++pattern){
            std::vector<float> values(batch*k);
            for(size_t row=0;row<batch;++row)for(size_t i=0;i<k;++i){float v=query[i]*(row?-1.f:1.f);
                if(pattern==1)v=0;else if(pattern==2)v=1e-20f;else if(pattern==3)v=1e20f;
                else if(pattern==4)v=std::numeric_limits<float>::quiet_NaN();else if(pattern==5)v=std::numeric_limits<float>::infinity();
                else if(pattern==6&&row==0)v=std::numeric_limits<float>::infinity();else if(pattern==7)v*=.5f;
                values[row*k+i]=v;
            }
            auto input=gpu.Upload(values,batch,k),full=gpu.Allocate(batch,n),control=gpu.Allocate(batch,2),candidate=gpu.Allocate(batch,2);
            gpu.Begin();gpu.EncodeBF16MatVec(input,weight,full);gpu.EncodeArgmax(full,control);gpu.SubmitAndWait();auto reference=gpu.Download(full),expected=gpu.Download(control);
            if(pattern==0)need(expected[0]==8192,"Outside-seed lowest tie fixture did not exercise intended winner");
            gpu.Begin();need(gpu.LastScreenedHeadDispatches()==0&&gpu.LastScreenedHeadSeedRows()==0&&gpu.LastInterleavedScreenDispatches()==0,"Begin retained screen count");need(gpu.EncodeScreenedArgmax(input,weight,candidate),"Prepared head did not screen");
            bool rejected=false;try{gpu.ScreenedDiagnostics();}catch(const std::logic_error&){rejected=true;}need(rejected,"Pending diagnostics became readable");
            gpu.SubmitAndWait();auto actual=gpu.Download(candidate);auto diagnostics=gpu.ScreenedDiagnostics();need(gpu.LastScreenedHeadDispatches()==1,"Actual head count missing");need(gpu.LastScreenedHeadSeedRows()==Garnet::kScreenedHeadSeedRows,"Actual seed rows differ");need(gpu.LastInterleavedScreenDispatches()==size_t(Garnet::kScreenedHeadInterleaved),"Actual index layout count differs");
            need(diagnostics.rows==batch&&diagnostics.columns==n,"Diagnostic shape mismatch");
            for(size_t row=0;row<batch;++row){bool unsafe=(pattern>=1&&pattern<=5)||(pattern==6&&row==0);
                need(diagnostics.querySafe[row]==(unsafe?0.f:1.f),"Query admission/fallback mismatch");
                need(actual[row*2+1]==expected[row*2+1]&&(actual[row*2+1]!=0||actual[row*2]==expected[row*2]),"Greedy/error semantics changed");
                float threshold=-std::numeric_limits<float>::infinity();for(size_t c=0;c<Garnet::kScreenedHeadSeedRows;++c)threshold=std::max(threshold,reference[row*n+c]);
                for(size_t c=0;c<n;++c){size_t i=row*n+c;need(!unsafe||diagnostics.kept[i],"Fallback pruned a logit");
                    if(diagnostics.kept[i]){float x=diagnostics.maskedLogits[i],wanted=reference[i];
                        if(std::isfinite(wanted))need(std::memcmp(&x,&wanted,4)==0,"Retained FP32 logit bits changed");
                        else need(std::isnan(wanted)?std::isnan(x):x==wanted,"Nonfinite classification changed");
                    }else need(reference[i]<threshold,"Rejected row can beat exact seed");
                }
            }
            gpu.Begin();need(gpu.EncodeScreenedArgmax(input,weight,candidate),"Abort preparation failed");gpu.Abort();need(gpu.LastScreenedHeadDispatches()==0&&gpu.LastScreenedHeadSeedRows()==0&&gpu.LastInterleavedScreenDispatches()==0,"Abort retained screen count");
            rejected=false;try{gpu.Download(candidate);}catch(const std::logic_error&){rejected=true;}need(rejected,"Aborted prediction readable");
            rejected=false;try{gpu.ScreenedDiagnostics();}catch(const std::logic_error&){rejected=true;}need(rejected,"Aborted diagnostics readable");
            gpu.Begin();need(gpu.EncodeScreenedArgmax(input,weight,candidate),"Scratch failed to recover");gpu.SubmitAndWait();need(gpu.Download(candidate)==actual,"Recovery changed prediction/error flags");++checked;
        }
        auto input=gpu.Upload(query,1,k),prediction=gpu.Allocate(1,2),wrong=gpu.Allocate(1,3),uninitialized=gpu.Allocate(1,k);
        gpu.Begin();bool rejected=false;try{gpu.EncodeScreenedArgmax(uninitialized,weight,prediction);}catch(const std::logic_error&){rejected=true;}need(rejected,"Uninitialized screen input accepted");gpu.Abort();
        gpu.Begin();rejected=false;try{gpu.EncodeScreenedArgmax(input,weight,wrong);}catch(const std::invalid_argument&){rejected=true;}need(rejected,"Wrong prediction shape accepted");gpu.Abort();
        Garnet::MetalContext other;gpu.Begin();need(gpu.EncodeScreenedArgmax(input,weight,prediction),"Ownership setup failed");other.Begin();rejected=false;
        try{other.EncodeScreenedArgmax(input,weight,prediction);}catch(const std::logic_error&){rejected=true;}need(rejected,"Foreign pending output stolen");other.Abort();gpu.Abort();
        auto unsupported=gpu.Upload(std::vector<float>(3*k,1),3,k),unchanged=gpu.Upload({9,8,7,6,5,4},3,2);
        gpu.Begin();need(!gpu.EncodeScreenedArgmax(unsupported,weight,unchanged),"Batch-three screen admitted");gpu.SubmitAndWait();need(gpu.Download(unchanged)==std::vector<float>({9,8,7,6,5,4}),"Fallback changed output");
    }
    std::cout<<"Screened head retained bits, every rejection, outside-seed ties, bounded/nonfinite/mixed fallback, padded strides, ownership and abort recovery PASS "<<checked<<" cases\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
