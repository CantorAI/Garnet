// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
static void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
static std::vector<uint16_t> weights(size_t count,int phase){std::vector<uint16_t> result(count);for(size_t i=0;i<count;++i){float value=float(int((i+phase)%7)-3)*.03125f;uint32_t bits;std::memcpy(&bits,&value,4);result[i]=uint16_t(bits>>16);}return result;}
static void compare(const std::vector<float>& actual,const std::vector<float>& expected){need(actual.size()==expected.size(),"MLP output size mismatch");for(size_t i=0;i<actual.size();++i)need(std::isfinite(actual[i])&&std::abs(double(actual[i])-expected[i])<=5e-6+5e-5*std::abs(double(expected[i])),"MLP output exceeded original native comparison bound");}
int main(){try{
    Garnet::MetalContext gpu,other;size_t checked=0;
    std::vector<size_t> rows={3,24,26,44,52,80,128};
#ifdef GARNET_METAL_QWEN35_WIDE_BATCH_PREFILL
    rows.push_back(160);
#endif
    for(size_t width:{size_t(8),size_t(16)}){size_t intermediate=width*2;
        auto g=weights(width*intermediate,0),u=weights(width*intermediate,2),d=weights(width*intermediate,4);
        auto gateWeight=gpu.UploadBF16(g.data(),g.size()*2,intermediate,width),upWeight=gpu.UploadBF16(u.data(),u.size()*2,intermediate,width),downWeight=gpu.UploadBF16(d.data(),d.size()*2,width,intermediate);
        for(size_t count:rows)for(int pattern=0;pattern<4;++pattern){
            std::vector<float> values(count*width);for(size_t i=0;i<values.size();++i){float v=std::sin(float(i%509)*.13f)*.02f;if(pattern==1)v*=float(i%2?-16:16);if(pattern==2)v=i%width==3?1.f:0.f;if(pattern==3)v=std::ldexp(v,int(i%7)-3);values[i]=v;}
            auto input=gpu.Upload(values,count,width),prepared=gpu.Allocate(count,width),gate=gpu.Allocate(count,intermediate),up=gpu.Allocate(count,intermediate),activated=gpu.Allocate(count,intermediate),control=gpu.Allocate(count,width),output=gpu.Allocate(count,width),consumer=gpu.Allocate(count,width),consumerControl=gpu.Allocate(count,width);
            gpu.Begin();gpu.EncodeAdd(input,input,prepared);gpu.EncodeBF16MatVec(prepared,gateWeight,gate);gpu.EncodeBF16MatVec(prepared,upWeight,up);gpu.EncodeSiluMul(gate,up,activated);gpu.EncodeBF16MatVec(activated,downWeight,control);gpu.EncodeAdd(control,input,consumerControl);gpu.SubmitAndWait();auto expected=gpu.Download(control);auto expectedConsumer=gpu.Download(consumerControl);
            auto run=[&](){gpu.Begin();need(gpu.LastMlpGraphDispatches()==0,"Graph counter survived Begin");gpu.EncodeAdd(input,input,prepared);need(gpu.EncodeMlpPrefillGraph(prepared,gateWeight,upWeight,downWeight,output),"Admitted graph unexpectedly fell back");need(gpu.LastMlpGraphDispatches()==1&&gpu.LastPartialSubmissions()==0,"Command-buffer graph/native-flush counts wrong");gpu.EncodeAdd(output,input,consumer);};
            run();bool rejected=false;try{gpu.Download(output);}catch(const std::logic_error&){rejected=true;}need(rejected,"Pending graph output readable");
            other.Begin();rejected=false;try{other.EncodeMlpPrefillGraph(input,gateWeight,upWeight,downWeight,output);}catch(const std::logic_error&){rejected=true;}need(rejected,"Foreign graph writer accepted");other.Abort();
            gpu.SubmitAndWait();compare(gpu.Download(output),expected);compare(gpu.Download(consumer),expectedConsumer);
            run();gpu.Abort();need(gpu.LastMlpGraphDispatches()==0&&gpu.LastPartialSubmissions()==0,"Graph counters survived Abort");rejected=false;try{gpu.Download(output);}catch(const std::logic_error&){rejected=true;}need(rejected,"Aborted graph output readable");
            run();gpu.SubmitAndWait();compare(gpu.Download(output),expected);compare(gpu.Download(consumer),expectedConsumer);
            auto uninitialized=gpu.Allocate(count,width);gpu.Begin();rejected=false;try{gpu.EncodeMlpPrefillGraph(uninitialized,gateWeight,upWeight,downWeight,output);}catch(const std::logic_error&){rejected=true;}need(rejected&&gpu.LastMlpGraphDispatches()==0&&gpu.LastPartialSubmissions()==0,"Uninitialized graph input accepted");gpu.Abort();
            gpu.Begin();rejected=false;try{gpu.EncodeMlpPrefillGraph(input,gateWeight,upWeight,downWeight,input);}catch(const std::invalid_argument&){rejected=true;}need(rejected&&gpu.LastPartialSubmissions()==0,"Graph input/output alias accepted");gpu.Abort();++checked;
        }
        for(size_t count:{size_t(1),size_t(2),size_t(129),size_t(161)}){
            auto input=gpu.Upload(std::vector<float>(count*width,.125f),count,width),output=gpu.Allocate(count,width);gpu.Begin();need(!gpu.EncodeMlpPrefillGraph(input,gateWeight,upWeight,downWeight,output),"Unsupported graph rows admitted");need(gpu.LastMlpGraphDispatches()==0&&gpu.LastPartialSubmissions()==0,"Fallback encoded graph work");gpu.Abort();
        }
    }
    std::printf("MLP command-buffer graph producer/consumer, ownership, abort/recovery and original native bound PASS: %zu cases\n",checked);return 0;
}catch(const std::exception& error){std::fprintf(stderr,"MLP graph test failed: %s\n",error.what());return 1;}}
