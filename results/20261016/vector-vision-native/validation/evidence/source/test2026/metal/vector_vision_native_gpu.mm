// Actual native library, public API: serving/fallback, changed inputs, pending,
// abort invalidation, counter reset and recovery. Full trained grids run separately.
#import <Foundation/Foundation.h>
#include "metal_context.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
static void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){@autoreleasepool{try{
    Garnet::MetalContext gpu;size_t checks=0,serving=0,fallback=0;
    for(size_t batch:{1,2})for(size_t tokens:{4,24,216})for(size_t heads:{2,12})for(size_t dim:{8,64}){
        const size_t rows=batch*tokens,hidden=heads*dim;
        const bool selected=tokens==216&&hidden==768&&heads==12;
        auto a=gpu.Allocate(rows,heads*tokens),b=gpu.Allocate(rows,3*hidden),c=gpu.Upload({0.f},1,1),out=gpu.Allocate(rows,hidden);
        std::vector<float> probs(rows*heads*tokens),qkv(rows*3*hidden),reference(rows*hidden);
        for(size_t pattern=0;pattern<4;++pattern){
            for(size_t i=0;i<probs.size();++i)probs[i]=float(int((i*7+pattern*3)%17)-8)/256.f;
            for(size_t i=0;i<qkv.size();++i)qkv[i]=float(int((i*13+pattern*11)%31)-15)/16.f;
            for(size_t row=0;row<rows;++row)for(size_t col=0;col<hidden;++col){
                const size_t head=col/dim,base=row/tokens*tokens;float value=0;
                for(size_t key=0;key<tokens;++key)value=std::fma(probs[row*heads*tokens+head*tokens+key],qkv[(base+key)*3*hidden+2*hidden+col],value);
                reference[row*hidden+col]=value;
            }
            gpu.Update(a,probs);gpu.Update(b,qkv);gpu.Begin();
            need(gpu.LastVisionVectorDispatches()==0,"Begin retained vector counter");
            gpu.EncodeVision(8,{a,b,c},out,tokens,hidden,heads,1);
            need(gpu.LastVisionVectorDispatches()==size_t(selected),"Vector serving/fallback admission mismatch");
            need(gpu.LastVisionPrefetchDispatches()==size_t(selected),"Scalar-prefetch umbrella count changed");
            bool pending=false;try{gpu.Download(out);}catch(const std::exception&){pending=true;}
            need(pending,"Pending output became readable");gpu.SubmitAndWait();
            auto actual=gpu.Download(out);need(actual.size()==reference.size(),"Result shape mismatch");
            need(std::memcmp(actual.data(),reference.data(),actual.size()*sizeof(float))==0,"Original ordered FMA reference bits changed");
            ++checks;if(selected)++serving;else ++fallback;
        }
        gpu.Begin();gpu.EncodeVision(8,{a,b,c},out,tokens,hidden,heads,1);gpu.Abort();
        need(gpu.LastVisionVectorDispatches()==0&&gpu.LastVisionPrefetchDispatches()==0,"Abort retained counters");
        bool rejected=false;try{gpu.Download(out);}catch(const std::exception&){rejected=true;}
        need(rejected,"Aborted output remained readable");
        gpu.Begin();gpu.EncodeVision(8,{a,b,c},out,tokens,hidden,heads,1);gpu.SubmitAndWait();
        auto recovered=gpu.Download(out);need(std::memcmp(recovered.data(),reference.data(),reference.size()*sizeof(float))==0,"Abort recovery bits changed");
    }
    need(checks==96&&serving==8&&fallback==88,"Native coverage count changed");
    puts("Vector vision native original-FMA bits/update/pending/abort-recovery/B1-B2 checks=96 serving=8 fallback=88 PASS");return 0;
}catch(const std::exception& e){fprintf(stderr,"vector_vision_native_failed: %s\n",e.what());return 1;}}}
