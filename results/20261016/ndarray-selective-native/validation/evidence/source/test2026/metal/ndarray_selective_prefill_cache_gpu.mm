// Actual selective library: unsupported tiny-normal fixture, native half,
// both serving shapes, updates, cache cap, fallback and abort/recovery.
#import <Foundation/Foundation.h>
#include "metal_context.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
static void need(bool v,const char* message){if(!v)throw std::runtime_error(message);}
int main(){@autoreleasepool{try{
    Garnet::MetalContext gpu;const size_t k=8,n=16,m=80;
    std::vector<uint16_t> bits(k*n,0x3f80);bits[0]=0x0200; // tiny NORMAL BF16; retained exactly
    auto weight=gpu.UploadBF16(bits.data(),bits.size()*2,n,k,false);
    need(!weight.ExactHalfReady(),"Selective fixture unexpectedly admitted half");
    auto first=gpu.Allocate(m,k),output=gpu.Allocate(m,n);std::vector<float> x(m*k,1.f);
    auto check=[&](const Garnet::MetalMatrix& input,const Garnet::MetalMatrix& target,size_t expected){
        gpu.Begin();gpu.EncodeBF16MatVec(input,weight,target);gpu.SubmitAndWait();
        need(gpu.LastNDArrayPrefillDispatches()==expected,"Actual selective NDArray route mismatch");
        auto values=gpu.Download(target);for(size_t i=0;i<values.size();++i)
            need(std::isfinite(values[i])&&values[i]==x[0]*(i%n==0?k-1:k),"Exact selective fixture mismatch");};
    gpu.Update(first,x.data(),x.size());check(first,output,1);
    std::fill(x.begin(),x.end(),-3.f);gpu.Update(first,x.data(),x.size());check(first,output,1);
    size_t admitted=1,fallback=0;
    for(size_t i=1;i<270;++i){auto a=gpu.Upload(x,m,k),o=gpu.Allocate(m,n);size_t expected=i<256?1:0;check(a,o,expected);admitted+=expected;fallback+=!expected;}
    need(admitted==256&&fallback==14,"Cache-bound coverage mismatch");check(first,output,1);
    gpu.Begin();gpu.EncodeBF16MatVec(first,weight,output);need(gpu.LastNDArrayPrefillDispatches()==1,"Abort precondition mismatch");gpu.Abort();
    need(gpu.LastNDArrayPrefillDispatches()==0,"Abort retained route counter");
    bool rejected=false;try{gpu.Download(output);}catch(const std::exception&){rejected=true;}need(rejected,"Aborted output remained readable");check(first,output,1);
    auto a=gpu.Upload(std::vector<float>(79*k,-3.f),79,k),o=gpu.Allocate(79,n);check(a,o,0);
    // Independent context leaves room to admit the batch-two serving shape.
    Garnet::MetalContext second;auto w=second.UploadBF16(bits.data(),bits.size()*2,n,k,false);
    auto input=second.Upload(std::vector<float>(160*k,1.f),160,k),target=second.Allocate(160,n);
    second.Begin();second.EncodeBF16MatVec(input,w,target);second.SubmitAndWait();
    need(second.LastNDArrayPrefillDispatches()==1,"Batch-two selective route mismatch");
    auto y=second.Download(target);for(size_t i=0;i<y.size();++i)need(y[i]==float(i%n==0?k-1:k),"Batch-two exact fixture mismatch");
    std::fill(bits.begin(),bits.end(),0x3f80);auto half=second.UploadBF16(bits.data(),bits.size()*2,n,k,false);
    need(half.ExactHalfReady(),"Ordinary half fixture rejected");
    second.Begin();second.EncodeBF16MatVec(input,half,target);second.SubmitAndWait();
    need(second.LastNDArrayPrefillDispatches()==0,"Selective path replaced admitted exact half");
    for(float v:second.Download(target))need(v==float(k),"Original half fixture mismatch");
    puts("Selective NDArray native exact/update/cache256/fallback14/cached-after-cap/abort-recovery/shape80-160/native-half PASS");return 0;
}catch(const std::exception& e){fprintf(stderr,"ndarray_selective_cache_failed: %s\n",e.what());return 1;}}}
