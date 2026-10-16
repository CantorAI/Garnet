// Actual private native library: exact fixture, updates, cache cap, fallback and abort.
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
    std::vector<uint16_t> bits(k*n,0x3f80);auto weight=gpu.UploadBF16(bits.data(),bits.size()*2,n,k,false);
    auto first=gpu.Allocate(m,k),output=gpu.Allocate(m,n);std::vector<float> x(m*k,1.f);
    auto check=[&](const Garnet::MetalMatrix& input,const Garnet::MetalMatrix& target,size_t expected){
        gpu.Begin();gpu.EncodeBF16MatVec(input,weight,target);gpu.SubmitAndWait();
        need(gpu.LastNDArrayPrefillDispatches()==expected,"Actual NDArray route mismatch");
        auto values=gpu.Download(target);for(float v:values)need(std::isfinite(v)&&v==x[0]*k,"Exact fixture mismatch");};
    gpu.Update(first,x.data(),x.size());check(first,output,1);
    std::fill(x.begin(),x.end(),-3.f);gpu.Update(first,x.data(),x.size());check(first,output,1);
    size_t admitted=1,fallback=0;
    for(size_t i=1;i<270;++i){auto a=gpu.Upload(x,m,k),o=gpu.Allocate(m,n);size_t expected=i<256?1:0;check(a,o,expected);admitted+=expected;fallback+=!expected;}
    need(admitted==256&&fallback==14,"Cache-bound coverage mismatch");check(first,output,1);
    gpu.Begin();gpu.EncodeBF16MatVec(first,weight,output);need(gpu.LastNDArrayPrefillDispatches()==1,"Abort precondition mismatch");gpu.Abort();
    need(gpu.LastNDArrayPrefillDispatches()==0,"Abort retained route counter");
    bool rejected=false;try{gpu.Download(output);}catch(const std::exception&){rejected=true;}need(rejected,"Aborted output remained readable");
    check(first,output,1);
    auto a=gpu.Upload(std::vector<float>(79*k,-3.f),79,k),o=gpu.Allocate(79,n);check(a,o,0);
    puts("NDArray native exact/update/cache256/fallback14/cached-after-cap/abort-recovery/shape-fallback PASS");return 0;
}catch(const std::exception& e){fprintf(stderr,"ndarray_cache_failed: %s\n",e.what());return 1;}}}
