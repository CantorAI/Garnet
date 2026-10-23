// SPDX-License-Identifier: Apache-2.0
#include "qwen35_prefix_commit.cuh"
#include <cuda_runtime.h>
#include <cstdint>
namespace Garnet { namespace {
__global__ void mask(uint32_t* value,const float* active,size_t elements,int hidden){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<elements&&active[i/hidden]==0)value[i]=0;
}
__global__ void restore(uint32_t* cache,const uint32_t* backup,const float* spans,size_t elements,int kv,int capacity,int dim){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=elements)return;
    int row=i/(size_t(kv)*capacity*dim),slot=(i/dim)%capacity;
    float start=spans[row*2],count=spans[row*2+1];
    if(!(slot>=start&&slot<start+count))cache[i]=backup[i];
}
bool overlap(const void* a,size_t na,const void* b,size_t nb){
    auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    return x>=y?x-y<nb:y-x<na;
}
}
cudaError_t MaskQwen35Chunk(float* value,const float* active,int b,int t,int h,cudaStream_t stream){
    if(!value||!active||b<1||b>8||t<1||t>16||h<1||h>16384)return cudaErrorInvalidValue;
    size_t elements=size_t(b)*t*h;
    if(overlap(value,elements*4,active,size_t(b)*t*4))return cudaErrorInvalidValue;
    mask<<<(elements+255)/256,256,0,stream>>>((uint32_t*)value,active,elements,h);return cudaPeekAtLastError();
}
cudaError_t RestoreQwen35KVSuffix(float* cache,const float* backup,const float* spans,int b,int kv,int c,int d,cudaStream_t stream){
    if(!cache||!backup||!spans||b<1||b>8||kv<1||kv>32||c<1||c>8192||d<1||d>256)return cudaErrorInvalidValue;
    size_t elements=size_t(b)*kv*c*d,bytes=elements*4;
    if((cache!=backup&&overlap(cache,bytes,backup,bytes))||overlap(cache,bytes,spans,size_t(b)*8)||overlap(backup,bytes,spans,size_t(b)*8))return cudaErrorInvalidValue;
    restore<<<(elements+255)/256,256,0,stream>>>((uint32_t*)cache,(const uint32_t*)backup,spans,elements,kv,c,d);return cudaPeekAtLastError();
}
}
