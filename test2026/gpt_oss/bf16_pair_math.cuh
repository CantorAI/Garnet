// SPDX-License-Identifier: Apache-2.0
// Private arithmetic prerequisite only: no peer transport or inference selector.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace GarnetPrototype {
constexpr size_t kMaximumPairElements=size_t(2880)*4096;
__device__ __forceinline__ float roundedPair(unsigned a,unsigned b) {
    const float x=__bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(a)));
    const float y=__bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(b)));
    return __bfloat162float(__float2bfloat16_rn(x+y));
}
__global__ void packedBf16PairMath(const uint4* first,const uint4* second,
                                  float4* output,size_t vectors) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<vectors;
        i+=size_t(gridDim.x)*blockDim.x) {
        const uint4 a=first[i],b=second[i];
        output[2*i]=make_float4(roundedPair(a.x,b.x),roundedPair(a.x>>16,b.x>>16),
                               roundedPair(a.y,b.y),roundedPair(a.y>>16,b.y>>16));
        output[2*i+1]=make_float4(roundedPair(a.z,b.z),roundedPair(a.z>>16,b.z>>16),
                                 roundedPair(a.w,b.w),roundedPair(a.w>>16,b.w>>16));
    }
}
inline cudaError_t pairMath(const void* a,const void* b,float* output,
                           size_t count,cudaStream_t stream) {
    if(!a||!b||!output||!count||count%8||count>kMaximumPairElements||
        reinterpret_cast<uintptr_t>(a)%16||reinterpret_cast<uintptr_t>(b)%16||
        reinterpret_cast<uintptr_t>(output)%16)return cudaErrorInvalidValue;
    packedBf16PairMath<<<128,256,0,stream>>>(static_cast<const uint4*>(a),
        static_cast<const uint4*>(b),reinterpret_cast<float4*>(output),count/8);
    return cudaGetLastError();
}
}
