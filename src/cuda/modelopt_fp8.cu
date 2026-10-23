// SPDX-License-Identifier: Apache-2.0
#include "modelopt_fp8.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdint>
namespace Garnet {
namespace {
__global__ void Quantize(const __nv_bfloat16* input,unsigned char* output,float scale,uint64_t count){
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)output[i]=__nv_cvt_float_to_fp8(__bfloat162float(input[i])/scale,__NV_SATFINITE,__NV_E4M3);
}
}
cudaError_t QuantizeModelOptFP8Activation(const void* input,unsigned char* output,
    float scale,int rows,int columns,cudaStream_t stream){
    if(!input||!output||rows<1||rows>4096||columns<1||columns>65536||
        !std::isfinite(scale)||scale<=0)return cudaErrorInvalidValue;
    const uint64_t count=uint64_t(rows)*columns;
    Quantize<<<unsigned((count+255)/256),256,0,stream>>>(static_cast<const __nv_bfloat16*>(input),output,scale,count);
    return cudaGetLastError();
}
}
