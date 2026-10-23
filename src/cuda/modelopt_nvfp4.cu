// SPDX-License-Identifier: Apache-2.0
#include "modelopt_nvfp4.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#if CUDART_VERSION >= 12080
#include <cuda_fp4.h>
#endif
#include <climits>
#include <cmath>
#include <cstdint>
namespace Garnet {
namespace {
__device__ uint64_t ScaleOffset(int row,int col,int paddedColumns){
    return uint64_t(row/128)*128*paddedColumns+uint64_t(col/4)*512+
        (row%32)*16+((row%128)/32)*4+col%4;
}
__global__ void RepackScales(const unsigned char* source,unsigned char* tiled,
    int rows,int blocks,int paddedColumns,uint64_t total){
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=total)return;
    const int row=int(i/paddedColumns),col=int(i%paddedColumns);
    tiled[ScaleOffset(row,col,paddedColumns)]=(row<rows&&col<blocks)?source[uint64_t(row)*blocks+col]:0;
}
#if CUDART_VERSION >= 12080
__global__ void QuantizeActivation(const __nv_bfloat16* input,unsigned char* packed,
    unsigned char* tiled,float global,int rows,int columns,int paddedColumns,uint64_t total){
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=total)return;
    const int row=int(i/paddedColumns),block=int(i%paddedColumns);
    const auto offset=ScaleOffset(row,block,paddedColumns);
    if(row>=rows||block>=columns/16){tiled[offset]=0;return;}
    float values[16],maximum=0;
    for(int j=0;j<16;++j){values[j]=__bfloat162float(input[uint64_t(row)*columns+block*16+j]);maximum=fmaxf(maximum,fabsf(values[j]));}
    const unsigned char code=__nv_cvt_float_to_fp8(maximum/(6.f*global),__NV_SATFINITE,__NV_E4M3);
    tiled[offset]=code;
    const float scale=__half2float(__half(__nv_cvt_fp8_to_halfraw(code,__NV_E4M3)))*global;
    for(int pair=0;pair<8;++pair){
        const auto low=__nv_cvt_float_to_fp4(scale?values[2*pair]/scale:0.f,__NV_E2M1,cudaRoundNearest);
        const auto high=__nv_cvt_float_to_fp4(scale?values[2*pair+1]/scale:0.f,__NV_E2M1,cudaRoundNearest);
        packed[uint64_t(row)*(columns/2)+block*8+pair]=low|(high<<4);
    }
}
#endif
__device__ float E2M1(unsigned int code) {
    const float values[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return (code & 8) ? -values[code & 7] : values[code & 7];
}
__global__ void Projection(const __nv_bfloat16* input, const unsigned char* weight,
    const unsigned char* scales, float scale2, __nv_bfloat16* output, int rows, int columns) {
    const int row = blockIdx.x, batch = blockIdx.y;
    float sum = 0;
    const uint64_t base = uint64_t(row) * (columns / 2);
    const uint64_t scaleBase = uint64_t(row) * (columns / 16);
    for (int pair = threadIdx.x; pair < columns / 2; pair += blockDim.x) {
        const unsigned int packed = weight[base + pair];
        const __half_raw scaleRaw = __nv_cvt_fp8_to_halfraw(scales[scaleBase + pair / 8], __NV_E4M3);
        const float scale = __half2float(__half(scaleRaw)) * scale2;
        const uint64_t offset = uint64_t(batch) * columns + pair * 2;
        sum += __bfloat162float(input[offset]) * (E2M1(packed & 15) * scale);
        sum += __bfloat162float(input[offset + 1]) * (E2M1(packed >> 4) * scale);
    }
    for (int delta = 16; delta; delta >>= 1) sum += __shfl_down_sync(0xffffffff, sum, delta);
    __shared__ float warps[4];
    if (!(threadIdx.x & 31)) warps[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < 4 ? warps[threadIdx.x] : 0.f;
        for (int delta = 16; delta; delta >>= 1) sum += __shfl_down_sync(0xffffffff, sum, delta);
        if (!threadIdx.x) output[uint64_t(batch) * rows + row] = __float2bfloat16_rn(sum);
    }
}
}
size_t ModelOptNVFP4ScaleBytes(int rows,int columns){
    if(rows<=0||rows>INT_MAX-127||columns<=0||columns%16)return 0;
    return size_t((rows+127)/128)*128*size_t((columns/16+3)/4)*4;
}
cudaError_t RepackModelOptNVFP4Scales(const unsigned char* source,unsigned char* tiled,
    int rows,int columns,cudaStream_t stream){
    const auto total=ModelOptNVFP4ScaleBytes(rows,columns);
    if(!source||!tiled||source==tiled||!total||(total+127)/128>INT_MAX)return cudaErrorInvalidValue;
    RepackScales<<<unsigned((total+127)/128),128,0,stream>>>(source,tiled,rows,columns/16,((columns/16+3)/4)*4,total);
    return cudaGetLastError();
}
cudaError_t QuantizeModelOptNVFP4Activation(const void* bf16Input,unsigned char* packed,
    unsigned char* tiledScales,float globalScale,int rows,int columns,cudaStream_t stream){
#if CUDART_VERSION >= 12080
    const auto total=ModelOptNVFP4ScaleBytes(rows,columns);
    if(!bf16Input||!packed||!tiledScales||!total||(total+127)/128>INT_MAX||
        !std::isfinite(globalScale)||globalScale<=0)return cudaErrorInvalidValue;
    QuantizeActivation<<<unsigned((total+127)/128),128,0,stream>>>(static_cast<const __nv_bfloat16*>(bf16Input),
        packed,tiledScales,globalScale,rows,columns,((columns/16+3)/4)*4,total);
    return cudaGetLastError();
#else
    return cudaErrorNotSupported;
#endif
}
cudaError_t RunModelOptNVFP4Projection(const void* input, const unsigned char* weight,
    const unsigned char* scales, float scale2, void* output,
    int batch, int rows, int columns, cudaStream_t stream) {
    if (!input || !weight || !scales || !output || batch <= 0 || batch > 65535 ||
        rows <= 0 || columns <= 0 || columns % 16 || !std::isfinite(scale2) || scale2 <= 0)
        return cudaErrorInvalidValue;
    Projection<<<dim3(rows, batch), 128, 0, stream>>>(static_cast<const __nv_bfloat16*>(input),
        weight, scales, scale2, static_cast<__nv_bfloat16*>(output), rows, columns);
    return cudaGetLastError();
}
}
