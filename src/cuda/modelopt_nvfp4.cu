// SPDX-License-Identifier: Apache-2.0
#include "modelopt_nvfp4.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdint>
namespace Garnet {
namespace {
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
