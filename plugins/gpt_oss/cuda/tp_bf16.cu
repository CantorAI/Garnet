// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_kernels.h"
#include <cuda_bf16.h>
#include <climits>

namespace Garnet {
namespace {
__global__ void packBf16(const float* input, __nv_bfloat16* output, size_t count) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __float2bfloat16(input[index]);
}
__global__ void unpackBf16(const __nv_bfloat16* input, float* output, size_t count) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __bfloat162float(input[index]);
}
}
cudaError_t GptOssTpPackBf16(const float* input, void* output,
    size_t count, cudaStream_t stream) {
    if (!input || !output || !count || count > size_t(INT_MAX) * 256)
        return cudaErrorInvalidValue;
    packBf16<<<(count + 255) / 256, 256, 0, stream>>>(
        input, static_cast<__nv_bfloat16*>(output), count);
    return cudaGetLastError();
}
cudaError_t GptOssTpUnpackBf16(const void* input, float* output,
    size_t count, cudaStream_t stream) {
    if (!input || !output || !count || count > size_t(INT_MAX) * 256)
        return cudaErrorInvalidValue;
    unpackBf16<<<(count + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input), output, count);
    return cudaGetLastError();
}
}
