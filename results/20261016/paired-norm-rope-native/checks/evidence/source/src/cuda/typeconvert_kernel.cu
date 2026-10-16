// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "cuda_lib.h"
// Templated kernel for element-wise type conversion.
template <typename InType, typename OutType>
__global__ void astypeKernel(const InType* __restrict__ input,
    OutType* __restrict__ output,
    int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = static_cast<OutType>(input[idx]);
    }
}

// Extern "C" interface for converting from __nv_bfloat16 to float.
extern "C" {

    cudaError_t runConvertFP32ToBF16Async(
        const float* input,
        __nv_bfloat16* output,
        int num_elements,
        cudaStream_t stream)
    {
        if (!input || !output || num_elements <= 0) return cudaErrorInvalidValue;
        int blockSize = 256;
        int gridSize = (num_elements + blockSize - 1) / blockSize;
        astypeKernel<float, __nv_bfloat16><<<gridSize, blockSize, 0, stream>>>(
            input, output, num_elements);
        return cudaGetLastError();
    }

    // Launch function for converting __nv_bfloat16 to float.
    void runAstype_bf16_to_fp32(const __nv_bfloat16* input, float* output, int num_elements)
    {
        int blockSize = 256;
        int gridSize = (num_elements + blockSize - 1) / blockSize;
        astypeKernel<__nv_bfloat16, float> << <gridSize, blockSize >> > (input, output, num_elements);
        cudaDeviceSynchronize();
    }

    // You can add additional conversion functions if needed.
    // For example, converting from __half to float:
    void runAstype_fp16_to_fp32(const __half* input, float* output, int num_elements)
    {
        int blockSize = 256;
        int gridSize = (num_elements + blockSize - 1) / blockSize;
        astypeKernel<__half, float> << <gridSize, blockSize >> > (input, output, num_elements);
        cudaDeviceSynchronize();
    }
}
