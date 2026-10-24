// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
namespace Garnet {
// Packed offset-binary codes, low nibble first, row-major [N,K].
// BF16 input [M,K], positive BF16 scales [N,K/128], BF16 output [M,N].
// No full dequantized weight matrix or workspace; FP32 accumulation.
cudaError_t ProjectCTInt4G128(const void* input,const unsigned char* packed,
    const void* scales,void* output,int m,int n,int k,cudaStream_t stream);
}
