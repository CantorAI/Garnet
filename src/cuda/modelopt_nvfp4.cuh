// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
namespace Garnet {
// ModelOpt row-major packed E2M1, low nibble first, block16 E4M3 scales.
// BF16 activations/output, FP32 accumulation. This is W4A16, not W4A4.
// No full-sized dequantized weight allocation; caller owns validated device data.
cudaError_t RunModelOptNVFP4Projection(const void* input, const unsigned char* weight,
    const unsigned char* scales, float scale2, void* output,
    int batch, int rows, int columns, cudaStream_t stream);
}
