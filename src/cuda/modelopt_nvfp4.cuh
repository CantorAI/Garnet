// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace Garnet {
// ModelOpt row-major packed E2M1, low nibble first, block16 E4M3 scales.
// BF16 activations/output, FP32 accumulation. This is W4A16, not W4A4.
// No full-sized dequantized weight allocation; caller owns validated device data.
cudaError_t RunModelOptNVFP4Projection(const void* input, const unsigned char* weight,
    const unsigned char* scales, float scale2, void* output,
    int batch, int rows, int columns, cudaStream_t stream);
// Required scale allocation includes zero-filled 128-row / 4-column tiles.
size_t ModelOptNVFP4ScaleBytes(int rows,int columns);
cudaError_t RepackModelOptNVFP4Scales(const unsigned char* source,unsigned char* tiled,
    int rows,int columns,cudaStream_t stream);
cudaError_t QuantizeModelOptNVFP4Activation(const void* bf16Input,unsigned char* packed,
    unsigned char* tiledScales,float globalScale,int rows,int columns,cudaStream_t stream);
}
