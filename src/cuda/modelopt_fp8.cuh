// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
namespace Garnet {
cudaError_t QuantizeModelOptFP8Activation(const void* input,unsigned char* output,
    float scale,int rows,int columns,cudaStream_t stream);
}
