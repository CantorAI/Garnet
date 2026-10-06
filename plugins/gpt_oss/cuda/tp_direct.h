// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace Garnet {
cudaError_t GptOssTpDirectAcquire();
void GptOssTpDirectRelease();
cudaError_t GptOssTpDirectAllReduce(const float* input, float* output,
    size_t count, int rank, cudaStream_t stream);
}
