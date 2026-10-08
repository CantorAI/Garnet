// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace Garnet {
// Private staged backend. Caller must provide normalized positive lengths and
// safe dense/repeated page indices. Complete-mask fallback belongs to adapter.
// Initialization must run on each context's device before capture; no global
// CUDA initialization or allocation occurs during module load/enqueue.
extern "C" cudaError_t GarnetGptOssXqaInitialize(uint32_t* dynamicSharedBytes);
extern "C" cudaError_t GarnetGptOssXqaLaunchBf16SingleBlock(
    const void* queries, const void* keys, const void* values, const float* sinks,
    const int32_t* table, const uint32_t* lengths, void* output,
    int batch, int kvHeads, int capacity, int window, uint32_t dynamicSharedBytes,
    cudaStream_t stream);
}
