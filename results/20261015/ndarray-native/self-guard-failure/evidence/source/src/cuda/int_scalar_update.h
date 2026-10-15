// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace Garnet {
struct IntScalarUpdate {
    void* destination;
    std::int64_t value;
    int bytes;
};

struct IntScalarUpdate4 {
    IntScalarUpdate items[4];
};

struct IntVectorUpdate4 {
    static constexpr int kMaxElements = 64;
    void* destinations[4];
    std::int64_t values[4][kMaxElements];
    int lengths[4];
    int bytes[4];
    int count;
};

// Short integer patterns are repeated over complete dense destinations. Keep
// the entire payload by value; a temporary host list is never borrowed by CUDA.
struct IntPatternUpdate4 {
    static constexpr int kMaxValues = 768;
    void* destinations[4];
    std::int32_t values[kMaxValues];
    int lengths[4];
    int patternLengths[4];
    int patternOffsets[4];
    int bytes[4];
    int count;
};
static_assert(sizeof(IntPatternUpdate4) <= 4096, "integer pattern CUDA payload must remain bounded");

cudaError_t LaunchIntScalarUpdate4(const IntScalarUpdate4& updates, cudaStream_t stream);
cudaError_t LaunchIntVectorUpdate4(const IntVectorUpdate4& updates, cudaStream_t stream);
cudaError_t LaunchIntPatternUpdate4(const IntPatternUpdate4& updates, cudaStream_t stream);
}
