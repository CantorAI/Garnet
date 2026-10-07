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

cudaError_t LaunchIntScalarUpdate4(const IntScalarUpdate4& updates, cudaStream_t stream);
cudaError_t LaunchIntVectorUpdate4(const IntVectorUpdate4& updates, cudaStream_t stream);
}
