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

cudaError_t LaunchIntScalarUpdate4(const IntScalarUpdate4& updates, cudaStream_t stream);
}
