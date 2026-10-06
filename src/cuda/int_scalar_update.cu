// SPDX-License-Identifier: Apache-2.0
#include "int_scalar_update.h"

namespace Garnet {
namespace {
__global__ void WriteIntScalarUpdate4(IntScalarUpdate4 updates) {
    const auto& item = updates.items[threadIdx.x];
    if (item.bytes == 4)
        *static_cast<std::int32_t*>(item.destination) = static_cast<std::int32_t>(item.value);
    else
        *static_cast<std::int64_t*>(item.destination) = item.value;
}
}

cudaError_t LaunchIntScalarUpdate4(const IntScalarUpdate4& updates, cudaStream_t stream) {
    WriteIntScalarUpdate4<<<1, 4, 0, stream>>>(updates);
    return cudaGetLastError();
}
}
