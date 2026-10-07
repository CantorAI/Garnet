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
__global__ void WriteIntVectorUpdate4(IntVectorUpdate4 updates) {
    const int item = threadIdx.x / IntVectorUpdate4::kMaxElements;
    const int element = threadIdx.x % IntVectorUpdate4::kMaxElements;
    if (item >= updates.count || element >= updates.lengths[item]) return;
    if (updates.bytes[item] == 4)
        static_cast<std::int32_t*>(updates.destinations[item])[element] =
            static_cast<std::int32_t>(updates.values[item][element]);
    else static_cast<std::int64_t*>(updates.destinations[item])[element] =
        updates.values[item][element];
}
}

cudaError_t LaunchIntScalarUpdate4(const IntScalarUpdate4& updates, cudaStream_t stream) {
    WriteIntScalarUpdate4<<<1, 4, 0, stream>>>(updates);
    return cudaGetLastError();
}
cudaError_t LaunchIntVectorUpdate4(const IntVectorUpdate4& updates, cudaStream_t stream) {
    WriteIntVectorUpdate4<<<1, 4 * IntVectorUpdate4::kMaxElements, 0, stream>>>(updates);
    return cudaGetLastError();
}
}
