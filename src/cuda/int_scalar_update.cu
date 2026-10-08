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
__global__ void WriteIntPatternUpdate4(IntPatternUpdate4 updates) {
    const int item = blockIdx.y;
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
    if (element >= updates.lengths[item]) return;
    const auto value = updates.values[updates.patternOffsets[item] + element % updates.patternLengths[item]];
    if (updates.bytes[item] == 4)
        static_cast<std::int32_t*>(updates.destinations[item])[element] = value;
    else static_cast<std::int64_t*>(updates.destinations[item])[element] = value;
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
cudaError_t LaunchIntPatternUpdate4(const IntPatternUpdate4& updates, cudaStream_t stream) {
    if (updates.count < 1 || updates.count > 4) return cudaErrorInvalidValue;
    int maximum = 0;
    for (int item = 0; item < updates.count; ++item) {
        if (!updates.destinations[item] || updates.lengths[item] < 1 ||
            updates.lengths[item] > INT32_MAX - 1024 || updates.patternLengths[item] < 1 ||
            updates.patternLengths[item] > IntPatternUpdate4::kMaxValues ||
            updates.patternOffsets[item] < 0 ||
            updates.patternOffsets[item] > IntPatternUpdate4::kMaxValues - updates.patternLengths[item] ||
            updates.lengths[item] % updates.patternLengths[item] ||
            (updates.bytes[item] != 4 && updates.bytes[item] != 8)) return cudaErrorInvalidValue;
        if (updates.lengths[item] > maximum) maximum = updates.lengths[item];
    }
    WriteIntPatternUpdate4<<<dim3((maximum + 255) / 256, updates.count), 256, 0, stream>>>(updates);
    return cudaGetLastError();
}
}
