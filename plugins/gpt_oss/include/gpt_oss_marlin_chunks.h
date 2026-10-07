// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstddef>

namespace Garnet {
inline int GptOssMarlinBoundedPrefillRows(int rows, int phase, int nativeLimit,
                                        bool enabled) noexcept {
    return enabled && phase == 1 && nativeLimit == 4096 &&
        rows > 4096 && rows <= 8192 ? 4096 : 0;
}
// Router, quantized constants, scales and biases are token-independent.
inline std::array<const void*,9> GptOssMarlinChunkInputs(const void* const* inputs,
                                                     std::size_t row, int hidden) {
    std::array<const void*,9> result;
    for (std::size_t i=0;i<result.size();++i) result[i]=inputs[i];
    result[0]=static_cast<const float*>(inputs[0])+row*std::size_t(hidden);
    return result;
}
}
