// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace Garnet {
// Dispatch support only. Arithmetic equivalence and speed require GPU gates.
constexpr bool GptOssDecodeTensorCoreRouterSupported(int phase, int rows,
    int hidden, int experts, int topK) noexcept {
    return phase == 0 && rows >= 128 && rows <= 512 &&
        hidden > 0 && hidden <= 4096 && experts > 0 && experts <= 128 &&
        topK > 0 && topK <= 8 && topK <= experts;
}
}
