// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_router_dispatch.h"
#include <cassert>
#include <climits>
#include <iostream>

using Garnet::GptOssDecodeTensorCoreRouterSupported;
int main() {
    for (int batch : {128, 129, 144, 256, 447, 448, 449, 511, 512})
        assert(GptOssDecodeTensorCoreRouterSupported(0, batch, 2880, 128, 4));
    for (int batch : {INT_MIN, -1, 0, 1, 8, 16, 127, 513, 1024, INT_MAX})
        assert(!GptOssDecodeTensorCoreRouterSupported(0, batch, 2880, 128, 4));
    // Native WMMA handles padded query/hidden/expert tiles, not just multiples.
    assert(GptOssDecodeTensorCoreRouterSupported(0, 129, 97, 65, 4));
    assert(GptOssDecodeTensorCoreRouterSupported(0, 512, 4096, 128, 8));
    assert(GptOssDecodeTensorCoreRouterSupported(0, 128, 1, 1, 1));
    for (int phase : {INT_MIN, -1, 1, 2, INT_MAX})
        assert(!GptOssDecodeTensorCoreRouterSupported(phase, 144, 2880, 128, 4));
    for (int hidden : {INT_MIN, -1, 0, 4097, INT_MAX})
        assert(!GptOssDecodeTensorCoreRouterSupported(0, 144, hidden, 128, 4));
    for (int experts : {INT_MIN, -1, 0, 129, INT_MAX})
        assert(!GptOssDecodeTensorCoreRouterSupported(0, 144, 2880, experts, 4));
    for (int topK : {INT_MIN, -1, 0, 9, INT_MAX})
        assert(!GptOssDecodeTensorCoreRouterSupported(0, 144, 2880, 128, topK));
    assert(!GptOssDecodeTensorCoreRouterSupported(0, 144, 2880, 3, 4));
    std::cout << "Decode router dispatch boundaries and phase/geometry rejection PASS; no GPU proof\n";
}
