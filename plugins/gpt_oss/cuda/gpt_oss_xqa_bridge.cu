// SPDX-License-Identifier: Apache-2.0
// Private native bridge to the marked single-block XQA adaptation. This staged
// translation unit is linked only by the opt-in native test, never inference.
#include "gpt_oss_xqa_bridge.h"
#include "mha.cu" // Generated include directory from prepare_xqa_sources.py.
#include <exception>

static_assert(validElemsPerHead == 64 && headGrpSize == 8 && tokensPerPage == 16);
static_assert(std::is_same_v<InputElem, __nv_bfloat16> && std::is_same_v<CacheElem, __nv_bfloat16>);
static_assert(!useInputKV && !useSpecDec && !lowPrecOutput && allowSlidingWindow);

namespace Garnet {
extern "C" cudaError_t GarnetGptOssXqaLaunchBf16SingleBlock(
    const void* queries, const void* keys, const void* values, const float* sinks,
    const int32_t* table, const uint32_t* lengths, void* output,
    int batch, int kvHeads, int capacity, int window, uint32_t dynamicSharedBytes,
    cudaStream_t stream) {
    if (!queries || !keys || !values || !sinks || !table || !lengths || !output ||
        batch <= 0 || batch > 512 || kvHeads <= 0 || kvHeads > 16 ||
        capacity <= 0 || capacity > 4096 || capacity % 16 || window < 0 ||
        !dynamicSharedBytes) return cudaErrorInvalidValue;
    try {
        // Single-block specialization does not use multiProcessorCount,
        // semaphores, scratch, PDL, quantization scales or speculative queries.
        ::GarnetGptOssXqaLaunch(dynamicSharedBytes, 0, uint32_t(kvHeads),
            uint32_t(window ? window : capacity), .125f, nullptr,
            static_cast<OutputHead*>(output), static_cast<const InputHead*>(queries), sinks,
            static_cast<GMemCacheHead*>(const_cast<void*>(keys)),
            static_cast<GMemCacheHead*>(const_cast<void*>(values)), table, uint32_t(capacity), lengths,
            uint32_t(batch), 1.f, nullptr, nullptr, nullptr, false,
            uint64_t(16) * kvHeads * 64, uint64_t(kvHeads) * 64, 64, stream);
        return cudaGetLastError();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "GPT-OSS XQA launch rejected: %s\n", error.what());
        return cudaErrorUnknown;
    } catch (...) { return cudaErrorUnknown; }
}
}
