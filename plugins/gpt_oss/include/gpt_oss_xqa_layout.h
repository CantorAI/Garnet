// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>

namespace Garnet {
#ifdef __CUDACC__
#define GARNET_GPT_OSS_XQA_HD __host__ __device__
#else
#define GARNET_GPT_OSS_XQA_HD
#endif
enum class GptOssXqaRowMode : int { Zero = 0, Dense = 1, HoleFallback = 2 };

GARNET_GPT_OSS_XQA_HD inline bool GptOssXqaValidRow(int end, int active, int logical) {
    return active != 0 && end > 0 && end <= logical * 16;
}
GARNET_GPT_OSS_XQA_HD inline bool GptOssXqaValidPage(int page, int physical) {
    return page >= 0 && page < physical;
}
GARNET_GPT_OSS_XQA_HD inline int GptOssXqaBegin(int end, int window) {
    return window > 0 && end > window ? end - window : 0;
}
// Called with a supported logical capacity. Invalid rows never read the table.
GARNET_GPT_OSS_XQA_HD inline GptOssXqaRowMode GptOssXqaClassifyRow(
    const int* table, int logical, int physical, int end, int active, int window) {
    if (!GptOssXqaValidRow(end, active, logical)) return GptOssXqaRowMode::Zero;
    const int first = GptOssXqaBegin(end, window) / 16;
    const int last = (end - 1) / 16;
    for (int page = first; page <= last; ++page)
        if (!GptOssXqaValidPage(table[page], physical)) return GptOssXqaRowMode::HoleFallback;
    return GptOssXqaRowMode::Dense;
}

inline bool GptOssXqaSupported(int batch, int tokens, int logical, int qHeads,
    int kvHeads, int headDim, int pageSize, int window, int prefill) {
    return batch > 0 && batch <= 512 && tokens == 1 && !prefill &&
        logical > 0 && logical <= 256 && kvHeads > 0 && kvHeads <= 16 &&
        qHeads == 8 * kvHeads && headDim == 64 && pageSize == 16 && window >= 0;
}
struct GptOssXqaLayout {
    std::size_t bytes = 0, query = 0, output = 0, table = 0, lengths = 0, modes = 0;
    std::size_t take(std::size_t count) {
        const auto offset = bytes;
        bytes = (bytes + count + 255) & ~std::size_t(255);
        return offset;
    }
    GptOssXqaLayout(int batch, int qHeads, int logical) {
        // All supported dimensions are bounded; reject before unsigned casts.
        if (batch <= 0 || batch > 512 || qHeads <= 0 || qHeads > 128 ||
            qHeads % 8 || logical <= 0 || logical > 256) return;
        const std::size_t queries = std::size_t(batch) * qHeads * 64;
        query = take(queries * 2); output = take(queries * 2);
        table = take(std::size_t(batch) * logical * 4);
        lengths = take(std::size_t(batch) * 4); modes = take(std::size_t(batch) * 4);
    }
};
#undef GARNET_GPT_OSS_XQA_HD
}
