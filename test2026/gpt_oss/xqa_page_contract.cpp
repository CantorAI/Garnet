// SPDX-License-Identifier: Apache-2.0
// Independent token-position oracle for the staged XQA row/page contract.
#include "gpt_oss_xqa_layout.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

int main() {
    using namespace Garnet;
    std::mt19937 random(52);
    std::size_t rows = 0, visibleTokens = 0, allPages = 0;
    for (int logical : {1, 2, 17, 160, 256}) {
        const int capacity = logical * 16;
        for (int physical : {1, 3, 18, 257}) {
            std::vector<std::vector<int>> tables;
            for (int pattern = 0; pattern < 8; ++pattern) {
                std::vector<int> table(logical);
                for (int p = 0; p < logical; ++p) {
                    table[p] = p % physical;
                    if (pattern == 1 && p % 3 == 0) table[p] = -1;
                    if (pattern == 2 && p % 5 == 0) table[p] = physical;
                    if (pattern == 3 && p % 7 == 0) table[p] = INT_MAX;
                    if (pattern == 4 && p % 2 == 0) table[p] = INT_MIN;
                    if (pattern == 5) table[p] = physical - 1;
                    if (pattern == 6) table[p] = -1;
                    if (pattern == 7) table[p] = int(random() % (physical + 4)) - 2;
                }
                tables.push_back(table);
            }
            std::vector<int> lengths{INT_MIN, -1, 0, 1, 15, 16, 17, capacity - 1,
                                     capacity, capacity + 1, INT_MAX};
            for (int i = 0; i < 35; ++i) lengths.push_back(int(random() % (capacity + 2)));
            for (const auto& table : tables) for (int end : lengths)
            for (int active : {-7, 0, 1}) for (int window : {0, 1, 15, 16, 17, 31, 64, 128, 512, 4096, INT_MAX}) {
                const bool valid = active != 0 && end >= 1 && end <= capacity;
                int expected = valid ? 1 : 0;
                std::vector<int> positions;
                if (valid) for (int position = 0; position < end; ++position) {
                    // Derive visibility from relative distance, independently
                    // of implementation's begin/page interval arithmetic.
                    if (window != 0 && end - position > window) continue;
                    positions.push_back(position);
                    const int page = table.at(position / 16);
                    if (page < 0 || page >= physical) expected = 2;
                    ++visibleTokens;
                }
                assert(int(GptOssXqaClassifyRow(table.data(), logical, physical, end, active, window)) == expected);
                const uint32_t normalized = expected == 1 ? uint32_t(end) : 1U;
                assert(normalized >= 1 && normalized <= uint32_t(capacity));
                for (int p = 0; p < logical; ++p) {
                    const bool safe = table[p] >= 0 && table[p] < physical;
                    assert(GptOssXqaValidPage(table[p], physical) == safe);
                    const int normalizedPage = GptOssXqaValidPage(table[p], physical) ? table[p] : 0;
                    assert(normalizedPage >= 0 && normalizedPage < physical);
                    if (safe) assert(normalizedPage == table[p]);
                    ++allPages;
                }
                for (int position : positions) {
                    const int original = table[position / 16];
                    if (original < 0 || original >= physical) continue;
                    // Repeated physical pages and nonzero partial-page ends
                    // preserve the original NHD cache address exactly.
                    const auto originalAddress = std::size_t(original) * 16 + position % 16;
                    const auto normalizedAddress = std::size_t(GptOssXqaValidPage(original, physical) ? original : 0) * 16 + position % 16;
                    assert(originalAddress == normalizedAddress);
                }
                ++rows;
            }
        }
    }
    std::size_t layouts = 0;
    for (int batch = 1; batch <= 512; ++batch) for (int heads : {8, 32, 128})
    for (int logical : {1, 17, 256}) {
        assert(GptOssXqaSupported(batch, 1, logical, heads, heads / 8, 64, 16, 17, 0));
        const GptOssXqaLayout layout(batch, heads, logical);
        const std::array<std::size_t, 5> offsets{layout.query, layout.output, layout.table, layout.lengths, layout.modes};
        const std::array<std::size_t, 5> counts{
            std::size_t(batch) * heads * 64 * sizeof(uint16_t),
            std::size_t(batch) * heads * 64 * sizeof(uint16_t),
            std::size_t(batch) * logical * sizeof(int32_t),
            std::size_t(batch) * sizeof(uint32_t), std::size_t(batch) * sizeof(int32_t)};
        std::size_t expectedBytes = 0;
        for (std::size_t i = 0; i < counts.size(); ++i) {
            assert(offsets[i] % 256 == 0 && offsets[i] >= expectedBytes);
            assert(offsets[i] + counts[i] <= layout.bytes);
            expectedBytes += ((counts[i] + 255) / 256) * 256;
            if (i + 1 < counts.size()) assert(offsets[i] + counts[i] <= offsets[i + 1]);
        }
        assert(layout.bytes == expectedBytes);
        ++layouts;
    }
    for (int batch : {INT_MIN, -1, 0, 513, INT_MAX})
        assert(!GptOssXqaSupported(batch, 1, 17, 32, 4, 64, 16, 0, 0) && !GptOssXqaLayout(batch, 32, 17).bytes);
    for (int logical : {INT_MIN, -1, 0, 257, INT_MAX})
        assert(!GptOssXqaSupported(128, 1, logical, 32, 4, 64, 16, 0, 0) && !GptOssXqaLayout(128, 32, logical).bytes);
    assert(!GptOssXqaSupported(128, 2, 17, 32, 4, 64, 16, 0, 0));
    assert(!GptOssXqaSupported(128, 1, 17, 32, 4, 64, 16, 0, 1));
    assert(!GptOssXqaSupported(128, 1, 17, 32, 4, 128, 16, 0, 0));
    assert(!GptOssXqaSupported(128, 1, 17, 31, 4, 64, 16, 0, 0));
    assert(!GptOssXqaSupported(128, 1, 17, 32, 4, 64, 16, -1, 0));
    assert(GptOssXqaClassifyRow(nullptr, 17, 1, 0, 1, 0) == GptOssXqaRowMode::Zero);
    assert(GptOssXqaClassifyRow(nullptr, 17, 1, 17, 0, 0) == GptOssXqaRowMode::Zero);
    std::cout << "Independent XQA row/page oracle: " << rows << " rows, " << visibleTokens
              << " visible token checks, " << allPages << " sanitized entries; " << layouts
              << " layouts and negative shape guards passed. CPU only; no GPU proof.\n";
}
