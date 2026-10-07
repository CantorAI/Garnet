// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>

namespace Garnet {
// V10/V11 contract: the same predicate governs reservation and runtime dispatch.
// The caller's eligibility declaration means the input is already BF16-rounded.
inline std::size_t GptOssTpBf16Workspace(int kind, int eligible, int phase,
    int rank, int hidden, int dimensions, std::int64_t batch,
    std::int64_t sequence, std::int64_t width) noexcept {
    if (kind != 3 || eligible != 1 || (phase != 0 && phase != 1) ||
        rank < 0 || rank > 1 || hidden <= 0 || dimensions != 3 ||
        batch <= 0 || sequence <= 0 || width != hidden ||
        batch > std::numeric_limits<int>::max() / sequence) return 0;
    const auto rows = batch * sequence;
    if (phase ? rows < 128 : (sequence != 1 || batch < 128 || batch > 512)) return 0;
    constexpr std::size_t bytesPerElement = 2 * sizeof(std::uint16_t);
    const auto sizeLimit = std::numeric_limits<std::size_t>::max() / bytesPerElement;
    // Native pack/unpack launches at most INT_MAX blocks of256 elements.
    const auto gridLimit = std::uint64_t(std::numeric_limits<int>::max()) * 256;
    const auto limit = sizeLimit < gridLimit ? sizeLimit : gridLimit;
    if (std::uint64_t(rows) > limit / std::uint64_t(hidden)) return 0;
    return std::size_t(rows) * std::size_t(hidden) * bytesPerElement;
}
}
