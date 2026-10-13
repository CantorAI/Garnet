// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

namespace Garnet {
// Explicitly opt in. The budget covers each engine's resident streamable
// weights, not tensors, KV pages or execution-context scratch memory.
struct TRTWeightStreamingPolicy {
    bool valid = true;
    int64_t budgetBytes = -1;
    bool Enabled() const noexcept { return budgetBytes >= 0; }
};

inline TRTWeightStreamingPolicy ParseTRTWeightStreamingPolicy(const char* value) {
    TRTWeightStreamingPolicy result;
    if (!value || !*value) return result;
    constexpr int64_t mib = 1024 * 1024;
    constexpr int64_t limit = std::numeric_limits<int64_t>::max() / mib;
    int64_t amount = 0;
    for (const char* cursor = value; *cursor; ++cursor) {
        if (*cursor < '0' || *cursor > '9' ||
            amount > (limit - (*cursor - '0')) / 10) {
            result.valid = false;
            return result;
        }
        amount = amount * 10 + (*cursor - '0');
    }
    result.budgetBytes = amount * mib;
    return result;
}

inline TRTWeightStreamingPolicy ReadTRTWeightStreamingPolicy() {
    return ParseTRTWeightStreamingPolicy(
        std::getenv("GARNET_TRT_WEIGHT_STREAMING_BUDGET_MIB"));
}

inline std::string TRTWeightStreamingFingerprint() {
    const auto policy = ReadTRTWeightStreamingPolicy();
    if (!policy.valid) return "weight_streaming_v1:invalid\n";
    if (!policy.Enabled()) return {};
    return "weight_streaming_v1:" + std::to_string(policy.budgetBytes) + "\n";
}
}
