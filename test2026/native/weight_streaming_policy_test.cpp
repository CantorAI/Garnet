// SPDX-License-Identifier: Apache-2.0
#include "weight_streaming_policy.h"
#include <iostream>
#include <stdexcept>

namespace {
void Require(bool condition, const char* description) {
    if (!condition) throw std::runtime_error(description);
}
void SetBudget(const char* value) {
#ifdef _WIN32
    _putenv_s("GARNET_TRT_WEIGHT_STREAMING_BUDGET_MIB", value ? value : "");
#else
    if (value) setenv("GARNET_TRT_WEIGHT_STREAMING_BUDGET_MIB", value, 1);
    else unsetenv("GARNET_TRT_WEIGHT_STREAMING_BUDGET_MIB");
#endif
}
struct RestoreEnvironment {
    bool existed;
    std::string value;
    RestoreEnvironment() {
        const char* old = std::getenv("GARNET_TRT_WEIGHT_STREAMING_BUDGET_MIB");
        existed = old != nullptr;
        if (old) value = old;
    }
    ~RestoreEnvironment() { SetBudget(existed ? value.c_str() : nullptr); }
};
}

int main() {
    try {
        RestoreEnvironment restore;
        for (const char* value : {static_cast<const char*>(nullptr), ""}) {
            const auto policy = Garnet::ParseTRTWeightStreamingPolicy(value);
            Require(policy.valid && !policy.Enabled(), "Unset policy must retain existing behavior");
        }
        const auto zero = Garnet::ParseTRTWeightStreamingPolicy("0");
        Require(zero.valid && zero.Enabled() && zero.budgetBytes == 0, "Zero must enable maximum streaming");
        const auto resident = Garnet::ParseTRTWeightStreamingPolicy("512");
        Require(resident.valid && resident.budgetBytes == 536870912, "MiB conversion failed");
        const auto max = Garnet::ParseTRTWeightStreamingPolicy("8796093022207");
        Require(max.valid && max.budgetBytes == 9223372036853727232LL, "Maximum safe MiB value rejected");
        for (const char* value : {"-1", "+1", "1.5", "1M", " 1", "1 ",
                "8796093022208", "18446744073709551615"}) {
            Require(!Garnet::ParseTRTWeightStreamingPolicy(value).valid, "Malformed or overflowing budget accepted");
        }
        SetBudget(nullptr);
        Require(Garnet::TRTWeightStreamingFingerprint().empty(), "Default cache fingerprint changed");
        SetBudget("0");
        const auto zeroFingerprint = Garnet::TRTWeightStreamingFingerprint();
        Require(!zeroFingerprint.empty(), "Streaming engine must use a distinct cache");
        SetBudget("00");
        Require(Garnet::TRTWeightStreamingFingerprint() == zeroFingerprint, "Equivalent budgets must share cache identity");
        SetBudget("512");
        Require(Garnet::TRTWeightStreamingFingerprint() != zeroFingerprint, "Different budgets reused cache identity");
        SetBudget("invalid");
        Require(!Garnet::ReadTRTWeightStreamingPolicy().valid, "Invalid environment silently disabled streaming");
        std::cout << "TRT weight streaming policy tests PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
