// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstring>
#include <string>
#include <vector>
#include <cstdint>

namespace Garnet {
// Derived names preserve enough information to regenerate the exact constant
// from the original checkpoint when loading a stripped/refittable engine.
inline std::string GptOssExpertShardName(const std::string& source, int rank) {
    return "__garnet_gpt_oss_ep2_v1_rank" + std::to_string(rank) + "__:" + source;
}
inline bool ParseGptOssExpertShardName(const std::string& name,
    std::string& source, int& rank) {
    for (int r = 0; r < 2; ++r) {
        const auto prefix = GptOssExpertShardName("", r);
        if (name.compare(0, prefix.size(), prefix) == 0 && name.size() > prefix.size()) {
            source = name.substr(prefix.size()); rank = r; return true;
        }
    }
    return false;
}
inline bool GatherGptOssExpertShard(const void* source, size_t bytes,
    int64_t experts, int rank, std::vector<unsigned char>& output,
    std::string& error) {
    if (!source || experts < 2 || (rank != 0 && rank != 1) ||
        !bytes || uint64_t(experts) > bytes || bytes % size_t(experts)) {
        error = "invalid GPT-OSS expert-axis checkpoint shard";
        return false;
    }
    const size_t stride = bytes / size_t(experts);
    const size_t localExperts = (size_t(experts) + 1 - rank) / 2;
    output.resize(localExperts * stride);
    const auto* original = static_cast<const unsigned char*>(source);
    for (size_t local = 0; local < localExperts; ++local)
        std::memcpy(output.data() + local * stride,
            original + (2 * local + rank) * stride, stride);
    return true;
}
}
