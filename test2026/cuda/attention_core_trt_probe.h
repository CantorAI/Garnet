// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../../src/cuda/qwen35_attention_core.cuh"
#include <memory>
class AttentionCoreTRTProbe {
    struct Impl;std::unique_ptr<Impl> impl_;
public:
    explicit AttentionCoreTRTProbe(Garnet::Qwen35AttentionCoreConfig);
    ~AttentionCoreTRTProbe();
    int enqueue(const void* const*,void* const*,cudaStream_t) noexcept;
};
