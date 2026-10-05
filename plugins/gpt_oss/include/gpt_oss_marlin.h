// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gpt_oss_kernels.h"
#include <memory>
namespace Garnet {
// One object per execution context. Repacked weights belong to this object
// and are released on its owning GPU, never cached by raw pointer globally.
class GptOssMarlin {
    struct State;
    std::unique_ptr<State> m_state;
public:
    explicit GptOssMarlin(const GptOssOptions&);
    ~GptOssMarlin();
    GptOssMarlin(const GptOssMarlin&) = delete;
    GptOssMarlin& operator=(const GptOssMarlin&) = delete;
    static size_t Workspace(int tokens, const GptOssOptions&);
    cudaError_t Run(const void* const*, float*, void*, int tokens, cudaStream_t);
};
}
