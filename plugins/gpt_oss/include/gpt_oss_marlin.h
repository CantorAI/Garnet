// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gpt_oss_kernels.h"
#include <memory>
namespace Garnet {
#ifdef GARNET_GPT_OSS_KERNEL_TEST
// Native parity/microbenchmark hook; scratch contains 2*expertCount integers.
cudaError_t TestGptOssMarlinMetadata(const int*,int*,int*,int*,int*,int slots,
    int expertCount,int rank,int block,bool parallel,cudaStream_t);
#endif
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
