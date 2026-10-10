// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gpt_oss_kernels.h"
#include <memory>
namespace Garnet {
#ifdef GARNET_GPT_OSS_KERNEL_TEST
cudaError_t TestGptOssMarlinRepack(const unsigned char*,const unsigned char*,
    unsigned char*,unsigned char*,int experts,int originalK,int originalN,
    int paddedK,int paddedN,cudaStream_t);
// Native parity/microbenchmark hook; scratch contains 2*expertCount integers.
cudaError_t TestGptOssMarlinMetadata(const int*,int*,int*,int*,int*,int slots,
    int expertCount,int rank,int block,bool parallel,cudaStream_t);
#endif
// One object per execution context. Legacy repacked weights are privately
// owned; prepacked constants are borrowed from the engine. Never cache raw
// pointers globally or free engine-owned constants during context teardown.
class GptOssMarlin {
    struct State;
    std::unique_ptr<State> m_state;
public:
    explicit GptOssMarlin(const GptOssOptions&);
    ~GptOssMarlin();
    GptOssMarlin(const GptOssMarlin&) = delete;
    GptOssMarlin& operator=(const GptOssMarlin&) = delete;
    static size_t Workspace(int tokens, const GptOssOptions&);
    // Optional direct BF16 partial output is a pre-fusion primitive. Existing
    // callers leave it null and retain the original FP32 output path.
    cudaError_t Run(const void* const*, float*, void*, int tokens, cudaStream_t,
        void* packedBf16Output = nullptr);
};
}
