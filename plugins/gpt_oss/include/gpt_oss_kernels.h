// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace Garnet {
// Serialized by the TensorRT plugin; bump plugin version when this contract changes.
struct GptOssOptions {
    int kind = 0; // 0: YaRN, 1: attention, 2: MoE, 3/4: TP collectives, 5: RMSNorm, 6: residual+RMSNorm
    int qHeads = 0, kvHeads = 0, headDim = 0;
    int layer = 0, pageSize = 16, window = 0, prefill = 0;
    int hidden = 0, intermediate = 0, experts = 0, topK = 0;
    int tpRank = -1; // -1: unsharded; 0/1: GPT-OSS tensor-parallel rank
    int bf16Communication = 0; // attention-only prefill all-reduce candidate
    float theta = 150000, factor = 32, initialContext = 4096;
    float betaFast = 32, betaSlow = 1, limit = 7, epsilon = 1.0e-5f;
};
cudaError_t GptOssTpAcquire();
void GptOssTpRelease();
cudaError_t GptOssTpAllReduce(const float*, float*, size_t, int, cudaStream_t);
cudaError_t GptOssTpAllReduceBf16(const float*, float*, void*, size_t,
    int, cudaStream_t);
cudaError_t GptOssTpPackBf16(const float*, void*, size_t, cudaStream_t);
cudaError_t GptOssTpUnpackBf16(const void*, float*, size_t, cudaStream_t);
cudaError_t GptOssTpAllGather(const float*, float*, float*, size_t, int, int,
    int, cudaStream_t);
cudaError_t RunGptOssRope(const float*, const std::int64_t*, float*, int,
    const GptOssOptions&, cudaStream_t);
cudaError_t RunGptOssRmsNorm(const float*, const float*, float*, int, int,
    float, int, cudaStream_t);
cudaError_t RunGptOssAddRmsNorm(const float*, const float*, const float*,
    float*, float*, int, int, float, cudaStream_t);
cudaError_t RunGptOssAttention(const void* const*, float*, void*, int, int, int, int,
    const GptOssOptions&, cudaStream_t);
size_t GptOssMoeWorkspace(int tokens, const GptOssOptions&);
cudaError_t RunGptOssMoe(const void* const*, float*, void*, int,
    const GptOssOptions&, cudaStream_t);
struct GptOssMarlinDecodeBuffers {
    void* convertedInput = nullptr;
    int paddedWidth = 0;
    int* locks = nullptr;
    int locksPerExpert = 0;
    int* sorted = nullptr;
    int* experts = nullptr;
    int* padded = nullptr;
    int block = 0;
};
cudaError_t RunGptOssMoeRoute(const void* const*, int*, float*, float*, int,
    const GptOssOptions&, cudaStream_t,
    const GptOssMarlinDecodeBuffers* = nullptr);
#ifdef GARNET_GPT_OSS_KERNEL_TEST
cudaError_t TestGptOssMxfp4Decode(const unsigned char*, const unsigned char*, float*, int);
size_t TestGptOssMoeWorkspace(int, const GptOssOptions&, bool grouped);
cudaError_t TestGptOssMoe(const void* const*, float*, void*, int,
    const GptOssOptions&, cudaStream_t, bool grouped);
#endif
}
