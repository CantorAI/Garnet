// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace Garnet {
// Serialized by the TensorRT plugin; bump plugin version when this contract changes.
struct GptOssOptions {
    int kind = 0; // 0: YaRN, 1: paged attention, 2: compressed MoE, 3: TP all-reduce
    int qHeads = 0, kvHeads = 0, headDim = 0;
    int layer = 0, pageSize = 16, window = 0, prefill = 0;
    int hidden = 0, intermediate = 0, experts = 0, topK = 0;
    int tpRank = -1; // -1: unsharded; 0/1: GPT-OSS MoE expert-parallel rank
    float theta = 150000, factor = 32, initialContext = 4096;
    float betaFast = 32, betaSlow = 1, limit = 7;
};
cudaError_t GptOssTpAcquire();
void GptOssTpRelease();
cudaError_t GptOssTpAllReduce(const float*, float*, size_t, int, cudaStream_t);
cudaError_t RunGptOssRope(const float*, const std::int64_t*, float*, int,
    const GptOssOptions&, cudaStream_t);
cudaError_t RunGptOssAttention(const void* const*, float*, int, int, int, int,
    const GptOssOptions&, cudaStream_t);
size_t GptOssMoeWorkspace(int tokens, const GptOssOptions&);
cudaError_t RunGptOssMoe(const void* const*, float*, void*, int,
    const GptOssOptions&, cudaStream_t);
cudaError_t RunGptOssMoeRoute(const void* const*, int*, float*, float*, int,
    const GptOssOptions&, cudaStream_t);
#ifdef GARNET_GPT_OSS_KERNEL_TEST
cudaError_t TestGptOssMxfp4Decode(const unsigned char*, const unsigned char*, float*, int);
size_t TestGptOssMoeWorkspace(int, const GptOssOptions&, bool grouped);
cudaError_t TestGptOssMoe(const void* const*, float*, void*, int,
    const GptOssOptions&, cudaStream_t, bool grouped);
#endif
}
