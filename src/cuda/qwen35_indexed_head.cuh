// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
namespace Garnet {
// Unwired proposal-only head. Target verification remains authoritative.
struct Qwen35IndexedHeadConfig {int rows,hidden,vocab,candidates;};
// Queries FP32, shared table BF16, per-row indices INT32, scores BF16.
// No gathered/duplicated table. Non-finite/invalid candidates never win;
// all-invalid rows return ID=-1/value=-Inf. Input/output ranges must not alias.
int RunQwen35IndexedHead(const Qwen35IndexedHeadConfig&,const float*,const void*,const int*,
    void*,long long*,float*,cudaStream_t) noexcept;
// Stable descending finite logits, ties use increasing global token ID.
// selectedRows is a HOST array validated in full before enqueue; BF16 logits.
// Nonfinite entries yield -1. Scratch is reusable across sequential rows.
size_t Qwen35HeadCandidatesWorkspace(int vocab) noexcept;
int RunQwen35HeadCandidates(int rows,int sourceRows,int vocab,int count,const int* selectedRows,
    const void* logits,int* output,void* workspace,size_t bytes,cudaStream_t) noexcept;
}
