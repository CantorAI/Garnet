// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
namespace Garnet {
// State-only replay. The caller validates and owns the complete packed layout.
// All layers use 128-wide K/V state; this deliberately has no general fallback.
struct Qwen35PackedCommitConfig {
    int batch,tokens,key_heads,value_heads,kernel,linear_layers,full_layers,kv_heads,capacity,head_dim;
    int64_t state_width,packed_width,hidden_width;
};
struct Qwen35PackedCommitSpans { int start[8],count[8]; };
struct Qwen35PackedLinearLayout { int64_t state_offset,payload_offset; };
size_t Qwen35PackedCommitWorkspace(const Qwen35PackedCommitConfig&) noexcept;
int RunQwen35PackedCommit(const Qwen35PackedCommitConfig&,Qwen35PackedCommitSpans,
    const float* packed,float* canonical,const Qwen35PackedLinearLayout* linear,
    const int64_t* full,const float* parameters,float* workspace,cudaStream_t) noexcept;
}
