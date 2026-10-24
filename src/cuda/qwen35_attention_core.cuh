// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
namespace Garnet {
// Experimental FP32 core: Q/K already
// normalized and rotated; projection storage/rounding remains a caller concern.
struct Qwen35AttentionCoreConfig { int batch,tokens,heads,kv,capacity,dim; };
size_t Qwen35AttentionCoreWorkspace(const Qwen35AttentionCoreConfig&) noexcept;
// Inputs: Q, K, V, gate [B,T,H-or-KV,D], oldK/oldV [B,KV,C,D],
// slots/active [B,T]. Outputs: gated context [B,T,H,D], nextK, nextV.
// Caller validates active controls, integer slot range, and unique active slots
// per row before enqueue. Outputs/workspace must not alias inputs or each other.
int RunQwen35AttentionCore(const Qwen35AttentionCoreConfig&,const void* const*,
    void* const*,void*,cudaStream_t) noexcept;
// Unwired research candidate. Zero global scratch; same input/output contract.
// Shared-memory use and long-prefix tradeoffs require independent qualification.
int RunQwen35AttentionCompact(const Qwen35AttentionCoreConfig&,const void* const*,
    void* const*,void*,cudaStream_t) noexcept;

// Separate output-tiled candidate; replicates scores to expose more query work.
int RunQwen35AttentionCompactTiled(const Qwen35AttentionCoreConfig&,const void* const*,
    void* const*,void*,cudaStream_t) noexcept;

}
