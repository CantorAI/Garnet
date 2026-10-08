// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gpt_oss_kernels.h"

namespace Garnet {
// Explicit context-owned initialization. No allocation/global CUDA initializer.
// The state must be initialized on its execution device before CUDA capture.
struct GptOssXqaContextState { int device = -1; uint32_t dynamicSharedBytes = 0; };
cudaError_t InitializeGptOssXqaContext(GptOssXqaContextState*);
std::size_t GptOssXqaAttentionWorkspace(int batch, int tokens, int logicalPages,
    const GptOssOptions&);
// Private staged adapter: KV already written on stream, workspace belongs to
// this invocation (256-byte aligned), and callers own valid layer/cache
// extents. Only a private opt-in native test uses it; no inference selector.
// Invalid rows zero; visible holes preserve
// the original complete-mask scalar semantics instead of dropping pages.
cudaError_t RunGptOssXqaAttention(const void* const*, float*, void*, int batch,
    int tokens, int logicalPages, int physicalPages, const GptOssOptions&,
    const GptOssXqaContextState&, cudaStream_t);
}
