// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
namespace Garnet {
// Unwired prefix-commit helpers. Spans are FP32 [B,2] start/count; caller admits
// integral bounds. Kernels never use span values for memory addresses.
cudaError_t MaskQwen35Chunk(float*,const float*,int batch,int tokens,int hidden,cudaStream_t);
cudaError_t RestoreQwen35KVSuffix(float*,const float*,const float*,int batch,int kv,int capacity,int dim,cudaStream_t);
}
