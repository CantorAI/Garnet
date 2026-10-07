// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gpt_oss_kernels.h"
namespace Garnet {
size_t GptOssFlashAttentionWorkspace(int batch,int tokens,int logicalPages,const GptOssOptions&);
// KV has already been written on stream. Workspace is invocation-owned and
// contains conversion buffers and a device-generated fixed-capacity schedule.
cudaError_t RunGptOssFlashAttention(const void* const*,float*,void*,int batch,int tokens,
    int logicalPages,int physicalPages,const GptOssOptions&,cudaStream_t);
}
