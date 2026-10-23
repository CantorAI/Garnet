// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cublasLt.h>
#include <cstddef>
namespace Garnet {
// X[batch,K] @ W[rows,K]^T, packed E2M1 input/weights, BF16 output.
// Scale arrays use cuBLASLt's 128x4 tiled E4M3 layout; tensor scales
// are combined in globalScale. Caller owns stream, workspace and handle.
cublasStatus_t RunModelOptNVFP4Matmul(cublasLtHandle_t handle,
    const void* weights,const void* weightScales,const void* input,const void* inputScales,
    void* output,int batch,int rows,int columns,float globalScale,
    void* workspace,size_t workspaceBytes,cudaStream_t stream) noexcept;
}
