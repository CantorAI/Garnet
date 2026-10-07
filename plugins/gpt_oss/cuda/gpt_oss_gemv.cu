// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_kernels.h"
#include <cuda_bf16.h>

namespace Garnet {
namespace {
__global__ void decodeGemv(const __nv_bfloat16* input,
                           const __nv_bfloat16* weight,
                           __nv_bfloat16* output, int rows, int columns,
                           int weightStride, int columnOffset, int qHeads,
                           int kvHeads, int headDim, int rank) {
    constexpr int rowsPerBlock = 4;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * rowsPerBlock + warp;
    if (row >= rows) return;
    int weightRow = row;
    if (qHeads > 0) {
        const int queryPart = (qHeads / 2) * headDim;
        const int kvPart = (kvHeads / 2) * headDim;
        if (row < queryPart)
            weightRow = rank * queryPart + row;
        else if (row < queryPart + kvPart)
            weightRow = qHeads * headDim + rank * kvPart + row - queryPart;
        else
            weightRow = (qHeads + kvHeads) * headDim + rank * kvPart +
                row - queryPart - kvPart;
    }
    const __nv_bfloat16* values =
        weight + size_t(weightRow) * weightStride + columnOffset;
    float sum = 0.0f;
    for (int col = lane; col < columns; col += 32)
        sum = fmaf(__bfloat162float(values[col]),
                   __bfloat162float(input[col]), sum);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0)
        output[row] = __float2bfloat16_rn(sum);
}
}

cudaError_t RunGptOssDecodeGemv(const void* input, const void* weight,
                               void* output, int rows, int columns,
                               int weightStride, int columnOffset,
                               int qHeads, int kvHeads, int headDim, int rank,
                               cudaStream_t stream) {
    if (!input || !weight || !output || rows <= 0 || columns <= 0 ||
        columns % 32 != 0 || weightStride < columnOffset + columns ||
        columnOffset < 0 || (qHeads > 0 &&
        (rank < 0 || rank > 1 || kvHeads <= 0 || headDim <= 0 ||
         qHeads % 2 || kvHeads % 2 ||
         rows != (qHeads / 2 + kvHeads) * headDim)))
        return cudaErrorInvalidValue;
    decodeGemv<<<(rows + 3) / 4, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(output), rows, columns, weightStride,
        columnOffset, qHeads, kvHeads, headDim, rank);
    return cudaGetLastError();
}
}
