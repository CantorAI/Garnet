// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_runtime.h>

__global__ void qwenVLVisionMetadataKernel(
    long long* bilinearIndices,
    __nv_bfloat16* bilinearWeights,
    long long* positionIds,
    int* cuSeqlens,
    int patchCount,
    int gridT,
    int gridH,
    int gridW,
    int mergeSize,
    int positionGridSide)
{
    int outputIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (outputIndex >= patchCount) return;

    if (outputIndex == 0) {
        cuSeqlens[0] = 0;
        for (int frame = 1; frame <= gridT; ++frame) {
            cuSeqlens[frame] = frame * gridH * gridW;
        }
    }

    const int framePatchCount = gridH * gridW;
    const int localIndex = outputIndex % framePatchCount;
    const int blocksWide = gridW / mergeSize;
    const int mergeUnit = mergeSize * mergeSize;
    const int blockIndex = localIndex / mergeUnit;
    const int innerIndex = localIndex % mergeUnit;
    const int blockRow = blockIndex / blocksWide;
    const int blockColumn = blockIndex % blocksWide;
    const int row = blockRow * mergeSize + innerIndex / mergeSize;
    const int column = blockColumn * mergeSize + innerIndex % mergeSize;

    positionIds[outputIndex * 2] = row;
    positionIds[outputIndex * 2 + 1] = column;

    const float rowPosition = gridH > 1
        ? static_cast<float>(row) * static_cast<float>(positionGridSide - 1) /
            static_cast<float>(gridH - 1)
        : 0.0F;
    const float columnPosition = gridW > 1
        ? static_cast<float>(column) * static_cast<float>(positionGridSide - 1) /
            static_cast<float>(gridW - 1)
        : 0.0F;
    const int rowFloor = static_cast<int>(floorf(rowPosition));
    const int columnFloor = static_cast<int>(floorf(columnPosition));
    const int rowCeil = min(rowFloor + 1, positionGridSide - 1);
    const int columnCeil = min(columnFloor + 1, positionGridSide - 1);
    const float rowFraction = rowPosition - rowFloor;
    const float columnFraction = columnPosition - columnFloor;

    const long long corners[4] = {
        rowFloor * positionGridSide + columnFloor,
        rowFloor * positionGridSide + columnCeil,
        rowCeil * positionGridSide + columnFloor,
        rowCeil * positionGridSide + columnCeil,
    };
    const float weights[4] = {
        (1.0F - rowFraction) * (1.0F - columnFraction),
        (1.0F - rowFraction) * columnFraction,
        rowFraction * (1.0F - columnFraction),
        rowFraction * columnFraction,
    };
    for (int corner = 0; corner < 4; ++corner) {
        const int index = outputIndex * 4 + corner;
        bilinearIndices[index] = corners[corner];
        bilinearWeights[index] = __float2bfloat16(weights[corner]);
    }
}

extern "C" cudaError_t runQwenVLVisionMetadata(
    long long* bilinearIndices,
    __nv_bfloat16* bilinearWeights,
    long long* positionIds,
    int* cuSeqlens,
    int gridT,
    int gridH,
    int gridW,
    int spatialMergeSize,
    int positionGridSide,
    cudaStream_t stream)
{
    const int patchCount = gridT * gridH * gridW;
    if (!bilinearIndices || !bilinearWeights || !positionIds || !cuSeqlens ||
        patchCount <= 0 || spatialMergeSize <= 0 || positionGridSide <= 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int blockSize = 256;
    const int gridSize = (patchCount + blockSize - 1) / blockSize;
    qwenVLVisionMetadataKernel<<<gridSize, blockSize, 0, stream>>>(
        bilinearIndices,
        bilinearWeights,
        positionIds,
        cuSeqlens,
        patchCount,
        gridT,
        gridH,
        gridW,
        spatialMergeSize,
        positionGridSide);
    return cudaGetLastError();
}
