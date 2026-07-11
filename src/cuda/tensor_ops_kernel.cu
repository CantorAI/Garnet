#include "cuda_lib.h"

namespace
{
    constexpr int kBlockSize = 256;

    __global__ void TensorAddFP32Kernel(
        const float* lhs,
        const float* rhs,
        float* output,
        int count)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < count) {
            output[index] = lhs[index] + rhs[index];
        }
    }

    __global__ void EmbeddingGatherInt64FP32Kernel(
        const float* weights,
        const long long* tokenIds,
        float* output,
        int tokenCount,
        int vocabSize,
        int hiddenSize)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int count = tokenCount * hiddenSize;
        if (index >= count) {
            return;
        }
        int tokenIndex = index / hiddenSize;
        int featureIndex = index % hiddenSize;
        long long tokenId = tokenIds[tokenIndex];
        if (tokenId < 0 || tokenId >= vocabSize) {
            output[index] = 0.0f;
            return;
        }
        output[index] = weights[tokenId * static_cast<long long>(hiddenSize) + featureIndex];
    }

    // The multimodal mask is short compared with model GEMMs. A single device
    // thread preserves replacement order without a host prefix-scan boundary.
    __global__ void ReplaceRowsByMaskInt64FP32Kernel(
        float* output,
        const long long* rowMask,
        const float* replacementRows,
        int rowCount,
        int replacementCount,
        int rowWidth,
        long long maskValue)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0) {
            return;
        }
        int replacementIndex = 0;
        for (int row = 0; row < rowCount; ++row) {
            if (rowMask[row] != maskValue) {
                continue;
            }
            if (replacementIndex >= replacementCount) {
                return;
            }
            float* destination = output + static_cast<long long>(row) * rowWidth;
            const float* source = replacementRows + static_cast<long long>(replacementIndex) * rowWidth;
            for (int column = 0; column < rowWidth; ++column) {
                destination[column] = source[column];
            }
            ++replacementIndex;
        }
    }
}

extern "C" cudaError_t runTensorAddFP32(
    const float* lhs,
    const float* rhs,
    float* output,
    int count,
    cudaStream_t stream)
{
    if (!lhs || !rhs || !output || count <= 0) {
        return cudaErrorInvalidValue;
    }
    int blocks = (count + kBlockSize - 1) / kBlockSize;
    TensorAddFP32Kernel<<<blocks, kBlockSize, 0, stream>>>(lhs, rhs, output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t runEmbeddingGatherInt64FP32(
    const float* weights,
    const long long* tokenIds,
    float* output,
    int tokenCount,
    int vocabSize,
    int hiddenSize,
    cudaStream_t stream)
{
    if (!weights || !tokenIds || !output || tokenCount <= 0 || vocabSize <= 0 || hiddenSize <= 0) {
        return cudaErrorInvalidValue;
    }
    int count = tokenCount * hiddenSize;
    int blocks = (count + kBlockSize - 1) / kBlockSize;
    EmbeddingGatherInt64FP32Kernel<<<blocks, kBlockSize, 0, stream>>>(
        weights, tokenIds, output, tokenCount, vocabSize, hiddenSize);
    return cudaGetLastError();
}

extern "C" cudaError_t runReplaceRowsByMaskInt64FP32(
    float* output,
    const long long* rowMask,
    const float* replacementRows,
    int rowCount,
    int replacementCount,
    int rowWidth,
    long long maskValue,
    cudaStream_t stream)
{
    if (!output || !rowMask || !replacementRows || rowCount <= 0 || replacementCount < 0 || rowWidth <= 0) {
        return cudaErrorInvalidValue;
    }
    ReplaceRowsByMaskInt64FP32Kernel<<<1, 1, 0, stream>>>(
        output, rowMask, replacementRows, rowCount, replacementCount, rowWidth, maskValue);
    return cudaGetLastError();
}
