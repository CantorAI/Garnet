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

    __global__ void EmbeddingGatherInt64BF16ToFP32Kernel(
        const __nv_bfloat16* weights,
        const long long* tokenIds,
        float* output,
        int tokenCount,
        int vocabSize,
        int hiddenSize)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int count = tokenCount * hiddenSize;
        if (index >= count) return;
        int tokenIndex = index / hiddenSize;
        int featureIndex = index % hiddenSize;
        long long tokenId = tokenIds[tokenIndex];
        output[index] = (tokenId < 0 || tokenId >= vocabSize)
            ? 0.0f
            : __bfloat162float(weights[tokenId * static_cast<long long>(hiddenSize) + featureIndex]);
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

    __global__ void AddRowsByMaskInt64FP32Kernel(
        float* output,
        const long long* rowMask,
        const float* additionRows,
        int rowCount,
        int additionCount,
        int rowWidth,
        long long maskValue)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0) {
            return;
        }
        int additionIndex = 0;
        for (int row = 0; row < rowCount; ++row) {
            if (rowMask[row] != maskValue) {
                continue;
            }
            if (additionIndex >= additionCount) {
                return;
            }
            float* destination = output + static_cast<long long>(row) * rowWidth;
            const float* source = additionRows + static_cast<long long>(additionIndex) * rowWidth;
            for (int column = 0; column < rowWidth; ++column) {
                destination[column] += source[column];
            }
            ++additionIndex;
        }
    }

    __global__ void GeluTanhFP32Kernel(const float* input, float* output, int count)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= count) {
            return;
        }
        float x = input[index];
        constexpr float kAlpha = 0.7978845608028654f;
        output[index] = 0.5f * x * (1.0f + tanhf(kAlpha * (x + 0.044715f * x * x * x)));
    }

    __global__ void VisionRoPEFP32Kernel(
        const float* qkv,
        const float* cos,
        const float* sin,
        float* output,
        int tokens,
        int hidden,
        int headDim)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int packedWidth = 3 * hidden;
        int count = tokens * packedWidth;
        if (index >= count) {
            return;
        }
        int token = index / packedWidth;
        int feature = index % packedWidth;
        if (feature >= 2 * hidden) {
            output[index] = qkv[index];
            return;
        }
        int tensorOffset = feature < hidden ? 0 : hidden;
        int localFeature = feature - tensorOffset;
        int head = localFeature / headDim;
        int dim = localFeature % headDim;
        int half = headDim / 2;
        int rotatedDim = dim < half ? dim + half : dim - half;
        float rotated = qkv[
            static_cast<long long>(token) * packedWidth + tensorOffset + head * headDim + rotatedDim];
        if (dim < half) {
            rotated = -rotated;
        }
        float c = cos[static_cast<long long>(token) * headDim + dim];
        float s = sin[static_cast<long long>(token) * headDim + dim];
        output[index] = qkv[index] * c + rotated * s;
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

extern "C" cudaError_t runEmbeddingGatherInt64BF16ToFP32(
    const __nv_bfloat16* weights,
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
    EmbeddingGatherInt64BF16ToFP32Kernel<<<blocks, kBlockSize, 0, stream>>>(
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

extern "C" cudaError_t runAddRowsByMaskInt64FP32(
    float* output,
    const long long* rowMask,
    const float* additionRows,
    int rowCount,
    int additionCount,
    int rowWidth,
    long long maskValue,
    cudaStream_t stream)
{
    if (!output || !rowMask || !additionRows || rowCount <= 0 || additionCount <= 0 || rowWidth <= 0) {
        return cudaErrorInvalidValue;
    }
    AddRowsByMaskInt64FP32Kernel<<<1, 1, 0, stream>>>(
        output, rowMask, additionRows, rowCount, additionCount, rowWidth, maskValue);
    return cudaGetLastError();
}

extern "C" cudaError_t runGeluTanhFP32(
    const float* input,
    float* output,
    int count,
    cudaStream_t stream)
{
    if (!input || !output || count <= 0) {
        return cudaErrorInvalidValue;
    }
    int blocks = (count + kBlockSize - 1) / kBlockSize;
    GeluTanhFP32Kernel<<<blocks, kBlockSize, 0, stream>>>(input, output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t runVisionRoPEFP32(
    const float* qkv,
    const float* cos,
    const float* sin,
    float* output,
    int tokens,
    int numHeads,
    int headDim,
    cudaStream_t stream)
{
    if (!qkv || !cos || !sin || !output || tokens <= 0 || numHeads <= 0 || headDim <= 0 || (headDim % 2) != 0) {
        return cudaErrorInvalidValue;
    }
    int hidden = numHeads * headDim;
    int count = tokens * 3 * hidden;
    int blocks = (count + kBlockSize - 1) / kBlockSize;
    VisionRoPEFP32Kernel<<<blocks, kBlockSize, 0, stream>>>(
        qkv, cos, sin, output, tokens, hidden, headDim);
    return cudaGetLastError();
}
