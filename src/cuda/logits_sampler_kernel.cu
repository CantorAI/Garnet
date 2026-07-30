#include "cuda_lib.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace
{
    __global__ void logitsTop1LastRowKernel(
        const float* logits,
        long long* outputTokenId,
        float* outputTokenValue,
        int rows,
        int vocabSize)
    {
        __shared__ float values[256];
        __shared__ int indices[256];

        int tid = threadIdx.x;
        const float* row = logits + static_cast<size_t>(rows - 1) * static_cast<size_t>(vocabSize);
        float bestValue = -3.4028234663852886e38f;
        int bestIndex = 0;

        for (int i = tid; i < vocabSize; i += blockDim.x) {
            float value = row[i];
            if (value > bestValue || (value == bestValue && i < bestIndex)) {
                bestValue = value;
                bestIndex = i;
            }
        }

        values[tid] = bestValue;
        indices[tid] = bestIndex;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float otherValue = values[tid + stride];
                int otherIndex = indices[tid + stride];
                if (otherValue > values[tid] || (otherValue == values[tid] && otherIndex < indices[tid])) {
                    values[tid] = otherValue;
                    indices[tid] = otherIndex;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            outputTokenId[0] = static_cast<long long>(indices[0]);
            if (outputTokenValue) {
                outputTokenValue[0] = values[0];
            }
        }
    }

    __global__ void logitsTop1LastRowBF16Kernel(
        const __nv_bfloat16* logits,
        long long* outputTokenId,
        float* outputTokenValue,
        int rows,
        int vocabSize)
    {
        __shared__ float values[256];
        __shared__ int indices[256];
        const int tid = threadIdx.x;
        const __nv_bfloat16* row = logits +
            static_cast<size_t>(rows - 1) * static_cast<size_t>(vocabSize);
        float bestValue = -3.4028234663852886e38f;
        int bestIndex = 0;
        for (int i = tid; i < vocabSize; i += blockDim.x) {
            const float value = __bfloat162float(row[i]);
            if (value > bestValue || (value == bestValue && i < bestIndex)) {
                bestValue = value;
                bestIndex = i;
            }
        }
        values[tid] = bestValue;
        indices[tid] = bestIndex;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float otherValue = values[tid + stride];
                const int otherIndex = indices[tid + stride];
                if (otherValue > values[tid] ||
                    (otherValue == values[tid] && otherIndex < indices[tid])) {
                    values[tid] = otherValue;
                    indices[tid] = otherIndex;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            outputTokenId[0] = static_cast<long long>(indices[0]);
            if (outputTokenValue) outputTokenValue[0] = values[0];
        }
    }

    __global__ void logitsTop1BatchKernel(
        const float* logits,
        long long* outputTokenIds,
        float* outputTokenValues,
        int vocabSize)
    {
        __shared__ float values[256];
        __shared__ int indices[256];
        const int rowIndex = blockIdx.x;
        const int tid = threadIdx.x;
        const float* row =
            logits + static_cast<size_t>(rowIndex) * vocabSize;
        float bestValue = -3.4028234663852886e38f;
        int bestIndex = 0;
        for (int index = tid; index < vocabSize; index += blockDim.x) {
            const float value = row[index];
            if (value > bestValue ||
                (value == bestValue && index < bestIndex)) {
                bestValue = value;
                bestIndex = index;
            }
        }
        values[tid] = bestValue;
        indices[tid] = bestIndex;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float otherValue = values[tid + stride];
                const int otherIndex = indices[tid + stride];
                if (otherValue > values[tid] ||
                    (otherValue == values[tid] &&
                     otherIndex < indices[tid])) {
                    values[tid] = otherValue;
                    indices[tid] = otherIndex;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            outputTokenIds[rowIndex] =
                static_cast<long long>(indices[0]);
            if (outputTokenValues) {
                outputTokenValues[rowIndex] = values[0];
            }
        }
    }

    __global__ void logitsTop1BatchBF16Kernel(
        const __nv_bfloat16* logits,
        long long* outputTokenIds,
        float* outputTokenValues,
        int vocabSize)
    {
        __shared__ float values[256];
        __shared__ int indices[256];
        const int rowIndex = blockIdx.x;
        const int tid = threadIdx.x;
        const __nv_bfloat16* row =
            logits + static_cast<size_t>(rowIndex) * vocabSize;
        float bestValue = -3.4028234663852886e38f;
        int bestIndex = 0;
        for (int index = tid; index < vocabSize; index += blockDim.x) {
            const float value = __bfloat162float(row[index]);
            if (value > bestValue ||
                (value == bestValue && index < bestIndex)) {
                bestValue = value;
                bestIndex = index;
            }
        }
        values[tid] = bestValue;
        indices[tid] = bestIndex;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float otherValue = values[tid + stride];
                const int otherIndex = indices[tid + stride];
                if (otherValue > values[tid] ||
                    (otherValue == values[tid] &&
                     otherIndex < indices[tid])) {
                    values[tid] = otherValue;
                    indices[tid] = otherIndex;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            outputTokenIds[rowIndex] =
                static_cast<long long>(indices[0]);
            if (outputTokenValues) {
                outputTokenValues[rowIndex] = values[0];
            }
        }
    }
}

extern "C" cudaError_t runLogitsTop1FP32(
    const float* logits,
    long long* outputTokenId,
    float* outputTokenValue,
    int rows,
    int vocabSize,
    cudaStream_t stream)
{
    if (!logits || !outputTokenId || rows <= 0 || vocabSize <= 0) {
        return cudaErrorInvalidValue;
    }
    logitsTop1LastRowKernel<<<1, 256, 0, stream>>>(
        logits,
        outputTokenId,
        outputTokenValue,
        rows,
        vocabSize);
    return cudaGetLastError();
}

extern "C" cudaError_t runLogitsTop1BF16(
    const bfloat16* logits,
    long long* outputTokenId,
    float* outputTokenValue,
    int rows,
    int vocabSize,
    cudaStream_t stream)
{
    if (!logits || !outputTokenId || rows <= 0 || vocabSize <= 0) {
        return cudaErrorInvalidValue;
    }
    logitsTop1LastRowBF16Kernel<<<1, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits),
        outputTokenId,
        outputTokenValue,
        rows,
        vocabSize);
    return cudaGetLastError();
}

extern "C" cudaError_t runLogitsTop1BatchFP32(
    const float* logits,
    long long* outputTokenIds,
    float* outputTokenValues,
    int rows,
    int vocabSize,
    cudaStream_t stream)
{
    if (!logits || !outputTokenIds || rows <= 0 || vocabSize <= 0) {
        return cudaErrorInvalidValue;
    }
    logitsTop1BatchKernel<<<rows, 256, 0, stream>>>(
        logits, outputTokenIds, outputTokenValues, vocabSize);
    return cudaGetLastError();
}

extern "C" cudaError_t runLogitsTop1BatchBF16(
    const bfloat16* logits,
    long long* outputTokenIds,
    float* outputTokenValues,
    int rows,
    int vocabSize,
    cudaStream_t stream)
{
    if (!logits || !outputTokenIds || rows <= 0 || vocabSize <= 0) {
        return cudaErrorInvalidValue;
    }
    logitsTop1BatchBF16Kernel<<<rows, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits),
        outputTokenIds, outputTokenValues, vocabSize);
    return cudaGetLastError();
}
