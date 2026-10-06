// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "cuda_lib.h"
#include "repetition_sampler.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace
{
    template<bool Penalize = false>
    __global__ void logitsTop1LastRowKernel(
        const float* logits,
        long long* outputTokenId,
        float* outputTokenValue,
        int rows,
        int vocabSize, const unsigned char* seen = nullptr, float penalty = 1.0f)
    {
        __shared__ float values[256];
        __shared__ int indices[256];

        int tid = threadIdx.x;
        const float* row = logits + static_cast<size_t>(rows - 1) * static_cast<size_t>(vocabSize);
        float bestValue = -3.4028234663852886e38f;
        int bestIndex = 0;

        for (int i = tid; i < vocabSize; i += blockDim.x) {
            float value = row[i];
            if constexpr (Penalize) {
                if (seen[i]) value = value < 0.0f ? value * penalty : value / penalty;
            }
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

    // For large vocabularies, many blocks scan independent chunks and publish
    // their best (value, lowest index) with one packed atomic maximum. The
    // zero sentinel means no logit exceeded the legacy -FLT_MAX initial value.
    __global__ void logitsTop1ChunksKernel(const float* logits,
        unsigned long long* packedBest, int rows, int vocabSize)
    {
        constexpr int chunkSize = 1024;
        __shared__ float values[256];
        __shared__ int indices[256];
        const int tid = threadIdx.x;
        const int first = blockIdx.x * chunkSize;
        const int end = min(first + chunkSize, vocabSize);
        const float* row = logits + static_cast<size_t>(rows - 1) * vocabSize;
        float bestValue = -3.4028234663852886e38f;
        int bestIndex = 0;
        for (int i = first + tid; i < end; i += blockDim.x) {
            const float value = row[i];
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
        if (tid == 0 && values[0] > -3.4028234663852886e38f) {
            // Treat -0 and +0 as equal, matching float comparison. The finish
            // kernel reads the original logit to retain its exact sign/value.
            const float value = values[0] == 0.0f ? 0.0f : values[0];
            const unsigned int raw = __float_as_uint(value);
            const unsigned int ordered = (raw & 0x80000000u) ? ~raw : raw ^ 0x80000000u;
            const unsigned long long packed =
                (static_cast<unsigned long long>(ordered) << 32) |
                (0xffffffffu - static_cast<unsigned int>(indices[0]));
            atomicMax(packedBest, packed);
        }
    }

    __global__ void logitsTop1FinishKernel(const float* logits,
        long long* outputTokenId, float* outputTokenValue, int rows, int vocabSize)
    {
        const unsigned long long packed =
            *reinterpret_cast<const unsigned long long*>(outputTokenId);
        const int index = packed ? static_cast<int>(0xffffffffu - static_cast<unsigned int>(packed)) : 0;
        outputTokenId[0] = static_cast<long long>(index);
        if (outputTokenValue) {
            outputTokenValue[0] = packed
                ? logits[static_cast<size_t>(rows - 1) * vocabSize + index]
                : -3.4028234663852886e38f;
        }
    }

    template<bool Penalize = false>
    __global__ void logitsTop1LastRowBF16Kernel(
        const __nv_bfloat16* logits,
        long long* outputTokenId,
        float* outputTokenValue,
        int rows,
        int vocabSize, const unsigned char* seen = nullptr, float penalty = 1.0f)
    {
        __shared__ float values[256];
        __shared__ int indices[256];
        const int tid = threadIdx.x;
        const __nv_bfloat16* row = logits +
            static_cast<size_t>(rows - 1) * static_cast<size_t>(vocabSize);
        float bestValue = -3.4028234663852886e38f;
        int bestIndex = 0;
        for (int i = tid; i < vocabSize; i += blockDim.x) {
            float value = __bfloat162float(row[i]);
            if constexpr (Penalize) {
                if (seen[i]) value = value < 0.0f ? value * penalty : value / penalty;
            }
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

cudaError_t sampleRepetitionFP32(const float* logits, const unsigned char* seen,
    float penalty, long long* token, float* score, int vocab, cudaStream_t stream)
{
    if (!logits || !seen || !token || vocab <= 0 || !isfinite(penalty) || penalty < 1.0f)
        return cudaErrorInvalidValue;
    logitsTop1LastRowKernel<true><<<1, 256, 0, stream>>>(
        logits, token, score, 1, vocab, seen, penalty);
    return cudaGetLastError();
}

cudaError_t sampleRepetitionBF16(const __nv_bfloat16* logits, const unsigned char* seen,
    float penalty, long long* token, float* score, int vocab, cudaStream_t stream)
{
    if (!logits || !seen || !token || vocab <= 0 || !isfinite(penalty) || penalty < 1.0f)
        return cudaErrorInvalidValue;
    logitsTop1LastRowBF16Kernel<true><<<1, 256, 0, stream>>>(
        logits, token, score, 1, vocab, seen, penalty);
    return cudaGetLastError();
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
    if (vocabSize >= 8192) {
        auto status = cudaMemsetAsync(outputTokenId, 0, sizeof(long long), stream);
        if (status != cudaSuccess) return status;
        logitsTop1ChunksKernel<<<(vocabSize + 1023) / 1024, 256, 0, stream>>>(
            logits, reinterpret_cast<unsigned long long*>(outputTokenId), rows, vocabSize);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        logitsTop1FinishKernel<<<1, 1, 0, stream>>>(
            logits, outputTokenId, outputTokenValue, rows, vocabSize);
    } else {
        logitsTop1LastRowKernel<false><<<1, 256, 0, stream>>>(
            logits, outputTokenId, outputTokenValue, rows, vocabSize);
    }
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
    logitsTop1LastRowBF16Kernel<false><<<1, 256, 0, stream>>>(
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
