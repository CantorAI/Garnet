// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "cuda_lib.h"

#include <cuda_runtime.h>
#include <float.h>

__global__ void garnet_vision_attention_scores_kernel(
    const float* __restrict__ qkv,
    float* __restrict__ scores,
    int tokens,
    int heads,
    int headDim,
    int head) {
    int key = blockIdx.x * blockDim.x + threadIdx.x;
    int query = blockIdx.y * blockDim.y + threadIdx.y;
    if (query >= tokens || key >= tokens) {
        return;
    }

    int hidden = heads * headDim;
    int rowStride = 3 * hidden;
    float dot = 0.0f;
    for (int d = 0; d < headDim; ++d) {
        float q = qkv[query * rowStride + head * headDim + d];
        float k = qkv[key * rowStride + hidden + head * headDim + d];
        dot += q * k;
    }
    scores[query * tokens + key] = dot * rsqrtf(static_cast<float>(headDim));
}

__global__ void garnet_softmax_rows_kernel(float* scores, int tokens) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float shared[];
    float* reduceMax = shared;
    float* reduceSum = shared + blockDim.x;

    float maxValue = -FLT_MAX;
    for (int col = tid; col < tokens; col += blockDim.x) {
        maxValue = fmaxf(maxValue, scores[row * tokens + col]);
    }
    reduceMax[tid] = maxValue;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduceMax[tid] = fmaxf(reduceMax[tid], reduceMax[tid + stride]);
        }
        __syncthreads();
    }
    maxValue = reduceMax[0];

    float sumValue = 0.0f;
    for (int col = tid; col < tokens; col += blockDim.x) {
        float value = expf(scores[row * tokens + col] - maxValue);
        scores[row * tokens + col] = value;
        sumValue += value;
    }
    reduceSum[tid] = sumValue;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduceSum[tid] += reduceSum[tid + stride];
        }
        __syncthreads();
    }
    sumValue = reduceSum[0];

    for (int col = tid; col < tokens; col += blockDim.x) {
        scores[row * tokens + col] /= sumValue;
    }
}

__global__ void garnet_vision_attention_context_kernel(
    const float* __restrict__ qkv,
    const float* __restrict__ scores,
    float* __restrict__ output,
    int tokens,
    int heads,
    int headDim,
    int head) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int query = blockIdx.y * blockDim.y + threadIdx.y;
    if (query >= tokens || d >= headDim) {
        return;
    }

    int hidden = heads * headDim;
    int rowStride = 3 * hidden;
    float sum = 0.0f;
    for (int key = 0; key < tokens; ++key) {
        float weight = scores[query * tokens + key];
        float v = qkv[key * rowStride + 2 * hidden + head * headDim + d];
        sum += weight * v;
    }
    output[query * hidden + head * headDim + d] = sum;
}

extern "C" cudaError_t runVisionAttentionFP32(
    const float* qkv,
    float* output,
    int tokens,
    int heads,
    int headDim,
    cudaStream_t stream) {
    if (!qkv || !output || tokens <= 0 || heads <= 0 || headDim <= 0) {
        return cudaErrorInvalidValue;
    }

    float* scores = nullptr;
    size_t scoreBytes = static_cast<size_t>(tokens) * static_cast<size_t>(tokens) * sizeof(float);
    cudaError_t err = cudaMalloc(&scores, scoreBytes);
    if (err != cudaSuccess) {
        return err;
    }

    dim3 scoreBlock(16, 16);
    dim3 scoreGrid(
        (tokens + scoreBlock.x - 1) / scoreBlock.x,
        (tokens + scoreBlock.y - 1) / scoreBlock.y);
    int softmaxThreads = 256;
    size_t softmaxSharedBytes = static_cast<size_t>(softmaxThreads) * 2 * sizeof(float);
    dim3 contextBlock(16, 16);
    dim3 contextGrid(
        (headDim + contextBlock.x - 1) / contextBlock.x,
        (tokens + contextBlock.y - 1) / contextBlock.y);

    for (int head = 0; head < heads; ++head) {
        garnet_vision_attention_scores_kernel<<<scoreGrid, scoreBlock, 0, stream>>>(
            qkv, scores, tokens, heads, headDim, head);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(scores);
            return err;
        }

        garnet_softmax_rows_kernel<<<tokens, softmaxThreads, softmaxSharedBytes, stream>>>(
            scores, tokens);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(scores);
            return err;
        }

        garnet_vision_attention_context_kernel<<<contextGrid, contextBlock, 0, stream>>>(
            qkv, scores, output, tokens, heads, headDim, head);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(scores);
            return err;
        }
    }

    err = cudaFree(scores);
    return err;
}
