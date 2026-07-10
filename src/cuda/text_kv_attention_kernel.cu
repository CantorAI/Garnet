#include "cuda_lib.h"

#include <cuda_runtime.h>
#include <float.h>

__device__ float garnet_text_kv_attention_score(
    const float* __restrict__ q,
    const float* __restrict__ keyCache,
    int position,
    int qHead,
    int kvHead,
    int sequenceLength,
    int qHeads,
    int kvHeads,
    int headDim) {
    float dot = 0.0f;
    const float* qRow = q + static_cast<size_t>(qHead) * headDim;
    const float* kRow = keyCache
        + (static_cast<size_t>(position) * kvHeads + kvHead) * headDim;
    for (int d = 0; d < headDim; ++d) {
        dot += qRow[d] * kRow[d];
    }
    return dot * rsqrtf(static_cast<float>(headDim));
}

__global__ void garnet_text_kv_cached_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ keyCache,
    const float* __restrict__ valueCache,
    float* __restrict__ output,
    int sequenceLength,
    int qHeads,
    int kvHeads,
    int headDim) {
    int qHead = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float shared[];
    float* reduceMax = shared;
    float* reduceSum = shared + blockDim.x;

    int groupSize = qHeads / kvHeads;
    int kvHead = qHead / groupSize;

    float maxValue = -FLT_MAX;
    for (int pos = tid; pos < sequenceLength; pos += blockDim.x) {
        float score = garnet_text_kv_attention_score(
            q, keyCache, pos, qHead, kvHead, sequenceLength, qHeads, kvHeads, headDim);
        maxValue = fmaxf(maxValue, score);
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
    for (int pos = tid; pos < sequenceLength; pos += blockDim.x) {
        float score = garnet_text_kv_attention_score(
            q, keyCache, pos, qHead, kvHead, sequenceLength, qHeads, kvHeads, headDim);
        sumValue += expf(score - maxValue);
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

    for (int d = tid; d < headDim; d += blockDim.x) {
        float out = 0.0f;
        for (int pos = 0; pos < sequenceLength; ++pos) {
            float score = garnet_text_kv_attention_score(
                q, keyCache, pos, qHead, kvHead, sequenceLength, qHeads, kvHeads, headDim);
            float weight = expf(score - maxValue) / sumValue;
            const float* vRow = valueCache
                + (static_cast<size_t>(pos) * kvHeads + kvHead) * headDim;
            out += weight * vRow[d];
        }
        output[static_cast<size_t>(qHead) * headDim + d] = out;
    }
}

extern "C" cudaError_t runTextKVCachedAttentionFP32(
    const float* q,
    const float* keyCache,
    const float* valueCache,
    float* output,
    int sequenceLength,
    int qHeads,
    int kvHeads,
    int headDim,
    cudaStream_t stream) {
    if (!q || !keyCache || !valueCache || !output ||
        sequenceLength <= 0 || qHeads <= 0 || kvHeads <= 0 || headDim <= 0 ||
        (qHeads % kvHeads) != 0) {
        return cudaErrorInvalidValue;
    }

    int threads = 256;
    size_t sharedBytes = static_cast<size_t>(threads) * 2 * sizeof(float);
    garnet_text_kv_cached_attention_kernel<<<qHeads, threads, sharedBytes, stream>>>(
        q,
        keyCache,
        valueCache,
        output,
        sequenceLength,
        qHeads,
        kvHeads,
        headDim);
    return cudaGetLastError();
}

__device__ const float* garnet_paged_kv_row(
    const float* __restrict__ pages,
    const int* __restrict__ pageTable,
    int position,
    int pageSize,
    int kvHead,
    int kvHeads,
    int headDim) {
    int logicalPage = position / pageSize;
    int pageOffset = position - logicalPage * pageSize;
    int physicalPage = pageTable[logicalPage];
    return pages
        + (((static_cast<size_t>(physicalPage) * pageSize + pageOffset) * kvHeads + kvHead) * headDim);
}

__device__ float garnet_text_paged_kv_attention_score(
    const float* __restrict__ q,
    const float* __restrict__ keyPages,
    const int* __restrict__ pageTable,
    int position,
    int pageSize,
    int qHead,
    int kvHead,
    int qHeads,
    int kvHeads,
    int headDim) {
    float dot = 0.0f;
    const float* qRow = q + static_cast<size_t>(qHead) * headDim;
    const float* kRow = garnet_paged_kv_row(
        keyPages, pageTable, position, pageSize, kvHead, kvHeads, headDim);
    for (int d = 0; d < headDim; ++d) {
        dot += qRow[d] * kRow[d];
    }
    return dot * rsqrtf(static_cast<float>(headDim));
}

__global__ void garnet_text_paged_kv_cached_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ keyPages,
    const float* __restrict__ valuePages,
    const int* __restrict__ pageTable,
    float* __restrict__ output,
    int sequenceLength,
    int pageSize,
    int qHeads,
    int kvHeads,
    int headDim) {
    int qHead = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float shared[];
    float* reduceMax = shared;
    float* reduceSum = shared + blockDim.x;

    int groupSize = qHeads / kvHeads;
    int kvHead = qHead / groupSize;

    float maxValue = -FLT_MAX;
    for (int pos = tid; pos < sequenceLength; pos += blockDim.x) {
        float score = garnet_text_paged_kv_attention_score(
            q, keyPages, pageTable, pos, pageSize, qHead, kvHead, qHeads, kvHeads, headDim);
        maxValue = fmaxf(maxValue, score);
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
    for (int pos = tid; pos < sequenceLength; pos += blockDim.x) {
        float score = garnet_text_paged_kv_attention_score(
            q, keyPages, pageTable, pos, pageSize, qHead, kvHead, qHeads, kvHeads, headDim);
        sumValue += expf(score - maxValue);
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

    for (int d = tid; d < headDim; d += blockDim.x) {
        float out = 0.0f;
        for (int pos = 0; pos < sequenceLength; ++pos) {
            float score = garnet_text_paged_kv_attention_score(
                q, keyPages, pageTable, pos, pageSize, qHead, kvHead, qHeads, kvHeads, headDim);
            float weight = expf(score - maxValue) / sumValue;
            const float* vRow = garnet_paged_kv_row(
                valuePages, pageTable, pos, pageSize, kvHead, kvHeads, headDim);
            out += weight * vRow[d];
        }
        output[static_cast<size_t>(qHead) * headDim + d] = out;
    }
}

extern "C" cudaError_t runTextPagedKVCachedAttentionFP32(
    const float* q,
    const float* keyPages,
    const float* valuePages,
    const int* pageTable,
    float* output,
    int sequenceLength,
    int pageSize,
    int qHeads,
    int kvHeads,
    int headDim,
    cudaStream_t stream) {
    if (!q || !keyPages || !valuePages || !pageTable || !output ||
        sequenceLength <= 0 || pageSize <= 0 || qHeads <= 0 || kvHeads <= 0 || headDim <= 0 ||
        (qHeads % kvHeads) != 0) {
        return cudaErrorInvalidValue;
    }

    int threads = 256;
    size_t sharedBytes = static_cast<size_t>(threads) * 2 * sizeof(float);
    garnet_text_paged_kv_cached_attention_kernel<<<qHeads, threads, sharedBytes, stream>>>(
        q,
        keyPages,
        valuePages,
        pageTable,
        output,
        sequenceLength,
        pageSize,
        qHeads,
        kvHeads,
        headDim);
    return cudaGetLastError();
}

__global__ void garnet_text_paged_kv_write_kernel(
    const float* __restrict__ qkv,
    float* __restrict__ keyPages,
    float* __restrict__ valuePages,
    const int* __restrict__ pageTable,
    int tokenCount,
    int startPosition,
    int pageSize,
    int qHeads,
    int kvHeads,
    int headDim) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int elementsPerToken = kvHeads * headDim;
    int totalElements = tokenCount * elementsPerToken;
    if (linear >= totalElements) {
        return;
    }

    int token = linear / elementsPerToken;
    int within = linear - token * elementsPerToken;
    int kvHead = within / headDim;
    int dim = within - kvHead * headDim;
    int logicalPosition = startPosition + token;
    int logicalPage = logicalPosition / pageSize;
    int pageOffset = logicalPosition - logicalPage * pageSize;
    int physicalPage = pageTable[logicalPage];

    int qWidth = qHeads * headDim;
    int kvWidth = kvHeads * headDim;
    int qkvStride = qWidth + 2 * kvWidth;
    const float* qkvRow = qkv + static_cast<size_t>(token) * qkvStride;
    float* keyRow = keyPages
        + (((static_cast<size_t>(physicalPage) * pageSize + pageOffset) * kvHeads + kvHead) * headDim);
    float* valueRow = valuePages
        + (((static_cast<size_t>(physicalPage) * pageSize + pageOffset) * kvHeads + kvHead) * headDim);

    keyRow[dim] = qkvRow[qWidth + kvHead * headDim + dim];
    valueRow[dim] = qkvRow[qWidth + kvWidth + kvHead * headDim + dim];
}

extern "C" cudaError_t runTextPagedKVWriteFP32(
    const float* qkv,
    float* keyPages,
    float* valuePages,
    const int* pageTable,
    int tokenCount,
    int startPosition,
    int pageSize,
    int qHeads,
    int kvHeads,
    int headDim,
    cudaStream_t stream) {
    if (!qkv || !keyPages || !valuePages || !pageTable ||
        tokenCount <= 0 || startPosition < 0 || pageSize <= 0 ||
        qHeads <= 0 || kvHeads <= 0 || headDim <= 0 || (qHeads % kvHeads) != 0) {
        return cudaErrorInvalidValue;
    }

    int totalElements = tokenCount * kvHeads * headDim;
    int threads = 256;
    int blocks = (totalElements + threads - 1) / threads;
    garnet_text_paged_kv_write_kernel<<<blocks, threads, 0, stream>>>(
        qkv,
        keyPages,
        valuePages,
        pageTable,
        tokenCount,
        startPosition,
        pageSize,
        qHeads,
        kvHeads,
        headDim);
    return cudaGetLastError();
}
