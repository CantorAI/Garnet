#include "cuda_lib.h"

#include <cuda_bf16.h>
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

namespace
{
    __device__ const __nv_bfloat16* garnet_paged_kv_row_bf16(
        const __nv_bfloat16* pages,
        const int* pageTable,
        int position,
        int pageSize,
        int kvHead,
        int kvHeads,
        int headDim)
    {
        const int logicalPage = position / pageSize;
        const int pageOffset = position - logicalPage * pageSize;
        const int physicalPage = pageTable[logicalPage];
        return pages +
            (((static_cast<size_t>(physicalPage) * pageSize + pageOffset) * kvHeads + kvHead) * headDim);
    }

    __device__ float garnet_text_paged_kv_attention_score_bf16(
        const __nv_bfloat16* q,
        const __nv_bfloat16* keyPages,
        const int* pageTable,
        int position,
        int pageSize,
        int qHead,
        int kvHead,
        int kvHeads,
        int headDim)
    {
        float dot = 0.0f;
        const __nv_bfloat16* qRow = q + static_cast<size_t>(qHead) * headDim;
        const __nv_bfloat16* kRow = garnet_paged_kv_row_bf16(
            keyPages, pageTable, position, pageSize, kvHead, kvHeads, headDim);
        for (int dimension = 0; dimension < headDim; ++dimension) {
            dot += __bfloat162float(qRow[dimension]) * __bfloat162float(kRow[dimension]);
        }
        return dot * rsqrtf(static_cast<float>(headDim));
    }

    __global__ void garnet_text_paged_kv_cached_attention_bf16_kernel(
        const __nv_bfloat16* q,
        const __nv_bfloat16* keyPages,
        const __nv_bfloat16* valuePages,
        const int* pageTable,
        __nv_bfloat16* output,
        int sequenceLength,
        int pageSize,
        int qHeads,
        int kvHeads,
        int headDim,
        const int* sequenceLengthDevice)
    {
        if (sequenceLengthDevice) sequenceLength = sequenceLengthDevice[0];
        const int qHead = blockIdx.x;
        const int tid = threadIdx.x;
        extern __shared__ float shared[];
        float* reduceMax = shared;
        float* reduceSum = shared + blockDim.x;
        const int kvHead = qHead / (qHeads / kvHeads);

        float maxValue = -FLT_MAX;
        for (int position = tid; position < sequenceLength; position += blockDim.x) {
            maxValue = fmaxf(maxValue, garnet_text_paged_kv_attention_score_bf16(
                q, keyPages, pageTable, position, pageSize, qHead, kvHead, kvHeads, headDim));
        }
        reduceMax[tid] = maxValue;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduceMax[tid] = fmaxf(reduceMax[tid], reduceMax[tid + stride]);
            __syncthreads();
        }
        maxValue = reduceMax[0];

        float sumValue = 0.0f;
        for (int position = tid; position < sequenceLength; position += blockDim.x) {
            const float score = garnet_text_paged_kv_attention_score_bf16(
                q, keyPages, pageTable, position, pageSize, qHead, kvHead, kvHeads, headDim);
            sumValue += expf(score - maxValue);
        }
        reduceSum[tid] = sumValue;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduceSum[tid] += reduceSum[tid + stride];
            __syncthreads();
        }
        sumValue = reduceSum[0];

        for (int dimension = tid; dimension < headDim; dimension += blockDim.x) {
            float value = 0.0f;
            for (int position = 0; position < sequenceLength; ++position) {
                const float score = garnet_text_paged_kv_attention_score_bf16(
                    q, keyPages, pageTable, position, pageSize, qHead, kvHead, kvHeads, headDim);
                const __nv_bfloat16* valueRow = garnet_paged_kv_row_bf16(
                    valuePages, pageTable, position, pageSize, kvHead, kvHeads, headDim);
                value += expf(score - maxValue) / sumValue * __bfloat162float(valueRow[dimension]);
            }
            output[static_cast<size_t>(qHead) * headDim + dimension] = __float2bfloat16(value);
        }
    }

    __global__ void garnet_text_paged_kv_write_bf16_kernel(
        const __nv_bfloat16* qkv,
        __nv_bfloat16* keyPages,
        __nv_bfloat16* valuePages,
        const int* pageTable,
        int tokenCount,
        int startPosition,
        int pageSize,
        int qHeads,
        int kvHeads,
        int headDim,
        const int* startPositionDevice)
    {
        if (startPositionDevice) startPosition = startPositionDevice[0];
        const int linear = blockIdx.x * blockDim.x + threadIdx.x;
        const int elementsPerToken = kvHeads * headDim;
        if (linear >= tokenCount * elementsPerToken) return;
        const int token = linear / elementsPerToken;
        const int within = linear - token * elementsPerToken;
        const int kvHead = within / headDim;
        const int dimension = within - kvHead * headDim;
        const int logicalPosition = startPosition + token;
        const int logicalPage = logicalPosition / pageSize;
        const int pageOffset = logicalPosition - logicalPage * pageSize;
        const int physicalPage = pageTable[logicalPage];
        const int qWidth = qHeads * headDim;
        const int kvWidth = kvHeads * headDim;
        const __nv_bfloat16* qkvRow = qkv +
            static_cast<size_t>(token) * (qWidth + 2 * kvWidth);
        const size_t cacheOffset =
            (((static_cast<size_t>(physicalPage) * pageSize + pageOffset) * kvHeads + kvHead) * headDim) +
            dimension;
        keyPages[cacheOffset] = qkvRow[qWidth + kvHead * headDim + dimension];
        valuePages[cacheOffset] = qkvRow[qWidth + kvWidth + kvHead * headDim + dimension];
    }
}

extern "C" cudaError_t runTextPagedKVCachedAttentionBF16(
    const bfloat16* q,
    const bfloat16* keyPages,
    const bfloat16* valuePages,
    const int* pageTable,
    bfloat16* output,
    int sequenceLength,
    int pageSize,
    int qHeads,
    int kvHeads,
    int headDim,
    cudaStream_t stream)
{
    if (!q || !keyPages || !valuePages || !pageTable || !output ||
        sequenceLength <= 0 || pageSize <= 0 || qHeads <= 0 || kvHeads <= 0 ||
        headDim <= 0 || qHeads % kvHeads != 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int threads = 256;
    garnet_text_paged_kv_cached_attention_bf16_kernel<<<
        qHeads, threads, threads * 2 * sizeof(float), stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q),
        reinterpret_cast<const __nv_bfloat16*>(keyPages),
        reinterpret_cast<const __nv_bfloat16*>(valuePages),
        pageTable,
        reinterpret_cast<__nv_bfloat16*>(output),
        sequenceLength, pageSize, qHeads, kvHeads, headDim, nullptr);
    return cudaGetLastError();
}

extern "C" cudaError_t runTextPagedKVWriteBF16(
    const bfloat16* qkv,
    bfloat16* keyPages,
    bfloat16* valuePages,
    const int* pageTable,
    int tokenCount,
    int startPosition,
    int pageSize,
    int qHeads,
    int kvHeads,
    int headDim,
    cudaStream_t stream)
{
    if (!qkv || !keyPages || !valuePages || !pageTable || tokenCount <= 0 ||
        startPosition < 0 || pageSize <= 0 || qHeads <= 0 || kvHeads <= 0 ||
        headDim <= 0 || qHeads % kvHeads != 0) {
        return cudaErrorInvalidValue;
    }
    const int elements = tokenCount * kvHeads * headDim;
    constexpr int threads = 256;
    garnet_text_paged_kv_write_bf16_kernel<<<(elements + threads - 1) / threads, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv),
        reinterpret_cast<__nv_bfloat16*>(keyPages),
        reinterpret_cast<__nv_bfloat16*>(valuePages),
        pageTable,
        tokenCount, startPosition, pageSize, qHeads, kvHeads, headDim, nullptr);
    return cudaGetLastError();
}

extern "C" cudaError_t runTextPagedKVWriteBF16DeviceStart(
    const bfloat16* qkv,
    bfloat16* keyPages,
    bfloat16* valuePages,
    const int* pageTable,
    const int* startPosition,
    int tokenCount,
    int pageSize,
    int qHeads,
    int kvHeads,
    int headDim,
    cudaStream_t stream)
{
    if (!qkv || !keyPages || !valuePages || !pageTable || !startPosition ||
        tokenCount <= 0 || pageSize <= 0 || qHeads <= 0 || kvHeads <= 0 ||
        headDim <= 0 || qHeads % kvHeads != 0) {
        return cudaErrorInvalidValue;
    }
    const int elements = tokenCount * kvHeads * headDim;
    constexpr int threads = 256;
    garnet_text_paged_kv_write_bf16_kernel<<<
        (elements + threads - 1) / threads, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv),
        reinterpret_cast<__nv_bfloat16*>(keyPages),
        reinterpret_cast<__nv_bfloat16*>(valuePages),
        pageTable,
        tokenCount, 0, pageSize, qHeads, kvHeads, headDim, startPosition);
    return cudaGetLastError();
}

extern "C" cudaError_t runTextPagedKVDecodeBF16DeviceMetadata(
    const bfloat16* qkv,
    bfloat16* keyPages,
    bfloat16* valuePages,
    const int* pageTable,
    const int* contextLength,
    const int* slotPosition,
    bfloat16* output,
    int pageSize,
    int qHeads,
    int kvHeads,
    int headDim,
    cudaStream_t stream)
{
    if (!qkv || !keyPages || !valuePages || !pageTable || !contextLength ||
        !slotPosition || !output || pageSize <= 0 || qHeads <= 0 || kvHeads <= 0 ||
        headDim <= 0 || qHeads % kvHeads != 0) {
        return cudaErrorInvalidValue;
    }
    const int writeElements = kvHeads * headDim;
    constexpr int threads = 256;
    garnet_text_paged_kv_write_bf16_kernel<<<
        (writeElements + threads - 1) / threads, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv),
        reinterpret_cast<__nv_bfloat16*>(keyPages),
        reinterpret_cast<__nv_bfloat16*>(valuePages),
        pageTable,
        1, 0, pageSize, qHeads, kvHeads, headDim, slotPosition);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    garnet_text_paged_kv_cached_attention_bf16_kernel<<<
        qHeads, threads, threads * 2 * sizeof(float), stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv),
        reinterpret_cast<const __nv_bfloat16*>(keyPages),
        reinterpret_cast<const __nv_bfloat16*>(valuePages),
        pageTable,
        reinterpret_cast<__nv_bfloat16*>(output),
        0, pageSize, qHeads, kvHeads, headDim, contextLength);
    return cudaGetLastError();
}
