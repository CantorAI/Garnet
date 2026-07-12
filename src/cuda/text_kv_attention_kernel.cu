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
        int maxSequenceLength,
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
        float* scores = shared;
        float* reduction = shared + maxSequenceLength;
        const int kvHead = qHead / (qHeads / kvHeads);
        if (sequenceLength <= 0 || sequenceLength > maxSequenceLength) {
            if (tid < headDim) {
                output[static_cast<size_t>(qHead) * headDim + tid] =
                    __float2bfloat16(0.0f);
            }
            return;
        }

        float maxValue = -FLT_MAX;
        for (int position = tid; position < sequenceLength; position += blockDim.x) {
            const float score = garnet_text_paged_kv_attention_score_bf16(
                q, keyPages, pageTable, position, pageSize,
                qHead, kvHead, kvHeads, headDim);
            scores[position] = score;
            maxValue = fmaxf(maxValue, score);
        }
        reduction[tid] = maxValue;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
            }
            __syncthreads();
        }
        maxValue = reduction[0];

        float sumValue = 0.0f;
        for (int position = tid; position < sequenceLength; position += blockDim.x) {
            const float weight = expf(scores[position] - maxValue);
            scores[position] = weight;
            sumValue += weight;
        }
        reduction[tid] = sumValue;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduction[tid] += reduction[tid + stride];
            __syncthreads();
        }
        sumValue = reduction[0];

        for (int dimension = tid; dimension < headDim; dimension += blockDim.x) {
            float value = 0.0f;
            for (int position = 0; position < sequenceLength; ++position) {
                const __nv_bfloat16* valueRow = garnet_paged_kv_row_bf16(
                    valuePages, pageTable, position, pageSize, kvHead, kvHeads, headDim);
                value += scores[position] / sumValue *
                    __bfloat162float(valueRow[dimension]);
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

    __global__ void garnet_text_paged_kv_score_split_bf16_kernel(
        const __nv_bfloat16* q,
        const __nv_bfloat16* keyPages,
        const int* pageTable,
        const int* sequenceLengthDevice,
        float* scores,
        int maxSequenceLength,
        int pageSize,
        int qHeads,
        int kvHeads,
        int headDim)
    {
        const int qHead = blockIdx.x;
        const int position = blockIdx.y * blockDim.x + threadIdx.x;
        const int sequenceLength = sequenceLengthDevice[0];
        if (position >= sequenceLength || sequenceLength > maxSequenceLength) return;
        const int kvHead = qHead / (qHeads / kvHeads);
        scores[static_cast<size_t>(qHead) * maxSequenceLength + position] =
            garnet_text_paged_kv_attention_score_bf16(
                q, keyPages, pageTable, position, pageSize,
                qHead, kvHead, kvHeads, headDim);
    }

    __global__ void garnet_text_paged_kv_softmax_split_bf16_kernel(
        float* scores,
        const int* sequenceLengthDevice,
        int maxSequenceLength)
    {
        const int qHead = blockIdx.x;
        const int tid = threadIdx.x;
        const int sequenceLength = sequenceLengthDevice[0];
        if (sequenceLength <= 0 || sequenceLength > maxSequenceLength) return;
        float* row = scores + static_cast<size_t>(qHead) * maxSequenceLength;
        extern __shared__ float reduction[];
        float maxValue = -FLT_MAX;
        for (int position = tid; position < sequenceLength; position += blockDim.x) {
            maxValue = fmaxf(maxValue, row[position]);
        }
        reduction[tid] = maxValue;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
            __syncthreads();
        }
        maxValue = reduction[0];
        float sumValue = 0.0f;
        for (int position = tid; position < sequenceLength; position += blockDim.x) {
            const float weight = expf(row[position] - maxValue);
            row[position] = weight;
            sumValue += weight;
        }
        reduction[tid] = sumValue;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduction[tid] += reduction[tid + stride];
            __syncthreads();
        }
        const float inverseSum = 1.0f / reduction[0];
        for (int position = tid; position < sequenceLength; position += blockDim.x) {
            row[position] *= inverseSum;
        }
    }

    __global__ void garnet_text_paged_kv_value_split_bf16_kernel(
        const __nv_bfloat16* valuePages,
        const int* pageTable,
        const int* sequenceLengthDevice,
        const float* scores,
        __nv_bfloat16* output,
        int maxSequenceLength,
        int pageSize,
        int qHeads,
        int kvHeads,
        int headDim)
    {
        const int qHead = blockIdx.x;
        const int dimension = threadIdx.x;
        const int sequenceLength = sequenceLengthDevice[0];
        if (dimension >= headDim || sequenceLength <= 0 || sequenceLength > maxSequenceLength) return;
        const int kvHead = qHead / (qHeads / kvHeads);
        const float* scoreRow = scores + static_cast<size_t>(qHead) * maxSequenceLength;
        float value = 0.0f;
        for (int position = 0; position < sequenceLength; ++position) {
            const __nv_bfloat16* valueRow = garnet_paged_kv_row_bf16(
                valuePages, pageTable, position, pageSize, kvHead, kvHeads, headDim);
            value = fmaf(scoreRow[position], __bfloat162float(valueRow[dimension]), value);
        }
        output[static_cast<size_t>(qHead) * headDim + dimension] = __float2bfloat16(value);
    }

    __global__ void garnet_text_paged_kv_value_partials_bf16_kernel(
        const __nv_bfloat16* valuePages,
        const int* pageTable,
        const int* sequenceLengthDevice,
        const float* scores,
        float* partials,
        int maxSequenceLength,
        int splitCount,
        int positionsPerSplit,
        int pageSize,
        int qHeads,
        int kvHeads,
        int headDim)
    {
        const int qHead = blockIdx.x;
        const int split = blockIdx.y;
        const int dimension = threadIdx.x;
        if (dimension >= headDim) return;
        const int sequenceLength = sequenceLengthDevice[0];
        const int start = split * positionsPerSplit;
        const int end = min(sequenceLength, start + positionsPerSplit);
        const int kvHead = qHead / (qHeads / kvHeads);
        const float* scoreRow = scores + static_cast<size_t>(qHead) * maxSequenceLength;
        float value = 0.0f;
        for (int position = start; position < end; ++position) {
            const __nv_bfloat16* valueRow = garnet_paged_kv_row_bf16(
                valuePages, pageTable, position, pageSize, kvHead, kvHeads, headDim);
            value = fmaf(scoreRow[position], __bfloat162float(valueRow[dimension]), value);
        }
        partials[(static_cast<size_t>(qHead) * splitCount + split) * headDim + dimension] = value;
    }

    __global__ void garnet_text_paged_kv_value_reduce_bf16_kernel(
        const float* partials,
        __nv_bfloat16* output,
        int splitCount,
        int headDim)
    {
        const int qHead = blockIdx.x;
        const int dimension = threadIdx.x;
        if (dimension >= headDim) return;
        float value = 0.0f;
        const float* row = partials + static_cast<size_t>(qHead) * splitCount * headDim;
        for (int split = 0; split < splitCount; ++split) {
            value += row[static_cast<size_t>(split) * headDim + dimension];
        }
        output[static_cast<size_t>(qHead) * headDim + dimension] = __float2bfloat16(value);
    }

    constexpr int kFlashDecodeHeadDim = 128;
    constexpr int kFlashDecodeWarps = 4;
    constexpr int kFlashDecodeThreads = kFlashDecodeWarps * 32;
    constexpr int kFlashDecodePositionsPerSplit = 128;

    __device__ __forceinline__ float garnet_warp_sum(float value)
    {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffU, value, offset);
        }
        return __shfl_sync(0xffffffffU, value, 0);
    }

    __global__ void garnet_text_paged_kv_write_decode_batched_bf16_kernel(
        const __nv_bfloat16* qkv,
        __nv_bfloat16* keyPages,
        __nv_bfloat16* valuePages,
        const int* pageTables,
        const int* slotPositions,
        int batchSize,
        int maxLogicalPages,
        int pageSize,
        int qHeads,
        int kvHeads,
        int headDim)
    {
        const int kvWidth = kvHeads * headDim;
        const int linear = blockIdx.x * blockDim.x + threadIdx.x;
        if (linear >= batchSize * kvWidth) return;
        const int batch = linear / kvWidth;
        const int within = linear - batch * kvWidth;
        const int kvHead = within / headDim;
        const int dimension = within - kvHead * headDim;
        const int slot = slotPositions[batch];
        const int logicalPage = slot / pageSize;
        if (logicalPage < 0 || logicalPage >= maxLogicalPages) return;
        const int physicalPage = pageTables[batch * maxLogicalPages + logicalPage];
        const int pageOffset = slot - logicalPage * pageSize;
        const int qWidth = qHeads * headDim;
        const int qkvWidth = qWidth + 2 * kvWidth;
        const __nv_bfloat16* row = qkv + static_cast<size_t>(batch) * qkvWidth;
        const size_t cacheOffset =
            (((static_cast<size_t>(physicalPage) * pageSize + pageOffset) * kvHeads + kvHead) * headDim) +
            dimension;
        keyPages[cacheOffset] = row[qWidth + kvHead * headDim + dimension];
        valuePages[cacheOffset] = row[qWidth + kvWidth + kvHead * headDim + dimension];
    }

    __global__ void garnet_text_paged_kv_flash_partials_bf16_kernel(
        const __nv_bfloat16* qkv,
        const __nv_bfloat16* keyPages,
        const __nv_bfloat16* valuePages,
        const int* pageTables,
        const int* contextLengths,
        float* partialStats,
        float* partialOutputs,
        int maxLogicalPages,
        int splitCount,
        int pageSize,
        int qHeads,
        int kvHeads)
    {
        const int batch = blockIdx.z;
        const int qHead = blockIdx.x;
        const int split = blockIdx.y;
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const int contextLength = contextLengths[batch];
        const int start = split * kFlashDecodePositionsPerSplit;
        const int end = min(contextLength, start + kFlashDecodePositionsPerSplit);
        const int kvHead = qHead / (qHeads / kvHeads);
        const int qWidth = qHeads * kFlashDecodeHeadDim;
        const int kvWidth = kvHeads * kFlashDecodeHeadDim;
        const int qkvWidth = qWidth + 2 * kvWidth;
        const __nv_bfloat16* qRow = qkv +
            static_cast<size_t>(batch) * qkvWidth + qHead * kFlashDecodeHeadDim;
        const int* pageTable = pageTables + batch * maxLogicalPages;

        float runningMax = -FLT_MAX;
        float runningSum = 0.0f;
        float output0 = 0.0f;
        float output1 = 0.0f;
        float output2 = 0.0f;
        float output3 = 0.0f;
        constexpr float scale = 0.08838834764831845f;

        for (int position = start + warp; position < end; position += kFlashDecodeWarps) {
            const __nv_bfloat16* keyRow = garnet_paged_kv_row_bf16(
                keyPages, pageTable, position, pageSize, kvHead, kvHeads,
                kFlashDecodeHeadDim);
            float dot =
                __bfloat162float(qRow[lane]) * __bfloat162float(keyRow[lane]) +
                __bfloat162float(qRow[lane + 32]) * __bfloat162float(keyRow[lane + 32]) +
                __bfloat162float(qRow[lane + 64]) * __bfloat162float(keyRow[lane + 64]) +
                __bfloat162float(qRow[lane + 96]) * __bfloat162float(keyRow[lane + 96]);
            const float score = garnet_warp_sum(dot) * scale;
            const float nextMax = fmaxf(runningMax, score);
            const float previousScale = runningSum == 0.0f ? 0.0f : __expf(runningMax - nextMax);
            const float weight = __expf(score - nextMax);
            const __nv_bfloat16* valueRow = garnet_paged_kv_row_bf16(
                valuePages, pageTable, position, pageSize, kvHead, kvHeads,
                kFlashDecodeHeadDim);
            output0 = output0 * previousScale + weight * __bfloat162float(valueRow[lane]);
            output1 = output1 * previousScale + weight * __bfloat162float(valueRow[lane + 32]);
            output2 = output2 * previousScale + weight * __bfloat162float(valueRow[lane + 64]);
            output3 = output3 * previousScale + weight * __bfloat162float(valueRow[lane + 96]);
            runningSum = runningSum * previousScale + weight;
            runningMax = nextMax;
        }

        __shared__ float warpMax[kFlashDecodeWarps];
        __shared__ float warpSum[kFlashDecodeWarps];
        __shared__ float warpOutput[kFlashDecodeWarps][kFlashDecodeHeadDim];
        if (lane == 0) {
            warpMax[warp] = runningMax;
            warpSum[warp] = runningSum;
        }
        warpOutput[warp][lane] = output0;
        warpOutput[warp][lane + 32] = output1;
        warpOutput[warp][lane + 64] = output2;
        warpOutput[warp][lane + 96] = output3;
        __syncthreads();

        const int dimension = threadIdx.x;
        float splitMax = -FLT_MAX;
        for (int sourceWarp = 0; sourceWarp < kFlashDecodeWarps; ++sourceWarp) {
            if (warpSum[sourceWarp] > 0.0f) splitMax = fmaxf(splitMax, warpMax[sourceWarp]);
        }
        float splitSum = 0.0f;
        float splitOutput = 0.0f;
        for (int sourceWarp = 0; sourceWarp < kFlashDecodeWarps; ++sourceWarp) {
            if (warpSum[sourceWarp] <= 0.0f) continue;
            const float mergeScale = __expf(warpMax[sourceWarp] - splitMax);
            splitSum += mergeScale * warpSum[sourceWarp];
            splitOutput += mergeScale * warpOutput[sourceWarp][dimension];
        }
        const size_t stateIndex =
            (static_cast<size_t>(batch) * qHeads + qHead) * splitCount + split;
        if (dimension == 0) {
            partialStats[stateIndex * 2] = splitMax;
            partialStats[stateIndex * 2 + 1] = splitSum;
        }
        partialOutputs[stateIndex * kFlashDecodeHeadDim + dimension] = splitOutput;
    }

    __global__ void garnet_text_paged_kv_flash_reduce_bf16_kernel(
        const float* partialStats,
        const float* partialOutputs,
        __nv_bfloat16* output,
        const int* contextLengths,
        int splitCount,
        int qHeads)
    {
        const int batchHead = blockIdx.x;
        const int batch = batchHead / qHeads;
        const int qHead = batchHead - batch * qHeads;
        const int activeSplits = min(
            splitCount,
            (contextLengths[batch] + kFlashDecodePositionsPerSplit - 1) /
                kFlashDecodePositionsPerSplit);
        __shared__ float mergeScales[32];
        __shared__ float inverseSum;
        if (threadIdx.x == 0) {
            float maximum = -FLT_MAX;
            for (int split = 0; split < activeSplits; ++split) {
                const size_t stateIndex =
                    static_cast<size_t>(batchHead) * splitCount + split;
                maximum = fmaxf(maximum, partialStats[stateIndex * 2]);
            }
            float sum = 0.0f;
            for (int split = 0; split < activeSplits; ++split) {
                const size_t stateIndex =
                    static_cast<size_t>(batchHead) * splitCount + split;
                const float scale = __expf(partialStats[stateIndex * 2] - maximum);
                mergeScales[split] = scale;
                sum += scale * partialStats[stateIndex * 2 + 1];
            }
            inverseSum = sum > 0.0f ? 1.0f / sum : 0.0f;
        }
        __syncthreads();
        const int dimension = threadIdx.x;
        float value = 0.0f;
        for (int split = 0; split < activeSplits; ++split) {
            const size_t stateIndex =
                static_cast<size_t>(batchHead) * splitCount + split;
            value += mergeScales[split] *
                partialOutputs[stateIndex * kFlashDecodeHeadDim + dimension];
        }
        output[(static_cast<size_t>(batch) * qHeads + qHead) * kFlashDecodeHeadDim + dimension] =
            __float2bfloat16(value * inverseSum);
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
        qHeads, threads,
        (static_cast<size_t>(sequenceLength) + threads) * sizeof(float), stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q),
        reinterpret_cast<const __nv_bfloat16*>(keyPages),
        reinterpret_cast<const __nv_bfloat16*>(valuePages),
        pageTable,
        reinterpret_cast<__nv_bfloat16*>(output),
        sequenceLength, sequenceLength, pageSize,
        qHeads, kvHeads, headDim, nullptr);
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
    int maxSequenceLength,
    int pageSize,
    int qHeads,
    int kvHeads,
    int headDim,
    cudaStream_t stream)
{
    if (!qkv || !keyPages || !valuePages || !pageTable || !contextLength ||
        !slotPosition || !output || maxSequenceLength <= 0 || pageSize <= 0 ||
        qHeads <= 0 || kvHeads <= 0 ||
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
        qHeads, threads,
        (static_cast<size_t>(maxSequenceLength) + threads) * sizeof(float), stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv),
        reinterpret_cast<const __nv_bfloat16*>(keyPages),
        reinterpret_cast<const __nv_bfloat16*>(valuePages),
        pageTable,
        reinterpret_cast<__nv_bfloat16*>(output),
        0, maxSequenceLength, pageSize,
        qHeads, kvHeads, headDim, contextLength);
    return cudaGetLastError();
}

extern "C" cudaError_t runTextPagedKVDecodeSplitKBF16DeviceMetadata(
    const bfloat16* qkv,
    bfloat16* keyPages,
    bfloat16* valuePages,
    const int* pageTable,
    const int* contextLength,
    const int* slotPosition,
    bfloat16* output,
    float* scores,
    float* valuePartials,
    int maxSequenceLength,
    int pageSize,
    int qHeads,
    int kvHeads,
    int headDim,
    int splitValue,
    cudaStream_t stream)
{
    if (!qkv || !keyPages || !valuePages || !pageTable || !contextLength ||
        !slotPosition || !output || !scores || !valuePartials || maxSequenceLength <= 0 || pageSize <= 0 ||
        qHeads <= 0 || kvHeads <= 0 || headDim <= 0 || qHeads % kvHeads != 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int writeThreads = 256;
    const int writeElements = kvHeads * headDim;
    garnet_text_paged_kv_write_bf16_kernel<<<
        (writeElements + writeThreads - 1) / writeThreads, writeThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv),
        reinterpret_cast<__nv_bfloat16*>(keyPages),
        reinterpret_cast<__nv_bfloat16*>(valuePages),
        pageTable, 1, 0, pageSize, qHeads, kvHeads, headDim, slotPosition);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;

    constexpr int scoreThreads = 128;
    const dim3 scoreGrid(qHeads, (maxSequenceLength + scoreThreads - 1) / scoreThreads);
    garnet_text_paged_kv_score_split_bf16_kernel<<<scoreGrid, scoreThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv),
        reinterpret_cast<const __nv_bfloat16*>(keyPages),
        pageTable, contextLength, scores, maxSequenceLength,
        pageSize, qHeads, kvHeads, headDim);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;

    constexpr int softmaxThreads = 256;
    garnet_text_paged_kv_softmax_split_bf16_kernel<<<
        qHeads, softmaxThreads, softmaxThreads * sizeof(float), stream>>>(
        scores, contextLength, maxSequenceLength);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;

    const int valueThreads = 1 << static_cast<int>(ceilf(log2f(static_cast<float>(headDim))));
    if (splitValue) {
        constexpr int positionsPerSplit = 128;
        const int splitCount = (maxSequenceLength + positionsPerSplit - 1) / positionsPerSplit;
        garnet_text_paged_kv_value_partials_bf16_kernel<<<
            dim3(qHeads, splitCount), valueThreads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(valuePages), pageTable, contextLength,
            scores, valuePartials, maxSequenceLength, splitCount, positionsPerSplit,
            pageSize, qHeads, kvHeads, headDim);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        garnet_text_paged_kv_value_reduce_bf16_kernel<<<qHeads, valueThreads, 0, stream>>>(
            valuePartials, reinterpret_cast<__nv_bfloat16*>(output), splitCount, headDim);
    }
    else {
        garnet_text_paged_kv_value_split_bf16_kernel<<<qHeads, valueThreads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(valuePages),
            pageTable, contextLength, scores,
            reinterpret_cast<__nv_bfloat16*>(output),
            maxSequenceLength, pageSize, qHeads, kvHeads, headDim);
    }
    return cudaGetLastError();
}

extern "C" cudaError_t runTextPagedKVDecodeFlashBF16DeviceMetadata(
    const bfloat16* qkv,
    bfloat16* keyPages,
    bfloat16* valuePages,
    const int* pageTables,
    const int* contextLengths,
    const int* slotPositions,
    bfloat16* output,
    float* partialStats,
    float* partialOutputs,
    int batchSize,
    int maxSequenceLength,
    int pageSize,
    int qHeads,
    int kvHeads,
    int headDim,
    cudaStream_t stream)
{
    if (!qkv || !keyPages || !valuePages || !pageTables || !contextLengths ||
        !slotPositions || !output || !partialStats || !partialOutputs ||
        batchSize <= 0 || maxSequenceLength <= 0 || pageSize <= 0 ||
        qHeads <= 0 || kvHeads <= 0 || qHeads % kvHeads != 0 ||
        headDim != kFlashDecodeHeadDim) {
        return cudaErrorInvalidValue;
    }
    const int maxLogicalPages = (maxSequenceLength + pageSize - 1) / pageSize;
    const int splitCount =
        (maxSequenceLength + kFlashDecodePositionsPerSplit - 1) /
        kFlashDecodePositionsPerSplit;
    if (splitCount > 32) return cudaErrorNotSupported;

    constexpr int writeThreads = 256;
    const int writeElements = batchSize * kvHeads * headDim;
    garnet_text_paged_kv_write_decode_batched_bf16_kernel<<<
        (writeElements + writeThreads - 1) / writeThreads,
        writeThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv),
        reinterpret_cast<__nv_bfloat16*>(keyPages),
        reinterpret_cast<__nv_bfloat16*>(valuePages),
        pageTables, slotPositions, batchSize, maxLogicalPages, pageSize,
        qHeads, kvHeads, headDim);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;

    garnet_text_paged_kv_flash_partials_bf16_kernel<<<
        dim3(qHeads, splitCount, batchSize), kFlashDecodeThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv),
        reinterpret_cast<const __nv_bfloat16*>(keyPages),
        reinterpret_cast<const __nv_bfloat16*>(valuePages),
        pageTables, contextLengths, partialStats, partialOutputs,
        maxLogicalPages, splitCount, pageSize, qHeads, kvHeads);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;

    garnet_text_paged_kv_flash_reduce_bf16_kernel<<<
        batchSize * qHeads, kFlashDecodeHeadDim, 0, stream>>>(
        partialStats, partialOutputs,
        reinterpret_cast<__nv_bfloat16*>(output), contextLengths,
        splitCount, qHeads);
    return cudaGetLastError();
}
