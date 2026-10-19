// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_kernels.h"
#include "gpt_oss_router_dispatch.h"
#ifdef GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL
#include "gpt_oss_flash_prefill.h"
#endif
#include <cuda_bf16.h>
#include <mma.h>
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstring>

namespace Garnet {
namespace {
__device__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ float warpSum(float value) {
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}
template<int Threads>
__global__ void rmsNorm(const float* x, const float* weight, float* y,
    int rows, int hidden, float epsilon) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float sum = 0.f;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        const float value = bf(x[size_t(row) * hidden + d]);
        sum += value * value;
    }
    sum = warpSum(sum);
    __shared__ float partial[Threads / 32];
    if (!lane) partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < Threads / 32 ? partial[lane] : 0.f;
        sum = warpSum(sum);
        if (!lane) partial[0] = rsqrtf(sum / hidden + epsilon);
    }
    __syncthreads();
    const float inverse = partial[0];
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        const size_t index = size_t(row) * hidden + d;
        const float value = bf(x[index]);
        y[index] = bf((value * inverse) * weight[d]);
    }
    (void)rows;
}
__device__ float fp4(const unsigned char* blocks, const unsigned char* scales,
    size_t row, int column, int width) {
    const unsigned char byte = blocks[row * (width / 2) + column / 2];
    const int code = (column & 1) ? byte >> 4 : byte & 15;
    const unsigned char exponent = scales[row * (width / 32) + column / 32];
    // E8M0 255 is NaN, not an ordinary exponent.
    if (exponent == 255) return nanf("");
    // Every scaled E2M1 value is exactly representable in BF16 (including
    // its subnormals), or overflows to infinity. Construct the IEEE bits
    // directly instead of a dynamically indexed per-thread LUT.
    const unsigned sign = unsigned(code & 8) << 28;
    const int magnitude = code & 7;
    if (!magnitude) return __uint_as_float(sign);
    const int adjusted = int(exponent) + (magnitude >> 1) - 1;
    const unsigned mantissa = magnitude >= 3 && (magnitude & 1) ? 0x00400000u : 0u;
    if (adjusted >= 255) return __uint_as_float(sign | 0x7f800000u);
    const unsigned bits = adjusted > 0
        ? (unsigned(adjusted) << 23) | mantissa
        : (0x00800000u | mantissa) >> (1 - adjusted);
    return __uint_as_float(sign | bits);
}
#ifdef GARNET_GPT_OSS_KERNEL_TEST
__global__ void testMxfp4Decode(const unsigned char* blocks, const unsigned char* scales,
    float* output, int rows) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows * 32) output[index] = fp4(blocks, scales, index / 32, index % 32, 32);
}
#endif
__global__ void rope(const float* x, const std::int64_t* positions, float* y,
    int tokens, GptOssOptions o) {
    const int packed = (o.qHeads + 2 * o.kvHeads) * o.headDim;
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(tokens) * packed) return;
    const int column = index % packed, token = index / packed;
    if (column >= (o.qHeads + o.kvHeads) * o.headDim) { y[index] = x[index]; return; }
    const int d = column % o.headDim, half = o.headDim / 2, frequency = d % half;
    const float pi = 3.14159265358979323846f;
    float inv = powf(o.theta, -2.f * frequency / o.headDim), concentration = 1;
    if (o.factor > 1) {
        const float low = half * logf(o.initialContext / (o.betaFast * 2 * pi)) / logf(o.theta);
        const float high = half * logf(o.initialContext / (o.betaSlow * 2 * pi)) / logf(o.theta);
        const float ramp = fminf(1, fmaxf(0, (frequency - low) / (high - low)));
        inv *= (1 - ramp) + ramp / o.factor;
        concentration = 1 + .1f * logf(o.factor);
    }
    const float angle = float(positions[token]) * inv;
    const float cosine = bf(cosf(angle) * concentration), sine = bf(sinf(angle) * concentration);
    const float rotated = d < half ? -x[index + half] : x[index - half];
    y[index] = bf(bf(x[index] * cosine) + bf(rotated * sine));
}
__device__ size_t cacheOffset(const int* table, int logicalPages, int physicalPages,
    int batch, int position, int head, int dimension, GptOssOptions o) {
    if (position < 0 || position / o.pageSize >= logicalPages) return SIZE_MAX;
    const int page = table[batch * logicalPages + position / o.pageSize];
    if (page < 0 || page >= physicalPages) return SIZE_MAX;
    return (((size_t(o.layer) * physicalPages + page) * o.pageSize + position % o.pageSize)
        * o.kvHeads + head) * o.headDim + dimension;
}
__global__ void writeKV(const float* qkv, __nv_bfloat16* keys, __nv_bfloat16* values,
    const int* table, const int* starts, const int* active, int batch, int tokens,
    int logicalPages, int physicalPages, GptOssOptions o) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int kvWidth = o.kvHeads * o.headDim, packed = (o.qHeads + 2 * o.kvHeads) * o.headDim;
    if (index >= size_t(batch) * tokens * kvWidth) return;
    const int row = index / kvWidth, b = row / tokens, t = row % tokens, d = index % kvWidth;
    if (!active[b]) return;
    const auto offset = cacheOffset(table, logicalPages, physicalPages, b, starts[b] + t,
        d / o.headDim, d % o.headDim, o);
    if (offset == SIZE_MAX) return;
    keys[offset] = __float2bfloat16(qkv[size_t(row) * packed + o.qHeads * o.headDim + d]);
    values[offset] = __float2bfloat16(qkv[size_t(row) * packed + (o.qHeads + o.kvHeads) * o.headDim + d]);
}
// One warp per query head keeps paged KV reads coalesced. The learned sink
// contributes to the softmax denominator, but never to the value accumulator.
template<bool FastExp>
__global__ void attention(const float* qkv, const __nv_bfloat16* keys,
    const __nv_bfloat16* values, const int* table, const int* lengths,
    const int* starts, const int* active, const float* sinks, float* output,
    int batch, int tokens, int logicalPages, int physicalPages, GptOssOptions o) {
    const int lane = threadIdx.x & 31;
    const int task = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (task >= batch * tokens * o.qHeads) return;
    const int head = task % o.qHeads, row = task / o.qHeads, b = row / tokens, t = row % tokens;
    const int packed = (o.qHeads + 2 * o.kvHeads) * o.headDim;
    float accum[4] = {};
    const int end = o.prefill ? starts[b] + t + 1 : lengths[b];
    const int begin = o.window ? max(0, end - o.window) : 0;
    float maximum = sinks[head], sum = 1;
    if (active[b] && end > 0 && end <= logicalPages * o.pageSize) {
        for (int p = begin; p < end; ++p) {
            const size_t offset = cacheOffset(table, logicalPages, physicalPages, b, p,
                head / (o.qHeads / o.kvHeads), 0, o);
            if (offset == SIZE_MAX) continue;
            float score = 0;
            for (int d = lane; d < o.headDim; d += 32)
                score += qkv[size_t(row) * packed + head * o.headDim + d] * __bfloat162float(keys[offset + d]);
            score = warpSum(score);
            score = __shfl_sync(0xffffffff, score, 0);
            score *= rsqrtf(float(o.headDim));
            const float next = fmaxf(maximum, score);
            const float old = FastExp ? __expf(maximum - next) : expf(maximum - next);
            const float weight = FastExp ? __expf(score - next) : expf(score - next);
            sum = sum * old + weight;
            for (int d = lane; d < o.headDim; d += 32)
                accum[d / 32] = accum[d / 32] * old + weight * __bfloat162float(values[offset + d]);
            maximum = next;
        }
    }
    for (int d = lane; d < o.headDim; d += 32)
        output[size_t(row) * o.qHeads * o.headDim + head * o.headDim + d] = bf(accum[d / 32] / sum);
}
// Prefill processes eight adjacent queries for one head in a CTA. Their KV
// ranges almost completely overlap, so stage sixteen paged entries once in
// shared memory and reuse them across the query warps. Keep the original
// online softmax order (including the learned sink) for each query.
template<bool FastExp, int QueryTile>
__global__ void tiledPrefillAttention64(const float* qkv,
    const __nv_bfloat16* keys, const __nv_bfloat16* values,
    const int* table, const int* starts, const int* active,
    const float* sinks, float* output, int tokens, int logicalPages,
    int physicalPages, GptOssOptions o) {
    static_assert(QueryTile == 8 || QueryTile == 16);
    constexpr int KeyTile = 16, Dim = 64;
    __shared__ __nv_bfloat16 tileK[KeyTile][Dim], tileV[KeyTile][Dim];
    __shared__ int tileValid[KeyTile];
    __shared__ size_t tileOffset[KeyTile];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int queryTiles = (tokens + QueryTile - 1) / QueryTile;
    const int head = blockIdx.x % o.qHeads;
    const int queryTile = (blockIdx.x / o.qHeads) % queryTiles;
    const int b = blockIdx.x / (o.qHeads * queryTiles);
    const int first = queryTile * QueryTile;
    const int t = first + warp;
    const int last = min(tokens - 1, first + QueryTile - 1);
    const int capacity = logicalPages * o.pageSize;
    const int end = starts[b] + t + 1;
    const int begin = o.window ? max(0, end - o.window) : 0;
    const bool validQuery = active[b] && t < tokens && end > 0 && end <= capacity;
    const int firstEnd = starts[b] + first + 1;
    const int firstBegin = o.window ? max(0, firstEnd - o.window) : 0;
    const int lastEnd = min(capacity, starts[b] + last + 1);
    const int packed = (o.qHeads + 2 * o.kvHeads) * Dim;
    float query[2] = {};
    if (validQuery) {
        const size_t base = size_t(b * tokens + t) * packed + head * Dim + lane;
        query[0] = qkv[base];
        query[1] = qkv[base + 32];
    }
    float accum[2] = {}, maximum = sinks[head], sum = 1.f;
    const int kvHead = head / (o.qHeads / o.kvHeads);
    if (active[b]) for (int p0 = firstBegin; p0 < lastEnd; p0 += KeyTile) {
        if (threadIdx.x < KeyTile) {
            const int p = p0 + threadIdx.x;
            const size_t offset = p < lastEnd ? cacheOffset(table, logicalPages,
                physicalPages, b, p, kvHead, 0, o) : SIZE_MAX;
            tileOffset[threadIdx.x] = offset;
            tileValid[threadIdx.x] = offset != SIZE_MAX;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < KeyTile * Dim; i += blockDim.x) {
            const int k = i / Dim, d = i % Dim;
            const size_t offset = tileOffset[k];
            tileK[k][d] = offset == SIZE_MAX ? __float2bfloat16(0.f) : keys[offset + d];
            tileV[k][d] = offset == SIZE_MAX ? __float2bfloat16(0.f) : values[offset + d];
        }
        __syncthreads();
        if (validQuery) for (int k = 0; k < KeyTile; ++k) {
            const int p = p0 + k;
            if (p < begin || p >= end || !tileValid[k]) continue;
            float score = query[0] * __bfloat162float(tileK[k][lane]);
            score += query[1] * __bfloat162float(tileK[k][lane + 32]);
            score = __shfl_sync(0xffffffff, warpSum(score), 0) * rsqrtf(float(Dim));
            const float next = fmaxf(maximum, score);
            const float old = FastExp ? __expf(maximum - next) : expf(maximum - next);
            const float weight = FastExp ? __expf(score - next) : expf(score - next);
            sum = sum * old + weight;
            accum[0] = accum[0] * old + weight * __bfloat162float(tileV[k][lane]);
            accum[1] = accum[1] * old + weight * __bfloat162float(tileV[k][lane + 32]);
            maximum = next;
        }
        __syncthreads();
    }
    if (t < tokens) {
        const size_t base = size_t(b * tokens + t) * o.qHeads * Dim + head * Dim + lane;
        output[base] = bf(accum[0] / sum);
        output[base + 32] = bf(accum[1] / sum);
    }
}
// Eight GQA heads share one KV tile. Each warp keeps independent online
// softmax states for two/four queries; arithmetic order within a query stays
// identical to tiledPrefillAttention64. All tail warps join CTA barriers.
template<bool FastExp, int QueriesPerWarp>
__global__ void gqaPrefillAttention64(const float* qkv,
    const __nv_bfloat16* keys, const __nv_bfloat16* values,
    const int* table, const int* starts, const int* active,
    const float* sinks, float* output, int tokens, int logicalPages,
    int physicalPages, GptOssOptions o) {
    static_assert(QueriesPerWarp == 2 || QueriesPerWarp == 4);
    constexpr int KeyTile = 16, Dim = 64, QueryTile = 2 * QueriesPerWarp;
    __shared__ __nv_bfloat16 tileK[KeyTile][Dim], tileV[KeyTile][Dim];
    __shared__ int tileValid[KeyTile];
    __shared__ size_t tileOffset[KeyTile];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int queryTiles = (tokens + QueryTile - 1) / QueryTile;
    const int kvHead = blockIdx.x % o.kvHeads;
    const int queryTile = (blockIdx.x / o.kvHeads) % queryTiles;
    const int b = blockIdx.x / (o.kvHeads * queryTiles);
    const int head = kvHead * 8 + warp % 8;
    const int first = queryTile * QueryTile;
    const int firstQuery = first + (warp / 8) * QueriesPerWarp;
    const int last = min(tokens - 1, first + QueryTile - 1);
    const int capacity = logicalPages * o.pageSize;
    const int firstEnd = starts[b] + first + 1;
    const int firstBegin = o.window ? max(0, firstEnd - o.window) : 0;
    const int lastEnd = min(capacity, starts[b] + last + 1);
    const int packed = (o.qHeads + 2 * o.kvHeads) * Dim;
    float query[QueriesPerWarp][2] = {}, accum[QueriesPerWarp][2] = {};
    float maximum[QueriesPerWarp], sum[QueriesPerWarp];
    int begin[QueriesPerWarp], end[QueriesPerWarp];
    bool valid[QueriesPerWarp];
#pragma unroll
    for (int q = 0; q < QueriesPerWarp; ++q) {
        const int t = firstQuery + q;
        end[q] = starts[b] + t + 1;
        begin[q] = o.window ? max(0, end[q] - o.window) : 0;
        valid[q] = active[b] && t < tokens && end[q] > 0 && end[q] <= capacity;
        maximum[q] = sinks[head]; sum[q] = 1.f;
        if (valid[q]) {
            const size_t base = size_t(b * tokens + t) * packed + head * Dim + lane;
            query[q][0] = qkv[base]; query[q][1] = qkv[base + 32];
        }
    }
    if (active[b]) for (int p0 = firstBegin; p0 < lastEnd; p0 += KeyTile) {
        if (threadIdx.x < KeyTile) {
            const int p = p0 + threadIdx.x;
            const size_t offset = p < lastEnd ? cacheOffset(table, logicalPages,
                physicalPages, b, p, kvHead, 0, o) : SIZE_MAX;
            tileOffset[threadIdx.x] = offset;
            tileValid[threadIdx.x] = offset != SIZE_MAX;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < KeyTile * Dim; i += blockDim.x) {
            const int k = i / Dim, d = i % Dim;
            const size_t offset = tileOffset[k];
            tileK[k][d] = offset == SIZE_MAX ? __float2bfloat16(0.f) : keys[offset + d];
            tileV[k][d] = offset == SIZE_MAX ? __float2bfloat16(0.f) : values[offset + d];
        }
        __syncthreads();
        for (int k = 0; k < KeyTile; ++k) {
            const int p = p0 + k;
#pragma unroll
            for (int q = 0; q < QueriesPerWarp; ++q) {
                if (!valid[q] || p < begin[q] || p >= end[q] || !tileValid[k]) continue;
                float score = query[q][0] * __bfloat162float(tileK[k][lane]);
                score += query[q][1] * __bfloat162float(tileK[k][lane + 32]);
                score = __shfl_sync(0xffffffff, warpSum(score), 0) * rsqrtf(float(Dim));
                const float next = fmaxf(maximum[q], score);
                const float old = FastExp ? __expf(maximum[q] - next) : expf(maximum[q] - next);
                const float weight = FastExp ? __expf(score - next) : expf(score - next);
                sum[q] = sum[q] * old + weight;
                accum[q][0] = accum[q][0] * old + weight * __bfloat162float(tileV[k][lane]);
                accum[q][1] = accum[q][1] * old + weight * __bfloat162float(tileV[k][lane + 32]);
                maximum[q] = next;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int q = 0; q < QueriesPerWarp; ++q) if (firstQuery + q < tokens) {
        const size_t base = size_t(b * tokens + firstQuery + q) * o.qHeads * Dim + head * Dim + lane;
        output[base] = bf(accum[q][0] / sum[q]);
        output[base + 32] = bf(accum[q][1] / sum[q]);
    }
}
// Decode has too few query rows to hide a serial context walk. Split each
// head's paged KV range across eight warps, then merge stable softmax states.
// Include the learned sink exactly once in the final merge.
template<int Warps>
__global__ void decodeAttention(const float* qkv, const __nv_bfloat16* keys,
    const __nv_bfloat16* values, const int* table, const int* lengths,
    const int* starts, const int* active, const float* sinks, float* output,
    int batch, int logicalPages, int physicalPages, GptOssOptions o) {
    static_assert(Warps > 0 && Warps <= 16);
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const int head = blockIdx.x % o.qHeads, b = blockIdx.x / o.qHeads;
    const int packed = (o.qHeads + 2 * o.kvHeads) * o.headDim;
    const int end = lengths[b], begin = o.window ? max(0, end - o.window) : 0;
    const bool valid = active[b] && end > 0 && end <= logicalPages * o.pageSize;
    const int chunk = (max(0, end - begin) + Warps - 1) / Warps;
    float accum[4] = {}, maximum = -FLT_MAX, sum = 0;
    if (valid) for (int p = begin + warp * chunk; p < min(end, begin + (warp + 1) * chunk); ++p) {
        const size_t offset = cacheOffset(table, logicalPages, physicalPages, b, p,
            head / (o.qHeads / o.kvHeads), 0, o);
        if (offset == SIZE_MAX) continue;
        float score = 0;
        for (int d = lane; d < o.headDim; d += 32)
            score += qkv[size_t(b) * packed + head * o.headDim + d] * __bfloat162float(keys[offset + d]);
        score = __shfl_sync(0xffffffff, warpSum(score), 0) * rsqrtf(float(o.headDim));
        const float next = fmaxf(maximum, score), old = expf(maximum - next), weight = expf(score - next);
        sum = sum * old + weight;
        for (int d = lane; d < o.headDim; d += 32)
            accum[d / 32] = accum[d / 32] * old + weight * __bfloat162float(values[offset + d]);
        maximum = next;
    }
    __shared__ float maxima[Warps], sums[Warps], partial[Warps][128];
    if (!lane) { maxima[warp] = maximum; sums[warp] = sum; }
    for (int d = lane; d < o.headDim; d += 32) partial[warp][d] = accum[d / 32];
    __syncthreads();
    if (warp) return;
    maximum = sinks[head];
    for (int w = 0; w < Warps; ++w) if (sums[w] > 0) maximum = fmaxf(maximum, maxima[w]);
    sum = expf(sinks[head] - maximum);
    float weights[Warps];
    for (int w = 0; w < Warps; ++w) {
        weights[w] = sums[w] > 0 ? expf(maxima[w] - maximum) : 0;
        sum += sums[w] * weights[w];
    }
    for (int d = lane; d < o.headDim; d += 32) {
        float value = 0;
        for (int w = 0; w < Warps; ++w) value += partial[w][d] * weights[w];
        output[size_t(b) * o.qHeads * o.headDim + head * o.headDim + d] = bf(value / sum);
    }
}
// A split owns four warps and writes a stable softmax state. Splitting across
// CTAs exposes more work when a single decode request has few query heads.
template<int Splits>
__global__ void decodeAttentionPartial(const float* qkv, const __nv_bfloat16* keys,
    const __nv_bfloat16* values, const int* table, const int* lengths,
    const int* active, float* scratch, int logicalPages, int physicalPages,
    GptOssOptions o) {
    constexpr int Warps = 4, Stride = 2 + 128;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int split = blockIdx.x % Splits;
    const int head = (blockIdx.x / Splits) % o.qHeads;
    const int b = blockIdx.x / (Splits * o.qHeads);
    const int packed = (o.qHeads + 2 * o.kvHeads) * o.headDim;
    const int end = lengths[b], begin = o.window ? max(0, end - o.window) : 0;
    const bool valid = active[b] && end > 0 && end <= logicalPages * o.pageSize;
    const int chunk = (max(0, end - begin) + Splits * Warps - 1) / (Splits * Warps);
    const int first = begin + (split * Warps + warp) * chunk;
    float accum[4] = {}, maximum = -FLT_MAX, sum = 0;
    if (valid) for (int p = first; p < min(end, first + chunk); ++p) {
        const size_t offset = cacheOffset(table, logicalPages, physicalPages, b, p,
            head / (o.qHeads / o.kvHeads), 0, o);
        if (offset == SIZE_MAX) continue;
        float score = 0;
        for (int d = lane; d < o.headDim; d += 32)
            score += qkv[size_t(b) * packed + head * o.headDim + d] *
                __bfloat162float(keys[offset + d]);
        score = __shfl_sync(0xffffffff, warpSum(score), 0) * rsqrtf(float(o.headDim));
        const float next = fmaxf(maximum, score);
        const float old = expf(maximum - next), weight = expf(score - next);
        sum = sum * old + weight;
        for (int d = lane; d < o.headDim; d += 32)
            accum[d / 32] = accum[d / 32] * old + weight *
                __bfloat162float(values[offset + d]);
        maximum = next;
    }
    __shared__ float maxima[Warps], sums[Warps], partial[Warps][128];
    if (!lane) { maxima[warp] = maximum; sums[warp] = sum; }
    for (int d = lane; d < o.headDim; d += 32) partial[warp][d] = accum[d / 32];
    __syncthreads();
    if (warp) return;
    maximum = -FLT_MAX;
    for (int w = 0; w < Warps; ++w)
        if (sums[w] > 0) maximum = fmaxf(maximum, maxima[w]);
    float weights[Warps];
    sum = 0;
    for (int w = 0; w < Warps; ++w) {
        weights[w] = sums[w] > 0 ? expf(maxima[w] - maximum) : 0;
        sum += sums[w] * weights[w];
    }
    float* state = scratch + size_t(blockIdx.x) * Stride;
    if (!lane) { state[0] = maximum; state[1] = sum; }
    for (int d = lane; d < o.headDim; d += 32) {
        float value = 0;
        for (int w = 0; w < Warps; ++w) value += partial[w][d] * weights[w];
        state[2 + d] = value;
    }
}
__global__ void vocabTop1(const float* logits, float* pairs, int width, int rank) {
    __shared__ float scores[256];
    __shared__ int indices[256];
    const int tid = threadIdx.x;
    float score = -FLT_MAX;
    int index = 0;
    for (int i = tid; i < width; i += 256) {
        const float value = logits[size_t(blockIdx.x) * width + i];
        if (value > score || (value == score && i < index)) {
            score = value; index = i;
        }
    }
    scores[tid] = score; indices[tid] = index;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride && (scores[tid + stride] > scores[tid] ||
            (scores[tid + stride] == scores[tid] && indices[tid + stride] < indices[tid]))) {
            scores[tid] = scores[tid + stride]; indices[tid] = indices[tid + stride];
        }
        __syncthreads();
    }
    if (!tid) {
        pairs[size_t(blockIdx.x) * 2] = scores[0];
        pairs[size_t(blockIdx.x) * 2 + 1] = float(rank * width + indices[0]);
    }
}
// Eight query heads share one KV head. Stage each paged tile once and reuse
// it across eight query warps, retaining FP32 scores/softmax and one sink.
template<int Splits>
__global__ void groupedDecodeAttention64(const float* qkv, const __nv_bfloat16* keys,
    const __nv_bfloat16* values, const int* table, const int* lengths,
    const int* active, float* scratch, int logicalPages, int physicalPages,
    GptOssOptions o) {
    constexpr int Tile = 16, Dim = 64, Stride = 130;
    __shared__ float tileK[Tile][Dim], tileV[Tile][Dim];
    __shared__ size_t offsets[Tile];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int split = blockIdx.x % Splits;
    const int kvHead = (blockIdx.x / Splits) % o.kvHeads;
    const int b = blockIdx.x / (Splits * o.kvHeads), head = kvHead * 8 + warp;
    const int packed = (o.qHeads + 2 * o.kvHeads) * Dim;
    const int end = lengths[b], begin = o.window ? max(0, end - o.window) : 0;
    const bool valid = active[b] && end > 0 && end <= logicalPages * o.pageSize;
    const int chunk = (max(0, end - begin) + Splits - 1) / Splits;
    const int first = begin + split * chunk, last = min(end, first + chunk);
    float query[2] = {}, accum[2] = {}, maximum = -FLT_MAX, sum = 0;
    if (valid) {
        query[0] = qkv[size_t(b) * packed + head * Dim + lane];
        query[1] = qkv[size_t(b) * packed + head * Dim + lane + 32];
    }
    if (valid) for (int p0 = first; p0 < last; p0 += Tile) {
        if (threadIdx.x < Tile) {
            const int p = p0 + threadIdx.x;
            offsets[threadIdx.x] = p < last ? cacheOffset(table, logicalPages,
                physicalPages, b, p, kvHead, 0, o) : SIZE_MAX;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < Tile * Dim; i += blockDim.x) {
            const int k = i / Dim, d = i % Dim;
            const size_t offset = offsets[k];
            tileK[k][d] = offset == SIZE_MAX ? 0.f : __bfloat162float(keys[offset + d]);
            tileV[k][d] = offset == SIZE_MAX ? 0.f : __bfloat162float(values[offset + d]);
        }
        __syncthreads();
        for (int k = 0; k < Tile; ++k) {
            if (p0 + k >= last || offsets[k] == SIZE_MAX) continue;
            float score = query[0] * tileK[k][lane];
            score += query[1] * tileK[k][lane + 32];
            score = __shfl_sync(0xffffffff, warpSum(score), 0) * .125f;
            const float next = fmaxf(maximum, score);
            const float old = expf(maximum - next), weight = expf(score - next);
            sum = sum * old + weight;
            accum[0] = accum[0] * old + weight * tileV[k][lane];
            accum[1] = accum[1] * old + weight * tileV[k][lane + 32];
            maximum = next;
        }
        __syncthreads();
    }
    float* state = scratch + ((size_t(b) * o.qHeads + head) * Splits + split) * Stride;
    if (!lane) { state[0] = maximum; state[1] = sum; }
    state[2 + lane] = accum[0]; state[2 + lane + 32] = accum[1];
}
template<int Splits>
__global__ void decodeAttentionMerge(const float* scratch, const float* sinks,
    float* output, GptOssOptions o) {
    constexpr int Stride = 2 + 128;
    const int lane = threadIdx.x, head = blockIdx.x % o.qHeads;
    const float* states = scratch + size_t(blockIdx.x) * Splits * Stride;
    float maximum = sinks[head];
    for (int s = 0; s < Splits; ++s)
        if (states[s * Stride + 1] > 0)
            maximum = fmaxf(maximum, states[s * Stride]);
    float sum = expf(sinks[head] - maximum);
    float weights[Splits];
    for (int s = 0; s < Splits; ++s) {
        const float partialSum = states[s * Stride + 1];
        weights[s] = partialSum > 0 ? expf(states[s * Stride] - maximum) : 0;
        sum += partialSum * weights[s];
    }
    for (int d = lane; d < o.headDim; d += 32) {
        float value = 0;
        for (int s = 0; s < Splits; ++s)
            value += states[s * Stride + 2 + d] * weights[s];
        output[size_t(blockIdx.x) * o.headDim + d] = bf(value / sum);
    }
}
template<int Threads, bool Vector4 = false>
__global__ void routeScores(const float* x, const float* weight, const float* bias,
    float* logits, int tokens, GptOssOptions o,
    __nv_bfloat16* converted, int paddedWidth, int* locks, int locksPerExpert) {
    const int token = blockIdx.x / o.experts, expert = blockIdx.x % o.experts;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (locks) {
        // One router CTA owns each expert. Clear its Marlin lock slice before
        // top-K and GEMM consume it; this removes a separate decode memset.
        auto* slice = reinterpret_cast<int4*>(locks + size_t(expert) * locksPerExpert);
        for (int i = threadIdx.x; i < locksPerExpert / 4; i += blockDim.x)
            slice[i] = make_int4(0, 0, 0, 0);
    }
    float value = 0;
    if constexpr (Vector4) {
        const auto* input = reinterpret_cast<const float4*>(x + size_t(token) * o.hidden);
        const auto* row = reinterpret_cast<const float4*>(weight + size_t(expert) * o.hidden);
        for (int d = threadIdx.x; d < o.hidden / 4; d += blockDim.x) {
            const float4 a = input[d], b = row[d];
            if (converted && expert == 0) {
                const int index = 4 * d;
                converted[index] = __float2bfloat16(a.x);
                converted[index + 1] = __float2bfloat16(a.y);
                converted[index + 2] = __float2bfloat16(a.z);
                converted[index + 3] = __float2bfloat16(a.w);
            }
            value = fmaf(a.x, b.x, value);
            value = fmaf(a.y, b.y, value);
            value = fmaf(a.z, b.z, value);
            value = fmaf(a.w, b.w, value);
        }
    } else {
        for (int d = threadIdx.x; d < o.hidden; d += blockDim.x) {
            const float inputValue = x[size_t(token) * o.hidden + d];
            if (converted && expert == 0)
                converted[d] = __float2bfloat16(inputValue);
            value += inputValue * weight[size_t(expert) * o.hidden + d];
        }
    }
    if (converted && expert == 0)
        for (int d = threadIdx.x + o.hidden; d < paddedWidth; d += blockDim.x)
            converted[d] = __float2bfloat16(0.f);
    value = warpSum(value);
    __shared__ float partial[Threads / 32];
    if (!lane) partial[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < Threads / 32 ? partial[lane] : 0.f;
        value = warpSum(value);
        if (!lane) logits[size_t(token) * o.experts + expert] = bf(value + bias[expert]);
    }
}
// Reuse one expert weight row across several queries, retaining each query's
// original 256-thread accumulation and reduction order exactly.
template<int Queries>
__global__ void routeScoresBatch(const float* x,const float* weight,const float* bias,
    float* logits,int tokens,GptOssOptions o) {
    __shared__ float row[4096],partial[Queries][8];
    const int expert=blockIdx.x%o.experts,first=blockIdx.x/o.experts*Queries;
    const int query=threadIdx.x/256,tid=threadIdx.x%256,lane=tid%32,warp=tid/32;
    for(int d=threadIdx.x;d<o.hidden;d+=Queries*256)row[d]=weight[size_t(expert)*o.hidden+d];
    __syncthreads();
    const int token=first+query;
    float value=0;
    if(token<tokens)for(int d=tid;d<o.hidden;d+=256)value+=x[size_t(token)*o.hidden+d]*row[d];
    value=warpSum(value);
    if(!lane)partial[query][warp]=value;
    __syncthreads();
    if(!warp) {
        value=lane<8?partial[query][lane]:0.f;
        value=warpSum(value);
        if(!lane && token<tokens)logits[size_t(token)*o.experts+expert]=bf(value+bias[expert]);
    }
}
// Opt-in BF16 tensor-core router for prefill or supported decode batches. Each CTA
// shares sixteen query rows across four warps, each owning sixteen experts.
// GPT-OSS normalized inputs and original router weights are BF16-representable;
// the accumulation order differs from the scalar path and remains opt-in.
__global__ void routeScoresTensorCore(const float* x,const float* weight,const float* bias,
    float* logits,int tokens,GptOssOptions o) {
    const int first=blockIdx.x*16,firstExpert=blockIdx.y*64;
#if __CUDA_ARCH__ >= 800
    namespace W=nvcuda::wmma;
    __shared__ __align__(32) __nv_bfloat16 a[16*32],b[64*32];
    __shared__ __align__(32) float result[4][16*16];
    const int warp=threadIdx.x/32;
    W::fragment<W::matrix_a,16,16,16,__nv_bfloat16,W::row_major> af;
    W::fragment<W::matrix_b,16,16,16,__nv_bfloat16,W::col_major> bfFragment;
    W::fragment<W::accumulator,16,16,16,float> acc;
    W::fill_fragment(acc,0.f);
    for(int start=0;start<o.hidden;start+=32) {
        for(int i=threadIdx.x;i<16*32;i+=128) {
            const int token=first+i/32,k=start+i%32;
            a[i]=__float2bfloat16(token<tokens && k<o.hidden?x[size_t(token)*o.hidden+k]:0.f);
        }
        for(int i=threadIdx.x;i<64*32;i+=128) {
            const int expert=firstExpert+i/32,k=start+i%32;
            b[i]=__float2bfloat16(expert<o.experts && k<o.hidden?weight[size_t(expert)*o.hidden+k]:0.f);
        }
        __syncthreads();
        for(int offset=0;offset<32;offset+=16) {
            W::load_matrix_sync(af,a+offset,32);
            W::load_matrix_sync(bfFragment,b+warp*16*32+offset,32);
            W::mma_sync(acc,af,bfFragment,acc);
        }
        __syncthreads();
    }
    W::store_matrix_sync(result[warp],acc,16,W::mem_row_major);
    __syncthreads();
    for(int i=threadIdx.x;i<16*64;i+=128) {
        const int token=first+i/64,e=i%64,expert=firstExpert+e;
        if(token<tokens && expert<o.experts)
            logits[size_t(token)*o.experts+expert]=bf(result[e/16][(i/64)*16+e%16]+bias[expert]);
    }
#else
    // Older architectures retain functional scores, without a tensor-core gain.
    for(int i=threadIdx.x;i<16*64;i+=128) {
        const int token=first+i/64,expert=firstExpert+i%64;
        if(token<tokens && expert<o.experts) {
            float value=0;
            for(int k=0;k<o.hidden;++k)value+=bf(x[size_t(token)*o.hidden+k])*bf(weight[size_t(expert)*o.hidden+k]);
            logits[size_t(token)*o.experts+expert]=bf(value+bias[expert]);
        }
    }
#endif
}

// Keep the original four-warp body/default path byte-for-byte above.
#include "tensor_router_geometry.cuh"
int tensorCoreRouterWarps() {
    static const int count=[] {
        const char* value=std::getenv("GARNET_GPT_OSS_TENSOR_ROUTER_EXPERT_WARPS");
        if(!value || std::strcmp(value,"4")==0)return 4;
        return std::strcmp(value,"2")==0?2:0;
    }();
    return count;
}
void launchTensorCoreRouter(const float* x,const float* weight,const float* bias,
    float* logits,int tokens,GptOssOptions o,int warps,cudaStream_t stream) {
    if(warps==2)
        routeScoresTensorCoreGeometry<2><<<dim3((tokens+15)/16,(o.experts+31)/32),64,0,stream>>>(x,weight,bias,logits,tokens,o);
    else
        routeScoresTensorCore<<<dim3((tokens+15)/16,(o.experts+63)/64),128,0,stream>>>(x,weight,bias,logits,tokens,o);
}

template<bool FusedMarlinDecode>
__global__ void routeTopK(float* logits, int* selected, float* probabilities,
    int tokens, GptOssOptions o, int* sorted, int* experts, int* padded,
    int marlinBlock) {
    const int token = blockIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ float warpValues[4], top[8];
    __shared__ int warpExperts[4];
    for (int k = 0; k < o.topK; ++k) {
        float value = threadIdx.x < o.experts
            ? logits[size_t(token) * o.experts + threadIdx.x] : -FLT_MAX;
        int expert = threadIdx.x < o.experts ? threadIdx.x : 0x7fffffff;
        for (int offset = 16; offset; offset >>= 1) {
            const float otherValue = __shfl_down_sync(0xffffffff, value, offset);
            const int otherExpert = __shfl_down_sync(0xffffffff, expert, offset);
            if (otherValue > value || (otherValue == value && otherExpert < expert)) {
                value = otherValue; expert = otherExpert;
            }
        }
        if (!lane) { warpValues[warp] = value; warpExperts[warp] = expert; }
        __syncthreads();
        if (warp == 0) {
            value = lane < 4 ? warpValues[lane] : -FLT_MAX;
            expert = lane < 4 ? warpExperts[lane] : 0x7fffffff;
            for (int offset = 16; offset; offset >>= 1) {
                const float otherValue = __shfl_down_sync(0xffffffff, value, offset);
                const int otherExpert = __shfl_down_sync(0xffffffff, expert, offset);
                if (otherValue > value || (otherValue == value && otherExpert < expert)) {
                    value = otherValue; expert = otherExpert;
                }
            }
            if (!lane) {
                selected[token * o.topK + k] = expert;
                top[k] = value;
                logits[size_t(token) * o.experts + expert] = -FLT_MAX;
            }
        }
        __syncthreads();
    }
    if (!threadIdx.x) {
        float total = 0;
        for (int k = 0; k < o.topK; ++k) total += expf(top[k] - top[0]);
        for (int k = 0; k < o.topK; ++k)
            probabilities[token * o.topK + k] = bf(expf(top[k] - top[0]) / total);
    }
    if constexpr (FusedMarlinDecode) {
        // Decode has one token. Reuse the top-K block to prepare Marlin's
        // expert rows; the score block converts its input in parallel.
        __syncthreads();
        __shared__ int counts[128], starts[128];
        const int e = threadIdx.x;
        const int localCount = o.tpRank < 0 ? o.experts :
            (o.experts + 1 - o.tpRank) / 2;
        const int globalExpert = o.tpRank < 0 ? e : 2 * e + o.tpRank;
        int count = 0;
        if (e < localCount)
            for (int s = 0; s < o.topK; ++s)
                count += selected[s] == globalExpert;
        counts[e] = count;
        __syncthreads();
        if (!e) {
            int total = 0;
            for (int i = 0; i < localCount; ++i) {
                starts[i] = total;
                total += (counts[i] + marlinBlock - 1) / marlinBlock * marlinBlock;
            }
            *padded = total;
        }
        __syncthreads();
        if (e < localCount && count) {
            int row = 0;
            for (int s = 0; s < o.topK; ++s)
                if (selected[s] == globalExpert) sorted[starts[e] + row++] = s;
            const int paddedCount = (count + marlinBlock - 1) / marlinBlock * marlinBlock;
            for (int i = count; i < paddedCount; ++i)
                sorted[starts[e] + i] = o.topK;
            for (int i = 0; i < count; i += marlinBlock)
                experts[(starts[e] + i) / marlinBlock] = e;
        }
    }
}
__global__ void gateUp(const float* x, const unsigned char* blocks, const unsigned char* scales,
    const float* bias, const int* selected, float* hidden, int tokens, GptOssOptions o) {
    // A warp owns each output: neighboring lanes read neighboring MXFP4
    // columns instead of issuing a separate serial dot product per thread.
    const int lane = threadIdx.x & 31;
    const size_t index = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (index >= size_t(tokens) * o.topK * o.intermediate) return;
    const int i = index % o.intermediate, slot = index / o.intermediate, token = slot / o.topK;
    const int expert = selected[slot];
    if (o.tpRank >= 0 && expert % 2 != o.tpRank) {
        if (!lane) hidden[index] = 0.f;
        return;
    }
    const int storedExpert = o.expertWeightsSharded ? expert / 2 : expert;
    const size_t row = size_t(storedExpert) * 2 * o.intermediate + 2 * i;
    float gate = 0, up = 0;
    for (int d = lane; d < o.hidden; d += 32) {
        const float v = x[size_t(token) * o.hidden + d];
        gate += v * fp4(blocks, scales, row, d, o.hidden);
        up += v * fp4(blocks, scales, row + 1, d, o.hidden);
    }
    gate = warpSum(gate);
    up = warpSum(up);
    if (lane) return;
    gate = fminf(bf(bf(gate) + bias[row]), o.limit);
    up = fminf(o.limit, fmaxf(-o.limit, bf(bf(up) + bias[row + 1])));
    const float glu = bf(gate * bf(1.f / (1.f + expf(-bf(1.702f * gate)))));
    hidden[index] = bf(glu * bf(up + 1));
}
__global__ void down(const float* hidden, const unsigned char* blocks, const unsigned char* scales,
    const float* bias, const int* selected, const float* probabilities, float* y,
    int tokens, GptOssOptions o) {
    const int lane = threadIdx.x & 31;
    const size_t index = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (index >= size_t(tokens) * o.hidden) return;
    const int token = index / o.hidden, d = index % o.hidden;
    float result = 0;
    for (int k = 0; k < o.topK; ++k) {
        const int slot = token * o.topK + k, expert = selected[slot];
        if (o.tpRank >= 0 && expert % 2 != o.tpRank) continue;
        const int storedExpert = o.expertWeightsSharded ? expert / 2 : expert;
        const size_t row = size_t(storedExpert) * o.hidden + d;
        float value = 0;
        for (int i = lane; i < o.intermediate; i += 32)
            value += hidden[size_t(slot) * o.intermediate + i] * fp4(blocks, scales, row, i, o.intermediate);
        value = warpSum(value);
        if (!lane) result += bf(bf(value) + bias[row]) * probabilities[slot];
    }
    if (!lane) y[index] = o.tpRank < 0 ? bf(result) : result;
}

__global__ void bucketExperts(const int* selected, int* counts, int* slots,
    int tokens, GptOssOptions o) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= tokens * o.topK) return;
    const int expert = selected[slot];
    if (o.tpRank >= 0 && expert % 2 != o.tpRank) return;
    const int row = atomicAdd(counts + expert, 1);
    slots[size_t(expert) * tokens * o.topK + row] = slot;
}
template<int Rows>
__global__ void expertTasks(const int* counts, int* taskCount, int* tasks,
    GptOssOptions o) {
    // Compact tiles rather than launching the worst-case token count for
    // every expert. Every routed slot has its own row, including repeated
    // selections if nonfinite router inputs reach this low-level operator.
    int total = 0;
    const int first = o.tpRank < 0 ? 0 : o.tpRank;
    const int stride = o.tpRank < 0 ? 1 : 2;
    for (int e = first; e < o.experts; e += stride)
        for (int row = 0; row < counts[e]; row += Rows) {
            tasks[2 * total] = e;
            tasks[2 * total + 1] = row;
            ++total;
        }
    *taskCount = total;
}
// W4A16 grouped prefill: unpack one weight tile into shared BF16, reuse it
// across thirty-two routed tokens, and accumulate on BF16 Tensor Cores. Weight
// storage remains MXFP4; there is no full-model BF16 materialization.
template<bool Up, int Rows>
__global__ void groupedExperts(const float* x, const unsigned char* blocks,
    const unsigned char* scales, const float* bias, const int* counts,
    const int* slots, const int* taskCount, const int* tasks, float* output,
    int tokens, GptOssOptions o) {
    if (blockIdx.x >= *taskCount) return;
    const int expert = tasks[2 * blockIdx.x], first = tasks[2 * blockIdx.x + 1];
    const int storedExpert = o.expertWeightsSharded ? expert / 2 : expert;
    const int nStart = blockIdx.y * 32;
    const int width = Up ? o.hidden : o.intermediate;
    const int outputs = Up ? 2 * o.intermediate : o.hidden;
    __shared__ __align__(32) float c[Rows * 32];
#if __CUDA_ARCH__ >= 800
    __shared__ __align__(32) __nv_bfloat16 a[Rows * 32];
    __shared__ __align__(32) __nv_bfloat16 b[32 * 32];
    const int warp = threadIdx.x / 32;
    const int mStart = (warp / 2) * 16, nSub = warp % 2;
    using namespace nvcuda;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    for (int kStart = 0; kStart < width; kStart += 32) {
        for (int index = threadIdx.x; index < Rows * 32; index += blockDim.x) {
            const int m = index / 32, k = index % 32;
            float value = 0;
            if (first + m < counts[expert] && kStart + k < width) {
                const int slot = slots[size_t(expert) * tokens * o.topK + first + m];
                const int inputRow = Up ? slot / o.topK : slot;
                value = x[size_t(inputRow) * width + kStart + k];
            }
            a[index] = __float2bfloat16(value);
        }
        for (int index = threadIdx.x; index < 32 * 32; index += blockDim.x) {
            const int k = index % 32, n = index / 32;
            const float value = nStart + n < outputs && kStart + k < width
                ? fp4(blocks, scales, size_t(storedExpert) * outputs + nStart + n,
                    kStart + k, width) : 0;
            b[index] = __float2bfloat16(value);
        }
        __syncthreads();
        for (int sub = 0; sub < 32; sub += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> aTile;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bTile;
            wmma::load_matrix_sync(aTile, a + mStart * 32 + sub, 32);
            wmma::load_matrix_sync(bTile, b + nSub * 16 * 32 + sub, 32);
            wmma::mma_sync(acc, aTile, bTile, acc);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(c + mStart * 32 + nSub * 16, acc, 32, wmma::mem_row_major);
#else
    // BF16 WMMA fragments are unavailable before SM80. Preserve the grouped
    // kernel's rounded-input/weight semantics with a CUDA-core fallback.
    for (int index = threadIdx.x; index < Rows * 32; index += blockDim.x) {
        const int m = index / 32, n = index % 32;
        if (first + m >= counts[expert] || nStart + n >= outputs) {
            c[index] = 0.f;
            continue;
        }
        const int slot = slots[size_t(expert) * tokens * o.topK + first + m];
        const int inputRow = Up ? slot / o.topK : slot;
        const size_t weightRow = size_t(storedExpert) * outputs + nStart + n;
        float value = 0.f;
        for (int k = 0; k < width; ++k) {
            const float input = bf(x[size_t(inputRow) * width + k]);
            const float weight = bf(fp4(blocks, scales, weightRow, k, width));
            value += input * weight;
        }
        c[index] = value;
    }
#endif
    __syncthreads();
    for (int index = threadIdx.x; index < Rows * 32; index += blockDim.x) {
        const int m = index / 32, n = index % 32;
        if (first + m >= counts[expert] || nStart + n >= outputs) continue;
        const int slot = slots[size_t(expert) * tokens * o.topK + first + m];
        const size_t weightRow = size_t(storedExpert) * outputs + nStart + n;
        if constexpr (Up) {
            if (n & 1) continue;
            const float gate = fminf(bf(bf(c[index]) + bias[weightRow]), o.limit);
            const float up = fminf(o.limit, fmaxf(-o.limit, bf(bf(c[index + 1]) + bias[weightRow + 1])));
            const float glu = bf(gate * bf(1.f / (1.f + expf(-bf(1.702f * gate)))));
            output[size_t(slot) * o.intermediate + (nStart + n) / 2] = bf(glu * bf(up + 1));
        } else {
            output[size_t(slot) * o.hidden + nStart + n] = bf(bf(c[index]) + bias[weightRow]);
        }
    }
}
__global__ void combineExperts(const float* values, const float* probabilities, const int* selected,
    float* y, int tokens, GptOssOptions o) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(tokens) * o.hidden) return;
    const int token = index / o.hidden, d = index % o.hidden;
    float result = 0;
    // Preserve router order regardless of atomic bucketing order.
    for (int k = 0; k < o.topK; ++k) {
        const int slot = token * o.topK + k;
        const int expert = selected[slot];
        if (o.tpRank >= 0 && expert % 2 != o.tpRank) continue;
        result += values[size_t(slot) * o.hidden + d] * probabilities[slot];
    }
    y[index] = o.tpRank < 0 ? bf(result) : result;
}
}
#ifdef GARNET_GPT_OSS_KERNEL_TEST
cudaError_t TestGptOssMxfp4Decode(const unsigned char* blocks, const unsigned char* scales,
    float* output, int rows) {
    testMxfp4Decode<<<(rows * 32 + 127) / 128, 128>>>(blocks, scales, output, rows);
    return cudaGetLastError();
}
#endif
cudaError_t RunGptOssVocabTop1(const float* x, float* pairs, int rows, int width,
    int rank, cudaStream_t stream) {
    if (!x || !pairs || rows <= 0 || width <= 0 || width > (1 << 23) ||
        (rank != 0 && rank != 1)) return cudaErrorInvalidValue;
    vocabTop1<<<rows, 256, 0, stream>>>(x, pairs, width, rank);
    return cudaGetLastError();
}
cudaError_t RunGptOssRope(const float* x, const std::int64_t* p, float* y, int tokens,
    const GptOssOptions& o, cudaStream_t stream) {
    const size_t n = size_t(tokens) * (o.qHeads + 2 * o.kvHeads) * o.headDim;
    rope<<<(n + 255) / 256, 256, 0, stream>>>(x, p, y, tokens, o);
    return cudaGetLastError();
}
cudaError_t RunGptOssRmsNorm(const float* x, const float* weight, float* y,
    int rows, int hidden, float epsilon, int threads, cudaStream_t stream) {
    if (!x || !weight || !y || rows <= 0 || hidden <= 0 || epsilon <= 0.f)
        return cudaErrorInvalidValue;
    switch (threads) {
    case 128: rmsNorm<128><<<rows, 128, 0, stream>>>(x, weight, y, rows, hidden, epsilon); break;
    case 256: rmsNorm<256><<<rows, 256, 0, stream>>>(x, weight, y, rows, hidden, epsilon); break;
    case 512: rmsNorm<512><<<rows, 512, 0, stream>>>(x, weight, y, rows, hidden, epsilon); break;
    case 1024: rmsNorm<1024><<<rows, 1024, 0, stream>>>(x, weight, y, rows, hidden, epsilon); break;
    default: return cudaErrorInvalidConfiguration;
    }
    return cudaGetLastError();
}
template<bool FastExp>
cudaError_t launchGqaPrefill64(const void* const* in, float* y, int batch,
    int tokens, int logicalPages, int physicalPages, const GptOssOptions& o,
    int queriesPerWarp, cudaStream_t stream) {
#define PREFILL_GQA_CASE(Q) if (queriesPerWarp == Q) { \
    const int blocks = batch * ((tokens + 2 * Q - 1) / (2 * Q)) * o.kvHeads; \
    gqaPrefillAttention64<FastExp, Q><<<blocks, 512, 0, stream>>>( \
        (const float*)in[0], (const __nv_bfloat16*)in[1], \
        (const __nv_bfloat16*)in[2], (const int*)in[3], \
        (const int*)in[5], (const int*)in[6], (const float*)in[7], \
        y, tokens, logicalPages, physicalPages, o); \
    return cudaGetLastError(); }
    PREFILL_GQA_CASE(2)
    PREFILL_GQA_CASE(4)
#undef PREFILL_GQA_CASE
    return cudaErrorInvalidValue;
}
#ifdef GARNET_GPT_OSS_KERNEL_TEST
cudaError_t TestGptOssGqaPrefill64(const void* const* in, float* y, int batch,
    int tokens, int logicalPages, int physicalPages, const GptOssOptions& o,
    int queriesPerWarp, cudaStream_t stream) {
    if (!o.prefill || o.headDim != 64 || o.kvHeads <= 0 || o.qHeads != 8 * o.kvHeads ||
        batch <= 0 || tokens <= 0 || (queriesPerWarp != -1 && queriesPerWarp != 0 && queriesPerWarp != 2 && queriesPerWarp != 4))
        return cudaErrorInvalidValue;
    const size_t n = size_t(batch) * tokens * o.kvHeads * o.headDim;
    writeKV<<<(n + 255) / 256, 256, 0, stream>>>((const float*)in[0],
        (__nv_bfloat16*)in[1], (__nv_bfloat16*)in[2], (const int*)in[3],
        (const int*)in[5], (const int*)in[6], batch, tokens, logicalPages, physicalPages, o);
    auto status = cudaGetLastError(); if (status != cudaSuccess) return status;
    if(queriesPerWarp==-1)return cudaSuccess; // Benchmark write-only preparation.
    if (queriesPerWarp) return launchGqaPrefill64<false>(in, y, batch,
        tokens, logicalPages, physicalPages, o, queriesPerWarp, stream);
    const int blocks = batch * ((tokens + 15) / 16) * o.qHeads;
    tiledPrefillAttention64<false, 16><<<blocks, 512, 0, stream>>>(
        (const float*)in[0], (const __nv_bfloat16*)in[1], (const __nv_bfloat16*)in[2],
        (const int*)in[3], (const int*)in[5], (const int*)in[6], (const float*)in[7],
        y, tokens, logicalPages, physicalPages, o);
    return cudaGetLastError();
}
#endif
#ifdef GARNET_GPT_OSS_KERNEL_TEST
cudaError_t TestGptOssWriteKV(const void* const* in,int batch,int tokens,
    int logicalPages,int physicalPages,const GptOssOptions& o,cudaStream_t stream) {
    if (!in || batch<1 || batch>512 || tokens<1 || tokens>4096 ||
        logicalPages<1 || logicalPages>256 || physicalPages<1 || physicalPages>batch*logicalPages || o.kind!=1 ||
        o.pageSize!=16 || o.layer<0 || o.layer>=36 || o.kvHeads<1 || o.kvHeads>16 ||
        o.qHeads!=8*o.kvHeads || o.headDim!=64)
        return cudaErrorInvalidValue;
    for (int i=0;i<7;++i) if (i!=4 && !in[i]) return cudaErrorInvalidValue;
    const size_t n=size_t(batch)*tokens*o.kvHeads*o.headDim;
    writeKV<<<(n+255)/256,256,0,stream>>>(static_cast<const float*>(in[0]),
        (__nv_bfloat16*)in[1],(__nv_bfloat16*)in[2],(const int*)in[3],
        (const int*)in[5],(const int*)in[6],batch,tokens,logicalPages,physicalPages,o);
    return cudaGetLastError();
}
#endif
cudaError_t RunGptOssAttention(const void* const* in, float* y, void* workspace,
    int batch, int tokens,
    int logicalPages, int physicalPages, const GptOssOptions& o, cudaStream_t stream) {
    const size_t n = size_t(batch) * tokens * o.kvHeads * o.headDim;
    writeKV<<<(n + 255) / 256, 256, 0, stream>>>(static_cast<const float*>(in[0]),
        (__nv_bfloat16*)in[1], (__nv_bfloat16*)in[2], (const int*)in[3],
        (const int*)in[5], (const int*)in[6], batch, tokens, logicalPages, physicalPages, o);
    auto status = cudaGetLastError(); if (status != cudaSuccess) return status;
    if (tokens == 1 && !o.prefill) {
#ifdef GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL
        static const bool flashDecode=[] {
            const char* value=std::getenv("GARNET_GPT_OSS_DECODE_FLASHINFER");
            return value&&std::strcmp(value,"1")==0;
        }();
        if(flashDecode&&GptOssFlashAttentionWorkspace(batch,tokens,logicalPages,o))
            return RunGptOssFlashAttention(in,y,workspace,batch,tokens,logicalPages,physicalPages,o,stream);
#endif
        static const bool grouped = [] {
            const char* value = std::getenv("GARNET_GPT_OSS_DECODE_GQA_TILED");
            return value && std::strcmp(value, "1") == 0;
        }();
        static const int groupSplits = [] {
            const char* value = std::getenv("GARNET_GPT_OSS_DECODE_GQA_SPLITS");
            const int requested = value ? std::atoi(value) : 8;
            return requested == 4 || requested == 8 || requested == 16 ? requested : 8;
        }();
        if (grouped && o.headDim == 64 && o.qHeads == 8 * o.kvHeads) {
            if (!workspace) return cudaErrorInvalidValue;
#define GQA_CASE(S) if (groupSplits == S) { \
            groupedDecodeAttention64<S><<<batch * o.kvHeads * S, 256, 0, stream>>>( \
                (const float*)in[0], (const __nv_bfloat16*)in[1], (const __nv_bfloat16*)in[2], \
                (const int*)in[3], (const int*)in[4], (const int*)in[6], \
                (float*)workspace, logicalPages, physicalPages, o); \
            status = cudaGetLastError(); if (status != cudaSuccess) return status; \
            decodeAttentionMerge<S><<<batch * o.qHeads, 32, 0, stream>>>( \
                (const float*)workspace, (const float*)in[7], y, o); \
            return cudaGetLastError(); }
            GQA_CASE(4) GQA_CASE(8) GQA_CASE(16)
#undef GQA_CASE
        }
        static const int splits = [] {
            const char* value = std::getenv("GARNET_GPT_OSS_DECODE_SPLITS");
            if (!value) return 16;
            if (std::strcmp(value, "0") == 0) return 0;
            const int requested = std::atoi(value);
            return requested == 8 || requested == 16 || requested == 32 ||
                requested == 64 ? requested : 16;
        }();
        if (splits && !workspace) return cudaErrorInvalidValue;
        static const int warps = [] {
            const char* value = std::getenv("GARNET_GPT_OSS_DECODE_WARPS");
            if (value) {
                const int requested = std::atoi(value);
                if (requested == 1 || requested == 2 || requested == 4 ||
                    requested == 8 || requested == 16)
                    return requested;
            }
            return 16;
        }();
        const float* qkv = (const float*)in[0];
        const auto* keys = (const __nv_bfloat16*)in[1];
        const auto* values = (const __nv_bfloat16*)in[2];
        const auto* table = (const int*)in[3];
        const auto* lengths = (const int*)in[4];
        const auto* starts = (const int*)in[5];
        const auto* active = (const int*)in[6];
        const auto* sinks = (const float*)in[7];
        if (splits == 8) {
            decodeAttentionPartial<8><<<batch * o.qHeads * 8, 128, 0, stream>>>(
                qkv, keys, values, table, lengths, active, (float*)workspace,
                logicalPages, physicalPages, o);
            status = cudaGetLastError(); if (status != cudaSuccess) return status;
            decodeAttentionMerge<8><<<batch * o.qHeads, 32, 0, stream>>>(
                (const float*)workspace, sinks, y, o);
            return cudaGetLastError();
        }
        if (splits == 16) {
            decodeAttentionPartial<16><<<batch * o.qHeads * 16, 128, 0, stream>>>(
                qkv, keys, values, table, lengths, active, (float*)workspace,
                logicalPages, physicalPages, o);
            status = cudaGetLastError(); if (status != cudaSuccess) return status;
            decodeAttentionMerge<16><<<batch * o.qHeads, 32, 0, stream>>>(
                (const float*)workspace, sinks, y, o);
            return cudaGetLastError();
        }
        if (splits == 32) {
            decodeAttentionPartial<32><<<batch * o.qHeads * 32, 128, 0, stream>>>(
                qkv, keys, values, table, lengths, active, (float*)workspace,
                logicalPages, physicalPages, o);
            status = cudaGetLastError(); if (status != cudaSuccess) return status;
            decodeAttentionMerge<32><<<batch * o.qHeads, 32, 0, stream>>>(
                (const float*)workspace, sinks, y, o);
            return cudaGetLastError();
        }
        if (splits == 64) {
            decodeAttentionPartial<64><<<batch * o.qHeads * 64, 128, 0, stream>>>(
                qkv, keys, values, table, lengths, active, (float*)workspace,
                logicalPages, physicalPages, o);
            status = cudaGetLastError(); if (status != cudaSuccess) return status;
            decodeAttentionMerge<64><<<batch * o.qHeads, 32, 0, stream>>>(
                (const float*)workspace, sinks, y, o);
            return cudaGetLastError();
        }
        switch (warps) {
        case 1:
            decodeAttention<1><<<batch * o.qHeads, 32, 0, stream>>>(qkv, keys, values,
                table, lengths, starts, active, sinks, y, batch, logicalPages, physicalPages, o);
            break;
        case 2:
            decodeAttention<2><<<batch * o.qHeads, 64, 0, stream>>>(qkv, keys, values,
                table, lengths, starts, active, sinks, y, batch, logicalPages, physicalPages, o);
            break;
        case 4:
            decodeAttention<4><<<batch * o.qHeads, 128, 0, stream>>>(qkv, keys, values,
                table, lengths, starts, active, sinks, y, batch, logicalPages, physicalPages, o);
            break;
        case 16:
            decodeAttention<16><<<batch * o.qHeads, 512, 0, stream>>>(qkv, keys, values,
                table, lengths, starts, active, sinks, y, batch, logicalPages, physicalPages, o);
            break;
        default:
            decodeAttention<8><<<batch * o.qHeads, 256, 0, stream>>>(qkv, keys, values,
                table, lengths, starts, active, sinks, y, batch, logicalPages, physicalPages, o);
            break;
        }
        return cudaGetLastError();
    }
    static const bool fastExp = [] {
        const char* value = std::getenv("GARNET_GPT_OSS_PREFILL_FAST_EXP");
        return value && std::strcmp(value, "1") == 0;
    }();
#ifdef GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL
    static const bool flashPrefill = [] {
        const char* value=std::getenv("GARNET_GPT_OSS_PREFILL_FLASHINFER");
        return value && std::strcmp(value,"1")==0;
    }();
    if(flashPrefill && GptOssFlashAttentionWorkspace(batch,tokens,logicalPages,o))
        return RunGptOssFlashAttention(in,y,workspace,batch,tokens,logicalPages,physicalPages,o,stream);
#endif
    static const bool tiledPrefill = [] {
        const char* value = std::getenv("GARNET_GPT_OSS_PREFILL_TILED_64");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (tiledPrefill && o.prefill && o.headDim == 64 &&
        o.qHeads > 0 && o.kvHeads > 0 && o.qHeads % o.kvHeads == 0) {
        static const int gqaQueries = [] {
            const char* value = std::getenv("GARNET_GPT_OSS_PREFILL_GQA_QUERY_TILE");
            const int requested = value ? std::atoi(value) : 0;
            return requested == 2 || requested == 4 ? requested : 0;
        }();
        if (gqaQueries && o.qHeads == 8 * o.kvHeads)
            return fastExp ? launchGqaPrefill64<true>(in, y, batch, tokens,
                logicalPages, physicalPages, o, gqaQueries, stream) :
                launchGqaPrefill64<false>(in, y, batch, tokens,
                logicalPages, physicalPages, o, gqaQueries, stream);
        static const int queryTile = [] {
            const char* value = std::getenv("GARNET_GPT_OSS_PREFILL_QUERY_TILE");
            return value && std::strcmp(value, "16") == 0 ? 16 : 8;
        }();
        const int blocks = batch * ((tokens + queryTile - 1) / queryTile) * o.qHeads;
        if (fastExp && queryTile == 16)
            tiledPrefillAttention64<true, 16><<<blocks, 512, 0, stream>>>(
                (const float*)in[0], (const __nv_bfloat16*)in[1],
                (const __nv_bfloat16*)in[2], (const int*)in[3],
                (const int*)in[5], (const int*)in[6], (const float*)in[7],
                y, tokens, logicalPages, physicalPages, o);
        else if (fastExp)
            tiledPrefillAttention64<true, 8><<<blocks, 256, 0, stream>>>(
                (const float*)in[0], (const __nv_bfloat16*)in[1],
                (const __nv_bfloat16*)in[2], (const int*)in[3],
                (const int*)in[5], (const int*)in[6], (const float*)in[7],
                y, tokens, logicalPages, physicalPages, o);
        else if (queryTile == 16)
            tiledPrefillAttention64<false, 16><<<blocks, 512, 0, stream>>>(
                (const float*)in[0], (const __nv_bfloat16*)in[1],
                (const __nv_bfloat16*)in[2], (const int*)in[3],
                (const int*)in[5], (const int*)in[6], (const float*)in[7],
                y, tokens, logicalPages, physicalPages, o);
        else
            tiledPrefillAttention64<false, 8><<<blocks, 256, 0, stream>>>(
                (const float*)in[0], (const __nv_bfloat16*)in[1],
                (const __nv_bfloat16*)in[2], (const int*)in[3],
                (const int*)in[5], (const int*)in[6], (const float*)in[7],
                y, tokens, logicalPages, physicalPages, o);
        return cudaGetLastError();
    }
    if (fastExp)
        attention<true><<<(batch * tokens * o.qHeads + 3) / 4, 128, 0, stream>>>(
            (const float*)in[0], (const __nv_bfloat16*)in[1],
            (const __nv_bfloat16*)in[2], (const int*)in[3], (const int*)in[4],
            (const int*)in[5], (const int*)in[6], (const float*)in[7], y,
            batch, tokens, logicalPages, physicalPages, o);
    else
        attention<false><<<(batch * tokens * o.qHeads + 3) / 4, 128, 0, stream>>>(
            (const float*)in[0], (const __nv_bfloat16*)in[1],
            (const __nv_bfloat16*)in[2], (const int*)in[3], (const int*)in[4],
            (const int*)in[5], (const int*)in[6], (const float*)in[7], y,
            batch, tokens, logicalPages, physicalPages, o);
    return cudaGetLastError();
}
static size_t moeWorkspace(int tokens, const GptOssOptions& o, bool grouped) {
    const size_t slots = size_t(tokens) * o.topK;
    size_t bytes = slots * (sizeof(int) + sizeof(float)) +
        size_t(tokens) * o.experts * sizeof(float) + slots * o.intermediate * sizeof(float);
    if (grouped) {
        const size_t tileRows = tokens > 512 ? 64 : 32;
        const size_t tasks = (slots + tileRows - 1) / tileRows + o.experts;
        bytes += sizeof(int) * (o.experts + size_t(o.experts) * tokens * o.topK + 1 + 2 * tasks);
        bytes += sizeof(float) * size_t(tokens) * o.topK * o.hidden;
    }
    return bytes;
}
static cudaError_t runMoe(const void* const* in, float* y, void* workspace, int tokens,
    const GptOssOptions& o, cudaStream_t stream, bool grouped) {
    auto* selected = (int*)workspace;
    auto* probabilities = (float*)(selected + size_t(tokens) * o.topK);
    auto* logits = probabilities + size_t(tokens) * o.topK;
    auto* hidden = logits + size_t(tokens) * o.experts;
    auto status = RunGptOssMoeRoute(in, selected, probabilities, logits, tokens, o, stream);
    if (status != cudaSuccess) return status;
    if (grouped) {
        const int tileRows = tokens > 512 ? 64 : 32;
        auto* counts = (int*)(hidden + size_t(tokens) * o.topK * o.intermediate);
        auto* slots = counts + o.experts;
        auto* taskCount = slots + size_t(o.experts) * tokens * o.topK;
        auto* tasks = taskCount + 1;
        const size_t maxTasks = (size_t(tokens) * o.topK + tileRows - 1) / tileRows + o.experts;
        auto* values = (float*)(tasks + 2 * maxTasks);
        status = cudaMemsetAsync(counts, 0, o.experts * sizeof(int), stream);
        if (status != cudaSuccess) return status;
        bucketExperts<<<(tokens * o.topK + 127) / 128, 128, 0, stream>>>(selected, counts, slots, tokens, o);
        status = cudaGetLastError(); if (status != cudaSuccess) return status;
        if (tileRows == 64)
            expertTasks<64><<<1, 1, 0, stream>>>(counts, taskCount, tasks, o);
        else
            expertTasks<32><<<1, 1, 0, stream>>>(counts, taskCount, tasks, o);
        status = cudaGetLastError(); if (status != cudaSuccess) return status;
        if (tileRows == 64)
            groupedExperts<true, 64><<<dim3(maxTasks, (2 * o.intermediate + 31) / 32), 256, 0, stream>>>(
                (const float*)in[0], (const unsigned char*)in[3], (const unsigned char*)in[4],
                (const float*)in[5], counts, slots, taskCount, tasks, hidden, tokens, o);
        else
            groupedExperts<true, 32><<<dim3(maxTasks, (2 * o.intermediate + 31) / 32), 128, 0, stream>>>(
                (const float*)in[0], (const unsigned char*)in[3], (const unsigned char*)in[4],
                (const float*)in[5], counts, slots, taskCount, tasks, hidden, tokens, o);
        status = cudaGetLastError(); if (status != cudaSuccess) return status;
        if (tileRows == 64)
            groupedExperts<false, 64><<<dim3(maxTasks, (o.hidden + 31) / 32), 256, 0, stream>>>(
                hidden, (const unsigned char*)in[6], (const unsigned char*)in[7],
                (const float*)in[8], counts, slots, taskCount, tasks, values, tokens, o);
        else
            groupedExperts<false, 32><<<dim3(maxTasks, (o.hidden + 31) / 32), 128, 0, stream>>>(
                hidden, (const unsigned char*)in[6], (const unsigned char*)in[7],
                (const float*)in[8], counts, slots, taskCount, tasks, values, tokens, o);
        status = cudaGetLastError(); if (status != cudaSuccess) return status;
        const size_t n = size_t(tokens) * o.hidden;
        combineExperts<<<(n + 127) / 128, 128, 0, stream>>>(values, probabilities, selected, y, tokens, o);
        return cudaGetLastError();
    }
    size_t n = size_t(tokens) * o.topK * o.intermediate;
    gateUp<<<(n + 3) / 4, 128, 0, stream>>>((const float*)in[0],
        (const unsigned char*)in[3], (const unsigned char*)in[4], (const float*)in[5], selected, hidden, tokens, o);
    status = cudaGetLastError(); if (status != cudaSuccess) return status;
    n = size_t(tokens) * o.hidden;
    down<<<(n + 3) / 4, 128, 0, stream>>>(hidden, (const unsigned char*)in[6],
        (const unsigned char*)in[7], (const float*)in[8], selected, probabilities, y, tokens, o);
    return cudaGetLastError();
}
size_t GptOssMoeWorkspace(int tokens, const GptOssOptions& o) {
    return moeWorkspace(tokens, o, tokens >= 16);
}
cudaError_t RunGptOssMoeRoute(const void* const* in, int* selected, float* probabilities,
    float* logits, int tokens, const GptOssOptions& o, cudaStream_t stream,
    const GptOssMarlinDecodeBuffers* fused) {
    const int routerWarps=tensorCoreRouterWarps();
    if(!routerWarps)return cudaErrorInvalidValue;
    const bool fuse = fused && tokens == 1 && o.experts <= 128;
    auto* converted = fuse ? static_cast<__nv_bfloat16*>(fused->convertedInput) : nullptr;
    const int paddedWidth = fuse ? fused->paddedWidth : 0;
    int* locks = fuse ? fused->locks : nullptr;
    const int locksPerExpert = fuse ? fused->locksPerExpert : 0;
    static const int routerThreads = [] {
        const char* value = std::getenv("GARNET_GPT_OSS_ROUTER_THREADS");
        return value && std::atoi(value) == 128 ? 128 : 256;
    }();
    static const bool vector4 = [] {
        const char* value = std::getenv("GARNET_GPT_OSS_ROUTER_VECTOR4");
        return !value || (value[0] == '1' && value[1] == '\0');
    }();
    static const int queryTile=[] {
        const char* value=std::getenv("GARNET_GPT_OSS_ROUTER_QUERY_TILE");
        const int parsed=value?std::atoi(value):0;
        return parsed==2 || parsed==4?parsed:0;
    }();
    const bool tiled=routerThreads==256 && queryTile && tokens>=16 && o.hidden<=4096 && o.experts<=128;
    static const bool tensorCorePrefill=[] {
        const char* value=std::getenv("GARNET_GPT_OSS_PREFILL_ROUTER_TENSORCORE");
        return value && value[0]=='1' && value[1]=='\0';
    }();
    static const bool tensorCoreDecode=[] {
        const char* value=std::getenv("GARNET_GPT_OSS_DECODE_ROUTER_TENSORCORE");
        return value && value[0]=='1' && value[1]=='\0';
    }();
    if((tensorCorePrefill && o.prefill && tokens>=1024 && o.hidden<=4096 && o.experts<=128) ||
        (tensorCoreDecode && GptOssDecodeTensorCoreRouterSupported(
            o.prefill,tokens,o.hidden,o.experts,o.topK)))
        launchTensorCoreRouter((const float*)in[0],(const float*)in[1],(const float*)in[2],
            logits,tokens,o,routerWarps,stream);
    else if(tiled && queryTile==4)
        routeScoresBatch<4><<<((tokens+3)/4)*o.experts,1024,0,stream>>>(
            (const float*)in[0],(const float*)in[1],(const float*)in[2],logits,tokens,o);
    else if(tiled)
        routeScoresBatch<2><<<((tokens+1)/2)*o.experts,512,0,stream>>>(
            (const float*)in[0],(const float*)in[1],(const float*)in[2],logits,tokens,o);
    else if (routerThreads == 256 && vector4 && o.hidden % 4 == 0 && tokens <= 8)
        routeScores<256, true><<<tokens * o.experts, 256, 0, stream>>>(
            (const float*)in[0], (const float*)in[1], (const float*)in[2], logits,
            tokens, o, converted, paddedWidth, locks, locksPerExpert);
    else if (routerThreads == 256)
        routeScores<256><<<tokens * o.experts, 256, 0, stream>>>(
            (const float*)in[0], (const float*)in[1], (const float*)in[2], logits,
            tokens, o, converted, paddedWidth, locks, locksPerExpert);
    else
        routeScores<128><<<tokens * o.experts, 128, 0, stream>>>(
            (const float*)in[0], (const float*)in[1], (const float*)in[2], logits,
            tokens, o, converted, paddedWidth, locks, locksPerExpert);
    auto status = cudaGetLastError(); if (status != cudaSuccess) return status;
    if (fuse)
        routeTopK<true><<<1, 128, 0, stream>>>(logits, selected, probabilities,
            tokens, o, fused->sorted, fused->experts, fused->padded, fused->block);
    else
        routeTopK<false><<<tokens, 128, 0, stream>>>(logits, selected, probabilities,
            tokens, o, nullptr, nullptr, nullptr, 0);
    return cudaGetLastError();
}
cudaError_t RunGptOssMoe(const void* const* in, float* y, void* workspace, int tokens,
    const GptOssOptions& o, cudaStream_t stream) {
    return runMoe(in, y, workspace, tokens, o, stream, tokens >= 16);
}
#ifdef GARNET_GPT_OSS_KERNEL_TEST
cudaError_t TestGptOssBatchRouter(const float* x,const float* weight,const float* bias,
    float* logits,int* selected,float* probabilities,int tokens,const GptOssOptions& o,
    int queryTile,bool topK,cudaStream_t stream) {
    if(tokens<=0 || o.hidden<=0 || o.hidden>4096 || o.experts<=0 || o.experts>128 ||
        o.topK<=0 || o.topK>8 || o.topK>o.experts ||
        (queryTile!=0 && queryTile!=2 && queryTile!=4 && queryTile!=16 && queryTile!=32))return cudaErrorInvalidValue;
    if(!tensorCoreRouterWarps())return cudaErrorInvalidValue;
    if(queryTile==32)launchTensorCoreRouter(x,weight,bias,logits,tokens,o,2,stream);
    else if(queryTile==16)launchTensorCoreRouter(x,weight,bias,logits,tokens,o,4,stream);
    else if(queryTile==4)routeScoresBatch<4><<<((tokens+3)/4)*o.experts,1024,0,stream>>>(x,weight,bias,logits,tokens,o);
    else if(queryTile==2)routeScoresBatch<2><<<((tokens+1)/2)*o.experts,512,0,stream>>>(x,weight,bias,logits,tokens,o);
    else routeScores<256><<<tokens*o.experts,256,0,stream>>>(x,weight,bias,logits,tokens,o,nullptr,0,nullptr,0);
    auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
    if(topK)routeTopK<false><<<tokens,128,0,stream>>>(logits,selected,probabilities,tokens,o,nullptr,nullptr,nullptr,0);
    return cudaGetLastError();
}
size_t TestGptOssMoeWorkspace(int tokens, const GptOssOptions& o, bool grouped) {
    return moeWorkspace(tokens, o, grouped);
}
cudaError_t TestGptOssMoe(const void* const* in, float* y, void* workspace, int tokens,
    const GptOssOptions& o, cudaStream_t stream, bool grouped) {
    return runMoe(in, y, workspace, tokens, o, stream, grouped);
}
#endif
}
