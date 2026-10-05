// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_kernels.h"
#include <cuda_bf16.h>
#include <mma.h>
#include <cmath>
#include <cfloat>

namespace Garnet {
namespace {
__device__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ float warpSum(float value) {
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
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
            const float next = fmaxf(maximum, score), old = expf(maximum - next), weight = expf(score - next);
            sum = sum * old + weight;
            for (int d = lane; d < o.headDim; d += 32)
                accum[d / 32] = accum[d / 32] * old + weight * __bfloat162float(values[offset + d]);
            maximum = next;
        }
    }
    for (int d = lane; d < o.headDim; d += 32)
        output[size_t(row) * o.qHeads * o.headDim + head * o.headDim + d] = bf(accum[d / 32] / sum);
}
// Decode has too few query rows to hide a serial context walk. Split each
// head's paged KV range across eight warps, then merge stable softmax states.
// Include the learned sink exactly once in the final merge.
__global__ void decodeAttention(const float* qkv, const __nv_bfloat16* keys,
    const __nv_bfloat16* values, const int* table, const int* lengths,
    const int* starts, const int* active, const float* sinks, float* output,
    int batch, int logicalPages, int physicalPages, GptOssOptions o) {
    constexpr int warps = 8;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const int head = blockIdx.x % o.qHeads, b = blockIdx.x / o.qHeads;
    const int packed = (o.qHeads + 2 * o.kvHeads) * o.headDim;
    const int end = lengths[b], begin = o.window ? max(0, end - o.window) : 0;
    const bool valid = active[b] && end > 0 && end <= logicalPages * o.pageSize;
    const int chunk = (max(0, end - begin) + warps - 1) / warps;
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
    __shared__ float maxima[warps], sums[warps], partial[warps][128];
    if (!lane) { maxima[warp] = maximum; sums[warp] = sum; }
    for (int d = lane; d < o.headDim; d += 32) partial[warp][d] = accum[d / 32];
    __syncthreads();
    if (warp) return;
    maximum = sinks[head];
    for (int w = 0; w < warps; ++w) if (sums[w] > 0) maximum = fmaxf(maximum, maxima[w]);
    sum = expf(sinks[head] - maximum);
    float weights[warps];
    for (int w = 0; w < warps; ++w) {
        weights[w] = sums[w] > 0 ? expf(maxima[w] - maximum) : 0;
        sum += sums[w] * weights[w];
    }
    for (int d = lane; d < o.headDim; d += 32) {
        float value = 0;
        for (int w = 0; w < warps; ++w) value += partial[w][d] * weights[w];
        output[size_t(b) * o.qHeads * o.headDim + head * o.headDim + d] = bf(value / sum);
    }
}
__global__ void route(const float* x, const float* weight, const float* bias,
    int* selected, float* probabilities, int tokens, GptOssOptions o) {
    const int token = blockIdx.x, lane = threadIdx.x & 31;
    extern __shared__ float logits[];
    for (int expert = threadIdx.x / 32; expert < o.experts; expert += blockDim.x / 32) {
        float value = 0;
        for (int d = lane; d < o.hidden; d += 32) value += x[size_t(token) * o.hidden + d] * weight[size_t(expert) * o.hidden + d];
        value = warpSum(value);
        if (!lane) logits[expert] = bf(value + bias[expert]);
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    float top[8];
    for (int k = 0; k < o.topK; ++k) {
        int best = 0;
        for (int e = 1; e < o.experts; ++e) if (logits[e] > logits[best]) best = e;
        selected[token * o.topK + k] = best; top[k] = logits[best]; logits[best] = -FLT_MAX;
    }
    float total = 0;
    for (int k = 0; k < o.topK; ++k) total += expf(top[k] - top[0]);
    for (int k = 0; k < o.topK; ++k) probabilities[token * o.topK + k] = bf(expf(top[k] - top[0]) / total);
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
    const size_t row = size_t(expert) * 2 * o.intermediate + 2 * i;
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
        const size_t row = size_t(expert) * o.hidden + d;
        float value = 0;
        for (int i = lane; i < o.intermediate; i += 32)
            value += hidden[size_t(slot) * o.intermediate + i] * fp4(blocks, scales, row, i, o.intermediate);
        value = warpSum(value);
        if (!lane) result += bf(bf(value) + bias[row]) * probabilities[slot];
    }
    if (!lane) y[index] = bf(result);
}

__global__ void bucketExperts(const int* selected, int* counts, int* slots,
    int tokens, GptOssOptions o) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= tokens * o.topK) return;
    const int expert = selected[slot];
    const int row = atomicAdd(counts + expert, 1);
    slots[size_t(expert) * tokens * o.topK + row] = slot;
}
__global__ void expertTasks(const int* counts, int* taskCount, int* tasks,
    GptOssOptions o) {
    // Compact tiles rather than launching the worst-case token count for
    // every expert. Every routed slot has its own row, including repeated
    // selections if nonfinite router inputs reach this low-level operator.
    int total = 0;
    for (int e = 0; e < o.experts; ++e)
        for (int row = 0; row < counts[e]; row += 16) {
            tasks[2 * total] = e;
            tasks[2 * total + 1] = row;
            ++total;
        }
    *taskCount = total;
}
// W4A16 grouped prefill: unpack one weight tile into shared BF16, reuse it
// across sixteen routed tokens, and accumulate on BF16 Tensor Cores. Weight
// storage remains MXFP4; there is no full-model BF16 materialization.
template<bool Up>
__global__ void groupedExperts(const float* x, const unsigned char* blocks,
    const unsigned char* scales, const float* bias, const int* counts,
    const int* slots, const int* taskCount, const int* tasks, float* output,
    int tokens, GptOssOptions o) {
    if (blockIdx.x >= *taskCount) return;
    const int expert = tasks[2 * blockIdx.x], first = tasks[2 * blockIdx.x + 1];
    const int nStart = blockIdx.y * 32;
    const int width = Up ? o.hidden : o.intermediate;
    const int outputs = Up ? 2 * o.intermediate : o.hidden;
    __shared__ __align__(32) __nv_bfloat16 a[16 * 32];
    __shared__ __align__(32) __nv_bfloat16 b[32 * 32];
    __shared__ __align__(32) float c[16 * 32];
    const int warp = threadIdx.x / 32;
    using namespace nvcuda;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    for (int kStart = 0; kStart < width; kStart += 32) {
        for (int index = threadIdx.x; index < 16 * 32; index += blockDim.x) {
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
                ? fp4(blocks, scales, size_t(expert) * outputs + nStart + n,
                    kStart + k, width) : 0;
            b[index] = __float2bfloat16(value);
        }
        __syncthreads();
        for (int sub = 0; sub < 32; sub += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
            wmma::load_matrix_sync(af, a + sub, 32);
            wmma::load_matrix_sync(bf, b + warp * 16 * 32 + sub, 32);
            wmma::mma_sync(acc, af, bf, acc);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(c + warp * 16, acc, 32, wmma::mem_row_major);
    __syncthreads();
    for (int index = threadIdx.x; index < 16 * 32; index += blockDim.x) {
        const int m = index / 32, n = index % 32;
        if (first + m >= counts[expert] || nStart + n >= outputs) continue;
        const int slot = slots[size_t(expert) * tokens * o.topK + first + m];
        const size_t weightRow = size_t(expert) * outputs + nStart + n;
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
__global__ void combineExperts(const float* values, const float* probabilities,
    float* y, int tokens, GptOssOptions o) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(tokens) * o.hidden) return;
    const int token = index / o.hidden, d = index % o.hidden;
    float result = 0;
    // Preserve router order regardless of atomic bucketing order.
    for (int k = 0; k < o.topK; ++k) {
        const int slot = token * o.topK + k;
        result += values[size_t(slot) * o.hidden + d] * probabilities[slot];
    }
    y[index] = bf(result);
}
}
#ifdef GARNET_GPT_OSS_KERNEL_TEST
cudaError_t TestGptOssMxfp4Decode(const unsigned char* blocks, const unsigned char* scales,
    float* output, int rows) {
    testMxfp4Decode<<<(rows * 32 + 127) / 128, 128>>>(blocks, scales, output, rows);
    return cudaGetLastError();
}
#endif
cudaError_t RunGptOssRope(const float* x, const std::int64_t* p, float* y, int tokens,
    const GptOssOptions& o, cudaStream_t stream) {
    const size_t n = size_t(tokens) * (o.qHeads + 2 * o.kvHeads) * o.headDim;
    rope<<<(n + 255) / 256, 256, 0, stream>>>(x, p, y, tokens, o);
    return cudaGetLastError();
}
cudaError_t RunGptOssAttention(const void* const* in, float* y, int batch, int tokens,
    int logicalPages, int physicalPages, const GptOssOptions& o, cudaStream_t stream) {
    const size_t n = size_t(batch) * tokens * o.kvHeads * o.headDim;
    writeKV<<<(n + 255) / 256, 256, 0, stream>>>(static_cast<const float*>(in[0]),
        (__nv_bfloat16*)in[1], (__nv_bfloat16*)in[2], (const int*)in[3],
        (const int*)in[5], (const int*)in[6], batch, tokens, logicalPages, physicalPages, o);
    auto status = cudaGetLastError(); if (status != cudaSuccess) return status;
    if (tokens == 1 && !o.prefill) {
        decodeAttention<<<batch * o.qHeads, 256, 0, stream>>>(
            (const float*)in[0], (const __nv_bfloat16*)in[1], (const __nv_bfloat16*)in[2],
            (const int*)in[3], (const int*)in[4], (const int*)in[5], (const int*)in[6],
            (const float*)in[7], y, batch, logicalPages, physicalPages, o);
        return cudaGetLastError();
    }
    attention<<<(batch * tokens * o.qHeads + 3) / 4, 128, 0, stream>>>(
        (const float*)in[0], (const __nv_bfloat16*)in[1], (const __nv_bfloat16*)in[2],
        (const int*)in[3], (const int*)in[4], (const int*)in[5], (const int*)in[6],
        (const float*)in[7], y, batch, tokens, logicalPages, physicalPages, o);
    return cudaGetLastError();
}
static size_t moeWorkspace(int tokens, const GptOssOptions& o, bool grouped) {
    size_t bytes = size_t(tokens) * o.topK * (sizeof(int) + sizeof(float) * (1 + o.intermediate));
    if (grouped) {
        const size_t tasks = (size_t(tokens) * o.topK + 15) / 16 + o.experts;
        bytes += sizeof(int) * (o.experts + size_t(o.experts) * tokens * o.topK + 1 + 2 * tasks);
        bytes += sizeof(float) * size_t(tokens) * o.topK * o.hidden;
    }
    return bytes;
}
static cudaError_t runMoe(const void* const* in, float* y, void* workspace, int tokens,
    const GptOssOptions& o, cudaStream_t stream, bool grouped) {
    auto* selected = (int*)workspace;
    auto* probabilities = (float*)(selected + size_t(tokens) * o.topK);
    auto* hidden = probabilities + size_t(tokens) * o.topK;
    route<<<tokens, 256, o.experts * sizeof(float), stream>>>((const float*)in[0],
        (const float*)in[1], (const float*)in[2], selected, probabilities, tokens, o);
    auto status = cudaGetLastError(); if (status != cudaSuccess) return status;
    if (grouped) {
        auto* counts = (int*)(hidden + size_t(tokens) * o.topK * o.intermediate);
        auto* slots = counts + o.experts;
        auto* taskCount = slots + size_t(o.experts) * tokens * o.topK;
        auto* tasks = taskCount + 1;
        const size_t maxTasks = (size_t(tokens) * o.topK + 15) / 16 + o.experts;
        auto* values = (float*)(tasks + 2 * maxTasks);
        status = cudaMemsetAsync(counts, 0, o.experts * sizeof(int), stream);
        if (status != cudaSuccess) return status;
        bucketExperts<<<(tokens * o.topK + 127) / 128, 128, 0, stream>>>(selected, counts, slots, tokens, o);
        status = cudaGetLastError(); if (status != cudaSuccess) return status;
        expertTasks<<<1, 1, 0, stream>>>(counts, taskCount, tasks, o);
        status = cudaGetLastError(); if (status != cudaSuccess) return status;
        groupedExperts<true><<<dim3(maxTasks, (2 * o.intermediate + 31) / 32), 64, 0, stream>>>(
            (const float*)in[0], (const unsigned char*)in[3], (const unsigned char*)in[4],
            (const float*)in[5], counts, slots, taskCount, tasks, hidden, tokens, o);
        status = cudaGetLastError(); if (status != cudaSuccess) return status;
        groupedExperts<false><<<dim3(maxTasks, (o.hidden + 31) / 32), 64, 0, stream>>>(
            hidden, (const unsigned char*)in[6], (const unsigned char*)in[7],
            (const float*)in[8], counts, slots, taskCount, tasks, values, tokens, o);
        status = cudaGetLastError(); if (status != cudaSuccess) return status;
        const size_t n = size_t(tokens) * o.hidden;
        combineExperts<<<(n + 127) / 128, 128, 0, stream>>>(values, probabilities, y, tokens, o);
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
    int tokens, const GptOssOptions& o, cudaStream_t stream) {
    route<<<tokens, 256, o.experts * sizeof(float), stream>>>((const float*)in[0],
        (const float*)in[1], (const float*)in[2], selected, probabilities, tokens, o);
    return cudaGetLastError();
}
cudaError_t RunGptOssMoe(const void* const* in, float* y, void* workspace, int tokens,
    const GptOssOptions& o, cudaStream_t stream) {
    return runMoe(in, y, workspace, tokens, o, stream, tokens >= 16);
}
#ifdef GARNET_GPT_OSS_KERNEL_TEST
size_t TestGptOssMoeWorkspace(int tokens, const GptOssOptions& o, bool grouped) {
    return moeWorkspace(tokens, o, grouped);
}
cudaError_t TestGptOssMoe(const void* const* in, float* y, void* workspace, int tokens,
    const GptOssOptions& o, cudaStream_t stream, bool grouped) {
    return runMoe(in, y, workspace, tokens, o, stream, grouped);
}
#endif
}
