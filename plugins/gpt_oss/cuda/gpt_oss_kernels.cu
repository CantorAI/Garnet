// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_kernels.h"
#include <cuda_bf16.h>
#include <cmath>
#include <cfloat>

namespace Garnet {
namespace {
__device__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ float fp4(const unsigned char* blocks, const unsigned char* scales,
    size_t row, int column, int width) {
    const unsigned char byte = blocks[row * (width / 2) + column / 2];
    const int code = (column & 1) ? byte >> 4 : byte & 15;
    const float values[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    const unsigned char exponent = scales[row * (width / 32) + column / 32];
    // E8M0 255 is NaN, not an ordinary exponent.
    if (exponent == 255) return nanf("");
    return bf(ldexpf((code & 8) ? -values[code & 7] : values[code], int(exponent) - 127));
}
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
// Correctness baseline: online stable softmax, one thread per query head.
// It includes the learned sink in the denominator but never as a value token.
__global__ void attention(const float* qkv, const __nv_bfloat16* keys,
    const __nv_bfloat16* values, const int* table, const int* lengths,
    const int* starts, const int* active, const float* sinks, float* output,
    int batch, int tokens, int logicalPages, int physicalPages, GptOssOptions o) {
    const int task = blockIdx.x * blockDim.x + threadIdx.x;
    if (task >= batch * tokens * o.qHeads) return;
    const int head = task % o.qHeads, row = task / o.qHeads, b = row / tokens, t = row % tokens;
    const int packed = (o.qHeads + 2 * o.kvHeads) * o.headDim;
    float accum[128];
    for (int d = 0; d < o.headDim; ++d) accum[d] = 0;
    const int end = o.prefill ? starts[b] + t + 1 : lengths[b];
    const int begin = o.window ? max(0, end - o.window) : 0;
    float maximum = sinks[head], sum = 1;
    if (active[b] && end > 0 && end <= logicalPages * o.pageSize) {
        for (int p = begin; p < end; ++p) {
            const size_t offset = cacheOffset(table, logicalPages, physicalPages, b, p,
                head / (o.qHeads / o.kvHeads), 0, o);
            if (offset == SIZE_MAX) continue;
            float score = 0;
            for (int d = 0; d < o.headDim; ++d)
                score += qkv[size_t(row) * packed + head * o.headDim + d] * __bfloat162float(keys[offset + d]);
            score *= rsqrtf(float(o.headDim));
            const float next = fmaxf(maximum, score), old = expf(maximum - next), weight = expf(score - next);
            sum = sum * old + weight;
            for (int d = 0; d < o.headDim; ++d)
                accum[d] = accum[d] * old + weight * __bfloat162float(values[offset + d]);
            maximum = next;
        }
    }
    for (int d = 0; d < o.headDim; ++d)
        output[size_t(row) * o.qHeads * o.headDim + head * o.headDim + d] = bf(accum[d] / sum);
}
__global__ void route(const float* x, const float* weight, const float* bias,
    int* selected, float* probabilities, int tokens, GptOssOptions o) {
    const int token = blockIdx.x, expert = threadIdx.x;
    extern __shared__ float logits[];
    if (expert < o.experts) {
        float value = 0;
        for (int d = 0; d < o.hidden; ++d) value += x[size_t(token) * o.hidden + d] * weight[size_t(expert) * o.hidden + d];
        logits[expert] = bf(value + bias[expert]);
    }
    __syncthreads();
    if (expert != 0) return;
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
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(tokens) * o.topK * o.intermediate) return;
    const int i = index % o.intermediate, slot = index / o.intermediate, token = slot / o.topK;
    const int expert = selected[slot];
    const size_t row = size_t(expert) * 2 * o.intermediate + 2 * i;
    float gate = 0, up = 0;
    for (int d = 0; d < o.hidden; ++d) {
        const float v = x[size_t(token) * o.hidden + d];
        gate += v * fp4(blocks, scales, row, d, o.hidden);
        up += v * fp4(blocks, scales, row + 1, d, o.hidden);
    }
    gate = fminf(bf(bf(gate) + bias[row]), o.limit);
    up = fminf(o.limit, fmaxf(-o.limit, bf(bf(up) + bias[row + 1])));
    const float glu = bf(gate * bf(1.f / (1.f + expf(-bf(1.702f * gate)))));
    hidden[index] = bf(glu * bf(up + 1));
}
__global__ void down(const float* hidden, const unsigned char* blocks, const unsigned char* scales,
    const float* bias, const int* selected, const float* probabilities, float* y,
    int tokens, GptOssOptions o) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(tokens) * o.hidden) return;
    const int token = index / o.hidden, d = index % o.hidden;
    float result = 0;
    for (int k = 0; k < o.topK; ++k) {
        const int slot = token * o.topK + k, expert = selected[slot];
        const size_t row = size_t(expert) * o.hidden + d;
        float value = 0;
        for (int i = 0; i < o.intermediate; ++i)
            value += hidden[size_t(slot) * o.intermediate + i] * fp4(blocks, scales, row, i, o.intermediate);
        result += bf(bf(value) + bias[row]) * probabilities[slot];
    }
    y[index] = bf(result);
}
}
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
    attention<<<(batch * tokens * o.qHeads + 31) / 32, 32, 0, stream>>>(
        (const float*)in[0], (const __nv_bfloat16*)in[1], (const __nv_bfloat16*)in[2],
        (const int*)in[3], (const int*)in[4], (const int*)in[5], (const int*)in[6],
        (const float*)in[7], y, batch, tokens, logicalPages, physicalPages, o);
    return cudaGetLastError();
}
size_t GptOssMoeWorkspace(int tokens, const GptOssOptions& o) {
    return size_t(tokens) * o.topK * (sizeof(int) + sizeof(float) * (1 + o.intermediate));
}
cudaError_t RunGptOssMoe(const void* const* in, float* y, void* workspace, int tokens,
    const GptOssOptions& o, cudaStream_t stream) {
    auto* selected = (int*)workspace;
    auto* probabilities = (float*)(selected + size_t(tokens) * o.topK);
    auto* hidden = probabilities + size_t(tokens) * o.topK;
    route<<<tokens, 256, o.experts * sizeof(float), stream>>>((const float*)in[0],
        (const float*)in[1], (const float*)in[2], selected, probabilities, tokens, o);
    auto status = cudaGetLastError(); if (status != cudaSuccess) return status;
    size_t n = size_t(tokens) * o.topK * o.intermediate;
    gateUp<<<(n + 127) / 128, 128, 0, stream>>>((const float*)in[0],
        (const unsigned char*)in[3], (const unsigned char*)in[4], (const float*)in[5], selected, hidden, tokens, o);
    status = cudaGetLastError(); if (status != cudaSuccess) return status;
    n = size_t(tokens) * o.hidden;
    down<<<(n + 127) / 128, 128, 0, stream>>>(hidden, (const unsigned char*)in[6],
        (const unsigned char*)in[7], (const float*)in[8], selected, probabilities, y, tokens, o);
    return cudaGetLastError();
}
}
