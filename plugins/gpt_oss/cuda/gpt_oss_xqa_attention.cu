// SPDX-License-Identifier: Apache-2.0
// Own mask/page/workspace adapter around the separately marked XQA backend.
// Staged and unlinked pending native/compiled/sanitizer/model proof.
#include "gpt_oss_xqa_attention.h"
#include "gpt_oss_xqa_layout.h"
#include "gpt_oss_xqa_bridge.h"
#include <cuda_bf16.h>
#include <cmath>
#include <limits>

namespace Garnet {
namespace {
using Bf16 = __nv_bfloat16;
template<class T> T* slot(void* workspace, std::size_t offset) {
    return reinterpret_cast<T*>(static_cast<char*>(workspace) + offset);
}
__global__ void gptOssXqaPrepare(const float* qkv, const int* rawTable,
    const int* rawLengths, const int* active, Bf16* query, int* table,
    uint32_t* lengths, int* modes, int logical, int physical, GptOssOptions o) {
    const int b = blockIdx.x;
    __shared__ int mode;
    if (!threadIdx.x) {
        mode = int(GptOssXqaClassifyRow(rawTable + std::size_t(b) * logical,
            logical, physical, rawLengths[b], active[b], o.window));
        modes[b] = mode;
        // XQA must always see a positive in-capacity length, even for a row
        // that will be zeroed or evaluated by complete-mask fallback.
        lengths[b] = mode == int(GptOssXqaRowMode::Dense) ? uint32_t(rawLengths[b]) : 1U;
    }
    __syncthreads();
    for (int page = threadIdx.x; page < logical; page += blockDim.x) {
        const auto i = std::size_t(b) * logical + page;
        const int raw = rawTable[i];
        // Sanitize ALL entries, not merely the visible interval. Upstream
        // tile prefetch can address masked and tail page-table entries.
        table[i] = GptOssXqaValidPage(raw, physical) ? raw : 0;
    }
    const int width = o.qHeads * 64, packed = (o.qHeads + 2 * o.kvHeads) * 64;
    for (int d = threadIdx.x; d < width; d += blockDim.x)
        query[std::size_t(b) * width + d] = __float2bfloat16(
            mode == int(GptOssXqaRowMode::Dense) ? qkv[std::size_t(b) * packed + d] : 0.f);
}
__device__ float xqaWarpSum(float value) {
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}
// One warp/head finalizes dense BF16 output, zeros invalid rows, or evaluates
// precisely the visible valid tokens with the original scalar online softmax.
// Dense output is preserved; holes outside the window do not force fallback.
__global__ void gptOssXqaFinalize(const Bf16* dense, const float* qkv,
    const Bf16* keys, const Bf16* values, const int* table, const int* lengths,
    const float* sinks, const int* modes, float* output, int batch,
    int logical, int physical, GptOssOptions o) {
    const int lane = threadIdx.x & 31;
    const int task = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (task >= batch * o.qHeads) return;
    const int b = task / o.qHeads, head = task % o.qHeads;
    const auto base = std::size_t(task) * 64;
    const int mode = modes[b];
    if (mode != int(GptOssXqaRowMode::HoleFallback)) {
        for (int d = lane; d < 64; d += 32)
            output[base + d] = mode == int(GptOssXqaRowMode::Dense)
                ? __bfloat162float(dense[base + d]) : 0.f;
        return;
    }
    const int end = lengths[b], begin = GptOssXqaBegin(end, o.window);
    const int packed = (o.qHeads + 2 * o.kvHeads) * 64, kvHead = head / 8;
    float accum[2] = {}, maximum = sinks[head], sum = 1.f;
    for (int position = begin; position < end; ++position) {
        const int page = table[std::size_t(b) * logical + position / 16];
        if (!GptOssXqaValidPage(page, physical)) continue;
        const auto offset = ((std::size_t(page) * 16 + position % 16) * o.kvHeads + kvHead) * 64;
        float score = 0.f;
        for (int d = lane; d < 64; d += 32)
            score += qkv[std::size_t(b) * packed + head * 64 + d] * __bfloat162float(keys[offset + d]);
        score = __shfl_sync(0xffffffff, xqaWarpSum(score), 0) * .125f;
        const float next = fmaxf(maximum, score);
        const float old = expf(maximum - next), weight = expf(score - next);
        sum = sum * old + weight;
        for (int d = lane; d < 64; d += 32)
            accum[d / 32] = accum[d / 32] * old + weight * __bfloat162float(values[offset + d]);
        maximum = next;
    }
    for (int d = lane; d < 64; d += 32)
        output[base + d] = __bfloat162float(__float2bfloat16(accum[d / 32] / sum));
}
}
cudaError_t InitializeGptOssXqaContext(GptOssXqaContextState* state) {
    if (!state) return cudaErrorInvalidValue;
    *state = {};
    auto status = cudaGetDevice(&state->device);
    if (status != cudaSuccess) return status;
    status = GarnetGptOssXqaInitialize(&state->dynamicSharedBytes);
    if (status != cudaSuccess || !state->dynamicSharedBytes) {
        *state = {};
        return status == cudaSuccess ? cudaErrorInvalidValue : status;
    }
    return cudaSuccess;
}
std::size_t GptOssXqaAttentionWorkspace(int batch, int tokens, int logical, const GptOssOptions& o) {
    return GptOssXqaSupported(batch, tokens, logical, o.qHeads, o.kvHeads,
        o.headDim, o.pageSize, o.window, o.prefill)
        ? GptOssXqaLayout(batch, o.qHeads, logical).bytes : 0;
}
cudaError_t RunGptOssXqaAttention(const void* const* in, float* y, void* workspace,
    int batch, int tokens, int logical, int physical, const GptOssOptions& o,
    const GptOssXqaContextState& state, cudaStream_t stream) {
    if (!GptOssXqaAttentionWorkspace(batch, tokens, logical, o) || physical <= 0 ||
        o.layer < 0 || !in || !y || !workspace || state.device < 0 || !state.dynamicSharedBytes)
        return cudaErrorInvalidValue;
    for (int i = 0; i < 8; ++i) if (!in[i]) return cudaErrorInvalidValue;
    int current = -1;
    auto status = cudaGetDevice(&current);
    if (status != cudaSuccess) return status;
    if (current != state.device) return cudaErrorInvalidDevice;
    // Tensor descriptors/caller own the full extent; guard arithmetic before
    // forming the current layer pointer. No allocation or initialization here.
    const auto perLayer = std::size_t(physical) * 16 * o.kvHeads * 64;
    if (std::size_t(o.layer) > std::numeric_limits<std::size_t>::max() / perLayer)
        return cudaErrorInvalidValue;
    const auto layerOffset = std::size_t(o.layer) * perLayer;
    const auto* keys = static_cast<const Bf16*>(in[1]) + layerOffset;
    const auto* values = static_cast<const Bf16*>(in[2]) + layerOffset;
    const GptOssXqaLayout layout(batch, o.qHeads, logical);
    gptOssXqaPrepare<<<batch, 256, 0, stream>>>(static_cast<const float*>(in[0]),
        static_cast<const int*>(in[3]), static_cast<const int*>(in[4]),
        static_cast<const int*>(in[6]), slot<Bf16>(workspace, layout.query),
        slot<int>(workspace, layout.table), slot<uint32_t>(workspace, layout.lengths),
        slot<int>(workspace, layout.modes), logical, physical, o);
    status = cudaGetLastError(); if (status != cudaSuccess) return status;
    status = GarnetGptOssXqaLaunchBf16SingleBlock(slot<Bf16>(workspace, layout.query),
        keys, values, static_cast<const float*>(in[7]), slot<int>(workspace, layout.table),
        slot<uint32_t>(workspace, layout.lengths), slot<Bf16>(workspace, layout.output),
        batch, o.kvHeads, logical * 16, o.window, state.dynamicSharedBytes, stream);
    if (status != cudaSuccess) return status;
    gptOssXqaFinalize<<<(batch * o.qHeads + 3) / 4, 128, 0, stream>>>(
        slot<Bf16>(workspace, layout.output), static_cast<const float*>(in[0]), keys, values,
        static_cast<const int*>(in[3]), static_cast<const int*>(in[4]),
        static_cast<const float*>(in[7]), slot<int>(workspace, layout.modes), y,
        batch, logical, physical, o);
    return cudaGetLastError();
}
}
