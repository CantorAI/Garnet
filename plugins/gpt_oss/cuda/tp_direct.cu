// SPDX-License-Identifier: Apache-2.0
// Experimental two-rank, small-message reduction for PCIe peer-access GPUs.
#include "tp_direct.h"
#include <cstdint>
#include <cstdlib>

namespace Garnet {
namespace {
constexpr size_t kCount = 2880; // GPT-OSS-120B hidden size, batch 1 decode.
constexpr int kMaxBlocks = 32;
struct alignas(128) Signal {
    alignas(128) unsigned start[kMaxBlocks][2];
    alignas(128) unsigned end[kMaxBlocks][2];
    alignas(128) unsigned seq[kMaxBlocks];
};
float* g_staging[2]{};
Signal* g_signal[2]{};
bool g_ready = false;
bool g_batchEnabled = false;
bool g_largeBatchEnabled = false;
int g_batchBlocks = 0;
size_t g_maxCount = kCount * 128;

__device__ __forceinline__ void storeRelease(unsigned* ptr, unsigned value) {
    asm volatile("st.release.sys.global.u32 [%1], %0;" :: "r"(value), "l"(ptr));
}
__device__ __forceinline__ unsigned loadAcquire(const unsigned* ptr) {
    unsigned value;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(ptr));
    return value;
}

__global__ void directReduce(const float* rank0, const float* rank1,
    float* output, Signal* mine, Signal* other, int rank, int count) {
    const int block = blockIdx.x;
    const unsigned epoch = mine->seq[block] + 1;
    if (threadIdx.x < 2) {
        Signal* target = threadIdx.x == rank ? mine : other;
        storeRelease(&target->start[block][rank], epoch);
        while (loadAcquire(&mine->start[block][threadIdx.x]) != epoch)
            __nanosleep(32);
    }
    __syncthreads();
    if (threadIdx.x == 0) mine->seq[block] = epoch;
    const auto* a = reinterpret_cast<const float4*>(rank0);
    const auto* b = reinterpret_cast<const float4*>(rank1);
    auto* y = reinterpret_cast<float4*>(output);
    for (int i = block * blockDim.x + threadIdx.x; i < count / 4;
         i += gridDim.x * blockDim.x) {
        const float4 x = a[i], z = b[i];
        y[i] = make_float4(x.x + z.x, x.y + z.y, x.z + z.z, x.w + z.w);
    }
    __syncthreads();
    if (threadIdx.x < 2) {
        Signal* target = threadIdx.x == rank ? mine : other;
        storeRelease(&target->end[block][rank], epoch);
        while (loadAcquire(&mine->end[block][threadIdx.x]) != epoch)
            __nanosleep(32);
    }
}

void cleanup() {
    for (int rank = 0; rank < 2; ++rank) {
        if (cudaSetDevice(rank) != cudaSuccess) continue;
        if (g_staging[rank]) cudaFree(g_staging[rank]);
        if (g_signal[rank]) cudaFree(g_signal[rank]);
        g_staging[rank] = nullptr;
        g_signal[rank] = nullptr;
    }
    g_ready = false;
    g_batchEnabled = false;
    g_largeBatchEnabled = false;
    g_batchBlocks = 0;
    g_maxCount = kCount * 128;
}
}

cudaError_t GptOssTpDirectAcquire() {
    if (g_ready) return cudaSuccess;
    int original = -1;
    auto status = cudaGetDevice(&original);
    if (status != cudaSuccess) return status;
    const char* capacity = std::getenv("GARNET_GPT_OSS_DIRECT_MAX_BATCH");
    const int maximum = capacity ? std::atoi(capacity) : 128;
    if (maximum != 128 && maximum != 256 && maximum != 512)
        return cudaErrorInvalidValue;
    g_maxCount = kCount * size_t(maximum);
    for (int rank = 0; rank < 2; ++rank) {
        status = cudaSetDevice(rank);
        if (status != cudaSuccess) break;
        int canPeer = 0;
        status = cudaDeviceCanAccessPeer(&canPeer, rank, 1 - rank);
        if (status != cudaSuccess || !canPeer) {
            if (status == cudaSuccess) status = cudaErrorNotSupported;
            break;
        }
        status = cudaDeviceEnablePeerAccess(1 - rank, 0);
        if (status == cudaErrorPeerAccessAlreadyEnabled) {
            cudaGetLastError();
            status = cudaSuccess;
        }
        if (status != cudaSuccess) break;
        status = cudaMalloc(&g_staging[rank], g_maxCount * sizeof(float));
        if (status != cudaSuccess) break;
        status = cudaMalloc(&g_signal[rank], sizeof(Signal));
        if (status != cudaSuccess) break;
        status = cudaMemset(g_signal[rank], 0, sizeof(Signal));
        if (status != cudaSuccess) break;
    }
    if (status != cudaSuccess) cleanup();
    else {
        g_ready = true;
        const char* batch = std::getenv("GARNET_GPT_OSS_DIRECT_BATCH_ALLREDUCE");
        g_batchEnabled = batch && batch[0] == '1' && batch[1] == '\0';
        const char* largeBatch = std::getenv("GARNET_GPT_OSS_DIRECT_LARGE_BATCH_ALLREDUCE");
        g_largeBatchEnabled = largeBatch && largeBatch[0] == '1' && largeBatch[1] == '\0';
        if (const char* blocks = std::getenv("GARNET_GPT_OSS_DIRECT_BATCH_CTAS")) {
            const int value = std::atoi(blocks);
            if (value == 2 || value == 4 || value == 8 || value == 16 || value == 32)
                g_batchBlocks = value;
        }
    }
    const auto restore = cudaSetDevice(original);
    return status == cudaSuccess ? restore : status;
}

void GptOssTpDirectRelease() {
    int original = -1;
    if (cudaGetDevice(&original) != cudaSuccess) return;
    cleanup();
    cudaSetDevice(original);
}

cudaError_t GptOssTpDirectAllReduce(const float* input, float* output,
    size_t count, int rank, cudaStream_t stream) {
    if (!g_ready || (count != kCount && (!g_batchEnabled || !count ||
        count > g_maxCount || count % kCount ||
        (count > kCount * 64 && !g_largeBatchEnabled)))) return cudaErrorNotSupported;
    if (!input || !output || rank < 0 || rank > 1) return cudaErrorInvalidValue;
    auto status = cudaMemcpyAsync(g_staging[rank], input,
        count * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    if (status != cudaSuccess) return status;
    int blocks = 2;
    if (count != kCount) {
        if (g_batchBlocks) blocks = g_batchBlocks;
        else while (blocks < kMaxBlocks && size_t(blocks * 512 * 4) < count)
            blocks *= 2;
    }
    // The paired rank streams must serialize all collectives. This opt-in
    // screen reuses communicator-wide staging and is not safe for overlapping
    // independent model executions; the exclusive benchmark runner enforces
    // that restriction. Each CTA's end handshake protects its staging reads
    // before either rank may enqueue the next staging copy.
    directReduce<<<blocks, 512, 0, stream>>>(g_staging[0], g_staging[1], output,
        g_signal[rank], g_signal[1 - rank], rank, static_cast<int>(count));
    return cudaGetLastError();
}
}
