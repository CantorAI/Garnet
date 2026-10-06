// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_kernels.h"
#include <array>
#include <mutex>

#ifdef GARNET_GPT_OSS_ENABLE_NCCL
#include <nccl.h>
#endif

namespace Garnet {
namespace {
std::mutex g_tpMutex;
int g_tpUsers = 0;
bool g_tpReady = false;
#ifdef GARNET_GPT_OSS_ENABLE_NCCL
std::array<ncclComm_t, 2> g_tpComms{};
cudaError_t NcclStatus(ncclResult_t status) {
    return status == ncclSuccess ? cudaSuccess : cudaErrorUnknown;
}
__global__ void interleaveVocab(const float* gathered, float* output,
    size_t localVocab, size_t rows) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = rows * localVocab * 2;
    if (index >= total) return;
    const size_t row = index / (localVocab * 2);
    const size_t within = index % (localVocab * 2);
    const size_t rank = within / localVocab;
    const size_t column = within % localVocab;
    output[index] = gathered[rank * rows * localVocab + row * localVocab + column];
}
#endif
}

cudaError_t GptOssTpAcquire() {
#ifdef GARNET_GPT_OSS_ENABLE_NCCL
    std::lock_guard<std::mutex> guard(g_tpMutex);
    if (g_tpReady) {
        ++g_tpUsers;
        return cudaSuccess;
    }
    int count = 0;
    auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) return status;
    if (count < 2) return cudaErrorNoDevice;
    const int devices[] = {0, 1};
    const auto result = ncclCommInitAll(g_tpComms.data(), 2, devices);
    if (result != ncclSuccess) {
        for (auto& comm : g_tpComms) {
            if (comm) ncclCommDestroy(comm);
            comm = nullptr;
        }
        return NcclStatus(result);
    }
    g_tpReady = true;
    g_tpUsers = 1;
    return cudaSuccess;
#else
    return cudaErrorNotSupported;
#endif
}

void GptOssTpRelease() {
#ifdef GARNET_GPT_OSS_ENABLE_NCCL
    std::lock_guard<std::mutex> guard(g_tpMutex);
    if (!g_tpReady || --g_tpUsers > 0) return;
    for (auto& comm : g_tpComms) {
        if (comm) ncclCommDestroy(comm);
        comm = nullptr;
    }
    g_tpReady = false;
#endif
}

cudaError_t GptOssTpAllReduce(const float* input, float* output, size_t count,
    int rank, cudaStream_t stream) {
#ifdef GARNET_GPT_OSS_ENABLE_NCCL
    if (!g_tpReady || rank < 0 || rank >= static_cast<int>(g_tpComms.size()) ||
        (count && (!input || !output))) return cudaErrorInvalidValue;
    int device = -1;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    if (device != rank) return cudaErrorInvalidDevice;
    return NcclStatus(ncclAllReduce(input, output, count, ncclFloat32, ncclSum,
        g_tpComms[rank], stream));
#else
    (void)input; (void)output; (void)count; (void)rank; (void)stream;
    return cudaErrorNotSupported;
#endif
}

cudaError_t GptOssTpAllGather(const float* input, float* output, float* scratch,
    size_t count, int rows, int localVocab, int rank, cudaStream_t stream) {
#ifdef GARNET_GPT_OSS_ENABLE_NCCL
    if (!g_tpReady || rank < 0 || rank >= static_cast<int>(g_tpComms.size()) ||
        rows <= 0 || localVocab <= 0 || count != size_t(rows) * localVocab ||
        !input || !output || !scratch) return cudaErrorInvalidValue;
    int device = -1;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    if (device != rank) return cudaErrorInvalidDevice;
    auto result = ncclAllGather(input, scratch, count, ncclFloat, g_tpComms[rank], stream);
    if (result != ncclSuccess) return NcclStatus(result);
    const size_t total = count * 2;
    interleaveVocab<<<(total + 255) / 256, 256, 0, stream>>>(
        scratch, output, size_t(localVocab), size_t(rows));
    return cudaGetLastError();
#else
    (void)input; (void)output; (void)scratch; (void)count;
    (void)rows; (void)localVocab; (void)rank; (void)stream;
    return cudaErrorNotSupported;
#endif
}
}
