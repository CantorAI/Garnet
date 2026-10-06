// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_kernels.h"
#include "tp_direct.h"
#include <array>
#include <cstdlib>
#include <mutex>

#ifdef GARNET_GPT_OSS_ENABLE_NCCL
#include <nccl.h>
#endif

namespace Garnet {
namespace {
std::mutex g_tpMutex;
int g_tpUsers = 0;
bool g_tpReady = false;
bool g_directReady = false;
#ifdef GARNET_GPT_OSS_ENABLE_NCCL
std::array<ncclComm_t, 2> g_tpComms{};
cudaError_t NcclStatus(ncclResult_t status) {
    return status == ncclSuccess ? cudaSuccess : cudaErrorUnknown;
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
    if (const char* direct = std::getenv("GARNET_GPT_OSS_DIRECT_ALLREDUCE")) {
        if (direct[0] == '1' && direct[1] == '\0')
            g_directReady = GptOssTpDirectAcquire() == cudaSuccess;
    }
    return cudaSuccess;
#else
    return cudaErrorNotSupported;
#endif
}

void GptOssTpRelease() {
#ifdef GARNET_GPT_OSS_ENABLE_NCCL
    std::lock_guard<std::mutex> guard(g_tpMutex);
    if (!g_tpReady || --g_tpUsers > 0) return;
    if (g_directReady) GptOssTpDirectRelease();
    g_directReady = false;
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
    if (g_directReady && count == 2880) {
        status = GptOssTpDirectAllReduce(input, output, count, rank, stream);
        if (status != cudaErrorNotSupported) return status;
    }
    return NcclStatus(ncclAllReduce(input, output, count, ncclFloat32, ncclSum,
        g_tpComms[rank], stream));
#else
    (void)input; (void)output; (void)count; (void)rank; (void)stream;
    return cudaErrorNotSupported;
#endif
}

cudaError_t GptOssTpAllReduceBf16(const float* input, float* output,
    void* workspace, size_t count, int rank, cudaStream_t stream) {
#ifdef GARNET_GPT_OSS_ENABLE_NCCL
    if (!g_tpReady || rank < 0 || rank >= static_cast<int>(g_tpComms.size()) ||
        !input || !output || !workspace || !count ||
        count > SIZE_MAX / (2 * sizeof(uint16_t))) return cudaErrorInvalidValue;
    int device = -1;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    if (device != rank) return cudaErrorInvalidDevice;
    auto* packed = static_cast<unsigned char*>(workspace);
    auto* reduced = packed + count * sizeof(uint16_t);
    status = GptOssTpPackBf16(input, packed, count, stream);
    if (status != cudaSuccess) return status;
    const auto result = ncclAllReduce(packed, reduced, count, ncclBfloat16,
        ncclSum, g_tpComms[rank], stream);
    if (result != ncclSuccess) return NcclStatus(result);
    return GptOssTpUnpackBf16(reduced, output, count, stream);
#else
    (void)input; (void)output; (void)workspace; (void)count;
    (void)rank; (void)stream;
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
    const size_t localVocabBytes = size_t(localVocab) * sizeof(float);
    const size_t fullVocabBytes = localVocabBytes * 2;
    status = cudaMemcpy2DAsync(output, fullVocabBytes, scratch,
        localVocabBytes, localVocabBytes, size_t(rows), cudaMemcpyDeviceToDevice, stream);
    if (status != cudaSuccess) return status;
    return cudaMemcpy2DAsync(output + localVocab, fullVocabBytes,
        scratch + count, localVocabBytes, localVocabBytes, size_t(rows),
        cudaMemcpyDeviceToDevice, stream);
#else
    (void)input; (void)output; (void)scratch; (void)count;
    (void)rows; (void)localVocab; (void)rank; (void)stream;
    return cudaErrorNotSupported;
#endif
}
}
