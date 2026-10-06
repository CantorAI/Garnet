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
}
