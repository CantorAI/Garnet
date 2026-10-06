// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_kernels.h"
#include <cuda_runtime.h>
#include <array>
#include <cstdio>
#include <thread>
#include <vector>

namespace {
bool Check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    return false;
}
}

int main() {
    constexpr size_t count = 4096;
    if (!Check(Garnet::GptOssTpAcquire(), "initialize TP communicator")) return 1;
    std::array<cudaStream_t, 2> streams{};
    std::array<float*, 2> inputs{}, outputs{};
    std::array<std::vector<float>, 2> hostInputs{
        std::vector<float>(count, 1.25f), std::vector<float>(count, -0.5f)};
    std::vector<float> hostOutput(count);
    cudaError_t status = cudaSuccess;
    for (int rank = 0; rank < 2 && status == cudaSuccess; ++rank) {
        status = cudaSetDevice(rank);
        if (status == cudaSuccess) status = cudaStreamCreate(&streams[rank]);
        if (status == cudaSuccess) status = cudaMalloc((void**)&inputs[rank], count * sizeof(float));
        if (status == cudaSuccess) status = cudaMalloc((void**)&outputs[rank], count * sizeof(float));
        if (status == cudaSuccess) status = cudaMemcpyAsync(inputs[rank], hostInputs[rank].data(),
            count * sizeof(float), cudaMemcpyHostToDevice, streams[rank]);
    }
    std::array<cudaError_t, 2> collectiveStatus{cudaSuccess, cudaSuccess};
    std::array<std::thread, 2> workers;
    for (int rank = 0; rank < 2 && status == cudaSuccess; ++rank) {
        workers[rank] = std::thread([&, rank] {
            collectiveStatus[rank] = cudaSetDevice(rank);
            if (collectiveStatus[rank] == cudaSuccess)
                collectiveStatus[rank] = Garnet::GptOssTpAllReduce(
                    inputs[rank], outputs[rank], count, rank, streams[rank]);
            if (collectiveStatus[rank] == cudaSuccess)
                collectiveStatus[rank] = cudaStreamSynchronize(streams[rank]);
        });
    }
    for (auto& worker : workers) if (worker.joinable()) worker.join();
    for (auto result : collectiveStatus) if (status == cudaSuccess && result != cudaSuccess) status = result;
    if (status == cudaSuccess) status = cudaSetDevice(0);
    if (status == cudaSuccess) status = cudaMemcpy(hostOutput.data(), outputs[0],
        count * sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        Check(status, "run TP all-reduce");
        return 1;
    }
    for (float value : hostOutput) {
        if (value != 0.75f) {
            std::fprintf(stderr, "TP all-reduce mismatch: %.9g\n", value);
            return 1;
        }
    }
    for (int rank = 0; rank < 2; ++rank) {
        cudaSetDevice(rank);
        if (inputs[rank]) cudaFree(inputs[rank]);
        if (outputs[rank]) cudaFree(outputs[rank]);
        if (streams[rank]) cudaStreamDestroy(streams[rank]);
    }
    Garnet::GptOssTpRelease();
    std::puts("NCCL TP2 all-reduce parity passed.");
    return 0;
}
