// SPDX-License-Identifier: Apache-2.0
// Paired-stream screen: no overlapping independent model executions.
#include "gpt_oss_kernels.h"
#include "tp_direct.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#define CUDA_OK(call) do { const auto s = (call); if (s != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(s)); std::exit(1); } } while (0)

__global__ void changeInput(float* input, int count, float increment) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) input[i] += increment;
}

int main(int argc, char** argv) {
    const int batch = argc > 1 ? std::atoi(argv[1]) : 8;
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 50;
    constexpr int operations = 72;
    if (batch < 1 || batch > 128 || repeats < 1 || repeats > 1000) return 2;
    const int count = batch * 2880;
    const bool direct = std::getenv("GARNET_GPT_OSS_DIRECT_ALLREDUCE") &&
        std::atoi(std::getenv("GARNET_GPT_OSS_DIRECT_ALLREDUCE")) == 1;
    if (direct && batch > 1 && (!std::getenv("GARNET_GPT_OSS_DIRECT_BATCH_ALLREDUCE") ||
        std::atoi(std::getenv("GARNET_GPT_OSS_DIRECT_BATCH_ALLREDUCE")) != 1)) return 2;
    if (direct && batch > 64 && (!std::getenv("GARNET_GPT_OSS_DIRECT_LARGE_BATCH_ALLREDUCE") ||
        std::atoi(std::getenv("GARNET_GPT_OSS_DIRECT_LARGE_BATCH_ALLREDUCE")) != 1)) return 2;
    CUDA_OK(Garnet::GptOssTpAcquire());
    std::array<cudaStream_t, 2> streams{};
    std::array<float*, 2> inputs{}, outputs{};
    std::array<cudaGraph_t, 2> graphs{};
    std::array<cudaGraphExec_t, 2> executions{};
    std::array<std::vector<float>, 2> host{
        std::vector<float>(count), std::vector<float>(count)};
    auto paired = [&](auto work) {
        std::array<std::thread, 2> workers;
        for (int rank = 0; rank < 2; ++rank) workers[rank] = std::thread([&, rank] {
            CUDA_OK(cudaSetDevice(rank));
            work(rank);
        });
        for (auto& thread : workers) thread.join();
    };
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_OK(cudaSetDevice(rank));
        CUDA_OK(cudaStreamCreateWithFlags(&streams[rank], cudaStreamNonBlocking));
        CUDA_OK(cudaMalloc(&inputs[rank], count * sizeof(float)));
        CUDA_OK(cudaMalloc(&outputs[rank], count * sizeof(float)));
    }
    auto resetInputs = [&](int trial) {
        for (int i = 0; i < count; ++i) {
            host[0][i] = (i % 97) * .125f + trial;
            host[1][i] = (i % 31) * .25f - trial * .5f;
        }
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_OK(cudaSetDevice(rank));
            CUDA_OK(cudaMemcpyAsync(inputs[rank], host[rank].data(),
                count * sizeof(float), cudaMemcpyHostToDevice, streams[rank]));
            CUDA_OK(cudaStreamSynchronize(streams[rank]));
        }
    };
    auto check = [&](float increment) {
        std::vector<float> result(count);
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_OK(cudaSetDevice(rank));
            CUDA_OK(cudaMemcpy(result.data(), outputs[rank], count * sizeof(float),
                cudaMemcpyDeviceToHost));
            for (int i = 0; i < count; ++i) {
                const float expected = host[0][i] + host[1][i] + increment;
                if (result[i] != expected) {
                    std::fprintf(stderr, "rank=%d index=%d got=%g expected=%g\n",
                        rank, i, result[i], expected);
                    std::exit(3);
                }
            }
        }
    };
    resetInputs(0);
    paired([&](int rank) {
        // A requested direct screen must not silently measure NCCL fallback.
        if (direct) CUDA_OK(Garnet::GptOssTpDirectAllReduce(inputs[rank],
            outputs[rank], count, rank, streams[rank]));
        for (int i = 0; i < 10; ++i) CUDA_OK(Garnet::GptOssTpAllReduce(
            inputs[rank], outputs[rank], count, rank, streams[rank]));
        CUDA_OK(cudaStreamSynchronize(streams[rank]));
    });
    check(0.f);
    auto capture = [&](bool changing) {
        paired([&](int rank) {
            CUDA_OK(cudaStreamBeginCapture(streams[rank], cudaStreamCaptureModeThreadLocal));
            for (int i = 0; i < operations; ++i) {
                if (changing) {
                    changeInput<<<32, 256, 0, streams[rank]>>>(inputs[rank], count,
                        rank == 0 ? .25f : -.125f);
                    CUDA_OK(cudaGetLastError());
                }
                CUDA_OK(Garnet::GptOssTpAllReduce(inputs[rank], outputs[rank],
                    count, rank, streams[rank]));
            }
            CUDA_OK(cudaStreamEndCapture(streams[rank], &graphs[rank]));
            CUDA_OK(cudaGraphInstantiate(&executions[rank], graphs[rank], 0));
        });
    };
    auto replay = [&](int times) {
        const auto start = std::chrono::steady_clock::now();
        paired([&](int rank) {
            for (int i = 0; i < times; ++i)
                CUDA_OK(cudaGraphLaunch(executions[rank], streams[rank]));
            CUDA_OK(cudaStreamSynchronize(streams[rank]));
        });
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count() / (times * operations);
    };
    auto releaseGraph = [&]() {
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_OK(cudaSetDevice(rank));
            CUDA_OK(cudaGraphExecDestroy(executions[rank]));
            CUDA_OK(cudaGraphDestroy(graphs[rank]));
        }
    };
    capture(true);
    for (int trial = 0; trial < 5; ++trial) {
        resetInputs(trial * 3);
        replay(3);
        check(3 * operations * .125f);
    }
    releaseGraph();
    capture(false);
    replay(10);
    std::array<double, 5> timings{};
    for (int trial = 0; trial < 5; ++trial) {
        resetInputs(trial);
        timings[trial] = replay(repeats);
        check(0.f);
    }
    auto sorted = timings;
    std::sort(sorted.begin(), sorted.end());
    std::printf("batch=%d count=%d mode=%s parity=PASS graph_ops=%d us_per_op=",
        batch, count, direct ? "direct" : "nccl", operations);
    for (double value : timings) std::printf(" %.3f", value);
    std::printf(" median=%.3f\n", sorted[2]);
    releaseGraph();
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_OK(cudaSetDevice(rank));
        CUDA_OK(cudaFree(inputs[rank]));
        CUDA_OK(cudaFree(outputs[rank]));
        CUDA_OK(cudaStreamDestroy(streams[rank]));
    }
    Garnet::GptOssTpRelease();
}
