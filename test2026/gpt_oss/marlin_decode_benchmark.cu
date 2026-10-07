// SPDX-License-Identifier: Apache-2.0
// One synthetic GPT-OSS-sized MoE invocation, captured as a CUDA graph.
// Optional ROWS prefill|decode OUTPUT_BIN; separate processes per setting.
#include "gpt_oss_marlin.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace Garnet;

static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct Buffer {
    void* data = nullptr;
    explicit Buffer(size_t bytes, int fill = 0) {
        check(cudaMalloc(&data, bytes));
        check(cudaMemset(data, fill, bytes));
    }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

__global__ void flushL2(unsigned* data, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) data[i] += unsigned(i) + 1;
}

int main(int argc, char** argv) { try {
    const int tokens = argc > 1 ? std::stoi(argv[1]) : 1;
    const bool prefill = argc > 2 && std::strcmp(argv[2], "prefill") == 0;
    if (argc > 4 || tokens < 1 || tokens > 4096 ||
        (argc > 2 && !prefill && std::strcmp(argv[2], "decode") != 0)) return 2;
    if(argc==4 && std::filesystem::exists(argv[3]))
        throw std::runtime_error("Synthetic output evidence already exists");
    GptOssOptions options;
    options.kind = 2;
    options.hidden = 2880;
    options.intermediate = 2880;
    options.experts = 128;
    options.topK = 8;
    options.prefill = prefill;
    constexpr size_t h = 2880, intermediate = 2880, experts = 128;
    Buffer x(size_t(tokens) * h * 4), router(experts * h * 4), routerBias(experts * 4);
    // Diverse deterministic rows/router, rather than only tied zero routes.
    // This is synthetic scheduling evidence, not a pretrained benchmark.
    auto values = [](size_t count, uint32_t seed, float scale) {
        std::vector<float> result(count);
        for (size_t i = 0; i < count; ++i) {
            uint32_t v = uint32_t(i) + seed;
            v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15;
            v *= 0x846ca68bu; v ^= v >> 16;
            result[i] = (int(v % 65) - 32) * scale;
        }
        return result;
    };
    auto hostX=values(size_t(tokens)*h,17,1.f/128);
    auto hostRouter=values(experts*h,991,1.f/4096);
    check(cudaMemcpy(x.data,hostX.data(),hostX.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(router.data,hostRouter.data(),hostRouter.size()*4,cudaMemcpyHostToDevice));
    Buffer up(experts * 2 * intermediate * h / 2, 0x31);
    Buffer upScales(experts * 2 * intermediate * h / 32, 120);
    Buffer upBias(experts * 2 * intermediate * 4);
    Buffer down(experts * h * intermediate / 2, 0x24);
    Buffer downScales(experts * h * intermediate / 32, 120);
    Buffer downBias(experts * h * 4), output(size_t(tokens) * h * 4);
    const auto workspaceBytes=GptOssMarlin::Workspace(tokens, options);
    if(!workspaceBytes) throw std::runtime_error("Requested Marlin shape is unsupported");
    Buffer workspace(workspaceBytes);
    const void* inputs[]{x.data, router.data, routerBias.data, up.data,
        upScales.data, upBias.data, down.data, downScales.data, downBias.data};
    GptOssMarlin marlin(options);
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(marlin.Run(inputs, static_cast<float*>(output.data), workspace.data, tokens, stream));
    check(cudaDeviceSynchronize());
    std::vector<float> reference(size_t(tokens)*h), actual(reference.size());
    check(cudaMemcpy(reference.data(),output.data,reference.size()*4,cudaMemcpyDeviceToHost));
    for(float value:reference)if(!std::isfinite(value))throw std::runtime_error("Nonfinite synthetic output");

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    check(marlin.Run(inputs, static_cast<float*>(output.data), workspace.data, tokens, stream));
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    size_t graphNodes = 0;
    check(cudaGraphGetNodes(graph, nullptr, &graphNodes));
    cudaDeviceProp device{};
    check(cudaGetDeviceProperties(&device, 0));
    const size_t flushBytes = std::max(size_t(256) << 20,
        size_t(device.l2CacheSize) * 4);
    Buffer flush(flushBytes);
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin));
    check(cudaEventCreate(&end));
    for (int i = 0; i < 5; ++i) check(cudaGraphLaunch(executable, stream));
    check(cudaDeviceSynchronize());
    std::vector<float> milliseconds;
    for (int i = 0; i < 25; ++i) {
        flushL2<<<(flushBytes / 4 + 255) / 256, 256, 0, stream>>>(
            static_cast<unsigned*>(flush.data), flushBytes / 4);
        check(cudaGetLastError());
        check(cudaEventRecord(begin, stream));
        check(cudaGraphLaunch(executable, stream));
        check(cudaEventRecord(end, stream));
        check(cudaEventSynchronize(end));
        float elapsed = 0;
        check(cudaEventElapsedTime(&elapsed, begin, end));
        milliseconds.push_back(elapsed);
    }
    std::sort(milliseconds.begin(), milliseconds.end());
    check(cudaMemcpy(actual.data(),output.data,actual.size()*4,cudaMemcpyDeviceToHost));
    if(actual!=reference)throw std::runtime_error("Warm graph replay differs from eager output");
    if(argc==4) {
        std::ofstream outputFile(argv[3],std::ios::binary);
        outputFile.write(reinterpret_cast<const char*>(actual.data()),actual.size()*sizeof(float));
        outputFile.close();
        if(!outputFile)throw std::runtime_error("Failed to preserve synthetic output evidence");
    }
    std::cout << "measurement=synthetic_single_moe device=" << device.name
              << " rows=" << tokens << " phase=" << (prefill?"prefill":"decode")
              << " workspace_bytes=" << workspaceBytes << " graph_replay_exact=PASS graph_nodes=" << graphNodes
              << " median_ms="
              << milliseconds[milliseconds.size() / 2]
              << " min_ms=" << milliseconds.front() << '\n';
    check(cudaEventDestroy(begin));
    check(cudaEventDestroy(end));
    check(cudaGraphExecDestroy(executable));
    check(cudaGraphDestroy(graph));
    check(cudaStreamDestroy(stream));
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} }
