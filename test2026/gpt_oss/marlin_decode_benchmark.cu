// SPDX-License-Identifier: Apache-2.0
// One GPT-OSS-sized decode MoE invocation, captured as a CUDA graph. Run in
// separate processes for each GARNET_GPT_OSS_FUSED_* environment setting.
#include "gpt_oss_marlin.h"
#include <cuda_runtime.h>
#include <algorithm>
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

int main() { try {
    GptOssOptions options;
    options.kind = 2;
    options.hidden = 2880;
    options.intermediate = 2880;
    options.experts = 128;
    options.topK = 8;
    constexpr size_t h = 2880, intermediate = 2880, experts = 128;
    Buffer x(h * 4), router(experts * h * 4), routerBias(experts * 4);
    Buffer up(experts * 2 * intermediate * h / 2, 0x31);
    Buffer upScales(experts * 2 * intermediate * h / 32, 120);
    Buffer upBias(experts * 2 * intermediate * 4);
    Buffer down(experts * h * intermediate / 2, 0x24);
    Buffer downScales(experts * h * intermediate / 32, 120);
    Buffer downBias(experts * h * 4), output(h * 4);
    Buffer workspace(GptOssMarlin::Workspace(1, options));
    const void* inputs[]{x.data, router.data, routerBias.data, up.data,
        upScales.data, upBias.data, down.data, downScales.data, downBias.data};
    GptOssMarlin marlin(options);
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(marlin.Run(inputs, static_cast<float*>(output.data), workspace.data, 1, stream));
    check(cudaDeviceSynchronize());

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    check(marlin.Run(inputs, static_cast<float*>(output.data), workspace.data, 1, stream));
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
        flushL2<<<(flushBytes / 4 + 255) / 256, 256>>>(
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
    std::cout << "device=" << device.name << " graph_nodes=" << graphNodes
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
