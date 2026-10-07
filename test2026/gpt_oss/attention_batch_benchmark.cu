// SPDX-License-Identifier: Apache-2.0
// Isolate a 36-layer full/sliding decode attention pass, with distinct KV
// storage per layer. This is not a full-model or two-GPU serving benchmark.
#include "gpt_oss_kernels.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace Garnet;
static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t bytes) { check(cudaMalloc(&p, bytes)); }
    ~Buffer() { cudaFree(p); }
};
static float pattern(size_t i, bool value) {
    return (int(i % (value ? 29 : 31)) - (value ? 14 : 15)) * .03125f;
}
__global__ void fillKv(__nv_bfloat16* keys, __nv_bfloat16* values, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(gridDim.x) * blockDim.x) {
        keys[i] = __float2bfloat16((int(i % 31) - 15) * .03125f);
        values[i] = __float2bfloat16((int(i % 29) - 14) * .03125f);
    }
}
template<class T> static void copy(Buffer& destination, const std::vector<T>& values) {
    check(cudaMemcpy(destination.p, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
}
int main(int argc, char** argv) { try {
    const int batch = argc > 1 ? std::atoi(argv[1]) : 128;
    const int length = argc > 2 ? std::atoi(argv[2]) : 767;
    const int repeats = argc > 3 ? std::atoi(argv[3]) : 20;
    if (batch < 2 || batch > 128 || length < 64 || length > 4096 || repeats < 1 || repeats > 100) return 2;
    constexpr int layers = 36, dimension = 64, qHeads = 32, kvHeads = 4;
    const int logical = (length + 15) / 16, pages = batch * logical;
    const int packed = (qHeads + 2 * kvHeads) * dimension;
    const size_t cacheElements = size_t(layers) * pages * 16 * kvHeads * dimension;
    Buffer keys(cacheElements * 2), values(cacheElements * 2), input(size_t(batch) * packed * 4),
        output(size_t(batch) * qHeads * dimension * 4), sinks(qHeads * 4),
        table(size_t(batch) * logical * 4), lengths(batch * 4), starts(batch * 4), active(batch * 4),
        scratch(size_t(batch) * qHeads * 64 * (dimension + 2) * 4);
    fillKv<<<4096, 256>>>((__nv_bfloat16*)keys.p, (__nv_bfloat16*)values.p, cacheElements);
    check(cudaGetLastError());
    std::vector<float> x(size_t(batch) * packed), sink(qHeads);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (int(i % 23) - 11) * .03125f;
    for (int h = 0; h < qHeads; ++h) sink[h] = (h % 13) * .25f;
    std::vector<int> pageTable(size_t(batch) * logical), mask(batch, 1);
    for (int b = 0; b < batch; ++b) for (int p = 0; p < logical; ++p)
        pageTable[b * logical + p] = p == 2 ? -1 : b * logical + logical - p - 1;
    mask.back() = 0;
    copy(input, x); copy(sinks, sink); copy(table, pageTable); copy(active, mask);
    copy(lengths, std::vector<int>(batch, length)); copy(starts, std::vector<int>(batch, length - 1));
    const void* in[]{input.p, keys.p, values.p, table.p, lengths.p, starts.p, active.p, sinks.p};
    GptOssOptions o; o.kind = 1; o.qHeads = qHeads; o.kvHeads = kvHeads;
    o.headDim = dimension; o.pageSize = 16; o.layer = layers - 1; o.prefill = 0;
    float maximumError = 0;
    auto validate = [&](int window) {
        std::vector<float> actual(size_t(batch) * qHeads * dimension);
        check(cudaMemcpy(actual.data(), output.p, actual.size() * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i])) throw std::runtime_error("Nonfinite attention output");
            if (i >= size_t(batch - 1) * qHeads * dimension && actual[i] != 0)
                throw std::runtime_error("Inactive attention slot is nonzero");
        }
        for (int b : {0, batch / 2, batch - 1}) for (int h : {0, 7, 8, 31}) {
            double sum = std::exp(double(sink[h])), accum[dimension] = {};
            if (mask[b]) for (int p = window ? std::max(0, length - window) : 0; p < length; ++p) {
                const int page = pageTable[b * logical + p / 16];
                if (page < 0) continue;
                const int kvHead = h / (qHeads / kvHeads);
                const size_t offset = ((size_t(o.layer) * pages + page) * 16 + p % 16) * kvHeads * dimension + kvHead * dimension;
                double score = 0;
                for (int d = 0; d < dimension; ++d) {
                    float key = p == length - 1 ? x[b * packed + (qHeads + kvHead) * dimension + d] : pattern(offset + d, false);
                    score += double(x[b * packed + h * dimension + d]) * key;
                }
                const double weight = std::exp(score / 8.);
                sum += weight;
                for (int d = 0; d < dimension; ++d) {
                    float value = p == length - 1 ? x[b * packed + (qHeads + kvHeads + kvHead) * dimension + d] : pattern(offset + d, true);
                    accum[d] += weight * value;
                }
            }
            for (int d = 0; d < dimension; ++d) {
                const float expected = __bfloat162float(__float2bfloat16(float(accum[d] / sum)));
                const float error = std::abs(actual[(b * qHeads + h) * dimension + d] - expected);
                maximumError = std::max(maximumError, error);
                if (error > .006f * (1 + std::abs(expected)))
                    throw std::runtime_error("Sampled paged GQA attention reference disagrees");
            }
        }
    };
    for (int window : {0, 128}) {
        o.window = window;
        check(RunGptOssAttention(in, (float*)output.p, scratch.p, batch, 1, logical, pages, o, nullptr));
        check(cudaDeviceSynchronize()); validate(window);
    }
    cudaStream_t stream; check(cudaStreamCreate(&stream));
    cudaGraph_t graph; cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int layer = 0; layer < layers; ++layer) {
        o.layer = layer; o.window = layer % 2 ? 128 : 0;
        check(RunGptOssAttention(in, (float*)output.p, scratch.p, batch, 1, logical, pages, o, stream));
    }
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, 0));
    for (int i = 0; i < 10; ++i) check(cudaGraphLaunch(executable, stream));
    check(cudaStreamSynchronize(stream));
    cudaEvent_t begin, end; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
    std::vector<float> times;
    for (int trial = 0; trial < 5; ++trial) {
        check(cudaEventRecord(begin, stream));
        for (int i = 0; i < repeats; ++i) check(cudaGraphLaunch(executable, stream));
        check(cudaEventRecord(end, stream)); check(cudaEventSynchronize(end));
        float milliseconds; check(cudaEventElapsedTime(&milliseconds, begin, end));
        times.push_back(milliseconds / repeats);
        validate(128);
    }
    auto sorted = times; std::sort(sorted.begin(), sorted.end());
    cudaDeviceProp device; check(cudaGetDeviceProperties(&device, 0));
    std::cout << "device=" << device.name << " batch=" << batch << " length=" << length
        << " KV_bytes=" << cacheElements * 4 << " layers=36 full=18 sliding=18"
        << " splits=" << (std::getenv("GARNET_GPT_OSS_DECODE_SPLITS") ? std::getenv("GARNET_GPT_OSS_DECODE_SPLITS") : "16")
        << " warps=" << (std::getenv("GARNET_GPT_OSS_DECODE_WARPS") ? std::getenv("GARNET_GPT_OSS_DECODE_WARPS") : "16")
        << " sampled_reference_max_abs=" << maximumError << " parity=PASS gpu_ms_per_36_layers=";
    for (auto value : times) std::cout << ' ' << value;
    std::cout << " median=" << sorted[2] << '\n';
    check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph)); check(cudaStreamDestroy(stream));
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; } }
