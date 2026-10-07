// SPDX-License-Identifier: Apache-2.0
// Isolated decode projection screen. Not part of the serving implementation.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
void Check(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(status));
        std::exit(1);
    }
}

__global__ void Init(__nv_bfloat16* data, size_t count, unsigned seed) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned n = static_cast<unsigned>(i) * 1664525u + seed * 1013904223u;
    float value = (float(n & 65535u) / 65536.0f - 0.5f) * 0.125f;
    data[i] = __float2bfloat16_rn(value);
}

template<int Rows>
__global__ void Gemv(const __nv_bfloat16* x, const __nv_bfloat16* w,
                     const __nv_bfloat16* bias, __nv_bfloat16* y,
                     int outputs, int inputs) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * Rows + warp;
    if (row >= outputs) return;
    const __nv_bfloat16* weights = w + size_t(row) * inputs;
    float sum = 0.0f;
    for (int col = lane; col < inputs; col += 32)
        sum = fmaf(__bfloat162float(weights[col]), __bfloat162float(x[col]), sum);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0)
        y[row] = __float2bfloat16_rn(sum + __bfloat162float(bias[row]));
}

struct Projection {
    int outputs;
    int inputs;
    size_t weightOffset;
    size_t outputOffset;
};

template<int Rows>
float TimeGraph(const std::vector<Projection>& projections,
                const __nv_bfloat16* weights, const __nv_bfloat16* xHidden,
                const __nv_bfloat16* xAttn, const __nv_bfloat16* bias,
                __nv_bfloat16* output, cudaStream_t stream) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t execution = nullptr;
    Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
    for (size_t i = 0; i < projections.size(); ++i) {
        const auto& p = projections[i];
        const auto* x = (i % 2 == 0) ? xHidden : xAttn;
        Gemv<Rows><<<(p.outputs + Rows - 1) / Rows, Rows * 32, 0, stream>>>(
            x, weights + p.weightOffset, bias + p.outputOffset,
            output + p.outputOffset, p.outputs, p.inputs);
    }
    Check(cudaGetLastError(), "launch capture");
    Check(cudaStreamEndCapture(stream, &graph), "end capture");
    Check(cudaGraphInstantiate(&execution, graph, nullptr, nullptr, 0), "instantiate");
    for (int i = 0; i < 20; ++i)
        Check(cudaGraphLaunch(execution, stream), "warmup graph");
    Check(cudaStreamSynchronize(stream), "warmup sync");
    cudaEvent_t begin{}, end{};
    Check(cudaEventCreate(&begin), "event begin");
    Check(cudaEventCreate(&end), "event end");
    float best = 1e9f;
    for (int trial = 0; trial < 5; ++trial) {
        Check(cudaEventRecord(begin, stream), "start event");
        for (int i = 0; i < 100; ++i)
            Check(cudaGraphLaunch(execution, stream), "timed graph");
        Check(cudaEventRecord(end, stream), "end event");
        Check(cudaEventSynchronize(end), "timed sync");
        float ms = 0;
        Check(cudaEventElapsedTime(&ms, begin, end), "elapsed");
        best = std::min(best, ms / 100.0f);
    }
    Check(cudaEventDestroy(begin), "destroy begin");
    Check(cudaEventDestroy(end), "destroy end");
    Check(cudaGraphExecDestroy(execution), "destroy execution");
    Check(cudaGraphDestroy(graph), "destroy graph");
    return best;
}

template<int Rows>
void Validate(const Projection& p, const __nv_bfloat16* weights,
              const __nv_bfloat16* x, const __nv_bfloat16* bias,
              __nv_bfloat16* output, cudaStream_t stream) {
    Gemv<Rows><<<(p.outputs + Rows - 1) / Rows, Rows * 32, 0, stream>>>(
        x, weights + p.weightOffset, bias + p.outputOffset,
        output + p.outputOffset, p.outputs, p.inputs);
    Check(cudaStreamSynchronize(stream), "validate sync");
    std::vector<__nv_bfloat16> hostX(p.inputs), hostW(size_t(16) * p.inputs);
    std::vector<__nv_bfloat16> hostBias(16), hostY(16);
    Check(cudaMemcpy(hostX.data(), x, p.inputs * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost), "read x");
    Check(cudaMemcpy(hostW.data(), weights + p.weightOffset,
                     hostW.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "read w");
    Check(cudaMemcpy(hostBias.data(), bias + p.outputOffset,
                     hostBias.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "read bias");
    Check(cudaMemcpy(hostY.data(), output + p.outputOffset,
                     hostY.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "read y");
    float maxAbs = 0, maxRel = 0;
    for (int row = 0; row < 16; ++row) {
        float sum = 0;
        for (int col = 0; col < p.inputs; ++col)
            sum = std::fma(__bfloat162float(hostW[size_t(row) * p.inputs + col]),
                           __bfloat162float(hostX[col]), sum);
        float expected = sum + __bfloat162float(hostBias[row]);
        float actual = __bfloat162float(hostY[row]);
        maxAbs = std::max(maxAbs, std::abs(expected - actual));
        maxRel = std::max(maxRel, std::abs(expected - actual) /
                         std::max(0.01f, std::abs(expected)));
    }
    std::printf("rows=%d max_abs_error=%.6f max_relative_error=%.6f\n",
                Rows, maxAbs, maxRel);
}
}

int main() {
    constexpr int layers = 36;
    constexpr int hidden = 2880;
    constexpr int localQkv = 2560;
    constexpr int localAttn = 2048;
    std::vector<Projection> projections;
    size_t weightElements = 0, outputElements = 0;
    for (int layer = 0; layer < layers; ++layer) {
        projections.push_back({localQkv, hidden, weightElements, outputElements});
        weightElements += size_t(localQkv) * hidden;
        outputElements += localQkv;
        projections.push_back({hidden, localAttn, weightElements, outputElements});
        weightElements += size_t(hidden) * localAttn;
        outputElements += hidden;
    }
    __nv_bfloat16 *weights{}, *xHidden{}, *xAttn{}, *bias{}, *output{};
    Check(cudaMalloc(&weights, weightElements * sizeof(__nv_bfloat16)), "weights");
    Check(cudaMalloc(&xHidden, hidden * sizeof(__nv_bfloat16)), "x hidden");
    Check(cudaMalloc(&xAttn, localAttn * sizeof(__nv_bfloat16)), "x attn");
    Check(cudaMalloc(&bias, outputElements * sizeof(__nv_bfloat16)), "bias");
    Check(cudaMalloc(&output, outputElements * sizeof(__nv_bfloat16)), "output");
    cudaStream_t stream{};
    Check(cudaStreamCreate(&stream), "stream");
    Init<<<(weightElements + 255) / 256, 256, 0, stream>>>(weights, weightElements, 1);
    Init<<<(hidden + 255) / 256, 256, 0, stream>>>(xHidden, hidden, 2);
    Init<<<(localAttn + 255) / 256, 256, 0, stream>>>(xAttn, localAttn, 3);
    Init<<<(outputElements + 255) / 256, 256, 0, stream>>>(bias, outputElements, 4);
    Check(cudaStreamSynchronize(stream), "init sync");
    std::printf("layers=%d projections=%zu weight_MiB=%.1f\n", layers,
                projections.size(), double(weightElements * sizeof(__nv_bfloat16)) / 1048576.0);
    Validate<4>(projections.front(), weights, xHidden, bias, output, stream);
    Validate<8>(projections.front(), weights, xHidden, bias, output, stream);
    std::printf("rows=4 graph_ms_per_decode=%.4f\n",
                TimeGraph<4>(projections, weights, xHidden, xAttn, bias, output, stream));
    std::printf("rows=8 graph_ms_per_decode=%.4f\n",
                TimeGraph<8>(projections, weights, xHidden, xAttn, bias, output, stream));
    Check(cudaStreamDestroy(stream), "destroy stream");
    Check(cudaFree(output), "free output");
    Check(cudaFree(bias), "free bias");
    Check(cudaFree(xAttn), "free x attn");
    Check(cudaFree(xHidden), "free x hidden");
    Check(cudaFree(weights), "free weights");
}
