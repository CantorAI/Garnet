// SPDX-License-Identifier: Apache-2.0
// Full-size expert microbenchmark. Flush L2 outside each timed invocation: one
// repeatedly cached layer is not representative of a 36-layer decode pass.
#include "gpt_oss_kernels.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace Garnet;
static void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t bytes, int value = 0) {
        check(cudaMalloc(&p, bytes)); check(cudaMemset(p, value, bytes));
    }
    ~Buffer() { cudaFree(p); }
};
__global__ void flushCache(unsigned* data, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) data[i] += unsigned(i) + 1;
}
int main() { try {
    cudaDeviceProp device; check(cudaGetDeviceProperties(&device, 0));
    std::cout << "device=" << device.name << ", L2_bytes=" << device.l2CacheSize << "\n";
    GptOssOptions o; o.hidden = o.intermediate = 2880; o.experts = 128; o.topK = 8;
    size_t h = o.hidden, inner = o.intermediate, e = o.experts;
    Buffer router(e*h*4), rb(e*4), up(e*2*inner*h/2, 0x31), us(e*2*inner*h/32, 120),
        ub(e*2*inner*4), down(e*h*inner/2, 0x24), ds(e*h*inner/32, 120), db(e*h*4);
    const size_t flushBytes = std::max(size_t(256) << 20, size_t(device.l2CacheSize) * 4);
    Buffer cache(flushBytes);
    cudaEvent_t begin, end; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
    for (int tokens : {1, 3, 17}) {
        Buffer x(tokens*h*4), output(tokens*h*4),
            workspace(TestGptOssMoeWorkspace(tokens, o, true));
        std::vector<float> input(tokens*h);
        for (size_t i=0; i<input.size(); ++i) input[i] = (int(i%7)-3)*.125f;
        check(cudaMemcpy(x.p, input.data(), input.size()*4, cudaMemcpyHostToDevice));
        const void* in[]{x.p,router.p,rb.p,up.p,us.p,ub.p,down.p,ds.p,db.p};
        std::vector<float> reference(tokens*h);
        for (bool grouped : {false, true}) {
            check(TestGptOssMoe(in, (float*)output.p, workspace.p, tokens, o, nullptr, grouped));
            check(cudaDeviceSynchronize());
            std::vector<float> actual(tokens*h);
            check(cudaMemcpy(actual.data(), output.p, actual.size()*4, cudaMemcpyDeviceToHost));
            if (!grouped) reference = actual;
            else for (size_t i=0; i<actual.size(); ++i)
                if (!std::isfinite(actual[i]) || std::abs(actual[i]-reference[i]) > .002f*(1+std::abs(reference[i])))
                    throw std::runtime_error("Grouped decode disagrees with warp reference");
            std::vector<float> times;
            for (int iteration=0; iteration<15; ++iteration) {
                flushCache<<<(flushBytes/4+255)/256,256>>>((unsigned*)cache.p,flushBytes/4);
                check(cudaGetLastError());
                check(cudaEventRecord(begin));
                check(TestGptOssMoe(in, (float*)output.p, workspace.p, tokens, o, nullptr, grouped));
                check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
                float milliseconds; check(cudaEventElapsedTime(&milliseconds,begin,end));
                times.push_back(milliseconds);
            }
            std::sort(times.begin(),times.end());
            std::cout << "tokens=" << tokens << ", grouped=" << grouped
                << ", median_ms=" << times[times.size()/2] << ", min_ms=" << times.front() << "\n";
        }
    }
    check(cudaEventDestroy(begin)); check(cudaEventDestroy(end)); return 0;
} catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; } }
