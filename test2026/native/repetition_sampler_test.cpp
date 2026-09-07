#include "repetition_sampler.h"
#include "cuda_lib.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template<class T> struct Device {
    T* data = nullptr;
    explicit Device(size_t count) { check(cudaMalloc(&data, count * sizeof(T))); }
    ~Device() { cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

static void verify(const std::vector<float>& values,
    const std::vector<unsigned char>& seen, float penalty, long long expected) {
    const int count = static_cast<int>(values.size());
    Device<float> logits(count), score(1);
    Device<__nv_bfloat16> bf16(count);
    Device<unsigned char> history(count);
    Device<long long> token(1);
    check(cudaMemcpy(logits.data, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(history.data, seen.data(), seen.size(), cudaMemcpyHostToDevice));
    std::vector<__nv_bfloat16> converted;
    for (float value : values) converted.push_back(__float2bfloat16(value));
    check(cudaMemcpy(bf16.data, converted.data(), converted.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    for (bool useBf16 : {false, true}) {
        check(useBf16 ? sampleRepetitionBF16(bf16.data, history.data, penalty, token.data, score.data, count, 0)
            : sampleRepetitionFP32(logits.data, history.data, penalty, token.data, score.data, count, 0));
        long long actual = -1;
        float actualScore = 0;
        check(cudaMemcpy(&actual, token.data, sizeof(actual), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&actualScore, score.data, sizeof(actualScore), cudaMemcpyDeviceToHost));
        float expectedScore = values[expected];
        if (seen[expected]) expectedScore = expectedScore < 0 ? expectedScore * penalty : expectedScore / penalty;
        if (actual != expected || std::abs(actualScore - expectedScore) > 1e-5f)
            throw std::runtime_error("incorrect penalized token or score");
    }
    std::vector<float> unchanged(values.size());
    check(cudaMemcpy(unchanged.data(), logits.data, values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (unchanged != values) throw std::runtime_error("sampling modified logits");
    std::vector<__nv_bfloat16> unchangedBf16(values.size());
    check(cudaMemcpy(unchangedBf16.data(), bf16.data, converted.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < values.size(); ++i)
        if (__bfloat162float(unchangedBf16[i]) != values[i]) throw std::runtime_error("sampling modified BF16 logits");
    if (sampleRepetitionFP32(logits.data, history.data, std::numeric_limits<float>::quiet_NaN(),
        token.data, score.data, count, 0) != cudaErrorInvalidValue)
        throw std::runtime_error("accepted nonfinite penalty");
}

int main() {
    try {
        verify({4, 3, 2}, {1, 0, 0}, 2, 1);
        verify({-1, -1.5f, -3}, {1, 0, 0}, 2, 1);
        verify({4, 3, 2}, {1, 0, 0}, 1, 0);
        verify({3, 3, 2}, {0, 0, 0}, 2, 0);
        verify({4, 3, 2}, {1, 1, 1}, 2, 0);
        std::cout << "PASS: FP32/BF16 repetition sampling, ties, identity, validation and immutable logits\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
