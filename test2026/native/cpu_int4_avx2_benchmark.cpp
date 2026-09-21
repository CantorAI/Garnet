// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <numeric>
#include <random>
#include <vector>

#include <omp.h>

namespace {
constexpr int kBlock = 128;
constexpr int kPackedBytes = kBlock / 2;

struct QuantizedActivation {
    std::vector<std::int8_t> values;
    std::vector<float> scales;
    std::vector<std::int32_t> sums;
};

QuantizedActivation QuantizeActivation(const float* input, int columns) {
    const int blocks = columns / kBlock;
    QuantizedActivation result;
    result.values.resize(columns);
    result.scales.resize(blocks);
    result.sums.resize(blocks);
    for (int block = 0; block < blocks; ++block) {
        const float* source = input + block * kBlock;
        float maximum = 0.0f;
        for (int index = 0; index < kBlock; ++index) maximum = std::max(maximum, std::abs(source[index]));
        const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        result.scales[block] = scale;
        std::int32_t sum = 0;
        for (int index = 0; index < kBlock; ++index) {
            const int quantized = std::clamp<int>(static_cast<int>(std::nearbyint(source[index] / scale)), -127, 127);
            result.values[block * kBlock + index] = static_cast<std::int8_t>(quantized);
            sum += quantized;
        }
        result.sums[block] = sum;
    }
    return result;
}

QuantizedActivation QuantizeActivationChannelwise(const float* input, int columns) {
    QuantizedActivation result;
    result.values.resize(columns);
    result.scales.resize(1);
    result.sums.resize(columns / kBlock);
    float maximum = 0.0f;
    for (int index = 0; index < columns; ++index) maximum = std::max(maximum, std::abs(input[index]));
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    result.scales[0] = scale;
    for (int block = 0; block < columns / kBlock; ++block) {
        std::int32_t sum = 0;
        for (int index = 0; index < kBlock; ++index) {
            const int offset = block * kBlock + index;
            const int quantized = std::clamp<int>(static_cast<int>(std::nearbyint(input[offset] / scale)), -127, 127);
            result.values[offset] = static_cast<std::int8_t>(quantized);
            sum += quantized;
        }
        result.sums[block] = sum;
    }
    return result;
}

inline std::int32_t HorizontalSum(__m256i value) {
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(value), _mm256_extracti128_si256(value, 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

inline std::int32_t Dot64(const std::uint8_t* packed, const std::int8_t* activation) {
    const __m256i weights = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed));
    const __m256i mask = _mm256_set1_epi8(0x0f);
    const __m256i low = _mm256_and_si256(weights, mask);
    const __m256i high = _mm256_and_si256(_mm256_srli_epi16(weights, 4), mask);
    const __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(activation));
    const __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(activation + 32));
    const __m256i pairs0 = _mm256_maddubs_epi16(low, a0);
    const __m256i pairs1 = _mm256_maddubs_epi16(high, a1);
    const __m256i ones = _mm256_set1_epi16(1);
    return HorizontalSum(_mm256_add_epi32(
        _mm256_madd_epi16(pairs0, ones), _mm256_madd_epi16(pairs1, ones)));
}

void MatVec(const std::uint8_t* weights, const float* weightScales,
            const QuantizedActivation& activation, float* output,
            int rows, int columns) {
    const int blocks = columns / kBlock;
#pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; ++row) {
        const std::uint8_t* rowWeights = weights + static_cast<std::size_t>(row) * blocks * kPackedBytes;
        const float* rowScales = weightScales + static_cast<std::size_t>(row) * blocks;
        float total = 0.0f;
        for (int block = 0; block < blocks; ++block) {
            const std::uint8_t* packed = rowWeights + block * kPackedBytes;
            const std::int8_t* act = activation.values.data() + block * kBlock;
            const std::int32_t dot = Dot64(packed, act) + Dot64(packed + 32, act + 64)
                - 8 * activation.sums[block];
            total += static_cast<float>(dot) * activation.scales[block] * rowScales[block];
        }
        output[row] = total;
    }
}

void MatVecChannelwise(const std::uint8_t* weights, const float* weightScales,
                       const QuantizedActivation& activation, float* output,
                       int rows, int columns) {
    const int blocks = columns / kBlock;
#pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; ++row) {
        const std::uint8_t* rowWeights = weights + static_cast<std::size_t>(row) * blocks * kPackedBytes;
        std::int32_t total = 0;
        for (int block = 0; block < blocks; ++block) {
            const std::uint8_t* packed = rowWeights + block * kPackedBytes;
            const std::int8_t* act = activation.values.data() + block * kBlock;
            total += Dot64(packed, act) + Dot64(packed + 32, act + 64)
                - 8 * activation.sums[block];
        }
        output[row] = static_cast<float>(total) * activation.scales[0] * weightScales[row];
    }
}

float ScalarRow(const std::uint8_t* weights, const float* weightScales,
                const QuantizedActivation& activation, int row,
                int columns, bool channelwise) {
    const int blocks = columns / kBlock;
    double total = 0.0;
    std::int32_t integerTotal = 0;
    const auto* rowWeights = weights + static_cast<std::size_t>(row) * blocks * kPackedBytes;
    for (int block = 0; block < blocks; ++block) {
        const auto* packed = rowWeights + block * kPackedBytes;
        std::int32_t dot = 0;
        for (int half = 0; half < 2; ++half) {
            for (int index = 0; index < 32; ++index) {
                const std::uint8_t pair = packed[half * 32 + index];
                const int w0 = int(pair & 15u) - 8;
                const int w1 = int(pair >> 4) - 8;
                const int base = block * kBlock + half * 64;
                dot += w0 * int(activation.values[base + index]);
                dot += w1 * int(activation.values[base + 32 + index]);
            }
        }
        if (channelwise) integerTotal += dot;
        else total += double(dot) * activation.scales[block] * weightScales[static_cast<std::size_t>(row) * blocks + block];
    }
    if (channelwise) return float(integerTotal) * activation.scales[0] * weightScales[row];
    return static_cast<float>(total);
}

void FillPackedWeights(std::vector<std::uint8_t>& weights,
                       std::vector<float>& scales, int rows, int columns) {
    const int blocks = columns / kBlock;
    std::uint32_t state = 0x12345678u;
    auto next = [&]() { state = state * 1664525u + 1013904223u; return state; };
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < blocks; ++block) {
            auto* packed = weights.data() + (static_cast<std::size_t>(row) * blocks + block) * kPackedBytes;
            scales[static_cast<std::size_t>(row) * blocks + block] = 0.003f + float(next() & 255u) * 0.000001f;
            for (int half = 0; half < 2; ++half) {
                for (int index = 0; index < 32; ++index) {
                    const std::uint8_t q0 = static_cast<std::uint8_t>((next() >> 28) & 15u);
                    const std::uint8_t q1 = static_cast<std::uint8_t>((next() >> 28) & 15u);
                    packed[half * 32 + index] = static_cast<std::uint8_t>(q0 | (q1 << 4));
                }
            }
        }
    }
}
}

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 151936;
    const int columns = argc > 2 ? std::atoi(argv[2]) : 2048;
    const int threads = argc > 3 ? std::atoi(argv[3]) : 4;
    const int iterations = argc > 4 ? std::atoi(argv[4]) : 20;
    const bool channelwise = argc > 5 && std::atoi(argv[5]) != 0;
    if (rows <= 0 || columns <= 0 || columns % kBlock != 0) return 2;
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    const int blocks = columns / kBlock;
    std::vector<std::uint8_t> weights(static_cast<std::size_t>(rows) * blocks * kPackedBytes);
    std::vector<float> weightScales(static_cast<std::size_t>(rows) * blocks);
    FillPackedWeights(weights, weightScales, rows, columns);
    if (channelwise) {
        for (int row = 0; row < rows; ++row) weightScales[row] = weightScales[static_cast<std::size_t>(row) * blocks];
    }
    std::vector<float> input(columns), output(rows);
    for (int index = 0; index < columns; ++index) input[index] = std::sin(index * 0.013f);
    auto activation = channelwise ? QuantizeActivationChannelwise(input.data(), columns) : QuantizeActivation(input.data(), columns);
    const auto run = [&]() {
        if (channelwise) MatVecChannelwise(weights.data(), weightScales.data(), activation, output.data(), rows, columns);
        else MatVec(weights.data(), weightScales.data(), activation, output.data(), rows, columns);
    };
    run();
    const float scalar = ScalarRow(weights.data(), weightScales.data(), activation, 0, columns, channelwise);
    const float tolerance = std::max(1.0e-5f, std::abs(scalar) * 2.0e-5f);
    if (std::abs(output[0] - scalar) > tolerance) {
        std::fprintf(stderr, "scalar parity failed: avx2=%.9g scalar=%.9g\n", output[0], scalar);
        return 4;
    }
    for (int warmup = 0; warmup < 3; ++warmup) run();
    std::vector<double> samples;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        run();
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(samples.begin(), samples.end());
    const double average = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    const double checksum = std::accumulate(output.begin(), output.end(), 0.0);
    const std::size_t scaleCount = channelwise ? static_cast<std::size_t>(rows) : weightScales.size();
    const double weightMB = (weights.size() + scaleCount * sizeof(float)) / 1048576.0;
    std::printf("rows=%d columns=%d threads=%d channelwise=%d weight_mb=%.2f median_ms=%.3f avg_ms=%.3f checksum=%.9g parity_error=%.3g\n",
                rows, columns, threads, channelwise ? 1 : 0, weightMB, samples[samples.size() / 2], average,
                checksum, std::abs(output[0] - scalar));
    return std::isfinite(checksum) ? 0 : 3;
}
