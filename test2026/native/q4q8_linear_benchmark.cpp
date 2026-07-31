#include "q4q8_linear.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace
{
    std::uint16_t FP16(float value)
    {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        const std::uint32_t sign = (bits >> 16) & 0x8000U;
        const int exponent = static_cast<int>((bits >> 23) & 0xFFU) - 127;
        if (exponent < -14) return static_cast<std::uint16_t>(sign);
        if (exponent > 15) return static_cast<std::uint16_t>(sign | 0x7C00U);
        return static_cast<std::uint16_t>(
            sign | (static_cast<std::uint32_t>(exponent + 15) << 10) |
            ((bits >> 13) & 0x03FFU));
    }

    Garnet::QuantizedWeight Weight(int rows, int columns, int group)
    {
        Garnet::QuantizedWeight result;
        result.shape = {rows, columns};
        result.blockSize = group;
        result.packedValues.resize(
            static_cast<std::size_t>(rows) * columns / 2);
        for (std::size_t index = 0;
             index < static_cast<std::size_t>(rows) * columns; ++index) {
            const int q = static_cast<int>((index * 13 + index / 17) % 16) - 8;
            result.packedValues[index / 2] |= static_cast<std::uint8_t>(
                (q & 15) << ((index & 1U) * 4U));
        }
        result.fp16Scales.assign(
            static_cast<std::size_t>(rows) * columns / group,
            FP16(0.01F));
        return result;
    }
}

int main(int argc, char** argv)
{
    std::fprintf(stderr, "garnet-q4q8-v2 avx2=%d\n",
        Garnet::Q4Q8Linear::AVX2Supported() ? 1 : 0);
    const int rows = argc > 1 ? std::atoi(argv[1]) : 151936;
    const int columns = argc > 2 ? std::atoi(argv[2]) : 2048;
    const int threads = argc > 3 ? std::atoi(argv[3]) : 4;
    const int group = argc > 4 ? std::atoi(argv[4]) : 128;
    const int iterations = argc > 5 ? std::atoi(argv[5]) : 10;
    auto weight = Weight(rows, columns, group);
    Garnet::Q4Q8Linear linear;
    std::string error;
    if (!linear.Prepare(weight, error)) {
        std::fprintf(stderr, "prepare: %s\n", error.c_str());
        return 2;
    }
    weight = {};
    std::vector<float> input(static_cast<std::size_t>(columns));
    std::vector<float> output(static_cast<std::size_t>(rows));
    for (int index = 0; index < columns; ++index) {
        input[static_cast<std::size_t>(index)] = std::sin(index * 0.017F);
    }
    for (int index = 0; index < 3; ++index) {
        if (!linear.Forward(input.data(), output.data(), threads, error)) {
            std::fprintf(stderr, "forward: %s\n", error.c_str());
            return 3;
        }
    }
    std::vector<double> times;
    for (int index = 0; index < iterations; ++index) {
        const auto begin = std::chrono::steady_clock::now();
        if (!linear.Forward(input.data(), output.data(), threads, error)) return 4;
        times.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count());
    }
    std::sort(times.begin(), times.end());
    std::printf(
        "signature=garnet-q4q8-v2 N=%d K=%d threads=%d group=%d packed_mb=%.2f median_ms=%.3f avg_ms=%.3f checksum=%.9g\n",
        rows, columns, threads, group,
        linear.PackedBytes() / 1048576.0,
        times[times.size() / 2],
        std::accumulate(times.begin(), times.end(), 0.0) / times.size(),
        std::accumulate(output.begin(), output.end(), 0.0));
    return 0;
}
