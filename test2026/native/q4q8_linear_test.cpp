// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "q4q8_linear.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace
{
    float FP16ToFloat(std::uint16_t value)
    {
        const std::uint32_t sign =
            static_cast<std::uint32_t>(value & 0x8000U) << 16;
        const std::uint32_t exponent = (value >> 10) & 0x1FU;
        const std::uint32_t mantissa = value & 0x03FFU;
        const std::uint32_t bits = exponent == 0
            ? sign
            : sign | ((exponent + 112U) << 23) | (mantissa << 13);
        float result = 0.0F;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }
}

int main()
{
    constexpr int rows = 7;
    constexpr int columns = 256;
    constexpr int groupSize = 128;
    std::vector<float> source(rows * columns);
    std::vector<float> input(columns);
    for (int index = 0; index < rows * columns; ++index) {
        source[index] = std::sin(index * 0.019F) * 0.08F;
    }
    for (int index = 0; index < columns; ++index) {
        input[index] = std::cos(index * 0.031F);
    }
    Garnet::SafeTensorMetadata metadata;
    metadata.dataType = "F32";
    metadata.shape = {rows, columns};
    metadata.dataSize = source.size() * sizeof(float);
    Garnet::QuantizedWeight quantized;
    std::string error;
    if (!Garnet::QuantizeSymmetricINT4(
            "model.layers.0.self_attn.q_proj.weight",
            metadata, source.data(), groupSize, quantized, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    Garnet::Q4Q8Linear linear;
    if (!linear.Prepare(quantized, error)) {
        std::cerr << error << '\n';
        return 2;
    }
    std::vector<float> actual(rows);
    if (!linear.Forward(input.data(), actual.data(), 4, error)) {
        std::cerr << error << '\n';
        return 3;
    }
    Garnet::Q4Q8Linear persistedView;
    if (!persistedView.BindPackedView(
            linear.Rows(), linear.Columns(), linear.GroupSize(),
            linear.PackedWeightData(), linear.PackedWeightBytes(),
            linear.PackedScaleData(), linear.PackedScaleCount(), error)) {
        std::cerr << error << '\n';
        return 5;
    }
    std::vector<float> viewActual(rows);
    if (!persistedView.Forward(
            input.data(), viewActual.data(), 4, error)) {
        std::cerr << error << '\n';
        return 6;
    }
    if (!std::equal(actual.begin(), actual.end(), viewActual.begin())) {
        std::cerr << "Q4Q8 persisted view parity failed\n";
        return 7;
    }
    double maximumError = 0.0;
    const int groups = columns / groupSize;
    for (int row = 0; row < rows; ++row) {
        float expected = 0.0F;
        for (int group = 0; group < groups; ++group) {
            const int base = group * groupSize;
            float maximum = 0.0F;
            for (int index = 0; index < groupSize; ++index) {
                maximum = std::max(maximum, std::abs(input[base + index]));
            }
            const float inputScale = maximum / 127.0F;
            std::int32_t dot = 0;
            for (int index = 0; index < groupSize; ++index) {
                const int qInput = std::clamp<int>(
                    static_cast<int>(std::nearbyint(
                        input[base + index] / inputScale)), -127, 127);
                const std::size_t logical =
                    static_cast<std::size_t>(row) * columns + base + index;
                const std::uint8_t byte =
                    quantized.packedValues[logical / 2];
                int qWeight =
                    (byte >> ((logical & 1U) * 4U)) & 15;
                if (qWeight >= 8) qWeight -= 16;
                dot += qInput * qWeight;
            }
            expected += static_cast<float>(dot) * inputScale *
                FP16ToFloat(quantized.fp16Scales[
                    static_cast<std::size_t>(row) * groups + group]);
        }
        maximumError = std::max(
            maximumError,
            static_cast<double>(std::abs(expected - actual[row])));
    }
    if (maximumError > 1.0e-5) {
        std::cerr << "Q4Q8 scalar parity error: " << maximumError << '\n';
        return 4;
    }
    std::cout << "Garnet Q4Q8 AVX2 scalar parity passed; max_error="
              << maximumError << " packed_bytes=" << linear.PackedBytes()
              << '\n';
    return 0;
}
