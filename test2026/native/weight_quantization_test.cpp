// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "weight_quantization.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace
{
    std::uint16_t FloatToBF16(float value)
    {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        const std::uint32_t rounding =
            0x7FFFU + ((bits >> 16) & 1U);
        return static_cast<std::uint16_t>((bits + rounding) >> 16);
    }

    float BF16ToFloat(std::uint16_t value)
    {
        std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
        float result = 0.0F;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }

    float FP16ToFloat(std::uint16_t value)
    {
        const std::uint32_t sign =
            static_cast<std::uint32_t>(value & 0x8000U) << 16;
        std::uint32_t exponent = (value >> 10) & 0x1FU;
        const std::uint32_t mantissa = value & 0x03FFU;
        std::uint32_t bits =
            exponent == 0
                ? sign
                : sign | ((exponent + 127 - 15) << 23) |
                    (mantissa << 13);
        float result = 0.0F;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }

    int Unpack(const std::vector<std::uint8_t>& packed, std::size_t index)
    {
        const unsigned shift = static_cast<unsigned>((index & 1U) * 4U);
        int value = static_cast<int>(
            (packed[index / 2] >> shift) & 0x0FU);
        return value >= 8 ? value - 16 : value;
    }
}

int main()
{
    Garnet::SafeTensorMetadata metadata;
    metadata.dataType = "BF16";
    metadata.shape = {2, 64};
    std::vector<std::uint16_t> source(128);
    for (std::size_t index = 0; index < source.size(); ++index) {
        const float value =
            std::sin(static_cast<float>(index) * 0.17F) *
            (index < 64 ? 0.75F : 2.5F);
        source[index] = FloatToBF16(value);
    }
    Garnet::QuantizedWeight quantized;
    std::string error;
    if (!Garnet::QuantizeSymmetricINT4(
            "model.layers.0.self_attn.q_proj.weight",
            metadata,
            source.data(),
            64,
            quantized,
            error)) {
        std::cerr << error << '\n';
        return 1;
    }
    if (quantized.packedValues.size() != 64 ||
        quantized.fp16Scales.size() != 2 ||
        quantized.ScaleShape() != std::vector<long long>({2, 1})) {
        std::cerr << "unexpected INT4 artifact shape\n";
        return 2;
    }
    for (std::size_t index = 0; index < source.size(); ++index) {
        const float scale =
            FP16ToFloat(quantized.fp16Scales[index / 64]);
        const float reconstructed =
            static_cast<float>(
                Unpack(quantized.packedValues, index)) *
            scale;
        if (std::abs(reconstructed - BF16ToFloat(source[index])) >
            scale * 0.501F + 1.0e-6F) {
            std::cerr << "INT4 reconstruction exceeded half a step\n";
            return 3;
        }
    }
    std::cout
        << "INT4 block-64 quantizer passed: packed_bytes="
        << quantized.packedValues.size()
        << ", scales=" << quantized.fp16Scales.size()
        << ", max_abs_error=" << quantized.maximumAbsoluteError
        << '\n';

    Garnet::SafeTensorMetadata lmHeadMetadata;
    lmHeadMetadata.dataType = "BF16";
    lmHeadMetadata.shape = {2, 128};
    std::vector<std::uint16_t> lmHeadSource(256);
    for (std::size_t index = 0; index < lmHeadSource.size(); ++index) {
        const float value =
            std::cos(static_cast<float>(index) * 0.11F) *
            (index < 128 ? 1.25F : 3.0F);
        lmHeadSource[index] = FloatToBF16(value);
    }
    Garnet::QuantizedWeight lmHeadQuantized;
    if (!Garnet::QuantizeSymmetricINT4(
            "lm_head.weight",
            lmHeadMetadata,
            lmHeadSource.data(),
            128,
            lmHeadQuantized,
            error)) {
        std::cerr << error << '\n';
        return 4;
    }
    if (lmHeadQuantized.packedValues.size() != 128 ||
        lmHeadQuantized.fp16Scales.size() != 2 ||
        lmHeadQuantized.ScaleShape() !=
            std::vector<long long>({2, 1}) ||
        lmHeadQuantized.blockSize != 128) {
        std::cerr << "unexpected block-128 LM-head artifact shape\n";
        return 5;
    }
    for (std::size_t index = 0; index < lmHeadSource.size(); ++index) {
        const float scale =
            FP16ToFloat(lmHeadQuantized.fp16Scales[index / 128]);
        const float reconstructed =
            static_cast<float>(
                Unpack(lmHeadQuantized.packedValues, index)) *
            scale;
        if (std::abs(
                reconstructed - BF16ToFloat(lmHeadSource[index])) >
            scale * 0.501F + 1.0e-6F) {
            std::cerr
                << "block-128 LM-head reconstruction exceeded half a step\n";
            return 6;
        }
    }
    std::cout
        << "INT4 block-128 LM-head quantizer passed: packed_bytes="
        << lmHeadQuantized.packedValues.size()
        << ", scales=" << lmHeadQuantized.fp16Scales.size()
        << ", max_abs_error="
        << lmHeadQuantized.maximumAbsoluteError
        << '\n';

    Garnet::SafeTensorMetadata channelMetadata;
    channelMetadata.dataType = "BF16";
    channelMetadata.shape = {2, 256};
    std::vector<std::uint16_t> channelSource(512);
    for (std::size_t index = 0; index < channelSource.size(); ++index) {
        channelSource[index] = FloatToBF16(
            std::sin(static_cast<float>(index) * 0.03F) * 2.0F);
    }
    Garnet::QuantizedWeight channelQuantized;
    if (!Garnet::QuantizeSymmetricINT4(
            "model.layers.0.self_attn.q_proj.weight",
            channelMetadata,
            channelSource.data(),
            256,
            channelQuantized,
            error)) {
        std::cerr << error << '\n';
        return 7;
    }
    if (channelQuantized.packedValues.size() != 256 ||
        channelQuantized.fp16Scales.size() != 2 ||
        channelQuantized.ScaleShape() !=
            std::vector<long long>({2, 1}) ||
        channelQuantized.blockSize != 256) {
        std::cerr << "unexpected channel-wise INT4 artifact shape\n";
        return 8;
    }
    std::cout
        << "INT4 channel-wise quantizer passed: packed_bytes="
        << channelQuantized.packedValues.size()
        << ", scales=" << channelQuantized.fp16Scales.size()
        << ", max_abs_error="
        << channelQuantized.maximumAbsoluteError
        << '\n';
    return 0;
}
