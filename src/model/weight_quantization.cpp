// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "weight_quantization.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace Garnet
{
    namespace
    {
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
            std::uint32_t mantissa = value & 0x03FFU;
            std::uint32_t bits = 0;
            if (exponent == 0) {
                if (mantissa == 0) {
                    bits = sign;
                }
                else {
                    int shift = 0;
                    while ((mantissa & 0x0400U) == 0) {
                        mantissa <<= 1;
                        ++shift;
                    }
                    mantissa &= 0x03FFU;
                    const std::uint32_t fp32Exponent =
                        static_cast<std::uint32_t>(127 - 15 - shift);
                    bits = sign | (fp32Exponent << 23) | (mantissa << 13);
                }
            }
            else if (exponent == 0x1FU) {
                bits = sign | 0x7F800000U | (mantissa << 13);
            }
            else {
                exponent += 127 - 15;
                bits = sign | (exponent << 23) | (mantissa << 13);
            }
            float result = 0.0F;
            std::memcpy(&result, &bits, sizeof(result));
            return result;
        }

        std::uint16_t FloatToFP16(float value)
        {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            const std::uint32_t sign = (bits >> 16) & 0x8000U;
            const std::uint32_t absolute = bits & 0x7FFFFFFFU;
            if (absolute >= 0x7F800000U) {
                return static_cast<std::uint16_t>(
                    sign | (absolute > 0x7F800000U ? 0x7E00U : 0x7C00U));
            }
            int exponent = static_cast<int>((absolute >> 23) & 0xFFU) - 127;
            std::uint32_t mantissa = absolute & 0x7FFFFFU;
            if (exponent > 15) {
                return static_cast<std::uint16_t>(sign | 0x7C00U);
            }
            if (exponent < -14) {
                if (exponent < -24) return static_cast<std::uint16_t>(sign);
                mantissa |= 0x800000U;
                const int shift = -exponent - 14;
                std::uint32_t rounded =
                    mantissa >> static_cast<unsigned>(shift + 13);
                const std::uint32_t remainderMask =
                    (1U << static_cast<unsigned>(shift + 13)) - 1U;
                const std::uint32_t remainder = mantissa & remainderMask;
                const std::uint32_t halfway =
                    1U << static_cast<unsigned>(shift + 12);
                if (remainder > halfway ||
                    (remainder == halfway && (rounded & 1U))) {
                    ++rounded;
                }
                return static_cast<std::uint16_t>(sign | rounded);
            }
            std::uint32_t rounded = mantissa >> 13;
            const std::uint32_t remainder = mantissa & 0x1FFFU;
            if (remainder > 0x1000U ||
                (remainder == 0x1000U && (rounded & 1U))) {
                ++rounded;
                if (rounded == 0x0400U) {
                    rounded = 0;
                    ++exponent;
                    if (exponent > 15) {
                        return static_cast<std::uint16_t>(sign | 0x7C00U);
                    }
                }
            }
            return static_cast<std::uint16_t>(
                sign |
                (static_cast<std::uint32_t>(exponent + 15) << 10) |
                rounded);
        }

        float ReadWeight(
            const SafeTensorMetadata& metadata,
            const void* source,
            std::uint64_t index)
        {
            if (metadata.dataType == "BF16") {
                return BF16ToFloat(
                    static_cast<const std::uint16_t*>(source)[index]);
            }
            if (metadata.dataType == "F16") {
                return FP16ToFloat(
                    static_cast<const std::uint16_t*>(source)[index]);
            }
            return static_cast<const float*>(source)[index];
        }

        std::uint8_t PackNibble(int value)
        {
            return static_cast<std::uint8_t>(value) & 0x0FU;
        }
    }

    std::uint64_t QuantizedWeight::LogicalElementCount() const
    {
        std::uint64_t result = 1;
        for (const long long dimension : shape) {
            result *= static_cast<std::uint64_t>(dimension);
        }
        return result;
    }

    std::vector<long long> QuantizedWeight::ScaleShape() const
    {
        if (shape.size() != 2 || blockSize <= 0) return {};
        return {shape[0], shape[1] / blockSize};
    }

    bool IsQwenLinearWeightForINT4(
        const std::string& name,
        const SafeTensorMetadata& metadata,
        int blockSize)
    {
        if (metadata.shape.size() != 2 ||
            metadata.shape[0] <= 0 ||
            metadata.shape[1] <= 0 ||
            blockSize <= 0 ||
            metadata.shape[1] % blockSize != 0 ||
            (metadata.dataType != "BF16" &&
             metadata.dataType != "F16" &&
             metadata.dataType != "F32")) {
            return false;
        }
        static constexpr const char* suffixes[] = {
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
            ".self_attn.o_proj.weight",
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.down_proj.weight",
            "lm_head.weight",
        };
        for (const char* suffix : suffixes) {
            if (name.size() >= std::strlen(suffix) &&
                name.compare(
                    name.size() - std::strlen(suffix),
                    std::strlen(suffix),
                    suffix) == 0) {
                return true;
            }
        }
        return false;
    }

    bool QuantizeSymmetricINT4(
        const std::string& name,
        const SafeTensorMetadata& metadata,
        const void* source,
        int blockSize,
        QuantizedWeight& output,
        std::string& errorMessage)
    {
        output = {};
        if (!source ||
            !IsQwenLinearWeightForINT4(name, metadata, blockSize)) {
            errorMessage =
                "INT4 quantization requires a supported rank-two Qwen weight";
            return false;
        }
        const std::uint64_t rows =
            static_cast<std::uint64_t>(metadata.shape[0]);
        const std::uint64_t columns =
            static_cast<std::uint64_t>(metadata.shape[1]);
        if (rows > std::numeric_limits<std::uint64_t>::max() / columns) {
            errorMessage = "INT4 weight element count overflows";
            return false;
        }
        const std::uint64_t elements = rows * columns;
        const std::uint64_t groupsPerRow =
            columns / static_cast<std::uint64_t>(blockSize);
        output.shape = metadata.shape;
        output.blockSize = blockSize;
        output.packedValues.assign(
            static_cast<size_t>((elements + 1) / 2), 0);
        output.fp16Scales.resize(
            static_cast<size_t>(rows * groupsPerRow));
        for (std::uint64_t row = 0; row < rows; ++row) {
            for (std::uint64_t group = 0;
                 group < groupsPerRow;
                 ++group) {
                const std::uint64_t start =
                    row * columns + group * output.blockSize;
                float minimum = 0.0F;
                float maximum = 0.0F;
                for (std::uint64_t offset = 0;
                     offset < static_cast<std::uint64_t>(output.blockSize);
                     ++offset) {
                    const float value =
                        ReadWeight(metadata, source, start + offset);
                    minimum = std::min(minimum, value);
                    maximum = std::max(maximum, value);
                }
                float scale = std::max(-minimum / 8.0F, maximum / 7.0F);
                if (!std::isfinite(scale) || scale <= 0.0F) {
                    scale = 1.0F;
                }
                const std::uint16_t scaleBits = FloatToFP16(scale);
                const float storedScale = FP16ToFloat(scaleBits);
                output.fp16Scales[
                    static_cast<size_t>(row * groupsPerRow + group)] =
                    scaleBits;
                for (std::uint64_t offset = 0;
                     offset < static_cast<std::uint64_t>(output.blockSize);
                     ++offset) {
                    const std::uint64_t index = start + offset;
                    const float value = ReadWeight(metadata, source, index);
                    int quantized = static_cast<int>(
                        std::nearbyint(value / storedScale));
                    quantized = std::max(-8, std::min(7, quantized));
                    const size_t byteIndex =
                        static_cast<size_t>(index / 2);
                    const unsigned shift =
                        static_cast<unsigned>((index & 1U) * 4U);
                    output.packedValues[byteIndex] |=
                        static_cast<std::uint8_t>(
                            PackNibble(quantized) << shift);
                    output.maximumAbsoluteError = std::max(
                        output.maximumAbsoluteError,
                        static_cast<double>(std::abs(
                            value -
                            static_cast<float>(quantized) * storedScale)));
                }
            }
        }
        errorMessage.clear();
        return true;
    }
}
