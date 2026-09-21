// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "safetensors_index.h"

#include <cstdint>
#include <string>
#include <vector>

namespace Garnet
{
    struct QuantizedWeight
    {
        std::vector<long long> shape;
        std::vector<std::uint8_t> packedValues;
        std::vector<std::uint16_t> fp16Scales;
        int blockSize = 0;
        int blockAxis = 1;
        double maximumAbsoluteError = 0.0;

        std::uint64_t LogicalElementCount() const;
        std::vector<long long> ScaleShape() const;
    };

    bool IsQwenLinearWeightForINT4(
        const std::string& name,
        const SafeTensorMetadata& metadata,
        int blockSize);

    bool QuantizeSymmetricINT4(
        const std::string& name,
        const SafeTensorMetadata& metadata,
        const void* source,
        int blockSize,
        QuantizedWeight& output,
        std::string& errorMessage);
}
