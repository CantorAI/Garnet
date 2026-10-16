// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "q4q8_linear.h"
#include "safetensors_index.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace Garnet
{
    class Qwen3NativeDecodeEngine
    {
        struct Implementation;
        std::unique_ptr<Implementation> m_implementation;

    public:
        struct DecodeInput
        {
            long long tokenId = -1;
            long long position = 0;
            std::uint16_t* keyCache = nullptr;
            std::uint16_t* valueCache = nullptr;
            const int* pageTable = nullptr;
            int pageTableSize = 0;
            int pages = 0;
            int pageSize = 0;
            int contextLength = 0;
            int slotPosition = 0;
        };

        Qwen3NativeDecodeEngine();
        ~Qwen3NativeDecodeEngine();
        bool Prepare(
            const SafeTensorsIndex& index,
            std::string& errorMessage);
        bool Prepare(
            const SafeTensorsIndex& index,
            const std::filesystem::path& packedCachePath,
            std::string& errorMessage);
        std::uint64_t PackedBytes() const;
        bool Decode(
            const DecodeInput& input,
            long long& outputTokenId,
            std::string& errorMessage) const;
    };
}
