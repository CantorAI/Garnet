// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Garnet::Tokenization
{
    class Tokenizer
    {
    public:
        virtual ~Tokenizer() = default;

        virtual bool LoadFromFolder(const std::string& modelDir, std::string* error = nullptr) = 0;
        virtual std::vector<int64_t> Encode(const std::string& text, bool addSpecialTokens = false) const = 0;
        virtual std::string Decode(const std::vector<int64_t>& tokenIds, bool skipSpecialTokens = true) const = 0;
        virtual int64_t TokenId(const std::string& token) const = 0;
    };
}
