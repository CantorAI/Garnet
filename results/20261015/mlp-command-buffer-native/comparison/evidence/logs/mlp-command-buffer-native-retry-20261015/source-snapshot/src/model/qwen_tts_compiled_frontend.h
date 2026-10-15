// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "xlang3/xlang3.h"

#include <string>
#include <vector>

namespace Garnet
{
    struct QwenTTSCompiledInputs
    {
        X::Value inputs;
        std::string error;
        int promptTokenCount = 0;
        int codecEosTokenId = 2150;
    };

    QwenTTSCompiledInputs BuildQwenTTSCompiledInputs(X3PackageHost* host,
        const std::string& modelDirectory,
        const std::string& text,
        const std::string& speaker,
        const std::string& language,
        const std::string& instruct,
        const std::vector<std::vector<int>>& profileShapes,
        X::Value reusableKeyCache = X::Value(),
        X::Value reusableValueCache = X::Value());
}
