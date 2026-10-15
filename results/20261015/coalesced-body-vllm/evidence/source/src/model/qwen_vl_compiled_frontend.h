// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "xlang3/xlang3.h"

#include <string>
#include <vector>

namespace Garnet
{
    struct QwenVLCompiledInputs
    {
        X::Value inputs;
        int promptTokenCount = 0;
        int visualTokenCount = 0;
        int sourceHeight = 0;
        int sourceWidth = 0;
        int resizedHeight = 0;
        int resizedWidth = 0;
        long long mropePositionDelta = 0;
        std::string error;
    };

    QwenVLCompiledInputs BuildQwenVLCompiledInputs(X3PackageHost* host,
        const std::string& modelDirectory,
        X::Value imageSource,
        const std::string& prompt,
        int minPixels,
        int maxPixels,
        const std::vector<std::vector<int>>& profileShapes,
        X::Value reusableKeyCache = X::Value(),
        X::Value reusableValueCache = X::Value());
}
