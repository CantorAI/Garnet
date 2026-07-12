#pragma once

#include "xlang.h"

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

    QwenVLCompiledInputs BuildQwenVLCompiledInputs(
        const std::string& modelDirectory,
        const std::string& imagePath,
        const std::string& prompt,
        int minPixels,
        int maxPixels,
        const std::vector<std::vector<int>>& profileShapes);
}
