#pragma once

#include "xlang.h"

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

    QwenTTSCompiledInputs BuildQwenTTSCompiledInputs(
        const std::string& modelDirectory,
        const std::string& text,
        const std::string& speaker,
        const std::string& language,
        const std::vector<std::vector<int>>& profileShapes,
        X::Value reusableKeyCache = X::Value(),
        X::Value reusableValueCache = X::Value());
}
