#pragma once

#include "xlang.h"

#include <string>
#include <vector>

namespace Garnet
{
    struct QwenTextCompiledInputs
    {
        X::Value inputs;
        int promptTokenCount = 0;
        std::string error;
    };

    QwenTextCompiledInputs BuildQwenTextCompiledInputs(
        const std::string& modelDirectory,
        const std::string& prompt,
        bool enableThinking,
        const std::vector<std::vector<int>>& profileShapes,
        X::Value reusableKeyCache = X::Value(),
        X::Value reusableValueCache = X::Value());
}
