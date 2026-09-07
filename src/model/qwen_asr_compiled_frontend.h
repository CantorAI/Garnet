#pragma once

#include "xlang3/xlang3.h"

#include <string>
#include <vector>

namespace Garnet
{
    struct QwenASRCompiledInputs
    {
        X::Value inputs;
        int promptTokenCount = 0;
        int audioTokenCount = 0;
        int audioSampleCount = 0;
        double audioDurationSeconds = 0.0;
        std::string error;
    };

    QwenASRCompiledInputs BuildQwenASRCompiledInputs(X3PackageHost* host,
        const std::string& modelDirectory,
        X::Value audioSource,
        const std::string& context,
        const std::string& language,
        const std::vector<std::vector<int>>& profileShapes,
        X::Value reusableKeyCache = X::Value(),
        X::Value reusableValueCache = X::Value());
}
