#pragma once

#include "xlang.h"

namespace Garnet::Image
{
    struct PreprocessResult
    {
        X::Value pixelValues;
        X::Value imageGridTHW;
        int resizedHeight = 0;
        int resizedWidth = 0;
        int patchSize = 0;
        int temporalPatchSize = 0;
        int mergeSize = 0;
    };
}
