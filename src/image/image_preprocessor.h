#pragma once

#include "xlang3/xlang3.h"

namespace Garnet::Image
{
    struct PreprocessResult
    {
        X::Value pixelValues;
        X::Value imageGridTHW;
        X::Value bilinearIndices;
        X::Value bilinearWeights;
        X::Value visionPositionIds;
        X::Value visionCuSeqlens;
        int sourceHeight = 0;
        int sourceWidth = 0;
        int resizedHeight = 0;
        int resizedWidth = 0;
        int patchSize = 0;
        int temporalPatchSize = 0;
        int mergeSize = 0;
    };
}
