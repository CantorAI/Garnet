#pragma once

#include "xlang.h"

namespace Garnet::Image::QwenVL
{
    struct VisionMetadataTensors
    {
        X::Value bilinearIndices;
        X::Value bilinearWeights;
        X::Value positionIds;
        X::Value cuSeqlens;
    };

    VisionMetadataTensors BuildVisionMetadataTensors(
        int gridT,
        int gridH,
        int gridW,
        int spatialMergeSize,
        int numPositionEmbeddings = 2304);
}
