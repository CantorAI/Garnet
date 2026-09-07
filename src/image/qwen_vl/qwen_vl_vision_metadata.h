#pragma once

#include "xlang3/xlang3.h"

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
        X3PackageHost* host,
        int gridT,
        int gridH,
        int gridW,
        int spatialMergeSize,
        int numPositionEmbeddings = 2304);
}
