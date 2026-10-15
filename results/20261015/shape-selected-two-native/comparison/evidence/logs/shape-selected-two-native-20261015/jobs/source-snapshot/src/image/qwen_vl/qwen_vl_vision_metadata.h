// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

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
