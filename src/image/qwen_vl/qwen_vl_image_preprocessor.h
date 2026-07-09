#pragma once

#include "../image_preprocessor.h"
#include "../image_source.h"
#include "xlang.h"
#include <string>

namespace Garnet::Image::QwenVL
{
    struct QwenVLImagePreprocessConfig
    {
        int patchSize = 16;
        int temporalPatchSize = 2;
        int mergeSize = 2;
        float mean[3] = {0.5f, 0.5f, 0.5f};
        float std[3] = {0.5f, 0.5f, 0.5f};
        float inputScale = 255.0f;
        PixelFormat pixelFormat = PixelFormat::RGB_FLOAT32_HWC;
    };

    struct SmartResizeResult
    {
        int height = 0;
        int width = 0;
    };

    SmartResizeResult SmartResize(
        int height,
        int width,
        int factor,
        int minPixels,
        int maxPixels);

    PreprocessResult PreprocessRawImageTensor(
        X::Value rawImageValue,
        int height,
        int width,
        const QwenVLImagePreprocessConfig& config);
}
