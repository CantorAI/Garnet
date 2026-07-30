#pragma once

#include "../image_preprocessor.h"
#include "../image_source.h"
#include "xlang.h"
#include <cstddef>
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

    struct DevicePreprocessResult
    {
        float* pixelValuesDevice = nullptr;
        long long imageGridTHW[3] = { 1, 0, 0 };
        int sourceHeight = 0;
        int sourceWidth = 0;
        int resizedHeight = 0;
        int resizedWidth = 0;
        int patchSize = 16;
        int temporalPatchSize = 2;
        int mergeSize = 2;
        int patchCount = 0;
        int featureDim = 0;
        size_t outputBytes = 0;
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

    PreprocessResult PreprocessJpegFileToTensor(
        const std::string& jpegPath,
        int minPixels,
        int maxPixels);

    PreprocessResult PreprocessJpegBytesToTensor(
        const unsigned char* jpegData,
        size_t jpegSize,
        int minPixels,
        int maxPixels);

    DevicePreprocessResult PreprocessJpegFileToDeviceBuffer(
        const std::string& jpegPath,
        int minPixels,
        int maxPixels);

    DevicePreprocessResult PreprocessJpegBytesToDeviceBuffer(
        const unsigned char* jpegData,
        size_t jpegSize,
        int minPixels,
        int maxPixels);
}
