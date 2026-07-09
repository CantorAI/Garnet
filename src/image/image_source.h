#pragma once

#include <string>

namespace Garnet::Image
{
    enum class PixelFormat
    {
        RGB_FLOAT32_HWC,
        BGR_FLOAT32_HWC,
        RGBA_FLOAT32_HWC,
        BGRA_FLOAT32_HWC,
    };

    inline PixelFormat PixelFormatFromString(const std::string& value)
    {
        if (value == "rgb" || value == "RGB" || value == "rgb_f32_hwc") {
            return PixelFormat::RGB_FLOAT32_HWC;
        }
        if (value == "bgr" || value == "BGR" || value == "bgr_f32_hwc") {
            return PixelFormat::BGR_FLOAT32_HWC;
        }
        if (value == "rgba" || value == "RGBA" || value == "rgba_f32_hwc") {
            return PixelFormat::RGBA_FLOAT32_HWC;
        }
        if (value == "bgra" || value == "BGRA" || value == "bgra_f32_hwc") {
            return PixelFormat::BGRA_FLOAT32_HWC;
        }
        return PixelFormat::RGB_FLOAT32_HWC;
    }

    inline int ChannelCount(PixelFormat format)
    {
        switch (format) {
        case PixelFormat::RGBA_FLOAT32_HWC:
        case PixelFormat::BGRA_FLOAT32_HWC:
            return 4;
        default:
            return 3;
        }
    }
}
