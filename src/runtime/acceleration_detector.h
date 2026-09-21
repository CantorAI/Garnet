// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <filesystem>

namespace Garnet
{
    class AccelerationDetector
    {
    public:
        // Uses only the operating-system NVIDIA driver entry point. It does
        // not load CUDA Runtime, cuBLAS, NPP, nvJPEG, or TensorRT.
        static std::string DetectJson(bool probeNvidia = true);
        static std::string PlatformId();
        static std::string ActivateJson(const std::filesystem::path& packageRoot);
    };
}
