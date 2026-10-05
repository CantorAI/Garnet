// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <NvInfer.h>
#include "gpt_oss_kernels.h"
#include <string>
namespace Garnet {
// Loads and retains the optional family library for the lifetime of the process.
nvinfer1::IPluginV2DynamicExt* CreateGptOssPlugin(const GptOssOptions&, std::string& error);
}
