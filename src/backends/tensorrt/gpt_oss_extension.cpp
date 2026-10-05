// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_extension.h"
#include "operator_plugins.h"
namespace Garnet {
nvinfer1::IPluginV2DynamicExt* CreateGptOssPlugin(const GptOssOptions& options, std::string& error) {
    using Factory = nvinfer1::IPluginV2DynamicExt* (*)(const void*, size_t);
    auto factory = reinterpret_cast<Factory>(OperatorPluginSymbol("gpt_oss", "GarnetCreateOperatorPlugin", error));
    if (!factory) return nullptr;
    auto* plugin = factory(&options, sizeof(options));
    if (!plugin) error = "GPT-OSS operator library rejected operator attributes";
    return plugin;
}
}
