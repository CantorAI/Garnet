// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <NvInfer.h>
#include "gpt_oss_kernels.h"
#include "gpt_oss_marlin.h"
#include <string>

namespace Garnet {
class GptOssPlugin final : public nvinfer1::IPluginV2DynamicExt {
    GptOssOptions m_options;
    std::unique_ptr<GptOssMarlin> m_marlin;
    std::string m_namespace;
    bool m_valid = false;
    bool m_tpInitialized = false;
public:
    explicit GptOssPlugin(const GptOssOptions& options);
    GptOssPlugin(const void*, size_t);
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int, const nvinfer1::DimsExprs*, int,
        nvinfer1::IExprBuilder&) noexcept override;
    bool supportsFormatCombination(int, const nvinfer1::PluginTensorDesc*, int, int) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc*, int,
        const nvinfer1::DynamicPluginTensorDesc*, int) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*, int,
        const nvinfer1::PluginTensorDesc*, int) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc*, const nvinfer1::PluginTensorDesc*,
        const void* const*, void* const*, void*, cudaStream_t) noexcept override;
    nvinfer1::DataType getOutputDataType(int, const nvinfer1::DataType*, int) const noexcept override;
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void*) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char*) noexcept override;
    const char* getPluginNamespace() const noexcept override;
};
bool EnsureGptOssPluginRegistered();
}
