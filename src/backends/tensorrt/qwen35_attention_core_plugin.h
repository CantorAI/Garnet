// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <NvInferRuntime.h>
#include "../../cuda/qwen35_attention_core.cuh"
#include <string>
namespace Garnet {
// Experimental wrapper, enabled only by the explicit fusedAttention profile.
class Qwen35AttentionCorePlugin final:public nvinfer1::IPluginV2DynamicExt {
    Qwen35AttentionCoreConfig c_;std::string ns_;
    bool matches(const nvinfer1::PluginTensorDesc*,int) const noexcept;
public:
    explicit Qwen35AttentionCorePlugin(Qwen35AttentionCoreConfig);
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int,const nvinfer1::DimsExprs*,int,nvinfer1::IExprBuilder&) noexcept override;
    bool supportsFormatCombination(int,const nvinfer1::PluginTensorDesc*,int,int) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc*,int,const nvinfer1::DynamicPluginTensorDesc*,int) noexcept override{}
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*,int,const nvinfer1::PluginTensorDesc*,int) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc*,const nvinfer1::PluginTensorDesc*,const void* const*,void* const*,void*,cudaStream_t) noexcept override;
    nvinfer1::DataType getOutputDataType(int,const nvinfer1::DataType*,int) const noexcept override{return nvinfer1::DataType::kFLOAT;}
    const char* getPluginType() const noexcept override{return "GarnetQwen35AttentionCore";}
    const char* getPluginVersion() const noexcept override{return "1";}
    int getNbOutputs() const noexcept override{return 3;}
    int initialize() noexcept override{return 0;}
    void terminate() noexcept override{}
    size_t getSerializationSize() const noexcept override{return sizeof(c_);}
    void serialize(void*) const noexcept override;
    void destroy() noexcept override{delete this;}
    void setPluginNamespace(const char* value) noexcept override{ns_=value?value:"";}
    const char* getPluginNamespace() const noexcept override{return ns_.c_str();}
};
bool EnsureQwen35AttentionCorePluginRegistered();
}
