// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <NvInferRuntime.h>
#include <memory>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
namespace Garnet {
std::shared_ptr<void> CreateBF16ProjectionContext();
int RunBF16Projection(const std::shared_ptr<void>& context,const void* input,const void* weight,
    void* output,int rows,int inner,int columns,cudaStream_t stream,void* workspace=nullptr,size_t workspaceBytes=0) noexcept;
class BF16ProjectionPlugin final:public nvinfer1::IPluginV2DynamicExt {
    std::string namespace_;
    std::shared_ptr<void> context_;
    bool workspace_=false;
public:
    explicit BF16ProjectionPlugin(bool workspace=false):workspace_(workspace){}
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int,const nvinfer1::DimsExprs*,int,nvinfer1::IExprBuilder&) noexcept override;
    bool supportsFormatCombination(int,const nvinfer1::PluginTensorDesc*,int,int) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc*,int,const nvinfer1::DynamicPluginTensorDesc*,int) noexcept override {}
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*,int,const nvinfer1::PluginTensorDesc*,int) const noexcept override{return workspace_?16384:0;}
    int enqueue(const nvinfer1::PluginTensorDesc*,const nvinfer1::PluginTensorDesc*,const void* const*,void* const*,void*,cudaStream_t) noexcept override;
    nvinfer1::DataType getOutputDataType(int,const nvinfer1::DataType*,int) const noexcept override{return nvinfer1::DataType::kBF16;}
    const char* getPluginType() const noexcept override{return "GarnetBF16Projection";}
    const char* getPluginVersion() const noexcept override{return workspace_?"2":"1";}
    int getNbOutputs() const noexcept override{return 1;}
    int initialize() noexcept override;
    void terminate() noexcept override{context_.reset();}
    size_t getSerializationSize() const noexcept override{return workspace_?sizeof(uint32_t):0;}
    void serialize(void* data) const noexcept override{if(workspace_&&data){uint32_t bytes=16384;std::memcpy(data,&bytes,sizeof(bytes));}}
    void destroy() noexcept override{delete this;}
    void setPluginNamespace(const char* value) noexcept override{namespace_=value?value:"";}
    const char* getPluginNamespace() const noexcept override{return namespace_.c_str();}
};
bool EnsureBF16ProjectionPluginRegistered();
}
