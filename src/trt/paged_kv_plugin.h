#pragma once

#include <NvInferRuntime.h>

#include <string>
#include <vector>

namespace Garnet
{
    class PagedKVDecodePlugin final : public nvinfer1::IPluginV2DynamicExt
    {
        int m_pageSize = 0;
        int m_qHeads = 0;
        int m_kvHeads = 0;
        int m_headDim = 0;
        int m_layerIndex = -1;
        std::string m_namespace;

    public:
        PagedKVDecodePlugin(int pageSize, int qHeads, int kvHeads, int headDim, int layerIndex = -1);
        PagedKVDecodePlugin(const void* data, size_t length);

        nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
        nvinfer1::DimsExprs getOutputDimensions(
            int outputIndex,
            const nvinfer1::DimsExprs* inputs,
            int nbInputs,
            nvinfer1::IExprBuilder& expressionBuilder) noexcept override;
        bool supportsFormatCombination(
            int position,
            const nvinfer1::PluginTensorDesc* inOut,
            int nbInputs,
            int nbOutputs) noexcept override;
        void configurePlugin(
            const nvinfer1::DynamicPluginTensorDesc* inputs,
            int nbInputs,
            const nvinfer1::DynamicPluginTensorDesc* outputs,
            int nbOutputs) noexcept override;
        size_t getWorkspaceSize(
            const nvinfer1::PluginTensorDesc* inputs,
            int nbInputs,
            const nvinfer1::PluginTensorDesc* outputs,
            int nbOutputs) const noexcept override;
        int enqueue(
            const nvinfer1::PluginTensorDesc* inputDesc,
            const nvinfer1::PluginTensorDesc* outputDesc,
            const void* const* inputs,
            void* const* outputs,
            void* workspace,
            cudaStream_t stream) noexcept override;
        nvinfer1::DataType getOutputDataType(
            int index,
            const nvinfer1::DataType* inputTypes,
            int nbInputs) const noexcept override;
        const char* getPluginType() const noexcept override;
        const char* getPluginVersion() const noexcept override;
        int getNbOutputs() const noexcept override;
        int initialize() noexcept override;
        void terminate() noexcept override;
        size_t getSerializationSize() const noexcept override;
        void serialize(void* buffer) const noexcept override;
        void destroy() noexcept override;
        void setPluginNamespace(const char* pluginNamespace) noexcept override;
        const char* getPluginNamespace() const noexcept override;
    };

    class PagedKVDecodePluginCreator final : public nvinfer1::IPluginCreator
    {
        std::string m_namespace;
        nvinfer1::PluginFieldCollection m_fields{};
        std::vector<nvinfer1::PluginField> m_attributes;

    public:
        PagedKVDecodePluginCreator();
        const char* getPluginName() const noexcept override;
        const char* getPluginVersion() const noexcept override;
        const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
        nvinfer1::IPluginV2* createPlugin(
            const char* name,
            const nvinfer1::PluginFieldCollection* fields) noexcept override;
        nvinfer1::IPluginV2* deserializePlugin(
            const char* name,
            const void* serialData,
            size_t serialLength) noexcept override;
        void setPluginNamespace(const char* pluginNamespace) noexcept override;
        const char* getPluginNamespace() const noexcept override;
    };

    class PagedKVPrefillWritePlugin final : public nvinfer1::IPluginV2DynamicExt
    {
        int m_pageSize = 0;
        int m_qHeads = 0;
        int m_kvHeads = 0;
        int m_headDim = 0;
        int m_layerIndex = -1;
        std::string m_namespace;

    public:
        PagedKVPrefillWritePlugin(int pageSize, int qHeads, int kvHeads, int headDim, int layerIndex = -1);
        PagedKVPrefillWritePlugin(const void* data, size_t length);
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

    class PagedKVPrefillWritePluginCreator final : public nvinfer1::IPluginCreator
    {
        std::string m_namespace;
        nvinfer1::PluginFieldCollection m_fields{};
        std::vector<nvinfer1::PluginField> m_attributes;
    public:
        PagedKVPrefillWritePluginCreator();
        const char* getPluginName() const noexcept override;
        const char* getPluginVersion() const noexcept override;
        const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
        nvinfer1::IPluginV2* createPlugin(const char*, const nvinfer1::PluginFieldCollection*) noexcept override;
        nvinfer1::IPluginV2* deserializePlugin(const char*, const void*, size_t) noexcept override;
        void setPluginNamespace(const char*) noexcept override;
        const char* getPluginNamespace() const noexcept override;
    };

    bool EnsurePagedKVDecodePluginRegistered();
}
