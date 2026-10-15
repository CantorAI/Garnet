// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "paged_kv_plugin.h"

#include "cuda_lib.h"

#include <NvInferPlugin.h>

#include <cstring>
#include <mutex>

using namespace nvinfer1;

namespace Garnet
{
    namespace
    {
        constexpr const char* kPluginName = "GarnetPagedKVDecodeBF16";
        constexpr const char* kPluginVersion = "2";
        constexpr const char* kPrefillPluginName = "GarnetPagedKVPrefillWriteBF16";

        template <typename T>
        void Write(char*& destination, const T& value)
        {
            std::memcpy(destination, &value, sizeof(T));
            destination += sizeof(T);
        }

        template <typename T>
        void Read(const char*& source, T& value)
        {
            std::memcpy(&value, source, sizeof(T));
            source += sizeof(T);
        }
    }

    PagedKVDecodePlugin::PagedKVDecodePlugin(
        int pageSize, int qHeads, int kvHeads, int headDim, int layerIndex,
        bool useActiveMask)
        : m_pageSize(pageSize), m_qHeads(qHeads), m_kvHeads(kvHeads),
          m_headDim(headDim), m_layerIndex(layerIndex),
          m_useActiveMask(useActiveMask)
    {
    }

    PagedKVDecodePlugin::PagedKVDecodePlugin(const void* data, size_t length)
    {
        if (length != sizeof(int) * 5 && length != sizeof(int) * 6) return;
        const char* source = static_cast<const char*>(data);
        Read(source, m_pageSize);
        Read(source, m_qHeads);
        Read(source, m_kvHeads);
        Read(source, m_headDim);
        Read(source, m_layerIndex);
        if (length == sizeof(int) * 6) {
            int useActiveMask = 0;
            Read(source, useActiveMask);
            m_useActiveMask = useActiveMask != 0;
        }
    }

    IPluginV2DynamicExt* PagedKVDecodePlugin::clone() const noexcept
    {
        auto* plugin = new PagedKVDecodePlugin(
            m_pageSize, m_qHeads, m_kvHeads, m_headDim, m_layerIndex,
            m_useActiveMask);
        plugin->setPluginNamespace(m_namespace.c_str());
        return plugin;
    }

    DimsExprs PagedKVDecodePlugin::getOutputDimensions(
        int outputIndex,
        const DimsExprs* inputs,
        int nbInputs,
        IExprBuilder& expressionBuilder) noexcept
    {
        DimsExprs output = inputs[0];
        const int expectedInputs = m_useActiveMask ? 7 : 6;
        if (outputIndex != 0 || nbInputs != expectedInputs || output.nbDims <= 0) {
            return output;
        }
        output.d[output.nbDims - 1] = expressionBuilder.constant(m_qHeads * m_headDim);
        return output;
    }

    bool PagedKVDecodePlugin::supportsFormatCombination(
        int position,
        const PluginTensorDesc* inOut,
        int nbInputs,
        int nbOutputs) noexcept
    {
        const int expectedInputs = m_useActiveMask ? 7 : 6;
        if (nbInputs != expectedInputs || nbOutputs != 1 ||
            position < 0 || position >= expectedInputs + 1) return false;
        const bool linear = inOut[position].format == TensorFormat::kLINEAR;
        return linear && ((position <= 2 || position == expectedInputs)
            ? inOut[position].type == DataType::kBF16
            : inOut[position].type == DataType::kINT32);
    }

    void PagedKVDecodePlugin::configurePlugin(
        const DynamicPluginTensorDesc*, int, const DynamicPluginTensorDesc*, int) noexcept
    {
    }

    size_t PagedKVDecodePlugin::getWorkspaceSize(
        const PluginTensorDesc* inputs, int inputCount, const PluginTensorDesc*, int) const noexcept
    {
        if (!inputs || inputCount <= 3) return 0;
        int pageTableElements = 1;
        const Dims& pageTableDimensions = inputs[3].dims;
        for (int index = 0; index < pageTableDimensions.nbDims; ++index) {
            if (pageTableDimensions.d[index] <= 0) return 0;
            pageTableElements *= pageTableDimensions.d[index];
        }
        const Dims& qkvDimensions = inputs[0].dims;
        const int batchSize = qkvDimensions.nbDims > 1 ? qkvDimensions.d[0] : 1;
        if (batchSize <= 0 || pageTableElements % batchSize != 0) return 0;
        const int logicalPages = pageTableElements / batchSize;
        const int maxSequenceLength = logicalPages * m_pageSize;
        constexpr int positionsPerSplit = 128;
        const int splitCount = (maxSequenceLength + positionsPerSplit - 1) / positionsPerSplit;
        const size_t stateCount = static_cast<size_t>(batchSize) * m_qHeads * splitCount;
        const size_t partialBytes = stateCount * m_headDim * sizeof(float);
        if (m_headDim == 128) {
            const size_t statsBytes = stateCount * 2 * sizeof(float);
            return statsBytes + partialBytes;
        }
        const size_t scoreBytes = static_cast<size_t>(batchSize) * m_qHeads *
            maxSequenceLength * sizeof(float);
        return scoreBytes + partialBytes;
    }

    int PagedKVDecodePlugin::enqueue(
        const PluginTensorDesc* inputDesc,
        const PluginTensorDesc*,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept
    {
        const auto layerPointer = [&](const void* pointer, int inputIndex) -> bfloat16* {
            auto* base = static_cast<bfloat16*>(const_cast<void*>(pointer));
            if (m_layerIndex < 0) return base;
            const Dims& dimensions = inputDesc[inputIndex].dims;
            if (dimensions.nbDims < 2 || m_layerIndex >= dimensions.d[0]) return nullptr;
            size_t stride = 1;
            for (int index = 1; index < dimensions.nbDims; ++index) {
                if (dimensions.d[index] <= 0) return nullptr;
                stride *= static_cast<size_t>(dimensions.d[index]);
            }
            return base + static_cast<size_t>(m_layerIndex) * stride;
        };
        bfloat16* keyPages = layerPointer(inputs[1], 1);
        bfloat16* valuePages = layerPointer(inputs[2], 2);
        if (!keyPages || !valuePages) return 1;
        const Dims& qkvDimensions = inputDesc[0].dims;
        const int batchSize = qkvDimensions.nbDims > 1 ? qkvDimensions.d[0] : 1;
        if (batchSize <= 0) return 1;
        int maxSequenceLength = m_pageSize;
        const Dims& pageTableDimensions = inputDesc[3].dims;
        if (pageTableDimensions.nbDims > 0) {
            int pageTableElements = 1;
            for (int index = 0; index < pageTableDimensions.nbDims; ++index) {
                if (pageTableDimensions.d[index] <= 0) return 1;
                pageTableElements *= pageTableDimensions.d[index];
            }
            if (pageTableElements % batchSize != 0) return 1;
            const int logicalPages = pageTableElements / batchSize;
            maxSequenceLength = logicalPages * m_pageSize;
        }
        const char* flash = std::getenv("GARNET_PAGED_KV_FLASH");
        const bool useFlash = m_headDim == 128 &&
            !(flash && flash[0] == '0' && flash[1] == '\0');
        const char* splitK = std::getenv("GARNET_PAGED_KV_SPLIT_K");
        const bool useSplitK = !(splitK && splitK[0] == '0' && splitK[1] == '\0');
        const char* splitValue = std::getenv("GARNET_PAGED_KV_SPLIT_VALUE");
        const bool useSplitValue = !(splitValue && splitValue[0] == '0' && splitValue[1] == '\0');
        constexpr int positionsPerSplit = 128;
        const int splitCount =
            (maxSequenceLength + positionsPerSplit - 1) / positionsPerSplit;
        float* scoreWorkspace = static_cast<float*>(workspace);
        float* valuePartialWorkspace = scoreWorkspace +
            (m_headDim == 128
                ? static_cast<size_t>(batchSize) * m_qHeads * splitCount * 2
                : static_cast<size_t>(batchSize) * m_qHeads * maxSequenceLength);
        cudaError_t status = cudaSuccess;
        if (useFlash && m_useActiveMask) {
            status = runTextPagedKVDecodeFlashMaskedBF16DeviceMetadata(
                static_cast<const bfloat16*>(inputs[0]), keyPages, valuePages,
                static_cast<const int*>(inputs[3]), static_cast<const int*>(inputs[4]),
                static_cast<const int*>(inputs[5]), static_cast<const int*>(inputs[6]),
                static_cast<bfloat16*>(outputs[0]), scoreWorkspace,
                valuePartialWorkspace, batchSize, maxSequenceLength,
                m_pageSize, m_qHeads, m_kvHeads, m_headDim, stream);
        }
        else if (useFlash) {
            status = runTextPagedKVDecodeFlashBF16DeviceMetadata(
                static_cast<const bfloat16*>(inputs[0]), keyPages, valuePages,
                static_cast<const int*>(inputs[3]), static_cast<const int*>(inputs[4]),
                static_cast<const int*>(inputs[5]), static_cast<bfloat16*>(outputs[0]),
                scoreWorkspace, valuePartialWorkspace, batchSize, maxSequenceLength,
                m_pageSize, m_qHeads, m_kvHeads, m_headDim, stream);
        }
        else if (batchSize != 1) {
            return 1;
        }
        else if (m_headDim == 128) {
            // The debug fallback is workspace-free. This keeps changing
            // GARNET_PAGED_KV_FLASH after engine creation memory-safe while
            // the normal head-128 path uses the compact flash workspace.
            status = runTextPagedKVDecodeBF16DeviceMetadata(
                static_cast<const bfloat16*>(inputs[0]), keyPages, valuePages,
                static_cast<const int*>(inputs[3]), static_cast<const int*>(inputs[4]),
                static_cast<const int*>(inputs[5]), static_cast<bfloat16*>(outputs[0]),
                maxSequenceLength, m_pageSize, m_qHeads, m_kvHeads, m_headDim, stream);
        }
        else if (useSplitK) {
            status = runTextPagedKVDecodeSplitKBF16DeviceMetadata(
                static_cast<const bfloat16*>(inputs[0]), keyPages, valuePages,
                static_cast<const int*>(inputs[3]), static_cast<const int*>(inputs[4]),
                static_cast<const int*>(inputs[5]), static_cast<bfloat16*>(outputs[0]),
                scoreWorkspace, valuePartialWorkspace, maxSequenceLength, m_pageSize,
                m_qHeads, m_kvHeads, m_headDim, useSplitValue ? 1 : 0, stream);
        }
        else {
            status = runTextPagedKVDecodeBF16DeviceMetadata(
                static_cast<const bfloat16*>(inputs[0]), keyPages, valuePages,
                static_cast<const int*>(inputs[3]), static_cast<const int*>(inputs[4]),
                static_cast<const int*>(inputs[5]), static_cast<bfloat16*>(outputs[0]),
                maxSequenceLength, m_pageSize, m_qHeads, m_kvHeads, m_headDim, stream);
        }
        return status == cudaSuccess ? 0 : 1;
    }

    DataType PagedKVDecodePlugin::getOutputDataType(
        int, const DataType*, int) const noexcept
    {
        return DataType::kBF16;
    }

    const char* PagedKVDecodePlugin::getPluginType() const noexcept { return kPluginName; }
    const char* PagedKVDecodePlugin::getPluginVersion() const noexcept { return kPluginVersion; }
    int PagedKVDecodePlugin::getNbOutputs() const noexcept { return 1; }
    int PagedKVDecodePlugin::initialize() noexcept { return 0; }
    void PagedKVDecodePlugin::terminate() noexcept {}
    size_t PagedKVDecodePlugin::getSerializationSize() const noexcept
    {
        return sizeof(int) * (m_useActiveMask ? 6 : 5);
    }

    void PagedKVDecodePlugin::serialize(void* buffer) const noexcept
    {
        char* destination = static_cast<char*>(buffer);
        Write(destination, m_pageSize);
        Write(destination, m_qHeads);
        Write(destination, m_kvHeads);
        Write(destination, m_headDim);
        Write(destination, m_layerIndex);
        if (m_useActiveMask) Write(destination, 1);
    }

    void PagedKVDecodePlugin::destroy() noexcept { delete this; }
    void PagedKVDecodePlugin::setPluginNamespace(const char* value) noexcept
    {
        m_namespace = value ? value : "";
    }
    const char* PagedKVDecodePlugin::getPluginNamespace() const noexcept { return m_namespace.c_str(); }

    PagedKVDecodePluginCreator::PagedKVDecodePluginCreator()
    {
        m_attributes.emplace_back("page_size", nullptr, PluginFieldType::kINT32, 1);
        m_attributes.emplace_back("q_heads", nullptr, PluginFieldType::kINT32, 1);
        m_attributes.emplace_back("kv_heads", nullptr, PluginFieldType::kINT32, 1);
        m_attributes.emplace_back("head_dim", nullptr, PluginFieldType::kINT32, 1);
        m_attributes.emplace_back("layer_idx", nullptr, PluginFieldType::kINT32, 1);
        m_fields.nbFields = static_cast<int>(m_attributes.size());
        m_fields.fields = m_attributes.data();
    }

    const char* PagedKVDecodePluginCreator::getPluginName() const noexcept { return kPluginName; }
    const char* PagedKVDecodePluginCreator::getPluginVersion() const noexcept { return kPluginVersion; }
    const PluginFieldCollection* PagedKVDecodePluginCreator::getFieldNames() noexcept { return &m_fields; }

    IPluginV2* PagedKVDecodePluginCreator::createPlugin(
        const char*, const PluginFieldCollection* fields) noexcept
    {
        int values[5]{0, 0, 0, 0, -1};
        if (!fields) return nullptr;
        for (int index = 0; index < fields->nbFields; ++index) {
            const PluginField& field = fields->fields[index];
            if (!field.data) continue;
            const int value = *static_cast<const int*>(field.data);
            if (std::strcmp(field.name, "page_size") == 0) values[0] = value;
            else if (std::strcmp(field.name, "q_heads") == 0) values[1] = value;
            else if (std::strcmp(field.name, "kv_heads") == 0) values[2] = value;
            else if (std::strcmp(field.name, "head_dim") == 0) values[3] = value;
            else if (std::strcmp(field.name, "layer_idx") == 0) values[4] = value;
        }
        auto* plugin = new PagedKVDecodePlugin(values[0], values[1], values[2], values[3], values[4]);
        plugin->setPluginNamespace(m_namespace.c_str());
        return plugin;
    }

    IPluginV2* PagedKVDecodePluginCreator::deserializePlugin(
        const char*, const void* data, size_t length) noexcept
    {
        auto* plugin = new PagedKVDecodePlugin(data, length);
        plugin->setPluginNamespace(m_namespace.c_str());
        return plugin;
    }

    void PagedKVDecodePluginCreator::setPluginNamespace(const char* value) noexcept
    {
        m_namespace = value ? value : "";
    }
    const char* PagedKVDecodePluginCreator::getPluginNamespace() const noexcept
    {
        return m_namespace.c_str();
    }

    PagedKVPrefillWritePlugin::PagedKVPrefillWritePlugin(
        int pageSize, int qHeads, int kvHeads, int headDim, int layerIndex)
        : m_pageSize(pageSize), m_qHeads(qHeads), m_kvHeads(kvHeads),
          m_headDim(headDim), m_layerIndex(layerIndex)
    {
    }

    PagedKVPrefillWritePlugin::PagedKVPrefillWritePlugin(const void* data, size_t length)
    {
        if (length != getSerializationSize()) return;
        const char* source = static_cast<const char*>(data);
        Read(source, m_pageSize);
        Read(source, m_qHeads);
        Read(source, m_kvHeads);
        Read(source, m_headDim);
        Read(source, m_layerIndex);
    }

    IPluginV2DynamicExt* PagedKVPrefillWritePlugin::clone() const noexcept
    {
        auto* plugin = new PagedKVPrefillWritePlugin(
            m_pageSize, m_qHeads, m_kvHeads, m_headDim, m_layerIndex);
        plugin->setPluginNamespace(m_namespace.c_str());
        return plugin;
    }

    DimsExprs PagedKVPrefillWritePlugin::getOutputDimensions(
        int, const DimsExprs* inputs, int, IExprBuilder&) noexcept
    {
        return inputs[0];
    }

    bool PagedKVPrefillWritePlugin::supportsFormatCombination(
        int position, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
    {
        if (nbInputs != 5 || nbOutputs != 1 || position < 0 || position >= 6) return false;
        const bool linear = inOut[position].format == TensorFormat::kLINEAR;
        return linear && ((position <= 2 || position == 5)
            ? inOut[position].type == DataType::kBF16
            : inOut[position].type == DataType::kINT32);
    }

    void PagedKVPrefillWritePlugin::configurePlugin(
        const DynamicPluginTensorDesc*, int, const DynamicPluginTensorDesc*, int) noexcept
    {
    }

    size_t PagedKVPrefillWritePlugin::getWorkspaceSize(
        const PluginTensorDesc*, int, const PluginTensorDesc*, int) const noexcept
    {
        return 0;
    }

    int PagedKVPrefillWritePlugin::enqueue(
        const PluginTensorDesc* inputDesc,
        const PluginTensorDesc*,
        const void* const* inputs,
        void* const* outputs,
        void*,
        cudaStream_t stream) noexcept
    {
        const Dims& dimensions = inputDesc[0].dims;
        if (dimensions.nbDims < 2) return 1;
        int tokenCount = 1;
        for (int index = 0; index + 1 < dimensions.nbDims; ++index) {
            if (dimensions.d[index] <= 0) return 1;
            tokenCount *= dimensions.d[index];
        }
        const auto layerPointer = [&](const void* pointer, int inputIndex) -> bfloat16* {
            auto* base = static_cast<bfloat16*>(const_cast<void*>(pointer));
            if (m_layerIndex < 0) return base;
            const Dims& cacheDimensions = inputDesc[inputIndex].dims;
            if (cacheDimensions.nbDims < 2 || m_layerIndex >= cacheDimensions.d[0]) return nullptr;
            size_t stride = 1;
            for (int index = 1; index < cacheDimensions.nbDims; ++index) {
                if (cacheDimensions.d[index] <= 0) return nullptr;
                stride *= static_cast<size_t>(cacheDimensions.d[index]);
            }
            return base + static_cast<size_t>(m_layerIndex) * stride;
        };
        bfloat16* keyPages = layerPointer(inputs[1], 1);
        bfloat16* valuePages = layerPointer(inputs[2], 2);
        if (!keyPages || !valuePages) return 1;
        cudaError_t status = runTextPagedKVWriteBF16DeviceStart(
            static_cast<const bfloat16*>(inputs[0]),
            keyPages,
            valuePages,
            static_cast<const int*>(inputs[3]),
            static_cast<const int*>(inputs[4]),
            tokenCount,
            m_pageSize,
            m_qHeads,
            m_kvHeads,
            m_headDim,
            stream);
        if (status != cudaSuccess) return 1;
        size_t elementCount = 1;
        for (int index = 0; index < dimensions.nbDims; ++index) {
            if (dimensions.d[index] <= 0) return 1;
            elementCount *= static_cast<size_t>(dimensions.d[index]);
        }
        status = cudaMemcpyAsync(
            outputs[0], inputs[0], elementCount * sizeof(bfloat16),
            cudaMemcpyDeviceToDevice, stream);
        return status == cudaSuccess ? 0 : 1;
    }

    DataType PagedKVPrefillWritePlugin::getOutputDataType(
        int, const DataType*, int) const noexcept
    {
        return DataType::kBF16;
    }
    const char* PagedKVPrefillWritePlugin::getPluginType() const noexcept { return kPrefillPluginName; }
    const char* PagedKVPrefillWritePlugin::getPluginVersion() const noexcept { return kPluginVersion; }
    int PagedKVPrefillWritePlugin::getNbOutputs() const noexcept { return 1; }
    int PagedKVPrefillWritePlugin::initialize() noexcept { return 0; }
    void PagedKVPrefillWritePlugin::terminate() noexcept {}
    size_t PagedKVPrefillWritePlugin::getSerializationSize() const noexcept { return sizeof(int) * 5; }
    void PagedKVPrefillWritePlugin::serialize(void* buffer) const noexcept
    {
        char* destination = static_cast<char*>(buffer);
        Write(destination, m_pageSize);
        Write(destination, m_qHeads);
        Write(destination, m_kvHeads);
        Write(destination, m_headDim);
        Write(destination, m_layerIndex);
    }
    void PagedKVPrefillWritePlugin::destroy() noexcept { delete this; }
    void PagedKVPrefillWritePlugin::setPluginNamespace(const char* value) noexcept
    {
        m_namespace = value ? value : "";
    }
    const char* PagedKVPrefillWritePlugin::getPluginNamespace() const noexcept
    {
        return m_namespace.c_str();
    }

    PagedKVPrefillWritePluginCreator::PagedKVPrefillWritePluginCreator()
    {
        m_attributes.emplace_back("page_size", nullptr, PluginFieldType::kINT32, 1);
        m_attributes.emplace_back("q_heads", nullptr, PluginFieldType::kINT32, 1);
        m_attributes.emplace_back("kv_heads", nullptr, PluginFieldType::kINT32, 1);
        m_attributes.emplace_back("head_dim", nullptr, PluginFieldType::kINT32, 1);
        m_attributes.emplace_back("layer_idx", nullptr, PluginFieldType::kINT32, 1);
        m_fields.nbFields = static_cast<int>(m_attributes.size());
        m_fields.fields = m_attributes.data();
    }
    const char* PagedKVPrefillWritePluginCreator::getPluginName() const noexcept { return kPrefillPluginName; }
    const char* PagedKVPrefillWritePluginCreator::getPluginVersion() const noexcept { return kPluginVersion; }
    const PluginFieldCollection* PagedKVPrefillWritePluginCreator::getFieldNames() noexcept { return &m_fields; }
    IPluginV2* PagedKVPrefillWritePluginCreator::createPlugin(
        const char*, const PluginFieldCollection* fields) noexcept
    {
        int values[5]{0, 0, 0, 0, -1};
        if (!fields) return nullptr;
        for (int index = 0; index < fields->nbFields; ++index) {
            const PluginField& field = fields->fields[index];
            if (!field.data) continue;
            const int value = *static_cast<const int*>(field.data);
            if (std::strcmp(field.name, "page_size") == 0) values[0] = value;
            else if (std::strcmp(field.name, "q_heads") == 0) values[1] = value;
            else if (std::strcmp(field.name, "kv_heads") == 0) values[2] = value;
            else if (std::strcmp(field.name, "head_dim") == 0) values[3] = value;
            else if (std::strcmp(field.name, "layer_idx") == 0) values[4] = value;
        }
        auto* plugin = new PagedKVPrefillWritePlugin(
            values[0], values[1], values[2], values[3], values[4]);
        plugin->setPluginNamespace(m_namespace.c_str());
        return plugin;
    }
    IPluginV2* PagedKVPrefillWritePluginCreator::deserializePlugin(
        const char*, const void* data, size_t length) noexcept
    {
        auto* plugin = new PagedKVPrefillWritePlugin(data, length);
        plugin->setPluginNamespace(m_namespace.c_str());
        return plugin;
    }
    void PagedKVPrefillWritePluginCreator::setPluginNamespace(const char* value) noexcept
    {
        m_namespace = value ? value : "";
    }
    const char* PagedKVPrefillWritePluginCreator::getPluginNamespace() const noexcept
    {
        return m_namespace.c_str();
    }

    bool EnsurePagedKVDecodePluginRegistered()
    {
        static PagedKVDecodePluginCreator creator;
        static PagedKVPrefillWritePluginCreator prefillCreator;
        static std::once_flag once;
        static bool registered = false;
        std::call_once(once, [&] {
            auto* registry = getPluginRegistry();
            registered = registry && registry->registerCreator(creator, "") &&
                registry->registerCreator(prefillCreator, "");
        });
        return registered;
    }
}
