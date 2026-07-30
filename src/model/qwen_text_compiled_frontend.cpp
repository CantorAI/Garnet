#include "qwen_text_compiled_frontend.h"

#include "garnet_tensor.h"
#include "tensor_helper.h"
#include "qwen_tokenizer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace Garnet
{
    namespace
    {
        X::Value MakeGpuTensor(
            X::TensorDataType dataType,
            const std::vector<int>& dimensions,
            const void* hostData,
            size_t bytes)
        {
            X::Tensor tensor(X::g_pXHost->CreateTensor());
            X::Port::vector<int> shape(static_cast<int>(dimensions.size()));
            for (const int dimension : dimensions) shape.push_back(dimension);
            tensor->SetDataType(dataType);
            tensor->SetShape(shape);
            void* deviceMemory = nullptr;
            if (cudaMalloc(&deviceMemory, bytes) != cudaSuccess) return X::Value();
            if (bytes > 0 && hostData &&
                cudaMemcpy(
                    deviceMemory, hostData, bytes, cudaMemcpyHostToDevice) !=
                    cudaSuccess) {
                cudaFree(deviceMemory);
                return X::Value();
            }
            if (TensorHelper::AttachGPUMemory(tensor, deviceMemory) !=
                TensorOpStatus::Success) {
                cudaFree(deviceMemory);
                return X::Value();
            }
            return X::Value(tensor);
        }

        X::Value MakeZeroGpuTensor(
            X::TensorDataType dataType,
            const std::vector<int>& dimensions,
            size_t elementBytes)
        {
            size_t elementCount = 1;
            for (const int dimension : dimensions) {
                if (dimension <= 0 ||
                    elementCount > std::numeric_limits<size_t>::max() /
                        static_cast<size_t>(dimension)) {
                    return X::Value();
                }
                elementCount *= static_cast<size_t>(dimension);
            }
            if (elementCount > std::numeric_limits<size_t>::max() / elementBytes) {
                return X::Value();
            }
            X::Tensor tensor(X::g_pXHost->CreateTensor());
            X::Port::vector<int> shape(static_cast<int>(dimensions.size()));
            for (const int dimension : dimensions) shape.push_back(dimension);
            tensor->SetDataType(dataType);
            tensor->SetShape(shape);
            void* deviceMemory = nullptr;
            const size_t bytes = elementCount * elementBytes;
            if (cudaMalloc(&deviceMemory, bytes) != cudaSuccess) return X::Value();
            if (cudaMemset(deviceMemory, 0, bytes) != cudaSuccess ||
                TensorHelper::AttachGPUMemory(tensor, deviceMemory) !=
                    TensorOpStatus::Success) {
                cudaFree(deviceMemory);
                return X::Value();
            }
            return X::Value(tensor);
        }
    }

    QwenTextCompiledInputs BuildQwenTextCompiledInputs(
        const std::string& modelDirectory,
        const std::string& prompt,
        bool enableThinking,
        const std::vector<std::vector<int>>& profileShapes)
    {
        QwenTextCompiledInputs result;
        try {
            if (profileShapes.size() != 7 ||
                profileShapes[0].size() != 2 ||
                profileShapes[1].size() != 3 ||
                profileShapes[2].size() != 2 ||
                profileShapes[3].size() != 5 ||
                profileShapes[4] != profileShapes[3] ||
                profileShapes[5].size() != 1 ||
                profileShapes[6] != std::vector<int>{1}) {
                throw std::invalid_argument(
                    "Qwen text frontend requires the seven-input paged-prefill profile");
            }
            const int maxTokens = profileShapes[0][1];
            if (profileShapes[0][0] != 1 ||
                profileShapes[1] != std::vector<int>({1, 1, maxTokens}) ||
                profileShapes[2] != std::vector<int>({1, maxTokens})) {
                throw std::invalid_argument(
                    "Qwen text frontend currently requires matching batch-1 token profiles");
            }
            const int pageSize = profileShapes[3][2];
            const int physicalPages = profileShapes[3][1];
            const int logicalPages = profileShapes[5][0];
            if (pageSize <= 0 || physicalPages <= 0 || logicalPages <= 0 ||
                logicalPages > physicalPages ||
                maxTokens > logicalPages * pageSize) {
                throw std::invalid_argument(
                    "Qwen text paged-KV profile cannot hold the compiled prompt length");
            }

            std::string tokenizerError;
            auto tokenizer =
                Tokenization::GetCachedQwenTokenizer(modelDirectory, &tokenizerError);
            if (!tokenizer) throw std::runtime_error(tokenizerError);
            std::string promptText =
                "<|im_start|>user\n" + prompt +
                "<|im_end|>\n<|im_start|>assistant\n";
            if (!enableThinking) promptText += "<think>\n\n</think>\n\n";
            const std::vector<int64_t> promptIds =
                tokenizer->Encode(promptText, false);
            if (promptIds.empty()) {
                throw std::invalid_argument("Qwen text prompt tokenized to no tokens");
            }
            if (promptIds.size() > static_cast<size_t>(maxTokens)) {
                throw std::invalid_argument(
                    "Qwen text prompt exceeds the compiled token profile");
            }
            result.promptTokenCount = static_cast<int>(promptIds.size());

            std::vector<int64_t> inputIds(static_cast<size_t>(maxTokens), 0);
            std::vector<int64_t> positions(static_cast<size_t>(maxTokens), 0);
            std::vector<int64_t> attentionMask(
                static_cast<size_t>(maxTokens), 0);
            std::copy(promptIds.begin(), promptIds.end(), inputIds.begin());
            for (int index = 0; index < result.promptTokenCount; ++index) {
                positions[index] = index;
                attentionMask[index] = 1;
            }
            std::vector<int> pageTable(static_cast<size_t>(logicalPages));
            for (int index = 0; index < logicalPages; ++index) {
                pageTable[index] = index;
            }
            const int startPosition = 0;

            X::V<X::XList> inputs;
            inputs->AddItem(MakeGpuTensor(
                X::TensorDataType::LONGLONG, profileShapes[0],
                inputIds.data(), inputIds.size() * sizeof(int64_t)));
            inputs->AddItem(MakeGpuTensor(
                X::TensorDataType::LONGLONG, profileShapes[1],
                positions.data(), positions.size() * sizeof(int64_t)));
            inputs->AddItem(MakeGpuTensor(
                X::TensorDataType::LONGLONG, profileShapes[2],
                attentionMask.data(), attentionMask.size() * sizeof(int64_t)));
            inputs->AddItem(MakeZeroGpuTensor(
                X::TensorDataType::BFLOAT16, profileShapes[3],
                sizeof(unsigned short)));
            inputs->AddItem(MakeZeroGpuTensor(
                X::TensorDataType::BFLOAT16, profileShapes[4],
                sizeof(unsigned short)));
            inputs->AddItem(MakeGpuTensor(
                X::TensorDataType::INT, profileShapes[5],
                pageTable.data(), pageTable.size() * sizeof(int)));
            inputs->AddItem(MakeGpuTensor(
                X::TensorDataType::INT, profileShapes[6],
                &startPosition, sizeof(startPosition)));
            for (long long index = 0; index < inputs->Size(); ++index) {
                if (!inputs->Get(index).IsTensor()) {
                    throw std::runtime_error(
                        "failed to construct Qwen text GPU input_" +
                        std::to_string(index));
                }
            }
            result.inputs = X::Value(inputs);
        }
        catch (const std::exception& exception) {
            result.inputs = X::Value();
            result.error = exception.what();
        }
        return result;
    }
}
