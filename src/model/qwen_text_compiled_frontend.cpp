// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "qwen_text_compiled_frontend.h"

#include "garnet_tensor.h"
#include "tensor_helper.h"
#include "qwen_tokenizer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace Garnet
{
    namespace
    {
        X::Value MakeGpuTensor(X3PackageHost* host, X3TensorDType type,
            const std::vector<int>& dimensions, const void* data, size_t bytes)
        {
            std::vector<int64_t> shape(dimensions.begin(), dimensions.end());
            uint64_t expected = TensorHelper::ItemSize(type);
            for (auto dimension : shape) {
                if (dimension < 0 || (dimension && expected > UINT64_MAX / dimension))
                    throw std::invalid_argument("invalid frontend tensor shape");
                expected *= dimension;
            }
            if (expected != bytes) throw std::invalid_argument("frontend tensor byte size mismatch");
            return TensorHelper::CreateGPU(host, type, shape, data);
        }

        X::Value MakeCpuTensor(X3PackageHost* host, X3TensorDType type,
            const std::vector<int>& dimensions, const void* data, size_t bytes)
        {
            const std::vector<int64_t> shape(dimensions.begin(), dimensions.end());
            auto tensor = X::Tensor::Create(host, type, shape, data, data ? bytes : 0);
            if (tensor.Info().byte_size != bytes)
                throw std::invalid_argument("frontend CPU tensor byte size mismatch");
            return tensor;
        }

        X::Value MakeZeroGpuTensor(X3PackageHost* host, X3TensorDType type,
            const std::vector<int>& dimensions, size_t elementBytes)
        {
            if (elementBytes != TensorHelper::ItemSize(type))
                throw std::invalid_argument("frontend tensor element size mismatch");
            const std::vector<int64_t> shape(dimensions.begin(), dimensions.end());
            return TensorHelper::CreateGPU(host, type, shape);
        }

        X::Value ReuseOrMakeZeroGpuTensor(X3PackageHost* host,
            X3TensorDType dataType,
            const std::vector<int>& dimensions,
            size_t elementBytes,
            X::Value reusable)
        {
            if ((X::Tensor::IsTensor(reusable))) {
                X::Tensor tensor(reusable);
                const auto info = tensor.Info();
                int device = -1;
                bool matching =
                    cudaGetDevice(&device) == cudaSuccess && info.device_id == device &&
                    !info.readonly && reusable.runtime() == host->runtime &&
                    tensor.Info().dtype == dataType &&
                    tensor.Info().rank == static_cast<int>(dimensions.size());
                for (int index = 0;
                     matching && index < tensor.Info().rank;
                     ++index) {
                    matching =
                        tensor.Info().shape[index] == dimensions[index];
                }
                void* deviceMemory =
                    nullptr;
                uint64_t span = elementBytes;
                for (size_t i = dimensions.size(); matching && i-- > 0;) {
                    matching = info.strides[i] == static_cast<int64_t>(span) &&
                        dimensions[i] >= 0 && (!dimensions[i] || span <= UINT64_MAX / dimensions[i]);
                    if (matching) span *= dimensions[i];
                }
                if (matching && span == info.byte_size) deviceMemory = TensorHelper::GetGPUMemory(tensor);
                if (deviceMemory) {
                    auto use = TensorHelper::AcquireGPU(tensor, X3_TENSOR_WRITE);
                    if (
                    cudaMemsetAsync(
                        deviceMemory,
                        0,
                        static_cast<size_t>(tensor.Info().byte_size),
                        cudaStreamPerThread) == cudaSuccess) {
                    use.Finish();
                    return reusable;
                    }
                }
            }
            return MakeZeroGpuTensor(host, dataType, dimensions, elementBytes);
        }
    }

    QwenTextCompiledInputs BuildQwenTextCompiledInputs(X3PackageHost* host,
        const std::string& modelDirectory,
        const std::string& prompt,
        bool enableThinking,
        const std::vector<std::vector<int>>& profileShapes,
        X::Value reusableKeyCache,
        X::Value reusableValueCache,
        bool cpuTensors)
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

            X::Value inputs = X::Value::List(host);
            auto makeTensor = [&](X3TensorDType type,
                                  const std::vector<int>& shape,
                                  const void* data,
                                  size_t bytes) {
                return cpuTensors
                    ? MakeCpuTensor(host, type, shape, data, bytes)
                    : MakeGpuTensor(host, type, shape, data, bytes);
            };
            inputs.Append(makeTensor(
                X3_TENSOR_INT64, profileShapes[0],
                inputIds.data(), inputIds.size() * sizeof(int64_t)));
            inputs.Append(makeTensor(
                X3_TENSOR_INT64, profileShapes[1],
                positions.data(), positions.size() * sizeof(int64_t)));
            inputs.Append(makeTensor(
                X3_TENSOR_INT64, profileShapes[2],
                attentionMask.data(), attentionMask.size() * sizeof(int64_t)));
            if (cpuTensors) {
                inputs.Append(MakeCpuTensor(host,
                    X3_TENSOR_BFLOAT16, profileShapes[3], nullptr,
                    static_cast<size_t>(
                        profileShapes[3][0]) * profileShapes[3][1] *
                        profileShapes[3][2] * profileShapes[3][3] *
                        profileShapes[3][4] * sizeof(unsigned short)));
                inputs.Append(MakeCpuTensor(host,
                    X3_TENSOR_BFLOAT16, profileShapes[4], nullptr,
                    static_cast<size_t>(
                        profileShapes[4][0]) * profileShapes[4][1] *
                        profileShapes[4][2] * profileShapes[4][3] *
                        profileShapes[4][4] * sizeof(unsigned short)));
            }
            else {
                inputs.Append(ReuseOrMakeZeroGpuTensor(host,
                    X3_TENSOR_BFLOAT16, profileShapes[3],
                    sizeof(unsigned short), reusableKeyCache));
                inputs.Append(ReuseOrMakeZeroGpuTensor(host,
                    X3_TENSOR_BFLOAT16, profileShapes[4],
                    sizeof(unsigned short), reusableValueCache));
            }
            inputs.Append(makeTensor(
                X3_TENSOR_INT32, profileShapes[5],
                pageTable.data(), pageTable.size() * sizeof(int)));
            inputs.Append(makeTensor(
                X3_TENSOR_INT32, profileShapes[6],
                &startPosition, sizeof(startPosition)));
            for (long long index = 0; index < inputs.Size(); ++index) {
                if (!X::Tensor::IsTensor(inputs.Get(index))) {
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
