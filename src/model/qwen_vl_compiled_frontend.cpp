// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "qwen_vl_compiled_frontend.h"

#include "cuda_lib.h"
#include "garnet_tensor.h"
#include "tensor_helper.h"
#include "qwen_tokenizer.h"
#include "qwen_vl_image_preprocessor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
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

        X::Value ConvertPixelsToBF16(X::Value sourceValue)
        {
            if (!X::Tensor::IsTensor(sourceValue)) return {};
            X::Tensor source(sourceValue);
            if (source.Info().dtype != X3_TENSOR_FLOAT32 ||
                TensorHelper::EnsureGPUMemory(source) != TensorOpStatus::Success) return {};
            const auto info = source.Info();
            const uint64_t count = info.byte_size / sizeof(float);
            if (!count || count > INT_MAX) return {};
            std::vector<int64_t> shape(info.shape, info.shape + info.rank);
            auto output = TensorHelper::CreateGPU(source.host(), X3_TENSOR_BFLOAT16, shape);
            auto use = TensorHelper::AcquireGPU({{source, X3_TENSOR_READ}, {output, X3_TENSOR_WRITE}});
            const auto status = runConvertFP32ToBF16Async(
                static_cast<const float*>(TensorHelper::GetGPUMemory(source)),
                static_cast<bfloat16*>(TensorHelper::GetGPUMemory(output)),
                static_cast<int>(count), cudaStreamPerThread);
            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
            use.Finish();
            // The temporary source may own an uploaded buffer. Finish before releasing it.
            const auto sync = cudaStreamSynchronize(cudaStreamPerThread);
            if (sync != cudaSuccess) throw std::runtime_error(cudaGetErrorString(sync));
            return output;
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

    QwenVLCompiledInputs BuildQwenVLCompiledInputs(X3PackageHost* host,
        const std::string& modelDirectory,
        X::Value imageSource,
        const std::string& prompt,
        int minPixels,
        int maxPixels,
        const std::vector<std::vector<int>>& profileShapes,
        X::Value reusableKeyCache,
        X::Value reusableValueCache)
    {
        QwenVLCompiledInputs result;
        try {
            if ((profileShapes.size() != 11 && profileShapes.size() != 15) ||
                profileShapes[0].size() != 2 ||
                profileShapes[1].size() != 2 || profileShapes[9].size() != 3) {
                throw std::invalid_argument("Qwen-VL compiled frontend requires an 11-input root or 15-input paged-prefill profile");
            }
            const int batch = profileShapes[0][0];
            const int maxTokens = profileShapes[0][1];
            if (batch != 1 || profileShapes[9][0] != 3 || profileShapes[9][1] != 1 ||
                profileShapes[9][2] != maxTokens) {
                throw std::invalid_argument("Qwen-VL frontend currently requires batch-1 matching token profiles");
            }

            Image::PreprocessResult image;
            if (imageSource.IsBin()) {
                uint64_t bytes = 0;
                const void* binary = imageSource.BytesData(&bytes);
                if (!binary || bytes == 0) {
                    throw std::invalid_argument("Qwen-VL image binary is empty");
                }
                image = Image::QwenVL::PreprocessJpegBytesToTensor(
                    host, static_cast<const unsigned char*>(binary),
                    static_cast<size_t>(bytes),
                    minPixels,
                    maxPixels);
            }
            else {
                const std::string imagePath = imageSource.ToString();
                if (imagePath.empty()) {
                    throw std::invalid_argument(
                        "Qwen-VL image must be a JPEG binary or file path");
                }
                image = Image::QwenVL::PreprocessJpegFileToTensor(
                    host, imagePath, minPixels, maxPixels);
            }
            X::Tensor gridTensor = TensorHelper::CopyToCPU(X::Tensor(image.imageGridTHW));
            auto gridUse = gridTensor.Acquire();
            const auto* gridData = reinterpret_cast<const long long*>(gridTensor.Info().data);
            const int64_t grid[3] = {gridData[0], gridData[1], gridData[2]};
            constexpr int mergeSize = 2;

            std::string tokenizerError;
            auto tokenizer = Tokenization::GetCachedQwenTokenizer(modelDirectory, &tokenizerError);
            if (!tokenizer) throw std::runtime_error(tokenizerError);
            const auto promptIds = Tokenization::QwenVLPromptBuilder::BuildSingleImagePromptIds(
                *tokenizer, prompt, grid, mergeSize);
            const int64_t imagePadId = tokenizer->TokenId("<|image_pad|>");
            if (imagePadId < 0) throw std::runtime_error("Qwen tokenizer is missing image_pad");
            if (promptIds.size() > static_cast<size_t>(maxTokens)) {
                throw std::invalid_argument("prompt and visual tokens exceed the compiled token profile");
            }

            std::vector<int64_t> inputIds(static_cast<size_t>(maxTokens), 0);
            std::vector<int64_t> mmTypes(static_cast<size_t>(maxTokens), 0);
            std::vector<int64_t> attentionMask(static_cast<size_t>(maxTokens), 0);
            std::vector<int64_t> unpaddedTypes;
            unpaddedTypes.reserve(promptIds.size());
            for (size_t index = 0; index < promptIds.size(); ++index) {
                inputIds[index] = promptIds[index];
                mmTypes[index] = promptIds[index] == imagePadId ? 1 : 0;
                attentionMask[index] = 1;
                unpaddedTypes.push_back(mmTypes[index]);
                result.visualTokenCount += static_cast<int>(mmTypes[index]);
            }
            result.promptTokenCount = static_cast<int>(promptIds.size());
            const int expectedVisualTokens = static_cast<int>(grid[0] * grid[1] * grid[2] / 4);
            if (result.visualTokenCount != expectedVisualTokens) {
                throw std::runtime_error("visual token count does not match image grid");
            }

            const auto mrope = Tokenization::QwenVLPromptBuilder::BuildSingleImageMRoPEMetadata(
                unpaddedTypes, grid, mergeSize);
            std::vector<int64_t> paddedPositions(static_cast<size_t>(3 * maxTokens), 0);
            for (int dimension = 0; dimension < 3; ++dimension) {
                std::copy_n(
                    mrope.positionIds.data() + static_cast<size_t>(dimension * result.promptTokenCount),
                    result.promptTokenCount,
                    paddedPositions.data() + static_cast<size_t>(dimension * maxTokens));
            }

            if (profileShapes[1][0] != grid[0] * grid[1] * grid[2] ||
                profileShapes[1][1] != 1536) {
                throw std::invalid_argument("image patch grid does not match the compiled vision profile");
            }

            X::Value convertedPixels = ConvertPixelsToBF16(image.pixelValues);
            image.pixelValues = X::Value();

            X::Value inputs = X::Value::List(host);
            inputs.Append(MakeGpuTensor(host, X3_TENSOR_INT64, profileShapes[0],
                inputIds.data(), inputIds.size() * sizeof(int64_t)));
            inputs.Append(convertedPixels);
            inputs.Append(MakeGpuTensor(host, X3_TENSOR_INT64, profileShapes[2],
                grid, sizeof(grid)));
            inputs.Append(image.bilinearIndices);
            inputs.Append(image.bilinearWeights);
            inputs.Append(image.visionPositionIds);
            inputs.Append(image.visionCuSeqlens);
            inputs.Append(MakeGpuTensor(host, X3_TENSOR_INT64, profileShapes[7],
                mmTypes.data(), mmTypes.size() * sizeof(int64_t)));
            inputs.Append(MakeGpuTensor(host, X3_TENSOR_INT64, profileShapes[8],
                attentionMask.data(), attentionMask.size() * sizeof(int64_t)));
            inputs.Append(MakeGpuTensor(host, X3_TENSOR_INT64, profileShapes[9],
                paddedPositions.data(), paddedPositions.size() * sizeof(int64_t)));
            inputs.Append(MakeGpuTensor(host, X3_TENSOR_INT64, profileShapes[10],
                &mrope.positionDelta, sizeof(mrope.positionDelta)));
            result.mropePositionDelta = mrope.positionDelta;
            if (profileShapes.size() == 15) {
                if (profileShapes[11] != profileShapes[12] || profileShapes[11].size() != 5 ||
                    profileShapes[13].size() != 1 || profileShapes[14] != std::vector<int>{1}) {
                    throw std::invalid_argument("Qwen-VL paged prefill cache profile is invalid");
                }
                inputs.Append(ReuseOrMakeZeroGpuTensor(host,
                    X3_TENSOR_BFLOAT16, profileShapes[11],
                    sizeof(bfloat16), reusableKeyCache));
                inputs.Append(ReuseOrMakeZeroGpuTensor(host,
                    X3_TENSOR_BFLOAT16, profileShapes[12],
                    sizeof(bfloat16), reusableValueCache));
                std::vector<int> pageTable(static_cast<size_t>(profileShapes[13][0]));
                for (int index = 0; index < profileShapes[13][0]; ++index) pageTable[index] = index;
                inputs.Append(MakeGpuTensor(host,
                    X3_TENSOR_INT32, profileShapes[13], pageTable.data(),
                    pageTable.size() * sizeof(int)));
                const int startPosition = 0;
                inputs.Append(MakeGpuTensor(host,
                    X3_TENSOR_INT32, profileShapes[14], &startPosition, sizeof(startPosition)));
            }
            for (long long index = 0; index < inputs.Size(); ++index) {
                if (!X::Tensor::IsTensor(inputs.Get(index))) {
                    throw std::runtime_error("failed to construct GPU input_" + std::to_string(index));
                }
            }

            result.inputs = X::Value(inputs);
            result.sourceHeight = image.sourceHeight;
            result.sourceWidth = image.sourceWidth;
            result.resizedHeight = image.resizedHeight;
            result.resizedWidth = image.resizedWidth;
        }
        catch (const std::exception& exception) {
            result.inputs = X::Value();
            result.error = exception.what();
        }
        return result;
    }
}
