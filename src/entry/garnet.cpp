// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "garnet.h"
#include "native_values.h"
#include "trt_builder.h"
#include "../image/qwen_vl/qwen_vl_image_preprocessor.h"
#include "../image/cuda/jpeg_decode_nvjpeg.h"
#include "../tokenizer/qwen_tokenizer.h"
#include "../cuda/cuda_lib.h"
#include "../tensor/tensor_helper.h"
#include "../model/model_catalog.h"
#include "../runtime/acceleration_detector.h"
#include "../include/garnet_serving.h"
#include "nlohmann/json.hpp"
#include "xlang3/xlang3.h"
#include <fstream>
#include <numeric>
#include <filesystem>
#include <regex>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <cuda_runtime.h>

namespace
{
    using json = nlohmann::json;
    using Garnet::FindField;
    using Garnet::CallChecked;
    using Garnet::TensorCount;

    class CudaDeviceScope
    {
        int m_previous = -1;
        bool m_changed = false;
    public:
        explicit CudaDeviceScope(int device)
        {
            if (cudaGetDevice(&m_previous) != cudaSuccess) m_previous = -1;
            if (m_previous != device) {
                if (cudaSetDevice(device) != cudaSuccess)
                    throw std::runtime_error("failed to select CUDA device " + std::to_string(device));
                m_changed = true;
            }
        }
        ~CudaDeviceScope()
        {
            if (m_changed && m_previous >= 0) cudaSetDevice(m_previous);
        }
    };

    std::string GarnetJsonError(const std::string& code, const std::string& message)
    {
        return json({
            {"status", "error"},
            {"error_code", code},
            {"error_message", message}
        }).dump();
    }

    int CopyJsonResult(
        const std::string& value,
        char* output,
        int outputCapacity,
        int* requiredCapacity)
    {
        const size_t required = value.size() + 1;
        if (requiredCapacity) {
            *requiredCapacity = required > static_cast<size_t>(INT_MAX)
                ? INT_MAX
                : static_cast<int>(required);
        }
        if (required > static_cast<size_t>(INT_MAX)) return 1;
        if (!output || outputCapacity < static_cast<int>(required)) return 2;
        std::memcpy(output, value.c_str(), required);
        return 0;
    }

    json GarnetServingStatus(X::Value modelValue,
        const std::string& modelRoot,
        const std::string& modelId,
        const std::string& inputCapability,
        const std::string& lastError,
        X3Runtime* runtime = nullptr)
    {
        json status = {
            {"state", modelValue.IsValid() ? "ready" :
                (lastError.empty() ? "stopped" : "error")},
            {"ready", modelValue.IsValid()},
            {"model_root", modelRoot},
            {"model_id", modelId},
            {"input_capability", inputCapability},
            {"error", lastError}
        };
        if (!modelValue.IsValid()) return status;
        X::Value runtimeStatusCallable = modelValue["runtime_status"];
        if (!runtimeStatusCallable.IsObject()) {
            status["state"] = "error";
            status["ready"] = false;
            status["error"] = "Garnet serving model handle is invalid";
            return status;
        }
        X::Value runtimeStatusValue = CallChecked(runtimeStatusCallable);
        if (runtimeStatusValue.IsDict()) {
            X::Value runtimeStatus(runtimeStatusValue);
            status["state"] = FindField(runtimeStatus, "state").ToString();
            status["ready"] = FindField(runtimeStatus, "ready").ToLongLong() != 0;
            if (FindField(runtimeStatus, "error_message").IsValid()) {
                status["error"] = FindField(runtimeStatus, "error_message").ToString();
            }
            for (const char* key : {
                     "backend", "precision", "frontend", "engine_path",
                     "cache_directory"}) {
                if (FindField(runtimeStatus, key).IsValid()) {
                    status[key] = FindField(runtimeStatus, key).ToString();
                }
            }
        }
        return status;
    }

    cudaError_t CreateEntryExecutionStream(cudaStream_t* stream)
    {
        if (!stream) return cudaErrorInvalidValue;
        *stream = cudaStreamPerThread;
        return cudaSuccess;
    }

    cudaError_t DestroyEntryExecutionStream(cudaStream_t)
    {
        return cudaSuccess;
    }
}

namespace Garnet
{
    std::shared_ptr<GarnetAPI::ServingInstance> GarnetAPI::FindServingInstance(
        const std::string& modelId, const std::string& capability) const
    {
        std::lock_guard<std::mutex> guard(m_servingMutex);
        if (!modelId.empty()) {
            const auto it = m_servingInstances.find(modelId);
            if (it != m_servingInstances.end() &&
                (capability.empty() || it->second->inputCapability == capability)) {
                return it->second;
            }
            return {};
        }
        if (!capability.empty()) {
            for (const auto& item : m_servingInstances) {
                if (item.second->inputCapability == capability) return item.second;
            }
            return {};
        }
        const auto current = m_servingInstances.find(m_defaultServingModelId);
        return current == m_servingInstances.end() ? nullptr : current->second;
    }

    namespace {
        std::mutex nativePackagesMutex;
        std::vector<GarnetAPI*> nativePackages;
    }

        void ValidateDenseTensor(const X::Tensor& tensor, bool writing)
        {
            const auto info = tensor.Info();
            if (info.symbolic || info.rank == UINT32_MAX) throw X::Error("native kernel requires a concrete tensor");
            if (writing && info.readonly) throw X::Error("tensor is readonly");
            uint64_t count = 1;
            for (uint32_t i = 0; i < info.rank; ++i) {
                if (info.shape[i] < 0 || info.shape[i] > INT32_MAX ||
                    (info.shape[i] && count > INT32_MAX / static_cast<uint64_t>(info.shape[i])))
                    throw X::Error("native tensor dimensions exceed int32 kernel limits");
                count *= info.shape[i];
            }
            if (count > INT32_MAX - 1024) throw X::Error("native tensor exceeds kernel indexing limits");
            uint64_t stride = TensorHelper::ItemSize(info.dtype);
            if (count) {
                for (uint32_t i = info.rank; i-- > 0;) {
                    if (info.shape[i] > 1 && info.strides[i] != static_cast<int64_t>(stride))
                        throw X::Error("native kernel requires a contiguous tensor");
                    stride *= info.shape[i];
                }
                if (!info.data || info.byte_size < stride) throw X::Error("tensor storage is too small");
                if (reinterpret_cast<uintptr_t>(info.data) % TensorHelper::ItemSize(info.dtype))
                    throw X::Error("native kernel requires aligned tensor storage");
            }
            if (info.device_type != 0) {
                int device = -1;
                if (info.device_type != TensorHelper::CudaDevice || cudaGetDevice(&device) != cudaSuccess || device != info.device_id)
                    throw X::Error("tensor must be on the active CUDA device");
            }
        }

    namespace {
        X::Tensor AdoptGpuOutput(X3PackageHost* host, X3TensorDType dtype,
            const std::vector<int64_t>& shape, void* allocation)
        {
            try {
                int device = 0;
                if (cudaGetDevice(&device) != cudaSuccess) throw X::Error("cannot query CUDA device");
                uint64_t bytes = TensorHelper::ItemSize(dtype);
                std::vector<int64_t> strides(shape.size());
                for (size_t i = shape.size(); i-- > 0;) {
                    if (shape[i] < 0 || bytes > INT64_MAX ||
                        (shape[i] && bytes > UINT64_MAX / shape[i])) throw X::Error("invalid GPU tensor shape");
                    strides[i] = static_cast<int64_t>(bytes);
                    bytes *= shape[i];
                }
                X3TensorInfo info{};
                info.size = sizeof(info); info.dtype = dtype; info.rank = static_cast<uint32_t>(shape.size());
                info.shape = shape.data(); info.strides = strides.data(); info.data = allocation;
                info.byte_size = bytes; info.device_type = TensorHelper::CudaDevice; info.device_id = device;
                return TensorHelper::WrapGPU(host, info, allocation, device);
            } catch (...) { if (allocation) cudaFree(allocation); throw; }
        }
    }

    GarnetAPI::GarnetAPI()
    {
        std::lock_guard<std::mutex> guard(nativePackagesMutex);
        nativePackages.push_back(this);
    }

    GarnetAPI::~GarnetAPI()
    {
        std::lock_guard<std::mutex> guard(nativePackagesMutex);
        nativePackages.erase(std::remove(nativePackages.begin(), nativePackages.end(), this), nativePackages.end());
        log.ResetForHost(Host());
    }

}

extern "C" GARNET_SERVING_API int GarnetListAvailableModelsJson(
    const char* catalogRoot,
    char* output,
    int outputCapacity,
    int* requiredCapacity)
{
    try {
        std::lock_guard<std::mutex> guard(Garnet::nativePackagesMutex);
        if (Garnet::nativePackages.size() > 1)
            throw std::runtime_error("C serving API is ambiguous with multiple Garnet runtimes");
        return CopyJsonResult(
            Garnet::nativePackages.empty() ? Garnet::EnumerateAvailableModelsJson(
                Garnet::ResolveModelCatalogRoot(catalogRoot ? catalogRoot : "", "")) :
                Garnet::nativePackages.front()->AvailableModelsJson(catalogRoot ? catalogRoot : ""),
            output,
            outputCapacity,
            requiredCapacity);
    }
    catch (const std::exception& exception) {
        return CopyJsonResult(
            GarnetJsonError("catalog_enumeration_failed", exception.what()),
            output,
            outputCapacity,
            requiredCapacity);
    }
}

extern "C" GARNET_SERVING_API int GarnetListLoadedModelsJson(
    char* output,
    int outputCapacity,
    int* requiredCapacity)
{
    try {
        std::lock_guard<std::mutex> guard(Garnet::nativePackagesMutex);
        if (Garnet::nativePackages.size() > 1)
            throw std::runtime_error("C serving API is ambiguous with multiple Garnet runtimes");
        return CopyJsonResult(
            Garnet::nativePackages.empty() ? json({{"schema_version", 1}, {"serving_mode", "single_instance"},
                {"models", json::array()}}).dump() : Garnet::nativePackages.front()->LoadedModelsJson(),
            output,
            outputCapacity,
            requiredCapacity);
    }
    catch (const std::exception& exception) {
        return CopyJsonResult(
            GarnetJsonError("loaded_model_enumeration_failed", exception.what()),
            output,
            outputCapacity,
            requiredCapacity);
    }
}

#define cudaStreamCreate CreateEntryExecutionStream
#define cudaStreamDestroy DestroyEntryExecutionStream

#if defined(_WIN32)
#define GARNET_ENTRY_EXPORT __declspec(dllexport)
#else
#define GARNET_ENTRY_EXPORT
#endif

extern "C" int GarnetQwenVLPreprocessJpegFile(
    const char* jpegPath,
    int minPixels,
    int maxPixels,
    float* output,
    long long* imageGridTHW,
    int* sourceHeight,
    int* sourceWidth,
    int* resizedHeight,
    int* resizedWidth,
    char* errorMessage,
    int errorMessageCapacity);

extern "C" int GarnetFreeDeviceBuffer(void* deviceBuffer);

extern "C" GARNET_ENTRY_EXPORT int GarnetCreateDevicePagedKVFP32(
    int physicalPageCount,
    int pageSize,
    int logicalPageCount,
    int qHeads,
    int kvHeads,
    int headDim,
    const int* pageTable,
    long long* outputHandle,
    char* errorMessage,
    int errorMessageCapacity);

extern "C" GARNET_ENTRY_EXPORT int GarnetDestroyDevicePagedKVFP32(
    long long handle,
    char* errorMessage,
    int errorMessageCapacity);

extern "C" GARNET_ENTRY_EXPORT int GarnetDevicePagedKVWriteDeviceFP32(
    long long handle,
    const float* deviceQKV,
    int tokenCount,
    int startPosition,
    char* errorMessage,
    int errorMessageCapacity);

extern "C" GARNET_ENTRY_EXPORT int GarnetDevicePagedKVAttentionDeviceFP32(
    long long handle,
    const float* deviceQ,
    float* deviceOutput,
    int sequenceLength,
    char* errorMessage,
    int errorMessageCapacity);

extern "C" GARNET_ENTRY_EXPORT int GarnetDebugSampleLogitsTop1FP32(
    const float* deviceLogits,
    int rows,
    int vocabSize,
    long long* deviceOutputTokenId,
    float* deviceOutputTokenValue,
    char* errorMessage,
    int errorMessageCapacity);

extern "C" GARNET_ENTRY_EXPORT int GarnetQwenVLPrepareJpegPrompt(
    const char* modelDir,
    const char* jpegPath,
    const char* prompt,
    int minPixels,
    int maxPixels,
    long long* outputInputIds,
    int inputIdCapacity,
    int* outputInputIdCount,
    long long* outputMmTokenTypes,
    int mmTokenTypeCapacity,
    float* outputPixelValues,
    int pixelValueCapacity,
    int* outputPixelValueCount,
    long long* outputImageGridTHW,
    int* sourceHeight,
    int* sourceWidth,
    int* resizedHeight,
    int* resizedWidth,
    char* errorMessage,
    int errorMessageCapacity)
{
    auto setError = [&](const std::string& message) {
        if (errorMessage != nullptr && errorMessageCapacity > 0) {
            std::snprintf(errorMessage, static_cast<size_t>(errorMessageCapacity), "%s", message.c_str());
        }
    };
    try {
        if (modelDir == nullptr || jpegPath == nullptr || prompt == nullptr) {
            setError("modelDir, jpegPath, and prompt are required");
            return 1;
        }
        constexpr int patchSize = 16;
        constexpr int temporalPatchSize = 2;
        constexpr int mergeSize = 2;
        int pixelBudget = minPixels > maxPixels ? minPixels : maxPixels;
        if (pixelBudget <= 0) {
            setError("pixel budget must be positive");
            return 1;
        }
        int maxPatchCount = (pixelBudget + patchSize * patchSize - 1) / (patchSize * patchSize);
        if (maxPatchCount < 1) {
            maxPatchCount = 1;
        }
        int featureDim = 3 * temporalPatchSize * patchSize * patchSize;
        std::vector<float> pixelScratch(static_cast<size_t>(maxPatchCount) * static_cast<size_t>(featureDim));
        long long grid[3] = {};
        int srcH = 0;
        int srcW = 0;
        int outH = 0;
        int outW = 0;
        char imageError[512] = {};
        int rc = GarnetQwenVLPreprocessJpegFile(
            jpegPath,
            minPixels,
            maxPixels,
            pixelScratch.data(),
            grid,
            &srcH,
            &srcW,
            &outH,
            &outW,
            imageError,
            static_cast<int>(sizeof(imageError)));
        if (rc != 0) {
            setError(imageError);
            return rc;
        }

        int patchCount = (outH / patchSize) * (outW / patchSize);
        int pixelValueCount = patchCount * featureDim;
        if (outputPixelValueCount != nullptr) {
            *outputPixelValueCount = pixelValueCount;
        }
        if (outputImageGridTHW != nullptr) {
            outputImageGridTHW[0] = grid[0];
            outputImageGridTHW[1] = grid[1];
            outputImageGridTHW[2] = grid[2];
        }
        if (sourceHeight != nullptr) *sourceHeight = srcH;
        if (sourceWidth != nullptr) *sourceWidth = srcW;
        if (resizedHeight != nullptr) *resizedHeight = outH;
        if (resizedWidth != nullptr) *resizedWidth = outW;

        std::string tokenizerError;
        auto tokenizer = Garnet::Tokenization::GetCachedQwenTokenizer(modelDir, &tokenizerError);
        if (!tokenizer) {
            setError(tokenizerError);
            return 1;
        }
        std::vector<int64_t> promptIds = Garnet::Tokenization::QwenVLPromptBuilder::BuildSingleImagePromptIds(
            *tokenizer,
            prompt,
            grid,
            mergeSize);
        int64_t imagePadId = tokenizer->TokenId("<|image_pad|>");
        if (imagePadId < 0 ||
            tokenizer->TokenId("<|vision_start|>") < 0 ||
            tokenizer->TokenId("<|vision_end|>") < 0 ||
            tokenizer->TokenId("<|im_start|>") < 0 ||
            tokenizer->TokenId("<|im_end|>") < 0) {
            setError("model tokenizer is missing required Qwen-VL special tokens");
            return 1;
        }
        if (outputInputIdCount != nullptr) {
            *outputInputIdCount = static_cast<int>(promptIds.size());
        }

        bool capacityOk = true;
        if (outputInputIds == nullptr || inputIdCapacity < static_cast<int>(promptIds.size())) {
            capacityOk = false;
        }
        if (outputMmTokenTypes == nullptr || mmTokenTypeCapacity < static_cast<int>(promptIds.size())) {
            capacityOk = false;
        }
        if (outputPixelValues == nullptr || pixelValueCapacity < pixelValueCount) {
            capacityOk = false;
        }
        if (!capacityOk) {
            setError("output buffer capacity is too small");
            return 3;
        }

        int visualTokenCount = 0;
        for (size_t i = 0; i < promptIds.size(); ++i) {
            outputInputIds[i] = static_cast<long long>(promptIds[i]);
            outputMmTokenTypes[i] = promptIds[i] == imagePadId ? 1LL : 0LL;
            if (outputMmTokenTypes[i] == 1LL) {
                ++visualTokenCount;
            }
        }
        int expectedVisualTokenCount = static_cast<int>((grid[0] * grid[1] * grid[2]) / (mergeSize * mergeSize));
        if (visualTokenCount != expectedVisualTokenCount) {
            setError("visual placeholder count does not match image grid");
            return 1;
        }
        std::memcpy(outputPixelValues, pixelScratch.data(), static_cast<size_t>(pixelValueCount) * sizeof(float));
        return 0;
    }
    catch (const std::exception& exc) {
        setError(exc.what());
        return 1;
    }
}

extern "C" GARNET_ENTRY_EXPORT int GarnetQwenVLPrepareJpegPromptDevice(
    const char* modelDir,
    const char* jpegPath,
    const char* prompt,
    int minPixels,
    int maxPixels,
    long long* outputInputIds,
    int inputIdCapacity,
    int* outputInputIdCount,
    long long* outputMmTokenTypes,
    int mmTokenTypeCapacity,
    void** outputPixelValuesDevice,
    size_t* outputPixelValueBytes,
    int* outputPixelValueCount,
    long long* outputImageGridTHW,
    int* sourceHeight,
    int* sourceWidth,
    int* resizedHeight,
    int* resizedWidth,
    long long* imagePreprocessUs,
    long long* tokenizeUs,
    long long* totalUs,
    char* errorMessage,
    int errorMessageCapacity)
{
    auto totalStart = std::chrono::steady_clock::now();
    auto setError = [&](const std::string& message) {
        if (errorMessage != nullptr && errorMessageCapacity > 0) {
            std::snprintf(errorMessage, static_cast<size_t>(errorMessageCapacity), "%s", message.c_str());
        }
    };

    auto setUs = [](long long* out, double ms) {
        if (out) {
            *out = static_cast<long long>(ms * 1000.0);
        }
    };
    auto msSince = [](std::chrono::steady_clock::time_point start) {
        auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    try {
        if (outputPixelValuesDevice) {
            *outputPixelValuesDevice = nullptr;
        }
        if (outputPixelValueBytes) {
            *outputPixelValueBytes = 0;
        }
        if (outputPixelValueCount) {
            *outputPixelValueCount = 0;
        }
        if (modelDir == nullptr || jpegPath == nullptr || prompt == nullptr ||
            outputInputIds == nullptr || outputMmTokenTypes == nullptr ||
            outputPixelValuesDevice == nullptr || outputPixelValueBytes == nullptr ||
            outputPixelValueCount == nullptr || outputImageGridTHW == nullptr) {
            setError("invalid GarnetQwenVLPrepareJpegPromptDevice arguments");
            return 1;
        }

        constexpr int mergeSize = 2;

        auto imageStart = std::chrono::steady_clock::now();
        auto image = Garnet::Image::QwenVL::PreprocessJpegFileToDeviceBuffer(jpegPath, minPixels, maxPixels);
        double imageMs = msSince(imageStart);
        setUs(imagePreprocessUs, imageMs);

        outputImageGridTHW[0] = image.imageGridTHW[0];
        outputImageGridTHW[1] = image.imageGridTHW[1];
        outputImageGridTHW[2] = image.imageGridTHW[2];
        if (sourceHeight) *sourceHeight = image.sourceHeight;
        if (sourceWidth) *sourceWidth = image.sourceWidth;
        if (resizedHeight) *resizedHeight = image.resizedHeight;
        if (resizedWidth) *resizedWidth = image.resizedWidth;
        *outputPixelValueBytes = image.outputBytes;
        *outputPixelValueCount = image.patchCount * image.featureDim;

        auto tokenStart = std::chrono::steady_clock::now();
        std::string tokenError;
        auto tokenizer = Garnet::Tokenization::GetCachedQwenTokenizer(modelDir, &tokenError);
        if (!tokenizer) {
            GarnetFreeDeviceBuffer(image.pixelValuesDevice);
            setError(tokenError.empty() ? "failed to load tokenizer" : tokenError);
            return 1;
        }
        std::vector<int64_t> promptIds = Garnet::Tokenization::QwenVLPromptBuilder::BuildSingleImagePromptIds(
            *tokenizer,
            prompt,
            image.imageGridTHW,
            mergeSize);
        int64_t imagePadId = tokenizer->TokenId("<|image_pad|>");
        if (imagePadId < 0 ||
            tokenizer->TokenId("<|vision_start|>") < 0 ||
            tokenizer->TokenId("<|vision_end|>") < 0 ||
            tokenizer->TokenId("<|im_start|>") < 0 ||
            tokenizer->TokenId("<|im_end|>") < 0) {
            GarnetFreeDeviceBuffer(image.pixelValuesDevice);
            setError("model tokenizer is missing required Qwen-VL special tokens");
            return 1;
        }
        double tokenMs = msSince(tokenStart);
        setUs(tokenizeUs, tokenMs);

        if (outputInputIdCount) {
            *outputInputIdCount = static_cast<int>(promptIds.size());
        }
        if (inputIdCapacity < static_cast<int>(promptIds.size()) ||
            mmTokenTypeCapacity < static_cast<int>(promptIds.size())) {
            GarnetFreeDeviceBuffer(image.pixelValuesDevice);
            setError("token output buffer capacity is too small");
            return 3;
        }

        int visualTokenCount = 0;
        for (size_t i = 0; i < promptIds.size(); ++i) {
            outputInputIds[i] = static_cast<long long>(promptIds[i]);
            long long mmType = promptIds[i] == imagePadId ? 1LL : 0LL;
            outputMmTokenTypes[i] = mmType;
            if (mmType == 1LL) {
                ++visualTokenCount;
            }
        }
        int expectedVisualTokenCount = static_cast<int>(
            (image.imageGridTHW[0] * image.imageGridTHW[1] * image.imageGridTHW[2]) / (mergeSize * mergeSize));
        if (visualTokenCount != expectedVisualTokenCount) {
            GarnetFreeDeviceBuffer(image.pixelValuesDevice);
            setError("visual placeholder count does not match image grid");
            return 1;
        }

        *outputPixelValuesDevice = image.pixelValuesDevice;
        setUs(totalUs, msSince(totalStart));
        setError("");
        return 0;
    }
    catch (const std::exception& exc) {
        setError(exc.what());
        return 1;
    }
}

extern "C" GARNET_ENTRY_EXPORT int GarnetQwenVLPrepareJpegPromptDeviceTensors(
    const char* modelDir,
    const char* jpegPath,
    const char* prompt,
    int minPixels,
    int maxPixels,
    void** outputInputIdsDevice,
    size_t* outputInputIdsBytes,
    int* outputInputIdCount,
    void** outputMmTokenTypesDevice,
    size_t* outputMmTokenTypesBytes,
    void** outputPixelValuesDevice,
    size_t* outputPixelValueBytes,
    int* outputPixelValueCount,
    long long* outputImageGridTHW,
    int* sourceHeight,
    int* sourceWidth,
    int* resizedHeight,
    int* resizedWidth,
    long long* imagePreprocessUs,
    long long* tokenizeUs,
    long long* tensorUploadUs,
    long long* totalUs,
    char* errorMessage,
    int errorMessageCapacity)
{
    auto totalStart = std::chrono::steady_clock::now();
    auto setError = [&](const std::string& message) {
        if (errorMessage != nullptr && errorMessageCapacity > 0) {
            std::snprintf(errorMessage, static_cast<size_t>(errorMessageCapacity), "%s", message.c_str());
        }
    };
    auto setUs = [](long long* out, double ms) {
        if (out) {
            *out = static_cast<long long>(ms * 1000.0);
        }
    };
    auto msSince = [](std::chrono::steady_clock::time_point start) {
        auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    };
    auto clearOutputs = [&]() {
        if (outputInputIdsDevice) *outputInputIdsDevice = nullptr;
        if (outputInputIdsBytes) *outputInputIdsBytes = 0;
        if (outputInputIdCount) *outputInputIdCount = 0;
        if (outputMmTokenTypesDevice) *outputMmTokenTypesDevice = nullptr;
        if (outputMmTokenTypesBytes) *outputMmTokenTypesBytes = 0;
        if (outputPixelValuesDevice) *outputPixelValuesDevice = nullptr;
        if (outputPixelValueBytes) *outputPixelValueBytes = 0;
        if (outputPixelValueCount) *outputPixelValueCount = 0;
    };

    clearOutputs();
    try {
        if (modelDir == nullptr || jpegPath == nullptr || prompt == nullptr ||
            outputInputIdsDevice == nullptr || outputInputIdsBytes == nullptr ||
            outputInputIdCount == nullptr || outputMmTokenTypesDevice == nullptr ||
            outputMmTokenTypesBytes == nullptr || outputPixelValuesDevice == nullptr ||
            outputPixelValueBytes == nullptr || outputPixelValueCount == nullptr ||
            outputImageGridTHW == nullptr) {
            setError("invalid GarnetQwenVLPrepareJpegPromptDeviceTensors arguments");
            return 1;
        }

        constexpr int mergeSize = 2;
        auto imageStart = std::chrono::steady_clock::now();
        auto image = Garnet::Image::QwenVL::PreprocessJpegFileToDeviceBuffer(jpegPath, minPixels, maxPixels);
        setUs(imagePreprocessUs, msSince(imageStart));

        outputImageGridTHW[0] = image.imageGridTHW[0];
        outputImageGridTHW[1] = image.imageGridTHW[1];
        outputImageGridTHW[2] = image.imageGridTHW[2];
        if (sourceHeight) *sourceHeight = image.sourceHeight;
        if (sourceWidth) *sourceWidth = image.sourceWidth;
        if (resizedHeight) *resizedHeight = image.resizedHeight;
        if (resizedWidth) *resizedWidth = image.resizedWidth;

        auto tokenStart = std::chrono::steady_clock::now();
        std::string tokenError;
        auto tokenizer = Garnet::Tokenization::GetCachedQwenTokenizer(modelDir, &tokenError);
        if (!tokenizer) {
            GarnetFreeDeviceBuffer(image.pixelValuesDevice);
            setError(tokenError.empty() ? "failed to load tokenizer" : tokenError);
            return 1;
        }
        std::vector<int64_t> promptIds = Garnet::Tokenization::QwenVLPromptBuilder::BuildSingleImagePromptIds(
            *tokenizer,
            prompt,
            image.imageGridTHW,
            mergeSize);
        int64_t imagePadId = tokenizer->TokenId("<|image_pad|>");
        if (imagePadId < 0 ||
            tokenizer->TokenId("<|vision_start|>") < 0 ||
            tokenizer->TokenId("<|vision_end|>") < 0 ||
            tokenizer->TokenId("<|im_start|>") < 0 ||
            tokenizer->TokenId("<|im_end|>") < 0) {
            GarnetFreeDeviceBuffer(image.pixelValuesDevice);
            setError("model tokenizer is missing required Qwen-VL special tokens");
            return 1;
        }

        std::vector<long long> inputIds;
        std::vector<long long> mmTypes;
        inputIds.reserve(promptIds.size());
        mmTypes.reserve(promptIds.size());
        int visualTokenCount = 0;
        for (int64_t id : promptIds) {
            inputIds.push_back(static_cast<long long>(id));
            long long mmType = id == imagePadId ? 1LL : 0LL;
            mmTypes.push_back(mmType);
            if (mmType == 1LL) {
                ++visualTokenCount;
            }
        }
        int expectedVisualTokenCount = static_cast<int>(
            (image.imageGridTHW[0] * image.imageGridTHW[1] * image.imageGridTHW[2]) / (mergeSize * mergeSize));
        if (visualTokenCount != expectedVisualTokenCount) {
            GarnetFreeDeviceBuffer(image.pixelValuesDevice);
            setError("visual placeholder count does not match image grid");
            return 1;
        }
        setUs(tokenizeUs, msSince(tokenStart));

        auto uploadStart = std::chrono::steady_clock::now();
        size_t idsBytes = inputIds.size() * sizeof(long long);
        size_t mmBytes = mmTypes.size() * sizeof(long long);
        void* dIds = nullptr;
        void* dMm = nullptr;
        cudaStream_t stream = nullptr;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err == cudaSuccess) err = cudaMalloc(&dIds, idsBytes);
        if (err == cudaSuccess) err = cudaMalloc(&dMm, mmBytes);
        if (err == cudaSuccess) err = cudaMemcpyAsync(dIds, inputIds.data(), idsBytes, cudaMemcpyHostToDevice, stream);
        if (err == cudaSuccess) err = cudaMemcpyAsync(dMm, mmTypes.data(), mmBytes, cudaMemcpyHostToDevice, stream);
        if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
        if (stream) cudaStreamDestroy(stream);
        if (err != cudaSuccess) {
            if (dIds) cudaFree(dIds);
            if (dMm) cudaFree(dMm);
            GarnetFreeDeviceBuffer(image.pixelValuesDevice);
            setError(cudaGetErrorString(err));
            return 1;
        }
        setUs(tensorUploadUs, msSince(uploadStart));

        *outputInputIdsDevice = dIds;
        *outputInputIdsBytes = idsBytes;
        *outputInputIdCount = static_cast<int>(inputIds.size());
        *outputMmTokenTypesDevice = dMm;
        *outputMmTokenTypesBytes = mmBytes;
        *outputPixelValuesDevice = image.pixelValuesDevice;
        *outputPixelValueBytes = image.outputBytes;
        *outputPixelValueCount = image.patchCount * image.featureDim;
        setUs(totalUs, msSince(totalStart));
        setError("");
        return 0;
    }
    catch (const std::exception& exc) {
        setError(exc.what());
        return 1;
    }
}

namespace
{
    struct QwenVLDeviceRequest
    {
        void* inputIdsDevice = nullptr;
        size_t inputIdsBytes = 0;
        int inputIdCount = 0;
        void* mmTokenTypesDevice = nullptr;
        size_t mmTokenTypesBytes = 0;
        void* pixelValuesDevice = nullptr;
        size_t pixelValuesBytes = 0;
        int pixelValueCount = 0;
        long long imageGridTHW[3] = { 1, 0, 0 };
        int sourceHeight = 0;
        int sourceWidth = 0;
        int resizedHeight = 0;
        int resizedWidth = 0;
        long long imagePreprocessUs = 0;
        long long tokenizeUs = 0;
        long long tensorUploadUs = 0;
        long long totalUs = 0;
        long long kvHandle = 0;
        int kvMaxTokens = 0;
        int kvLogicalLength = 0;
        int kvPageSize = 0;
        int kvLogicalPages = 0;
        int kvPhysicalPages = 0;
        int kvQHeads = 0;
        int kvHeads = 0;
        int kvHeadDim = 0;
    };

    std::mutex g_qwenVLDeviceRequestsMutex;
    std::unordered_map<long long, QwenVLDeviceRequest> g_qwenVLDeviceRequests;
    long long g_nextQwenVLDeviceRequestHandle = 1;

    void FreeQwenVLDeviceRequestBuffers(QwenVLDeviceRequest& request)
    {
        if (request.inputIdsDevice) {
            cudaFree(request.inputIdsDevice);
            request.inputIdsDevice = nullptr;
        }
        if (request.mmTokenTypesDevice) {
            cudaFree(request.mmTokenTypesDevice);
            request.mmTokenTypesDevice = nullptr;
        }
        if (request.pixelValuesDevice) {
            cudaFree(request.pixelValuesDevice);
            request.pixelValuesDevice = nullptr;
        }
        if (request.kvHandle > 0) {
            char error[1024] = {};
            GarnetDestroyDevicePagedKVFP32(request.kvHandle, error, static_cast<int>(sizeof(error)));
            request.kvHandle = 0;
        }
    }
}

extern "C" GARNET_ENTRY_EXPORT int GarnetCreateQwenVLDeviceRequest(
    const char* modelDir,
    const char* jpegPath,
    const char* prompt,
    int minPixels,
    int maxPixels,
    long long* outputHandle,
    char* errorMessage,
    int errorMessageCapacity)
{
    auto setError = [&](const std::string& message) {
        if (errorMessage != nullptr && errorMessageCapacity > 0) {
            std::snprintf(errorMessage, static_cast<size_t>(errorMessageCapacity), "%s", message.c_str());
        }
    };
    if (!outputHandle) {
        setError("outputHandle is required");
        return 1;
    }
    *outputHandle = 0;

    QwenVLDeviceRequest request;
    int rc = GarnetQwenVLPrepareJpegPromptDeviceTensors(
        modelDir,
        jpegPath,
        prompt,
        minPixels,
        maxPixels,
        &request.inputIdsDevice,
        &request.inputIdsBytes,
        &request.inputIdCount,
        &request.mmTokenTypesDevice,
        &request.mmTokenTypesBytes,
        &request.pixelValuesDevice,
        &request.pixelValuesBytes,
        &request.pixelValueCount,
        request.imageGridTHW,
        &request.sourceHeight,
        &request.sourceWidth,
        &request.resizedHeight,
        &request.resizedWidth,
        &request.imagePreprocessUs,
        &request.tokenizeUs,
        &request.tensorUploadUs,
        &request.totalUs,
        errorMessage,
        errorMessageCapacity);
    if (rc != 0) {
        FreeQwenVLDeviceRequestBuffers(request);
        return rc;
    }

    long long handle = 0;
    {
        std::lock_guard<std::mutex> lock(g_qwenVLDeviceRequestsMutex);
        handle = g_nextQwenVLDeviceRequestHandle++;
        g_qwenVLDeviceRequests.emplace(handle, request);
    }
    *outputHandle = handle;
    setError("");
    return 0;
}

extern "C" GARNET_ENTRY_EXPORT int GarnetDestroyQwenVLDeviceRequest(
    long long handle,
    char* errorMessage,
    int errorMessageCapacity)
{
    auto setError = [&](const std::string& message) {
        if (errorMessage != nullptr && errorMessageCapacity > 0) {
            std::snprintf(errorMessage, static_cast<size_t>(errorMessageCapacity), "%s", message.c_str());
        }
    };
    QwenVLDeviceRequest request;
    {
        std::lock_guard<std::mutex> lock(g_qwenVLDeviceRequestsMutex);
        auto found = g_qwenVLDeviceRequests.find(handle);
        if (found == g_qwenVLDeviceRequests.end()) {
            setError("invalid Qwen-VL device request handle");
            return 1;
        }
        request = found->second;
        g_qwenVLDeviceRequests.erase(found);
    }
    FreeQwenVLDeviceRequestBuffers(request);
    setError("");
    return 0;
}

extern "C" GARNET_ENTRY_EXPORT int GarnetGetQwenVLDeviceRequestInfo(
    long long handle,
    void** inputIdsDevice,
    size_t* inputIdsBytes,
    int* inputIdCount,
    void** mmTokenTypesDevice,
    size_t* mmTokenTypesBytes,
    void** pixelValuesDevice,
    size_t* pixelValuesBytes,
    int* pixelValueCount,
    long long* imageGridTHW,
    int* sourceHeight,
    int* sourceWidth,
    int* resizedHeight,
    int* resizedWidth,
    long long* imagePreprocessUs,
    long long* tokenizeUs,
    long long* tensorUploadUs,
    long long* totalUs,
    char* errorMessage,
    int errorMessageCapacity)
{
    auto setError = [&](const std::string& message) {
        if (errorMessage != nullptr && errorMessageCapacity > 0) {
            std::snprintf(errorMessage, static_cast<size_t>(errorMessageCapacity), "%s", message.c_str());
        }
    };
    QwenVLDeviceRequest request;
    {
        std::lock_guard<std::mutex> lock(g_qwenVLDeviceRequestsMutex);
        auto found = g_qwenVLDeviceRequests.find(handle);
        if (found == g_qwenVLDeviceRequests.end()) {
            setError("invalid Qwen-VL device request handle");
            return 1;
        }
        request = found->second;
    }

    if (inputIdsDevice) *inputIdsDevice = request.inputIdsDevice;
    if (inputIdsBytes) *inputIdsBytes = request.inputIdsBytes;
    if (inputIdCount) *inputIdCount = request.inputIdCount;
    if (mmTokenTypesDevice) *mmTokenTypesDevice = request.mmTokenTypesDevice;
    if (mmTokenTypesBytes) *mmTokenTypesBytes = request.mmTokenTypesBytes;
    if (pixelValuesDevice) *pixelValuesDevice = request.pixelValuesDevice;
    if (pixelValuesBytes) *pixelValuesBytes = request.pixelValuesBytes;
    if (pixelValueCount) *pixelValueCount = request.pixelValueCount;
    if (imageGridTHW) {
        imageGridTHW[0] = request.imageGridTHW[0];
        imageGridTHW[1] = request.imageGridTHW[1];
        imageGridTHW[2] = request.imageGridTHW[2];
    }
    if (sourceHeight) *sourceHeight = request.sourceHeight;
    if (sourceWidth) *sourceWidth = request.sourceWidth;
    if (resizedHeight) *resizedHeight = request.resizedHeight;
    if (resizedWidth) *resizedWidth = request.resizedWidth;
    if (imagePreprocessUs) *imagePreprocessUs = request.imagePreprocessUs;
    if (tokenizeUs) *tokenizeUs = request.tokenizeUs;
    if (tensorUploadUs) *tensorUploadUs = request.tensorUploadUs;
    if (totalUs) *totalUs = request.totalUs;
    setError("");
    return 0;
}

extern "C" GARNET_ENTRY_EXPORT int GarnetAllocateQwenVLDeviceRequestKV(
    long long requestHandle,
    int maxNewTokens,
    int pageSize,
    int qHeads,
    int kvHeads,
    int headDim,
    int physicalPageCount,
    long long* outputKVHandle,
    int* outputMaxTokens,
    int* outputLogicalPages,
    int* outputPhysicalPages,
    char* errorMessage,
    int errorMessageCapacity)
{
    auto setError = [&](const std::string& message) {
        if (errorMessage != nullptr && errorMessageCapacity > 0) {
            std::snprintf(errorMessage, static_cast<size_t>(errorMessageCapacity), "%s", message.c_str());
        }
    };
    if (outputKVHandle) *outputKVHandle = 0;
    if (outputMaxTokens) *outputMaxTokens = 0;
    if (outputLogicalPages) *outputLogicalPages = 0;
    if (outputPhysicalPages) *outputPhysicalPages = 0;
    if (pageSize <= 0 || qHeads <= 0 || kvHeads <= 0 || headDim <= 0 || (qHeads % kvHeads) != 0) {
        setError("invalid request KV geometry");
        return 1;
    }

    std::lock_guard<std::mutex> lock(g_qwenVLDeviceRequestsMutex);
    auto found = g_qwenVLDeviceRequests.find(requestHandle);
    if (found == g_qwenVLDeviceRequests.end()) {
        setError("invalid Qwen-VL device request handle");
        return 1;
    }

    QwenVLDeviceRequest& request = found->second;
    int maxTokens = request.inputIdCount + (maxNewTokens > 0 ? maxNewTokens : 0);
    if (maxTokens <= 0) {
        setError("request has no tokens for KV allocation");
        return 1;
    }
    int logicalPages = (maxTokens + pageSize - 1) / pageSize;
    int physicalPages = physicalPageCount > 0 ? physicalPageCount : logicalPages;
    if (physicalPages < logicalPages) {
        setError("physicalPageCount must cover logical pages for request KV allocation");
        return 1;
    }

    if (request.kvHandle > 0) {
        char destroyError[1024] = {};
        GarnetDestroyDevicePagedKVFP32(request.kvHandle, destroyError, static_cast<int>(sizeof(destroyError)));
        request.kvHandle = 0;
    }

    std::vector<int> pageTable(static_cast<size_t>(logicalPages));
    for (int i = 0; i < logicalPages; ++i) {
        pageTable[static_cast<size_t>(i)] = i;
    }

    long long kvHandle = 0;
    int rc = GarnetCreateDevicePagedKVFP32(
        physicalPages,
        pageSize,
        logicalPages,
        qHeads,
        kvHeads,
        headDim,
        pageTable.data(),
        &kvHandle,
        errorMessage,
        errorMessageCapacity);
    if (rc != 0) {
        return rc;
    }

    request.kvHandle = kvHandle;
    request.kvMaxTokens = maxTokens;
    request.kvLogicalLength = 0;
    request.kvPageSize = pageSize;
    request.kvLogicalPages = logicalPages;
    request.kvPhysicalPages = physicalPages;
    request.kvQHeads = qHeads;
    request.kvHeads = kvHeads;
    request.kvHeadDim = headDim;

    if (outputKVHandle) *outputKVHandle = kvHandle;
    if (outputMaxTokens) *outputMaxTokens = maxTokens;
    if (outputLogicalPages) *outputLogicalPages = logicalPages;
    if (outputPhysicalPages) *outputPhysicalPages = physicalPages;
    setError("");
    return 0;
}

extern "C" GARNET_ENTRY_EXPORT int GarnetGetQwenVLDeviceRequestKVInfo(
    long long requestHandle,
    long long* outputKVHandle,
    int* outputMaxTokens,
    int* outputPageSize,
    int* outputLogicalPages,
    int* outputPhysicalPages,
    int* outputQHeads,
    int* outputKVHeads,
    int* outputHeadDim,
    char* errorMessage,
    int errorMessageCapacity)
{
    auto setError = [&](const std::string& message) {
        if (errorMessage != nullptr && errorMessageCapacity > 0) {
            std::snprintf(errorMessage, static_cast<size_t>(errorMessageCapacity), "%s", message.c_str());
        }
    };
    QwenVLDeviceRequest request;
    {
        std::lock_guard<std::mutex> lock(g_qwenVLDeviceRequestsMutex);
        auto found = g_qwenVLDeviceRequests.find(requestHandle);
        if (found == g_qwenVLDeviceRequests.end()) {
            setError("invalid Qwen-VL device request handle");
            return 1;
        }
        request = found->second;
    }

    if (outputKVHandle) *outputKVHandle = request.kvHandle;
    if (outputMaxTokens) *outputMaxTokens = request.kvMaxTokens;
    if (outputPageSize) *outputPageSize = request.kvPageSize;
    if (outputLogicalPages) *outputLogicalPages = request.kvLogicalPages;
    if (outputPhysicalPages) *outputPhysicalPages = request.kvPhysicalPages;
    if (outputQHeads) *outputQHeads = request.kvQHeads;
    if (outputKVHeads) *outputKVHeads = request.kvHeads;
    if (outputHeadDim) *outputHeadDim = request.kvHeadDim;
    setError("");
    return 0;
}

extern "C" GARNET_ENTRY_EXPORT int GarnetGetQwenVLDeviceRequestKVState(
    long long requestHandle,
    long long* outputKVHandle,
    int* outputMaxTokens,
    int* outputLogicalLength,
    int* outputPageSize,
    int* outputLogicalPages,
    int* outputPhysicalPages,
    char* errorMessage,
    int errorMessageCapacity)
{
    auto setError = [&](const std::string& message) {
        if (errorMessage != nullptr && errorMessageCapacity > 0) {
            std::snprintf(errorMessage, static_cast<size_t>(errorMessageCapacity), "%s", message.c_str());
        }
    };
    QwenVLDeviceRequest request;
    {
        std::lock_guard<std::mutex> lock(g_qwenVLDeviceRequestsMutex);
        auto found = g_qwenVLDeviceRequests.find(requestHandle);
        if (found == g_qwenVLDeviceRequests.end()) {
            setError("invalid Qwen-VL device request handle");
            return 1;
        }
        request = found->second;
    }

    if (outputKVHandle) *outputKVHandle = request.kvHandle;
    if (outputMaxTokens) *outputMaxTokens = request.kvMaxTokens;
    if (outputLogicalLength) *outputLogicalLength = request.kvLogicalLength;
    if (outputPageSize) *outputPageSize = request.kvPageSize;
    if (outputLogicalPages) *outputLogicalPages = request.kvLogicalPages;
    if (outputPhysicalPages) *outputPhysicalPages = request.kvPhysicalPages;
    setError("");
    return 0;
}

extern "C" GARNET_ENTRY_EXPORT int GarnetQwenVLDeviceRequestKVWriteDevice(
    long long requestHandle,
    const float* deviceQKV,
    int tokenCount,
    int startPosition,
    char* errorMessage,
    int errorMessageCapacity)
{
    auto setError = [&](const std::string& message) {
        if (errorMessage != nullptr && errorMessageCapacity > 0) {
            std::snprintf(errorMessage, static_cast<size_t>(errorMessageCapacity), "%s", message.c_str());
        }
    };
    long long kvHandle = 0;
    int capacity = 0;
    {
        std::lock_guard<std::mutex> lock(g_qwenVLDeviceRequestsMutex);
        auto found = g_qwenVLDeviceRequests.find(requestHandle);
        if (found == g_qwenVLDeviceRequests.end()) {
            setError("invalid Qwen-VL device request handle");
            return 1;
        }
        kvHandle = found->second.kvHandle;
        capacity = found->second.kvMaxTokens;
    }
    if (kvHandle <= 0) {
        setError("request has no allocated device KV cache");
        return 1;
    }
    if (deviceQKV == nullptr || tokenCount <= 0 || startPosition < 0 ||
        startPosition + tokenCount > capacity) {
        setError("invalid request KV write range");
        return 1;
    }
    int rc = GarnetDevicePagedKVWriteDeviceFP32(
        kvHandle,
        deviceQKV,
        tokenCount,
        startPosition,
        errorMessage,
        errorMessageCapacity);
    if (rc == 0) {
        std::lock_guard<std::mutex> lock(g_qwenVLDeviceRequestsMutex);
        auto found = g_qwenVLDeviceRequests.find(requestHandle);
        if (found != g_qwenVLDeviceRequests.end()) {
            int endPosition = startPosition + tokenCount;
            if (endPosition > found->second.kvLogicalLength) {
                found->second.kvLogicalLength = endPosition;
            }
        }
    }
    return rc;
}

extern "C" GARNET_ENTRY_EXPORT int GarnetQwenVLDeviceRequestKVAttentionDevice(
    long long requestHandle,
    const float* deviceQ,
    float* deviceOutput,
    int sequenceLength,
    char* errorMessage,
    int errorMessageCapacity)
{
    auto setError = [&](const std::string& message) {
        if (errorMessage != nullptr && errorMessageCapacity > 0) {
            std::snprintf(errorMessage, static_cast<size_t>(errorMessageCapacity), "%s", message.c_str());
        }
    };
    long long kvHandle = 0;
    int capacity = 0;
    int logicalLength = 0;
    {
        std::lock_guard<std::mutex> lock(g_qwenVLDeviceRequestsMutex);
        auto found = g_qwenVLDeviceRequests.find(requestHandle);
        if (found == g_qwenVLDeviceRequests.end()) {
            setError("invalid Qwen-VL device request handle");
            return 1;
        }
        kvHandle = found->second.kvHandle;
        capacity = found->second.kvMaxTokens;
        logicalLength = found->second.kvLogicalLength;
    }
    if (kvHandle <= 0) {
        setError("request has no allocated device KV cache");
        return 1;
    }
    if (deviceQ == nullptr || deviceOutput == nullptr ||
        sequenceLength <= 0 || sequenceLength > capacity || sequenceLength > logicalLength) {
        setError("invalid request KV attention range");
        return 1;
    }
    return GarnetDevicePagedKVAttentionDeviceFP32(
        kvHandle,
        deviceQ,
        deviceOutput,
        sequenceLength,
        errorMessage,
        errorMessageCapacity);
}

extern "C" GARNET_ENTRY_EXPORT int GarnetRunTextKVCachedAttentionFP32(
    const float* q,
    const float* keyCache,
    const float* valueCache,
    float* output,
    int sequenceLength,
    int qHeads,
    int kvHeads,
    int headDim,
    char* errorMessage,
    int errorMessageCapacity)
{
    auto setError = [&](const char* message) {
        if (errorMessage != nullptr && errorMessageCapacity > 0) {
            std::snprintf(errorMessage, static_cast<size_t>(errorMessageCapacity), "%s", message ? message : "");
        }
    };
    if (q == nullptr || keyCache == nullptr || valueCache == nullptr || output == nullptr ||
        sequenceLength <= 0 || qHeads <= 0 || kvHeads <= 0 || headDim <= 0 || (qHeads % kvHeads) != 0) {
        setError("invalid cached attention arguments");
        return 1;
    }

    size_t qBytes = static_cast<size_t>(qHeads) * static_cast<size_t>(headDim) * sizeof(float);
    size_t kvBytes = static_cast<size_t>(sequenceLength) * static_cast<size_t>(kvHeads) * static_cast<size_t>(headDim) * sizeof(float);
    size_t outBytes = qBytes;
    float* dQ = nullptr;
    float* dK = nullptr;
    float* dV = nullptr;
    float* dOut = nullptr;
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err == cudaSuccess) err = cudaMalloc(&dQ, qBytes);
    if (err == cudaSuccess) err = cudaMalloc(&dK, kvBytes);
    if (err == cudaSuccess) err = cudaMalloc(&dV, kvBytes);
    if (err == cudaSuccess) err = cudaMalloc(&dOut, outBytes);
    if (err != cudaSuccess) {
        setError(cudaGetErrorString(err));
        if (dQ) cudaFree(dQ);
        if (dK) cudaFree(dK);
        if (dV) cudaFree(dV);
        if (dOut) cudaFree(dOut);
        if (stream) cudaStreamDestroy(stream);
        return 2;
    }

    err = cudaMemcpyAsync(dQ, q, qBytes, cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) err = cudaMemcpyAsync(dK, keyCache, kvBytes, cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) err = cudaMemcpyAsync(dV, valueCache, kvBytes, cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) {
        err = runTextKVCachedAttentionFP32(
            dQ, dK, dV, dOut, sequenceLength, qHeads, kvHeads, headDim, stream);
    }
    if (err == cudaSuccess) err = cudaMemcpyAsync(output, dOut, outBytes, cudaMemcpyDeviceToHost, stream);
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dOut);
    cudaStreamDestroy(stream);
    if (err != cudaSuccess) {
        setError(cudaGetErrorString(err));
        return 3;
    }
    setError("");
    return 0;
}

extern "C" GARNET_ENTRY_EXPORT int GarnetRunTextPagedKVCachedAttentionFP32(
    const float* q,
    const float* keyPages,
    const float* valuePages,
    const int* pageTable,
    float* output,
    int sequenceLength,
    int pageSize,
    int physicalPageCount,
    int qHeads,
    int kvHeads,
    int headDim,
    char* errorMessage,
    int errorMessageCapacity)
{
    auto setError = [&](const char* message) {
        if (errorMessage != nullptr && errorMessageCapacity > 0) {
            std::snprintf(errorMessage, static_cast<size_t>(errorMessageCapacity), "%s", message ? message : "");
        }
    };
    if (q == nullptr || keyPages == nullptr || valuePages == nullptr || pageTable == nullptr || output == nullptr ||
        sequenceLength <= 0 || pageSize <= 0 || physicalPageCount <= 0 ||
        qHeads <= 0 || kvHeads <= 0 || headDim <= 0 || (qHeads % kvHeads) != 0) {
        setError("invalid paged cached attention arguments");
        return 1;
    }

    int logicalPageCount = (sequenceLength + pageSize - 1) / pageSize;
    size_t qBytes = static_cast<size_t>(qHeads) * static_cast<size_t>(headDim) * sizeof(float);
    size_t pagesBytes = static_cast<size_t>(physicalPageCount) * static_cast<size_t>(pageSize)
        * static_cast<size_t>(kvHeads) * static_cast<size_t>(headDim) * sizeof(float);
    size_t tableBytes = static_cast<size_t>(logicalPageCount) * sizeof(int);
    size_t outBytes = qBytes;
    float* dQ = nullptr;
    float* dK = nullptr;
    float* dV = nullptr;
    float* dOut = nullptr;
    int* dPageTable = nullptr;
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err == cudaSuccess) err = cudaMalloc(&dQ, qBytes);
    if (err == cudaSuccess) err = cudaMalloc(&dK, pagesBytes);
    if (err == cudaSuccess) err = cudaMalloc(&dV, pagesBytes);
    if (err == cudaSuccess) err = cudaMalloc(&dPageTable, tableBytes);
    if (err == cudaSuccess) err = cudaMalloc(&dOut, outBytes);
    if (err != cudaSuccess) {
        setError(cudaGetErrorString(err));
        if (dQ) cudaFree(dQ);
        if (dK) cudaFree(dK);
        if (dV) cudaFree(dV);
        if (dPageTable) cudaFree(dPageTable);
        if (dOut) cudaFree(dOut);
        if (stream) cudaStreamDestroy(stream);
        return 2;
    }

    err = cudaMemcpyAsync(dQ, q, qBytes, cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) err = cudaMemcpyAsync(dK, keyPages, pagesBytes, cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) err = cudaMemcpyAsync(dV, valuePages, pagesBytes, cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) err = cudaMemcpyAsync(dPageTable, pageTable, tableBytes, cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) {
        err = runTextPagedKVCachedAttentionFP32(
            dQ, dK, dV, dPageTable, dOut, sequenceLength, pageSize, qHeads, kvHeads, headDim, stream);
    }
    if (err == cudaSuccess) err = cudaMemcpyAsync(output, dOut, outBytes, cudaMemcpyDeviceToHost, stream);
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dPageTable);
    cudaFree(dOut);
    cudaStreamDestroy(stream);
    if (err != cudaSuccess) {
        setError(cudaGetErrorString(err));
        return 3;
    }
    setError("");
    return 0;
}

extern "C" GARNET_ENTRY_EXPORT int GarnetRunTextPagedKVWriteFP32(
    const float* qkv,
    float* keyPages,
    float* valuePages,
    const int* pageTable,
    int tokenCount,
    int startPosition,
    int pageSize,
    int physicalPageCount,
    int qHeads,
    int kvHeads,
    int headDim,
    char* errorMessage,
    int errorMessageCapacity)
{
    auto setError = [&](const char* message) {
        if (errorMessage != nullptr && errorMessageCapacity > 0) {
            std::snprintf(errorMessage, static_cast<size_t>(errorMessageCapacity), "%s", message ? message : "");
        }
    };
    if (qkv == nullptr || keyPages == nullptr || valuePages == nullptr || pageTable == nullptr ||
        tokenCount <= 0 || startPosition < 0 || pageSize <= 0 || physicalPageCount <= 0 ||
        qHeads <= 0 || kvHeads <= 0 || headDim <= 0 || (qHeads % kvHeads) != 0) {
        setError("invalid paged KV write arguments");
        return 1;
    }

    int qWidth = qHeads * headDim;
    int kvWidth = kvHeads * headDim;
    int qkvStride = qWidth + 2 * kvWidth;
    int logicalPageCount = (startPosition + tokenCount + pageSize - 1) / pageSize;
    size_t qkvBytes = static_cast<size_t>(tokenCount) * static_cast<size_t>(qkvStride) * sizeof(float);
    size_t pagesBytes = static_cast<size_t>(physicalPageCount) * static_cast<size_t>(pageSize)
        * static_cast<size_t>(kvHeads) * static_cast<size_t>(headDim) * sizeof(float);
    size_t tableBytes = static_cast<size_t>(logicalPageCount) * sizeof(int);

    float* dQKV = nullptr;
    float* dK = nullptr;
    float* dV = nullptr;
    int* dPageTable = nullptr;
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err == cudaSuccess) err = cudaMalloc(&dQKV, qkvBytes);
    if (err == cudaSuccess) err = cudaMalloc(&dK, pagesBytes);
    if (err == cudaSuccess) err = cudaMalloc(&dV, pagesBytes);
    if (err == cudaSuccess) err = cudaMalloc(&dPageTable, tableBytes);
    if (err != cudaSuccess) {
        setError(cudaGetErrorString(err));
        if (dQKV) cudaFree(dQKV);
        if (dK) cudaFree(dK);
        if (dV) cudaFree(dV);
        if (dPageTable) cudaFree(dPageTable);
        if (stream) cudaStreamDestroy(stream);
        return 2;
    }

    err = cudaMemcpyAsync(dQKV, qkv, qkvBytes, cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) err = cudaMemcpyAsync(dK, keyPages, pagesBytes, cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) err = cudaMemcpyAsync(dV, valuePages, pagesBytes, cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) err = cudaMemcpyAsync(dPageTable, pageTable, tableBytes, cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) {
        err = runTextPagedKVWriteFP32(
            dQKV, dK, dV, dPageTable, tokenCount, startPosition, pageSize,
            qHeads, kvHeads, headDim, stream);
    }
    if (err == cudaSuccess) err = cudaMemcpyAsync(keyPages, dK, pagesBytes, cudaMemcpyDeviceToHost, stream);
    if (err == cudaSuccess) err = cudaMemcpyAsync(valuePages, dV, pagesBytes, cudaMemcpyDeviceToHost, stream);
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);

    cudaFree(dQKV);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dPageTable);
    cudaStreamDestroy(stream);
    if (err != cudaSuccess) {
        setError(cudaGetErrorString(err));
        return 3;
    }
    setError("");
    return 0;
}

namespace {
    struct DevicePagedKVFP32 {
        float* keyPages = nullptr;
        float* valuePages = nullptr;
        int* pageTable = nullptr;
        int physicalPageCount = 0;
        int logicalPageCount = 0;
        int pageSize = 0;
        int qHeads = 0;
        int kvHeads = 0;
        int headDim = 0;
    };

    std::mutex g_devicePagedKVCachesMutex;
    std::unordered_map<long long, DevicePagedKVFP32> g_devicePagedKVCaches;
    long long g_nextDevicePagedKVHandle = 1;

    void SetCError(char* errorMessage, int errorMessageCapacity, const char* message) {
        if (errorMessage != nullptr && errorMessageCapacity > 0) {
            std::snprintf(errorMessage, static_cast<size_t>(errorMessageCapacity), "%s", message ? message : "");
        }
    }

    bool GetDevicePagedKV(long long handle, DevicePagedKVFP32& out) {
        std::lock_guard<std::mutex> lock(g_devicePagedKVCachesMutex);
        auto found = g_devicePagedKVCaches.find(handle);
        if (found == g_devicePagedKVCaches.end()) {
            return false;
        }
        out = found->second;
        return true;
    }
}

extern "C" GARNET_ENTRY_EXPORT int GarnetCreateDevicePagedKVFP32(
    int physicalPageCount,
    int pageSize,
    int logicalPageCount,
    int qHeads,
    int kvHeads,
    int headDim,
    const int* pageTable,
    long long* outputHandle,
    char* errorMessage,
    int errorMessageCapacity)
{
    if (physicalPageCount <= 0 || pageSize <= 0 || logicalPageCount <= 0 ||
        qHeads <= 0 || kvHeads <= 0 || headDim <= 0 || (qHeads % kvHeads) != 0 ||
        pageTable == nullptr || outputHandle == nullptr) {
        SetCError(errorMessage, errorMessageCapacity, "invalid device paged KV create arguments");
        return 1;
    }

    DevicePagedKVFP32 cache;
    cache.physicalPageCount = physicalPageCount;
    cache.logicalPageCount = logicalPageCount;
    cache.pageSize = pageSize;
    cache.qHeads = qHeads;
    cache.kvHeads = kvHeads;
    cache.headDim = headDim;
    size_t pagesBytes = static_cast<size_t>(physicalPageCount) * static_cast<size_t>(pageSize)
        * static_cast<size_t>(kvHeads) * static_cast<size_t>(headDim) * sizeof(float);
    size_t tableBytes = static_cast<size_t>(logicalPageCount) * sizeof(int);
    cudaError_t err = cudaMalloc(&cache.keyPages, pagesBytes);
    if (err == cudaSuccess) err = cudaMalloc(&cache.valuePages, pagesBytes);
    if (err == cudaSuccess) err = cudaMalloc(&cache.pageTable, tableBytes);
    if (err == cudaSuccess) err = cudaMemset(cache.keyPages, 0, pagesBytes);
    if (err == cudaSuccess) err = cudaMemset(cache.valuePages, 0, pagesBytes);
    if (err == cudaSuccess) err = cudaMemcpy(cache.pageTable, pageTable, tableBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        if (cache.keyPages) cudaFree(cache.keyPages);
        if (cache.valuePages) cudaFree(cache.valuePages);
        if (cache.pageTable) cudaFree(cache.pageTable);
        SetCError(errorMessage, errorMessageCapacity, cudaGetErrorString(err));
        return 2;
    }

    long long handle = 0;
    {
        std::lock_guard<std::mutex> lock(g_devicePagedKVCachesMutex);
        handle = g_nextDevicePagedKVHandle++;
        g_devicePagedKVCaches.emplace(handle, cache);
    }
    *outputHandle = handle;
    SetCError(errorMessage, errorMessageCapacity, "");
    return 0;
}

extern "C" GARNET_ENTRY_EXPORT int GarnetDestroyDevicePagedKVFP32(
    long long handle,
    char* errorMessage,
    int errorMessageCapacity)
{
    DevicePagedKVFP32 cache;
    {
        std::lock_guard<std::mutex> lock(g_devicePagedKVCachesMutex);
        auto found = g_devicePagedKVCaches.find(handle);
        if (found == g_devicePagedKVCaches.end()) {
            SetCError(errorMessage, errorMessageCapacity, "invalid device paged KV handle");
            return 1;
        }
        cache = found->second;
        g_devicePagedKVCaches.erase(found);
    }
    if (cache.keyPages) cudaFree(cache.keyPages);
    if (cache.valuePages) cudaFree(cache.valuePages);
    if (cache.pageTable) cudaFree(cache.pageTable);
    SetCError(errorMessage, errorMessageCapacity, "");
    return 0;
}

extern "C" GARNET_ENTRY_EXPORT int GarnetDevicePagedKVWriteFP32(
    long long handle,
    const float* qkv,
    int tokenCount,
    int startPosition,
    char* errorMessage,
    int errorMessageCapacity)
{
    DevicePagedKVFP32 cache;
    if (!GetDevicePagedKV(handle, cache)) {
        SetCError(errorMessage, errorMessageCapacity, "invalid device paged KV handle");
        return 1;
    }
    if (qkv == nullptr || tokenCount <= 0 || startPosition < 0 ||
        startPosition + tokenCount > cache.logicalPageCount * cache.pageSize) {
        SetCError(errorMessage, errorMessageCapacity, "invalid device paged KV write arguments");
        return 2;
    }
    int qWidth = cache.qHeads * cache.headDim;
    int kvWidth = cache.kvHeads * cache.headDim;
    int qkvStride = qWidth + 2 * kvWidth;
    size_t qkvBytes = static_cast<size_t>(tokenCount) * static_cast<size_t>(qkvStride) * sizeof(float);
    float* dQKV = nullptr;
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err == cudaSuccess) err = cudaMalloc(&dQKV, qkvBytes);
    if (err == cudaSuccess) err = cudaMemcpyAsync(dQKV, qkv, qkvBytes, cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) {
        err = runTextPagedKVWriteFP32(
            dQKV, cache.keyPages, cache.valuePages, cache.pageTable,
            tokenCount, startPosition, cache.pageSize, cache.qHeads, cache.kvHeads, cache.headDim, stream);
    }
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
    if (dQKV) cudaFree(dQKV);
    if (stream) cudaStreamDestroy(stream);
    if (err != cudaSuccess) {
        SetCError(errorMessage, errorMessageCapacity, cudaGetErrorString(err));
        return 3;
    }
    SetCError(errorMessage, errorMessageCapacity, "");
    return 0;
}

extern "C" GARNET_ENTRY_EXPORT int GarnetDevicePagedKVWriteDeviceFP32(
    long long handle,
    const float* deviceQKV,
    int tokenCount,
    int startPosition,
    char* errorMessage,
    int errorMessageCapacity)
{
    DevicePagedKVFP32 cache;
    if (!GetDevicePagedKV(handle, cache)) {
        SetCError(errorMessage, errorMessageCapacity, "invalid device paged KV handle");
        return 1;
    }
    if (deviceQKV == nullptr || tokenCount <= 0 || startPosition < 0 ||
        startPosition + tokenCount > cache.logicalPageCount * cache.pageSize) {
        SetCError(errorMessage, errorMessageCapacity, "invalid device paged KV device-write arguments");
        return 2;
    }

    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err == cudaSuccess) {
        err = runTextPagedKVWriteFP32(
            deviceQKV,
            cache.keyPages,
            cache.valuePages,
            cache.pageTable,
            tokenCount,
            startPosition,
            cache.pageSize,
            cache.qHeads,
            cache.kvHeads,
            cache.headDim,
            stream);
    }
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
    if (stream) cudaStreamDestroy(stream);
    if (err != cudaSuccess) {
        SetCError(errorMessage, errorMessageCapacity, cudaGetErrorString(err));
        return 3;
    }
    SetCError(errorMessage, errorMessageCapacity, "");
    return 0;
}

extern "C" GARNET_ENTRY_EXPORT int GarnetDevicePagedKVAttentionFP32(
    long long handle,
    const float* q,
    float* output,
    int sequenceLength,
    char* errorMessage,
    int errorMessageCapacity)
{
    DevicePagedKVFP32 cache;
    if (!GetDevicePagedKV(handle, cache)) {
        SetCError(errorMessage, errorMessageCapacity, "invalid device paged KV handle");
        return 1;
    }
    if (q == nullptr || output == nullptr || sequenceLength <= 0 ||
        sequenceLength > cache.logicalPageCount * cache.pageSize) {
        SetCError(errorMessage, errorMessageCapacity, "invalid device paged KV attention arguments");
        return 2;
    }
    size_t qBytes = static_cast<size_t>(cache.qHeads) * static_cast<size_t>(cache.headDim) * sizeof(float);
    float* dQ = nullptr;
    float* dOut = nullptr;
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err == cudaSuccess) err = cudaMalloc(&dQ, qBytes);
    if (err == cudaSuccess) err = cudaMalloc(&dOut, qBytes);
    if (err == cudaSuccess) err = cudaMemcpyAsync(dQ, q, qBytes, cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) {
        err = runTextPagedKVCachedAttentionFP32(
            dQ, cache.keyPages, cache.valuePages, cache.pageTable, dOut,
            sequenceLength, cache.pageSize, cache.qHeads, cache.kvHeads, cache.headDim, stream);
    }
    if (err == cudaSuccess) err = cudaMemcpyAsync(output, dOut, qBytes, cudaMemcpyDeviceToHost, stream);
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
    if (dQ) cudaFree(dQ);
    if (dOut) cudaFree(dOut);
    if (stream) cudaStreamDestroy(stream);
    if (err != cudaSuccess) {
        SetCError(errorMessage, errorMessageCapacity, cudaGetErrorString(err));
        return 3;
    }
    SetCError(errorMessage, errorMessageCapacity, "");
    return 0;
}

extern "C" GARNET_ENTRY_EXPORT int GarnetDevicePagedKVAttentionDeviceFP32(
    long long handle,
    const float* deviceQ,
    float* deviceOutput,
    int sequenceLength,
    char* errorMessage,
    int errorMessageCapacity)
{
    DevicePagedKVFP32 cache;
    if (!GetDevicePagedKV(handle, cache)) {
        SetCError(errorMessage, errorMessageCapacity, "invalid device paged KV handle");
        return 1;
    }
    if (deviceQ == nullptr || deviceOutput == nullptr || sequenceLength <= 0 ||
        sequenceLength > cache.logicalPageCount * cache.pageSize) {
        SetCError(errorMessage, errorMessageCapacity, "invalid device paged KV device-attention arguments");
        return 2;
    }

    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err == cudaSuccess) {
        err = runTextPagedKVCachedAttentionFP32(
            deviceQ,
            cache.keyPages,
            cache.valuePages,
            cache.pageTable,
            deviceOutput,
            sequenceLength,
            cache.pageSize,
            cache.qHeads,
            cache.kvHeads,
            cache.headDim,
            stream);
    }
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
    if (stream) cudaStreamDestroy(stream);
    if (err != cudaSuccess) {
        SetCError(errorMessage, errorMessageCapacity, cudaGetErrorString(err));
        return 3;
    }
    SetCError(errorMessage, errorMessageCapacity, "");
    return 0;
}

extern "C" GARNET_ENTRY_EXPORT int GarnetDebugSampleLogitsTop1FP32(
    const float* deviceLogits,
    int rows,
    int vocabSize,
    long long* deviceOutputTokenId,
    float* deviceOutputTokenValue,
    char* errorMessage,
    int errorMessageCapacity)
{
    if (deviceLogits == nullptr || deviceOutputTokenId == nullptr || rows <= 0 || vocabSize <= 0) {
        SetCError(errorMessage, errorMessageCapacity, "invalid logits top-1 arguments");
        return 1;
    }

    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err == cudaSuccess) {
        err = runLogitsTop1FP32(
            deviceLogits,
            deviceOutputTokenId,
            deviceOutputTokenValue,
            rows,
            vocabSize,
            stream);
    }
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
    if (stream) cudaStreamDestroy(stream);
    if (err != cudaSuccess) {
        SetCError(errorMessage, errorMessageCapacity, cudaGetErrorString(err));
        return 2;
    }
    SetCError(errorMessage, errorMessageCapacity, "");
    return 0;
}

namespace Garnet
{
    std::string GarnetAPI::AvailableModelsJson(
        const std::string& catalogRoot) const
    {
        return Garnet::EnumerateAvailableModelsJson(
            ResolveModelCatalogRoot(catalogRoot, m_baseFolder));
    }

    std::string GarnetAPI::LoadedModelsJson()
    {
        std::lock_guard<std::mutex> guard(m_servingMutex);
        json response = {
            {"schema_version", 1},
            {"serving_mode", "multi_instance"},
            {"models", json::array()}
        };
        for (const auto& item : m_servingInstances) {
            const auto& instance = item.second;
            std::lock_guard<std::mutex> instanceGuard(instance->mutex);
            json model = GarnetServingStatus(
                instance->model,
                instance->modelRoot,
                instance->modelId,
                instance->inputCapability,
                instance->error);
            model["instance_id"] = instance->modelId;
            model["device_id"] = instance->deviceId;
            response["models"].push_back(std::move(model));
        }
        return response.dump();
    }

    X::Value GarnetAPI::ListAvailableModelsJson(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        const std::string catalogRoot = params.size() == 0
            ? std::string()
            : params[0].ToString();
        retValue = NativeValue(Host(), AvailableModelsJson(catalogRoot));
        return retValue;
    }

    X::Value GarnetAPI::ListLoadedModelsJson(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        retValue = NativeValue(Host(), LoadedModelsJson());
        return retValue;
    }

    X::Value GarnetAPI::ServeModel(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() == 0) {
            retValue = NativeValue(Host(), GarnetJsonError("model_root_required",
                "serve_model requires a Qwen model root"));
            return retValue;
        }
        namespace fs = std::filesystem;
        const fs::path modelRoot = fs::path(params[0].ToString());
        const fs::path xmodelRoot = params.size() > 1 && !params[1].ToString().empty()
            ? fs::path(params[1].ToString())
            : modelRoot / "xmodel";
        const std::string requestedModelId = params.size() > 4
            ? params[4].ToString()
            : std::string();
        const bool asrModel = requestedModelId == "Qwen3-ASR-0.6B";
        const bool tts06Model = requestedModelId ==
            "Qwen3-TTS-12Hz-0.6B-CustomVoice";
        const bool tts17Model = requestedModelId ==
            "Qwen3-TTS-12Hz-1.7B-CustomVoice";
        const bool ttsModel = tts06Model || tts17Model;
        const bool textModel = !asrModel && !ttsModel && (
            requestedModelId == "Qwen3-1.7B" ||
            (requestedModelId.empty() &&
                !fs::is_regular_file(xmodelRoot / "qwen_vl_prefill.py") &&
                fs::is_regular_file(xmodelRoot / "prefill.py")));
        const std::string modelId = asrModel
            ? "Qwen3-ASR-0.6B"
            : (ttsModel ? requestedModelId :
                (textModel ? "Qwen3-1.7B" : "Qwen3-VL-2B-Instruct"));
        if (!requestedModelId.empty() && requestedModelId != modelId) {
            retValue = NativeValue(Host(), GarnetJsonError(
                "model_unsupported",
                "The requested Garnet serving model is not supported"));
            return retValue;
        }
        const fs::path xmodelPath = xmodelRoot /
            (ttsModel ? "talker_prefill.py" :
                ((textModel || asrModel) ? "prefill.py" : "qwen_vl_prefill.py"));
        const fs::path cacheRoot = params.size() > 2 && !params[2].ToString().empty()
            ? fs::path(params[2].ToString())
            : modelRoot / "compiled_cache";
        int maxInputTokens = 1536;
        int patchCount = 3772;
        int kvPages = 128;
        int minPixels = 256 * 28 * 28;
        int maxPixels = 1280 * 28 * 28;
        int maxOutputTokens = 256;
        int audioChunks = 5;
        int deviceId = 0;
        if (params.size() > 3 && !params[3].ToString().empty()) {
            try {
                const json profile = json::parse(params[3].ToString());
                maxInputTokens = profile.value("maxInputTokens", maxInputTokens);
                patchCount = profile.value("patchCount", patchCount);
                kvPages = profile.value("kvPages", kvPages);
                minPixels = profile.value("minPixels", minPixels);
                maxPixels = profile.value("maxPixels", maxPixels);
                maxOutputTokens = profile.value(
                    "maxOutputTokens", maxOutputTokens);
                audioChunks = profile.value("audioChunks", audioChunks);
                deviceId = profile.value("deviceId", deviceId);
            }
            catch (const std::exception&) {
                retValue = NativeValue(Host(), GarnetJsonError(
                    "profile_invalid", "The Garnet inference profile is invalid"));
                return retValue;
            }
        }
        const bool fastProfile =
            !textModel &&
            maxInputTokens == 512 && patchCount == 240 && kvPages == 48 &&
            minPixels == 65536 && maxPixels == 65536;
        const bool textProfile = textModel &&
            maxInputTokens >= 128 && maxInputTokens <= 32768 &&
            kvPages >= 16 && kvPages <= 4096;
        const bool visionProfile = !textModel &&
            maxInputTokens >= 128 && maxInputTokens <= 32768 &&
            patchCount >= 1 && patchCount <= 8192 &&
            kvPages >= 16 && kvPages <= 4096 &&
            minPixels >= 1 && maxPixels >= minPixels;
        const bool asrProfile = asrModel &&
            maxInputTokens >= 128 && maxInputTokens <= 2048 &&
            kvPages >= 16 && kvPages <= 256 &&
            audioChunks >= 1 && audioChunks <= 30;
        const bool ttsProfile = ttsModel &&
            maxInputTokens >= 128 && maxInputTokens <= 2048 &&
            kvPages >= 16 && kvPages <= 256;
        if ((!asrProfile && !ttsProfile && !textProfile && !fastProfile && !visionProfile) ||
            maxOutputTokens < 1 || maxOutputTokens > 32768 || deviceId < 0) {
            retValue = NativeValue(Host(), GarnetJsonError(
                "profile_unsupported",
                "The requested Garnet inference profile is not supported"));
            return retValue;
        }
        if (!fs::is_regular_file(xmodelPath)) {
            retValue = NativeValue(Host(), GarnetJsonError("xmodel_missing",
                "The Python model entry was not found: " + xmodelPath.string()));
            return retValue;
        }
        const bool hasTokenizer =
            fs::is_regular_file(modelRoot / "tokenizer.json") ||
            (fs::is_regular_file(modelRoot / "vocab.json") &&
             fs::is_regular_file(modelRoot / "merges.txt"));
        if (!fs::is_regular_file(modelRoot / "config.json") || !hasTokenizer) {
            retValue = NativeValue(Host(), GarnetJsonError("model_incomplete",
                "The Qwen model configuration or tokenizer is missing"));
            return retValue;
        }

        std::lock_guard<std::mutex> guard(m_servingMutex);
        const auto existing = m_servingInstances.find(modelId);
        if (existing != m_servingInstances.end() && existing->second->model.IsValid()) {
            const auto& instance = existing->second;
            if (instance->modelRoot != modelRoot.string() ||
                instance->cacheRoot != cacheRoot.string()) {
                retValue = NativeValue(Host(), GarnetJsonError(
                    "model_instance_conflict",
                    "The model UID is already loaded from a different root"));
                return retValue;
            }
            m_defaultServingModelId = modelId;
            retValue = NativeValue(Host(), GarnetServingStatus(
                instance->model, instance->modelRoot, instance->modelId,
                instance->inputCapability, instance->error, rt).dump());
            return retValue;
        }
        auto instance = std::make_shared<ServingInstance>();
        instance->modelRoot = modelRoot.string();
        instance->cacheRoot = cacheRoot.string();
        instance->modelId = modelId;
        instance->inputCapability = asrModel
            ? "audio" : (ttsModel ? "speech" : (textModel ? "text" : "vision"));
        instance->minPixels = minPixels;
        instance->maxPixels = maxPixels;
        instance->maxOutputTokens = maxOutputTokens;
        instance->deviceId = deviceId;
        try {
            int deviceCount = 0;
            if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceId >= deviceCount)
                throw std::runtime_error("CUDA device is unavailable: " + std::to_string(deviceId));
            CudaDeviceScope deviceScope(deviceId);
            fs::create_directories(cacheRoot);
            auto modelValue = CallChecked(__xlang3_package_->GetValue("model"));
            Model& model = *modelValue.NativeData<Model>();
            auto emptyWeights = X::Value::Dict(Host());
            std::string modelDirectory = xmodelPath.parent_path().string();
            std::string emptyString;
            model.SetInfo(modelDirectory, emptyString, emptyString, emptyWeights);
            const std::vector<std::vector<int>> inputShapes = asrModel
                ? std::vector<std::vector<int>>{
                    {1, maxInputTokens}, {audioChunks, 1, 128, 100},
                    {audioChunks + 1}, {3, 1, maxInputTokens},
                    {1, maxInputTokens}, {28, kvPages, 16, 8, 128},
                    {28, kvPages, 16, 8, 128}, {kvPages}, {1}}
                : ttsModel
                ? std::vector<std::vector<int>>{
                    {1, maxInputTokens, 2}, {3, 1, maxInputTokens},
                    {1, maxInputTokens}, {28, kvPages, 16, 8, 128},
                    {28, kvPages, 16, 8, 128}, {kvPages}, {1}}
                : textModel
                ? std::vector<std::vector<int>>{
                    {1, maxInputTokens}, {1, 1, maxInputTokens},
                    {1, maxInputTokens},
                    {28, kvPages, 16, 8, 128},
                    {28, kvPages, 16, 8, 128},
                    {kvPages}, {1}}
                : std::vector<std::vector<int>>{
                    {1, maxInputTokens}, {patchCount, 1536}, {1, 3},
                    {patchCount, 4}, {patchCount, 4}, {patchCount, 2}, {2},
                    {1, maxInputTokens}, {1, maxInputTokens},
                    {3, 1, maxInputTokens}, {1, 1},
                    {28, kvPages, 16, 8, 128}, {28, kvPages, 16, 8, 128},
                    {kvPages}, {1}};
            const std::vector<std::string> inputDataTypes = asrModel
                ? std::vector<std::string>{
                    "int64", "float32", "int32", "int64", "int64",
                    "bfloat16", "bfloat16", "int32", "int32"}
                : ttsModel
                ? std::vector<std::string>{
                    "int64", "int64", "int64", "bfloat16", "bfloat16",
                    "int32", "int32"}
                : textModel
                ? std::vector<std::string>{
                    "int64", "int64", "int64", "bfloat16", "bfloat16",
                    "int32", "int32"}
                : std::vector<std::string>{
                    "int64", "bfloat16", "int64", "int64", "bfloat16",
                    "int64", "int32", "int64", "int64", "int64", "int64",
                    "bfloat16", "bfloat16", "int32", "int32"};
            FusionPartitionOptions partitionOptions;
            partitionOptions.builderWorkspaceBytes = 4096ULL << 20;
            if (!model.InitializeCompiledRuntime(
                    xmodelPath.string(), cacheRoot.string(), modelRoot.string(),
                    asrModel ? "Qwen3ASRPrefill" :
                        (ttsModel ? "Qwen3TTSTalkerPrefill" :
                            (textModel ? "Qwen3Prefill" : "Qwen3VLPrefill")),
                    asrModel ? "qwen3_asr" :
                        (ttsModel ? "qwen3_tts" :
                            (textModel ? "qwen3_text" : "qwen3_vl")),
                    inputShapes, inputDataTypes,
                    partitionOptions)) {
                std::string initializationError =
                    "Garnet failed to initialize the compiled Qwen runtime";
                X::Value statusValue = model.CompiledRuntimeStatus();
                if (statusValue.IsDict()) {
                    X::Value status(statusValue);
                    const std::string detail = FindField(status, "error_message").ToString();
                    const std::string code = FindField(status, "error_code").ToString();
                    if (!detail.empty()) {
                        initializationError += code.empty()
                            ? ": " + detail
                            : ": " + code + ": " + detail;
                    }
                }
                TRTBuilder::ReleaseCachedExecutions(cacheRoot.string());
                instance->error = initializationError;
                retValue = NativeValue(Host(), GarnetJsonError("model_load_failed", instance->error));
                return retValue;
            }
            instance->model = X::Value(modelValue);
            instance->error.clear();
            m_servingInstances[modelId] = instance;
            m_defaultServingModelId = modelId;
            retValue = NativeValue(Host(), GarnetServingStatus(
                instance->model, instance->modelRoot, instance->modelId,
                instance->inputCapability, instance->error, rt
            ).dump());
        }
        catch (const std::exception& exception) {
            TRTBuilder::ReleaseCachedExecutions(cacheRoot.string());
            instance->error = exception.what();
            retValue = NativeValue(Host(), GarnetJsonError("model_load_failed", instance->error));
        }
        return retValue;
    }

    X::Value GarnetAPI::ServeStatusJson(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        const std::string modelId = params.size() > 0 ? params[0].ToString() : std::string();
        const auto instance = FindServingInstance(modelId);
        if (!instance) {
            retValue = NativeValue(Host(), GarnetJsonError("serving_not_ready",
                modelId.empty() ? "Garnet serving is not started" : "The requested Garnet model is not loaded"));
            return retValue;
        }
        std::lock_guard<std::mutex> guard(instance->mutex);
        retValue = NativeValue(Host(), GarnetServingStatus(
            instance->model, instance->modelRoot, instance->modelId,
            instance->inputCapability, instance->error, rt
        ).dump());
        return retValue;
    }

    X::Value GarnetAPI::InferJson(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() == 0) {
            retValue = NativeValue(Host(), GarnetJsonError("request_invalid",
                "infer_json requires a prompt"));
            return retValue;
        }
        const std::string prompt = params[0].ToString();
        X::Value imageSource = params.size() > 1 ? params[1] : X::Value();
        const bool hasImageBinary =
            x3_value_object_kind(imageSource.raw()) == X3_OBJECT_KIND_BYTES && imageSource.Size() > 0;
        const bool hasImagePath =
            !hasImageBinary && !imageSource.ToString().empty();
        const int requestedMaxNewTokens = params.size() > 2
            ? CheckedInt(params[2], "argument")
            : 0;
        const std::string requestedModelId = params.size() > 3
            ? params[3].ToString() : std::string();
        if (prompt.empty()) {
            retValue = NativeValue(Host(), GarnetJsonError("request_invalid",
                "Garnet inference requires a prompt"));
            return retValue;
        }
        const auto instance = FindServingInstance(requestedModelId);
        if (!instance) {
            retValue = NativeValue(Host(), GarnetJsonError("serving_not_ready",
                requestedModelId.empty() ? "Garnet serving is not started" :
                    "The requested Garnet model is not loaded"));
            return retValue;
        }
        std::lock_guard<std::mutex> guard(instance->mutex);
        CudaDeviceScope deviceScope(instance->deviceId);
        if (instance->inputCapability == "vision" &&
            !hasImageBinary && !hasImagePath) {
            retValue = NativeValue(Host(), GarnetJsonError("request_invalid",
                "Garnet Qwen-VL inference requires JPEG binary data or an image path"));
            return retValue;
        }
        const int maxNewTokens = requestedMaxNewTokens > 0
            ? (std::max)(1, (std::min)(
                instance->maxOutputTokens, requestedMaxNewTokens))
            : instance->maxOutputTokens;
        if (!instance->model.IsValid()) {
            retValue = NativeValue(Host(), GarnetJsonError("serving_not_ready",
                instance->error.empty() ? "Garnet serving is not started" : instance->error));
            return retValue;
        }
        X::Value forwardCallable = instance->model["forward"];
        if (!forwardCallable.IsObject()) {
            retValue = NativeValue(Host(), GarnetJsonError("serving_not_ready",
                "Garnet serving model handle is invalid"));
            return retValue;
        }
        auto request = X::Value::Dict(Host());
        if (instance->inputCapability == "vision") {
            request.SetItem("image", imageSource);
            request.SetItem("min_pixels", X::Value(instance->minPixels));
            request.SetItem("max_pixels", X::Value(instance->maxPixels));
        }
        else {
            request.SetItem("enable_thinking", X::Value(0));
        }
        request.SetItem("prompt", X::Value::String(Host(), prompt));
        request.SetItem("max_new_tokens", X::Value(maxNewTokens));
        request.SetItem("reuse_output", X::Value(1));
        X::Value resultValue = CallChecked(forwardCallable, request);
        if (!resultValue.IsDict()) {
            retValue = NativeValue(Host(), GarnetJsonError("inference_failed",
                "Garnet returned an invalid inference result"));
            return retValue;
        }
        X::Value result(resultValue);
        json response = {
            {"status", FindField(result, "status").ToString()},
            {"model_id", instance->modelId},
            {"text", FindField(result, "text").ToString()},
            {"error_code", FindField(result, "error_code").ToString()},
            {"error_message", FindField(result, "error_message").ToString()},
            {"prompt_tokens", FindField(result, "prompt_token_count").IsValid()
                ? FindField(result, "prompt_token_count").ToLongLong() : 0},
            {"output_tokens", FindField(result, "generated_token_count").IsValid()
                ? FindField(result, "generated_token_count").ToLongLong() : 0},
            {"visual_tokens", FindField(result, "visual_token_count").IsValid()
                ? FindField(result, "visual_token_count").ToLongLong() : 0},
            {"duration_ms", FindField(result, "total_ms").IsValid()
                ? FindField(result, "total_ms").ToDouble() : 0.0},
            {"time_to_first_token_ms", FindField(result, "time_to_first_token_ms").IsValid()
                ? FindField(result, "time_to_first_token_ms").ToDouble() : 0.0},
            {"tokens_per_second", FindField(result, "decode_tokens_per_second").IsValid()
                ? FindField(result, "decode_tokens_per_second").ToDouble() : 0.0}
        };
        retValue = NativeValue(Host(), response.dump());
        return retValue;
    }

    X::Value GarnetAPI::TranscribeJson(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() == 0) {
            retValue = NativeValue(Host(), GarnetJsonError("request_invalid",
                "transcribe_json requires WAV audio"));
            return retValue;
        }
        X::Value audioSource = params[0];
        const bool hasAudioBinary =
            x3_value_object_kind(audioSource.raw()) == X3_OBJECT_KIND_BYTES && audioSource.Size() > 0;
        const bool hasAudioPath =
            !hasAudioBinary && !audioSource.ToString().empty();
        if (!hasAudioBinary && !hasAudioPath) {
            retValue = NativeValue(Host(), GarnetJsonError("request_invalid",
                "Garnet ASR requires WAV binary data or an audio path"));
            return retValue;
        }
        const std::string context = params.size() > 1
            ? params[1].ToString() : std::string();
        const std::string language = params.size() > 2
            ? params[2].ToString() : std::string("English");
        const int requestedMaxNewTokens = params.size() > 3
            ? CheckedInt(params[3], "argument") : 0;

        const auto instance = FindServingInstance(std::string(), "audio");
        if (!instance) {
            retValue = NativeValue(Host(), GarnetJsonError("serving_not_ready",
                "No loaded Garnet model accepts audio"));
            return retValue;
        }
        std::lock_guard<std::mutex> guard(instance->mutex);
        CudaDeviceScope deviceScope(instance->deviceId);
        if (!instance->model.IsValid()) {
            retValue = NativeValue(Host(), GarnetJsonError("serving_not_ready",
                instance->error.empty() ? "Garnet serving is not started" : instance->error));
            return retValue;
        }
        X::Value forwardCallable = instance->model["forward"];
        if (!forwardCallable.IsObject()) {
            retValue = NativeValue(Host(), GarnetJsonError("serving_not_ready",
                "Garnet serving model handle is invalid"));
            return retValue;
        }
        const int maxNewTokens = requestedMaxNewTokens > 0
            ? (std::max)(1, (std::min)(
                instance->maxOutputTokens, requestedMaxNewTokens))
            : instance->maxOutputTokens;
        auto request = X::Value::Dict(Host());
        request.SetItem("audio", audioSource);
        request.SetItem("context", X::Value::String(Host(), context));
        request.SetItem("language", X::Value::String(Host(), language));
        request.SetItem("max_new_tokens", X::Value(maxNewTokens));
        request.SetItem("reuse_output", X::Value(1));
        X::Value resultValue = CallChecked(forwardCallable, request);
        if (!resultValue.IsDict()) {
            retValue = NativeValue(Host(), GarnetJsonError("inference_failed",
                "Garnet returned an invalid ASR result"));
            return retValue;
        }
        X::Value result(resultValue);
        json response = {
            {"status", FindField(result, "status").ToString()},
            {"model_id", instance->modelId},
            {"text", FindField(result, "text").ToString()},
            {"error_code", FindField(result, "error_code").ToString()},
            {"error_message", FindField(result, "error_message").ToString()},
            {"prompt_tokens", FindField(result, "prompt_token_count").IsValid()
                ? FindField(result, "prompt_token_count").ToLongLong() : 0},
            {"output_tokens", FindField(result, "generated_token_count").IsValid()
                ? FindField(result, "generated_token_count").ToLongLong() : 0},
            {"audio_tokens", FindField(result, "audio_token_count").IsValid()
                ? FindField(result, "audio_token_count").ToLongLong() : 0},
            {"audio_samples", FindField(result, "audio_sample_count").IsValid()
                ? FindField(result, "audio_sample_count").ToLongLong() : 0},
            {"audio_duration_seconds", FindField(result, "audio_duration_seconds").IsValid()
                ? FindField(result, "audio_duration_seconds").ToDouble() : 0.0},
            {"duration_ms", FindField(result, "total_ms").IsValid()
                ? FindField(result, "total_ms").ToDouble() : 0.0}
        };
        retValue = NativeValue(Host(), response.dump());
        return retValue;
    }

    X::Value GarnetAPI::SynthesizeJson(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() < 5 || params[0].ToString().empty() ||
            params[4].ToString().empty()) {
            retValue = NativeValue(Host(), GarnetJsonError("request_invalid",
                "synthesize_json requires text, speaker, language, max frames, and output path"));
            return retValue;
        }
        const std::string text = params[0].ToString();
        const std::string speaker = params[1].ToString();
        const std::string language = params[2].ToString().empty()
            ? std::string("English") : params[2].ToString();
        const int requestedFrames = (std::max)(1,
            CheckedInt(params[3], "argument"));
        const std::filesystem::path outputPath(params[4].ToString());

        const auto instance = FindServingInstance(std::string(), "speech");
        if (!instance) {
            retValue = NativeValue(Host(), GarnetJsonError("serving_not_ready",
                "No loaded Garnet model synthesizes speech"));
            return retValue;
        }
        std::lock_guard<std::mutex> guard(instance->mutex);
        CudaDeviceScope deviceScope(instance->deviceId);
        if (!instance->model.IsValid()) {
            retValue = NativeValue(Host(), GarnetJsonError("serving_not_ready",
                instance->error.empty() ? "Garnet serving is not started" : instance->error));
            return retValue;
        }
        X::Value forwardCallable = instance->model["forward"];
        if (!forwardCallable.IsObject()) {
            retValue = NativeValue(Host(), GarnetJsonError("serving_not_ready",
                "Garnet serving model handle is invalid"));
            return retValue;
        }
        const int maxFrames = (std::min)(instance->maxOutputTokens,
            requestedFrames);
        auto request = X::Value::Dict(Host());
        request.SetItem("text", X::Value::String(Host(), text));
        request.SetItem("speaker", X::Value::String(Host(), speaker));
        request.SetItem("language", X::Value::String(Host(), language));
        request.SetItem("max_audio_frames", X::Value(maxFrames));
        request.SetItem("reuse_output", X::Value(1));
        if (params.size() > 6 && !params[6].ToString().empty()) {
            request.SetItem("instruct", X::Value::String(Host(), params[6].ToString()));
        }
        if (params.size() > 5 && !params[5].ToString().empty()) {
            try {
                const json sampling = json::parse(params[5].ToString());
                if (sampling.contains("do_sample")) {
                    request.SetItem("do_sample", X::Value(
                        sampling.at("do_sample").get<bool>() ? 1 : 0));
                }
                if (sampling.contains("top_k")) {
                    request.SetItem("top_k", X::Value(
                        sampling.at("top_k").get<int>()));
                }
                if (sampling.contains("temperature")) {
                    request.SetItem("temperature", X::Value(
                        sampling.at("temperature").get<double>()));
                }
                if (sampling.contains("repetition_penalty")) {
                    request.SetItem("repetition_penalty", X::Value(
                        sampling.at("repetition_penalty").get<double>()));
                }
                if (sampling.contains("seed")) {
                    request.SetItem("seed", X::Value(
                        sampling.at("seed").get<long long>()));
                }
            }
            catch (const std::exception& exception) {
                retValue = NativeValue(Host(), GarnetJsonError("request_invalid",
                    std::string("invalid TTS sampling options: ") + exception.what()));
                return retValue;
            }
        }
        X::Value resultValue = CallChecked(forwardCallable, request);
        if (!resultValue.IsDict()) {
            retValue = NativeValue(Host(), GarnetJsonError("inference_failed",
                "Garnet returned an invalid TTS result"));
            return retValue;
        }
        X::Value result(resultValue);
        if (FindField(result, "status").ToString() != "ok") {
            retValue = NativeValue(Host(), json({
                {"status", "error"},
                {"error_code", FindField(result, "error_code").ToString()},
                {"error_message", FindField(result, "error_message").ToString()}
            }).dump());
            return retValue;
        }
        X::Value audioValue = FindField(result, "audio");
        if (!X::Tensor::IsTensor(audioValue)) {
            retValue = NativeValue(Host(), GarnetJsonError("waveform_invalid",
                "Garnet TTS returned no waveform tensor"));
            return retValue;
        }
        X::Tensor gpuAudio(audioValue);
        X::Value cpuValue = TensorHelper::CopyToCPUTensor(gpuAudio);
        if (!X::Tensor::IsTensor(cpuValue)) {
            retValue = NativeValue(Host(), GarnetJsonError("waveform_download_failed",
                "Garnet could not copy the waveform from the GPU"));
            return retValue;
        }
        X::Tensor cpuAudio(cpuValue);
        auto audioUse = cpuAudio.Acquire();
        const long long reportedSamples = FindField(result, "audio_sample_count").IsValid()
            ? FindField(result, "audio_sample_count").ToLongLong() : 0;
        const long long tensorSamples = cpuAudio.Info().dtype ==
            X3_TENSOR_FLOAT32
            ? cpuAudio.Info().byte_size / static_cast<long long>(sizeof(float))
            : cpuAudio.Info().byte_size / static_cast<long long>(sizeof(uint16_t));
        const size_t sampleCount = static_cast<size_t>((std::max)(
            0LL, (std::min)(reportedSamples, tensorSamples)));
        if (!cpuAudio.Info().data || sampleCount == 0 ||
            (cpuAudio.Info().dtype != X3_TENSOR_FLOAT32 &&
             cpuAudio.Info().dtype != X3_TENSOR_BFLOAT16)) {
            retValue = NativeValue(Host(), GarnetJsonError("waveform_invalid",
                "Garnet TTS returned an unsupported waveform tensor"));
            return retValue;
        }
        std::vector<int16_t> pcm(sampleCount);
        for (size_t index = 0; index < sampleCount; ++index) {
            float value = 0.0F;
            if (cpuAudio.Info().dtype == X3_TENSOR_FLOAT32) {
                value = static_cast<const float*>(
                    static_cast<const void*>(cpuAudio.Info().data))[index];
            }
            else {
                const uint16_t bits = static_cast<const uint16_t*>(
                    static_cast<const void*>(cpuAudio.Info().data))[index];
                const uint32_t expanded = static_cast<uint32_t>(bits) << 16;
                std::memcpy(&value, &expanded, sizeof(value));
            }
            value = (std::max)(-1.0F, (std::min)(1.0F, value));
            pcm[index] = static_cast<int16_t>(std::lround(value * 32767.0F));
        }
        try {
            if (!outputPath.parent_path().empty()) {
                std::filesystem::create_directories(outputPath.parent_path());
            }
            std::ofstream wav(outputPath, std::ios::binary | std::ios::trunc);
            if (!wav) throw std::runtime_error("cannot open output WAV");
            const uint32_t dataBytes = static_cast<uint32_t>(pcm.size() * sizeof(int16_t));
            const uint32_t riffBytes = 36U + dataBytes;
            const uint32_t sampleRate = 24000U;
            const uint32_t byteRate = sampleRate * sizeof(int16_t);
            const uint16_t format = 1U;
            const uint16_t channels = 1U;
            const uint16_t blockAlign = sizeof(int16_t);
            const uint16_t bitsPerSample = 16U;
            const uint32_t fmtBytes = 16U;
            wav.write("RIFF", 4); wav.write(reinterpret_cast<const char*>(&riffBytes), 4);
            wav.write("WAVEfmt ", 8); wav.write(reinterpret_cast<const char*>(&fmtBytes), 4);
            wav.write(reinterpret_cast<const char*>(&format), 2);
            wav.write(reinterpret_cast<const char*>(&channels), 2);
            wav.write(reinterpret_cast<const char*>(&sampleRate), 4);
            wav.write(reinterpret_cast<const char*>(&byteRate), 4);
            wav.write(reinterpret_cast<const char*>(&blockAlign), 2);
            wav.write(reinterpret_cast<const char*>(&bitsPerSample), 2);
            wav.write("data", 4); wav.write(reinterpret_cast<const char*>(&dataBytes), 4);
            wav.write(reinterpret_cast<const char*>(pcm.data()), dataBytes);
            if (!wav) throw std::runtime_error("cannot write output WAV");
        }
        catch (const std::exception& exception) {
            retValue = NativeValue(Host(), GarnetJsonError("waveform_write_failed", exception.what()));
            return retValue;
        }
        retValue = NativeValue(Host(), json({
            {"status", "ok"},
            {"model_id", instance->modelId},
            {"output_path", outputPath.string()},
            {"sample_rate", 24000},
            {"audio_frames", FindField(result, "audio_frame_count").ToLongLong()},
            {"audio_samples", static_cast<long long>(sampleCount)},
            {"audio_duration_seconds", FindField(result, "audio_duration_seconds").ToDouble()},
            {"duration_ms", FindField(result, "total_ms").ToDouble()}
        }).dump());
        return retValue;
    }

    X::Value GarnetAPI::StopServing(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        const std::string modelId = params.size() > 0 ? params[0].ToString() : std::string();
        std::vector<std::shared_ptr<ServingInstance>> stopped;
        {
            std::lock_guard<std::mutex> guard(m_servingMutex);
            if (modelId.empty()) {
                for (const auto& item : m_servingInstances) stopped.push_back(item.second);
                m_servingInstances.clear();
                m_defaultServingModelId.clear();
            }
            else {
                const auto it = m_servingInstances.find(modelId);
                if (it != m_servingInstances.end()) {
                    stopped.push_back(it->second);
                    m_servingInstances.erase(it);
                }
                if (m_defaultServingModelId == modelId) {
                    m_defaultServingModelId = m_servingInstances.empty()
                        ? std::string() : m_servingInstances.begin()->first;
                }
            }
        }
        for (const auto& instance : stopped) {
            std::lock_guard<std::mutex> instanceGuard(instance->mutex);
            if (instance->model.IsValid()) {
                X::Value releaseCallable = instance->model["release_runtime"];
                if (releaseCallable.IsObject()) CallChecked(releaseCallable);
            }
            instance->model = X::Value();
            if (!instance->cacheRoot.empty()) {
                TRTBuilder::ReleaseCachedExecutions(instance->cacheRoot);
            }
        }
        Garnet::Image::Cuda::ShutdownThreadNvJpegDecoder();
        retValue = NativeValue(Host(), true);
        return retValue;
    }

    X::Value GarnetAPI::DetectAccelerationJson(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        try {
            const json options = params.size() == 0
                ? json::object()
                : json::parse(params[0].ToString());
            retValue = NativeValue(Host(), AccelerationDetector::DetectJson(
                options.value("enable_nvidia", true)));
        }
        catch (const std::exception& exception) {
            retValue = NativeValue(Host(), GarnetJsonError("acceleration_detection_failed", exception.what()));
        }
        return retValue;
    }

    X::Value GarnetAPI::ActivateAccelerationPathJson(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        if (params.size() == 0 || params[0].ToString().empty()) {
            retValue = NativeValue(Host(), GarnetJsonError(
                "package_path_required", "activate_acceleration_path_json requires a package path"));
            return retValue;
        }
        retValue = NativeValue(Host(), AccelerationDetector::ActivateJson(params[0].ToString()));
        return retValue;
    }

    namespace
    {
        std::vector<int> ReadIntList(X::Value value)
        {
            std::vector<int> result;
            if (!value.IsList()) return result;
            X::Value list(value);
            long long size = list.Size();
            result.reserve(static_cast<size_t>(size));
            for (long long i = 0; i < size; ++i) {
                result.push_back(static_cast<int>(list.Get(i).ToLongLong()));
            }
            return result;
        }

        std::vector<int> TensorShape(X::Value value)
        {
            std::vector<int> result;
            if (!X::Tensor::IsTensor(value)) return result;
            X::Tensor tensor(value);
            int dimCount = tensor.Info().rank;
            result.reserve(static_cast<size_t>(dimCount));
            for (int i = 0; i < dimCount; ++i) {
                result.push_back(static_cast<int>(tensor.Info().shape[i]));
            }
            return result;
        }

        X::Value GetKwarg(const X::KWARGS& kwParams, const char* name)
        {
            for (const auto& item : kwParams) if (item.first == name) return item.second;
            return X::Value();
        }

        int GetIntArg(const X::ARGS& params, const X::KWARGS& kwParams, size_t index, const char* name, int defaultValue)
        {
            X::Value value = GetKwarg(kwParams, name);
            if (value.IsValid()) {
                return CheckedInt(value, "argument");
            }
            if (params.size() > index) {
                return CheckedInt(params[index], "argument");
            }
            return defaultValue;
        }

        double GetDoubleArg(const X::ARGS& params, const X::KWARGS& kwParams, size_t index, const char* name, double defaultValue)
        {
            X::Value value = GetKwarg(kwParams, name);
            if (value.IsValid()) {
                return value.ToDouble();
            }
            if (params.size() > index) {
                return params[index].ToDouble();
            }
            return defaultValue;
        }

        std::string GetStringArg(const X::ARGS& params, const X::KWARGS& kwParams, size_t index, const char* name, const std::string& defaultValue)
        {
            X::Value value = GetKwarg(kwParams, name);
            if (value.IsValid()) {
                return value.ToString();
            }
            if (params.size() > index) {
                return params[index].ToString();
            }
            return defaultValue;
        }

        X::Value MakeInt64Tensor(X3PackageHost* host, const std::vector<int64_t>& values, bool ensureGpu = false)
        {
            const std::vector<int64_t> shape{static_cast<int64_t>(values.size())};
            return ensureGpu ? TensorHelper::CreateGPU(host, X3_TENSOR_INT64, shape, values.data()) :
                X::Tensor::Create(host, X3_TENSOR_INT64, shape, values.data(), values.size() * sizeof(long long));
        }

        X::Value MakeInt64List(X3PackageHost* host, const std::vector<int64_t>& values)
        {
            auto list = X::Value::List(host);
            for (long long value : values) {
                X::Value item(value);
                if (!list.Append(item)) throw X::Error("cannot append integer");
            }
            return list;
        }

        X::Value MakeInt64Tensor2D(
            X3PackageHost* host,
            const std::vector<int64_t>& values,
            int rows,
            int cols,
            bool ensureGpu = false)
        {
            if (rows < 0 || cols < 0 || static_cast<uint64_t>(rows) * cols != values.size())
                throw X::Error("integer tensor shape does not match data");
            const std::vector<int64_t> shape{rows, cols};
            return ensureGpu ? TensorHelper::CreateGPU(host, X3_TENSOR_INT64, shape, values.data()) :
                X::Tensor::Create(host, X3_TENSOR_INT64, shape, values.data(), values.size() * sizeof(long long));
        }

        X::Value MakeInt64Tensor3DGpu(
            X3PackageHost* host,
            const std::vector<int64_t>& values,
            int dimension0,
            int dimension1,
            int dimension2)
        {
            if (dimension0 < 0 || dimension1 < 0 || dimension2 < 0 ||
                static_cast<size_t>(dimension0) * dimension1 * dimension2 != values.size()) {
                return X::Value();
            }
            return TensorHelper::CreateGPU(host, X3_TENSOR_INT64,
                {dimension0, dimension1, dimension2}, values.data());
        }

        X::Value MakeFloatTensor2D(X3PackageHost* host, const float* data, int rows, int cols)
        {
            if (rows < 0 || cols < 0) throw X::Error("negative tensor dimension");
            return X::Tensor::Create(host, X3_TENSOR_FLOAT32, {rows, cols}, data,
                data ? static_cast<uint64_t>(rows) * cols * sizeof(float) : 0);
        }

        double MsSince(std::chrono::steady_clock::time_point start)
        {
            auto elapsed = std::chrono::steady_clock::now() - start;
            return std::chrono::duration<double, std::milli>(elapsed).count();
        }
    }

    X::Value QwenVLRequestContext::Stats(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        auto tensorGpu = [](X::Value value) {
            if (!X::Tensor::IsTensor(value)) {
                return false;
            }
            X::Tensor tensor(value);
            return TensorHelper::GetGPUMemory(tensor) != nullptr;
        };

        auto stats = X::Value::Dict(Host());
        stats.SetItem("source_height", X::Value(sourceHeight));
        stats.SetItem("source_width", X::Value(sourceWidth));
        stats.SetItem("height", X::Value(resizedHeight));
        stats.SetItem("width", X::Value(resizedWidth));
        stats.SetItem("prompt_token_count", X::Value(promptTokenCount));
        stats.SetItem("visual_token_count", X::Value(visualTokenCount));
        stats.SetItem("pixel_value_count", X::Value(pixelValueCount));
        stats.SetItem("patch_size", X::Value(patchSize));
        stats.SetItem("temporal_patch_size", X::Value(temporalPatchSize));
        stats.SetItem("merge_size", X::Value(mergeSize));
        stats.SetItem("input_ids_gpu", X::Value(tensorGpu(inputIds)));
        stats.SetItem("mm_token_type_ids_gpu", X::Value(tensorGpu(mmTokenTypeIds)));
        stats.SetItem("pixel_values_gpu", X::Value(tensorGpu(pixelValues)));
        stats.SetItem("kv_allocated", X::Value(kvHandle > 0));
        stats.SetItem("kv_handle", X::Value(kvHandle));
        stats.SetItem("kv_max_tokens", X::Value(kvMaxTokens));
        stats.SetItem("kv_logical_length", X::Value(kvLogicalLength));
        stats.SetItem("kv_page_size", X::Value(kvPageSize));
        stats.SetItem("kv_logical_pages", X::Value(kvLogicalPages));
        stats.SetItem("kv_physical_pages", X::Value(kvPhysicalPages));
        stats.SetItem("kv_q_heads", X::Value(kvQHeads));
        stats.SetItem("kv_heads", X::Value(kvHeads));
        stats.SetItem("kv_head_dim", X::Value(kvHeadDim));
        stats.SetItem("image_preprocess_us", X::Value(imagePreprocessUs));
        stats.SetItem("tokenize_us", X::Value(tokenizeUs));
        stats.SetItem("tensor_upload_us", X::Value(tensorUploadUs));
        stats.SetItem("total_us", X::Value(totalUs));
        retValue = NativeValue(Host(), stats);
        return retValue;
    }

    KVCacheManager::~KVCacheManager()
    {
        if (m_keyArena != nullptr) {
            cudaFree(m_keyArena);
            m_keyArena = nullptr;
        }
        if (m_valueArena != nullptr) {
            cudaFree(m_valueArena);
            m_valueArena = nullptr;
        }
    }

    int KVCacheManager::PagesForTokens(long long tokenCount) const
    {
        if (tokenCount <= 0) {
            return 0;
        }
        int pageSize = m_pageSize > 0 ? m_pageSize : 1;
        return static_cast<int>((tokenCount + pageSize - 1) / pageSize);
    }

    X::Value KVCacheManager::MakePageList(const std::vector<int>& pages) const
    {
        auto retList = X::Value::List(Host());
        for (int page : pages) {
            X::Value pageValue(page);
            if (!retList.Append(pageValue)) throw X::Error("cannot append page index");
        }
        return retList;
    }

    bool KVCacheManager::EnsurePages(long long sequenceId, long long tokenCount)
    {
        int pagesNeeded = PagesForTokens(tokenCount);
        auto& state = m_sequences[sequenceId];
        int existing = static_cast<int>(state.pages.size());
        if (pagesNeeded <= existing) {
            state.logicalLength = tokenCount;
            return true;
        }
        int extra = pagesNeeded - existing;
        if (extra > static_cast<int>(m_freePages.size())) {
            return false;
        }
        for (int i = 0; i < extra; ++i) {
            int page = m_freePages.front();
            m_freePages.pop_front();
            state.pages.push_back(page);
        }
        state.logicalLength = tokenCount;
        return true;
    }

    void KVCacheManager::Configure(int maxNumPages, int pageSize, int headDim, int numKVHeads,
        int numLayers, int dtypeBytes, int deviceId)
    {
        m_maxNumPages = maxNumPages > 0 ? maxNumPages : 0;
        m_pageSize = pageSize > 0 ? pageSize : 1;
        m_headDim = headDim > 0 ? headDim : 1;
        m_numKVHeads = numKVHeads > 0 ? numKVHeads : 1;
        m_numLayers = numLayers > 0 ? numLayers : 1;
        m_dtypeBytes = dtypeBytes > 0 ? dtypeBytes : 2;
        m_deviceId = deviceId >= 0 ? deviceId : 0;
        m_bytesPerPagePerLayer = static_cast<size_t>(m_pageSize)
            * static_cast<size_t>(m_numKVHeads)
            * static_cast<size_t>(m_headDim)
            * static_cast<size_t>(m_dtypeBytes);
        m_totalBytes = static_cast<size_t>(m_maxNumPages)
            * static_cast<size_t>(m_numLayers)
            * m_bytesPerPagePerLayer;

        if (m_keyArena != nullptr) {
            cudaFree(m_keyArena);
            m_keyArena = nullptr;
        }
        if (m_valueArena != nullptr) {
            cudaFree(m_valueArena);
            m_valueArena = nullptr;
        }
        if (m_totalBytes > 0) {
            cudaSetDevice(m_deviceId);
            cudaError_t keyErr = cudaMalloc(&m_keyArena, m_totalBytes);
            cudaError_t valueErr = cudaMalloc(&m_valueArena, m_totalBytes);
            if (keyErr != cudaSuccess || valueErr != cudaSuccess) {
                std::cout << "[KVCacheManager] cudaMalloc failed: key="
                    << cudaGetErrorString(keyErr) << ", value=" << cudaGetErrorString(valueErr) << std::endl;
                if (m_keyArena != nullptr) {
                    cudaFree(m_keyArena);
                    m_keyArena = nullptr;
                }
                if (m_valueArena != nullptr) {
                    cudaFree(m_valueArena);
                    m_valueArena = nullptr;
                }
                m_totalBytes = 0;
            }
        }
        m_freePages.clear();
        m_sequences.clear();
        for (int page = 0; page < m_maxNumPages; ++page) {
            m_freePages.push_back(page);
        }
    }

    X::Value KVCacheManager::Allocate(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        long long seqId = params.size() > 0 ? params[0].ToLongLong() : 0;
        X::Value sequenceLengthValue = GetKwarg(kwParams, "sequence_length");
        long long sequenceLength = sequenceLengthValue.IsValid()
            ? sequenceLengthValue.ToLongLong()
            : (params.size() > 1 ? params[1].ToLongLong() : 0);
        if (sequenceLength < 0) sequenceLength = 0;
        int pagesNeeded = PagesForTokens(sequenceLength);
        auto existing = m_sequences.find(seqId);
        if (existing != m_sequences.end()) {
            for (int page : existing->second.pages) {
                m_freePages.push_front(page);
            }
            m_sequences.erase(existing);
        }
        if (!EnsurePages(seqId, sequenceLength)) {
            std::cout << "[KVCacheManager] Not enough free pages: need " << pagesNeeded
                << ", have " << m_freePages.size() << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        retValue = NativeValue(Host(), MakePageList(m_sequences[seqId].pages));
        return retValue;
    }

    X::Value KVCacheManager::Append(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        long long seqId = params.size() > 0 ? params[0].ToLongLong() : 0;
        X::Value appendValue = GetKwarg(kwParams, "tokens");
        long long appendTokens = appendValue.IsValid()
            ? appendValue.ToLongLong()
            : (params.size() > 1 ? params[1].ToLongLong() : 1);
        if (appendTokens < 0) appendTokens = 0;
        auto existing = m_sequences.find(seqId);
        if (existing == m_sequences.end()) {
            if (!EnsurePages(seqId, appendTokens)) {
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }
        }
        else {
            long long newLength = existing->second.logicalLength + appendTokens;
            if (!EnsurePages(seqId, newLength)) {
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }
        }

        const auto& state = m_sequences[seqId];
        auto dict = X::Value::Dict(Host());
        dict.SetItem("sequence_id", X::Value(seqId));
        dict.SetItem("logical_length", X::Value(state.logicalLength));
        dict.SetItem("page_count", X::Value(static_cast<int>(state.pages.size())));
        dict.SetItem("pages", MakePageList(state.pages));
        retValue = NativeValue(Host(), dict);
        return retValue;
    }

    X::Value KVCacheManager::Free(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        long long seqId = params.size() > 0 ? params[0].ToLongLong() : 0;
        auto existing = m_sequences.find(seqId);
        bool released = existing != m_sequences.end();
        if (released) {
            for (int page : existing->second.pages) {
                m_freePages.push_back(page);
            }
            m_sequences.erase(existing);
        }
        retValue = NativeValue(Host(), X::Value(released));
        return retValue;
    }

    X::Value KVCacheManager::Stats(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        auto stats = X::Value::Dict(Host());
        stats.SetItem("max_num_pages", X::Value(m_maxNumPages));
        stats.SetItem("free_pages", X::Value(static_cast<int>(m_freePages.size())));
        stats.SetItem("used_pages", X::Value(m_maxNumPages - static_cast<int>(m_freePages.size())));
        stats.SetItem("page_size", X::Value(m_pageSize));
        stats.SetItem("head_dim", X::Value(m_headDim));
        stats.SetItem("num_kv_heads", X::Value(m_numKVHeads));
        stats.SetItem("num_layers", X::Value(m_numLayers));
        stats.SetItem("dtype_bytes", X::Value(m_dtypeBytes));
        stats.SetItem("device_id", X::Value(m_deviceId));
        stats.SetItem("bytes_per_page_per_layer", X::Value(static_cast<long long>(m_bytesPerPagePerLayer)));
        stats.SetItem("total_key_bytes", X::Value(static_cast<long long>(m_totalBytes)));
        stats.SetItem("total_value_bytes", X::Value(static_cast<long long>(m_totalBytes)));
        stats.SetItem("gpu_allocated", X::Value(m_keyArena != nullptr && m_valueArena != nullptr));
        stats.SetItem("sequence_count", X::Value(static_cast<int>(m_sequences.size())));
        retValue = NativeValue(Host(), stats);
        return retValue;
    }

    bool GarnetAPI::LoadModelFromFile(std::string modelPath, X::Value& model)
    {
        std::ifstream file(modelPath, std::ios::binary);

        if (!file.is_open()) {
            return false;
        }
        LOG << "LoadModelFromFile at: " << modelPath << LINE_END;

        // Read metadata
        std::string line;
        std::getline(file, line, '\0');  // Skip metadata

        while (!file.eof()) {
            std::string key;
            std::getline(file, key, '\0');  // Read key
            if (key.empty() || file.eof()) break;

            std::string dtype;
            std::getline(file, dtype, '\0');  // Read dtype

            // Determine the data type
            X3TensorDType tensor_data_type = X3_TENSOR_FLOAT32;  // Default to float32
            if (dtype == "torch.float32") {
                tensor_data_type = X3_TENSOR_FLOAT32;
            }
            else if (dtype == "torch.float64") {
                tensor_data_type = X3_TENSOR_FLOAT64;
            }
            else if (dtype == "torch.int32") {
                tensor_data_type = X3_TENSOR_INT32;
            }
            else if (dtype == "torch.int64") {
                tensor_data_type = X3_TENSOR_INT64;
            }
            else if (dtype == "torch.bfloat16") {
                tensor_data_type = X3_TENSOR_BFLOAT16;
            }
            else if (dtype == "torch.float8_e4m3fn")
            {
                tensor_data_type = X3_TENSOR_FLOAT8_E4M3FN;
            }
            else if (dtype == "torch.float8_e4m3fnuz")
            {
                tensor_data_type = X3_TENSOR_FLOAT8_E4M3FNUZ;
            }
            else if (dtype == "torch.float8_e5m2")
            {
                tensor_data_type = X3_TENSOR_FLOAT8_E5M2;
            }
            else if (dtype == "torch.float8_e5m2fnuz")
            {
                tensor_data_type = X3_TENSOR_FLOAT8_E5M2FNUZ;
            }
            else throw X::Error("unsupported model tensor dtype: " + dtype);
            // Read shape
            uint64_t num_dims = 0;
            if (!file.read(reinterpret_cast<char*>(&num_dims), sizeof(uint64_t)) || num_dims > 64)
                throw X::Error("invalid model tensor rank");
            std::vector<int64_t> shape;
            shape.reserve(static_cast<size_t>(num_dims));
            uint64_t num_bytes = TensorHelper::ItemSize(tensor_data_type);
            for (uint64_t i = 0; i < num_dims;i++) {
                int64_t d = 0;
                if (!file.read(reinterpret_cast<char*>(&d), sizeof(int64_t)) || d < 0 ||
                    (d && num_bytes > INT64_MAX / static_cast<uint64_t>(d)))
                    throw X::Error("invalid model tensor dimensions");
                shape.push_back(d);
                num_bytes *= d;
            }
            const auto offset = file.tellg();
            const auto fileSize = std::filesystem::file_size(modelPath);
            if (offset < 0 || static_cast<uint64_t>(offset) > fileSize ||
                num_bytes > fileSize - static_cast<uint64_t>(offset)) throw X::Error("truncated model tensor");
            std::vector<char> buffer(num_bytes);
            if (num_bytes && !file.read(buffer.data(), static_cast<std::streamsize>(num_bytes)))
                throw X::Error("cannot read model tensor payload");
            auto tensor = TensorHelper::CreateGPU(Host(), tensor_data_type, shape, buffer.data());

            // Store in dictionary
            if (!model.SetItem(key, tensor)) throw X::Error("cannot register model weight");
        }

        file.close();
        return true;
    }


    X::Value GarnetAPI::LoadModel(std::string modelPath)
    {
        auto dictModel = X::Value::Dict(Host());
        namespace fs = std::filesystem;

        std::string tokenizerJsonPath;
        std::string tokenizerConfigJsonPath;
        std::string strModelPath;
        // Check if the path contains a wildcard ('*' or '?')
        if (modelPath.find('*') != std::string::npos || modelPath.find('?') != std::string::npos)
        {
            // Split the path into directory and pattern parts.
            fs::path pathPattern(modelPath);
            fs::path directory = pathPattern.parent_path();
            if (directory.empty()) {
                directory = fs::current_path();
            }
			strModelPath = directory.string();
            std::string pattern = pathPattern.filename().string();

            // Convert wildcard pattern to a regular expression.
            // For example, "*.bin" becomes ".*\.bin"
            std::string regexPattern;
            for (char c : pattern)
            {
                if (c == '*')
                    regexPattern += ".*";
                else if (c == '?')
                    regexPattern += ".";
                // Escape regex special characters, except for alphanumerics
                else if (std::isalnum(c) || c == '_' || c == '-')
                    regexPattern += c;
                else
                    regexPattern += "\\" + std::string(1, c);
            }
            std::regex fileRegex(regexPattern, std::regex::icase);

            // Iterate over files in the target directory.
            for (const auto& entry : fs::directory_iterator(directory))
            {
                if (entry.is_regular_file())
                {
                    std::string filename = entry.path().filename().string();
                    if (std::regex_match(filename, fileRegex))
                    {
                        // Append the model data from each matching file.
                        LoadModelFromFile(entry.path().string(), dictModel);
                    }
                }
            }
        }
        else
        {
            // No wildcard found; check if it's a directory or file
            fs::path path(modelPath);

            if (fs::is_directory(path))
            {
				strModelPath = path.string();
                // Reset tokenizer paths
                tokenizerJsonPath = "";
                tokenizerConfigJsonPath = "";

                // Scan for *.bin files and tokenizer files
                for (const auto& entry : fs::directory_iterator(path))
                {
                    if (!entry.is_regular_file()) continue;

                    fs::path entryPath = entry.path();
                    std::string filename = entryPath.filename().string();
                    std::string extension = entryPath.extension().string();

                    if (extension == ".bin")
                    {
                        // Load .bin file
                        LoadModelFromFile(entryPath.string(), dictModel);
                    }
                    else if (filename == "tokenizer.json")
                    {
                        tokenizerJsonPath = fs::absolute(entryPath).string();
                        LOG << "Found tokenizer.json at: " << tokenizerJsonPath << LINE_END;
                    }
                    else if (filename == "tokenizer_config.json")
                    {
                        tokenizerConfigJsonPath = fs::absolute(entryPath).string();
                        LOG << "Found tokenizer_config.json at: " << tokenizerConfigJsonPath << LINE_END;
                    }
                }
            }
            else
            {
                // Single file
                fs::path fsPath(modelPath);
                fs::path directory = fsPath.parent_path();
                strModelPath = directory.string();
                LoadModelFromFile(modelPath, dictModel);
            }
        }

        auto varModel = CallChecked(__xlang3_package_->GetValue("model"));
        Model& model = *varModel.NativeData<Model>();
		model.SetInfo(strModelPath, tokenizerJsonPath, tokenizerConfigJsonPath, dictModel);
        return varModel;
    }

    X::Value GarnetAPI::CreateKVCacheManager(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        auto getInt = [&](const char* name, size_t pos, int defaultValue) -> int {
            X::Value value = GetKwarg(kwParams, name);
            if (value.IsValid()) return CheckedInt(value, "argument");
            if (params.size() > pos) return static_cast<int>(params[pos].ToLongLong());
            return defaultValue;
        };
        int maxNumPages = getInt("max_num_pages", 0, 0);
        int pageSize = getInt("page_size", 1, 16);
        int headDim = getInt("head_dim", 2, 128);
        int numKVHeads = getInt("num_kv_heads", 3, 1);
        int numLayers = getInt("num_layers", 4, 1);
        int dtypeBytes = getInt("dtype_bytes", 5, 2);
        int deviceId = getInt("device_id", 6, 0);
        auto manager = CallChecked(__xlang3_package_->GetValue("KVCacheManagerClass"));
        manager.NativeData<KVCacheManager>()->Configure(maxNumPages, pageSize, headDim, numKVHeads, numLayers, dtypeBytes, deviceId);
        retValue = NativeValue(Host(), manager);
        return retValue;
    }

    X::Value GarnetAPI::DevicePagedKVWriteTensor(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() < 4 || !X::Tensor::IsTensor(params[1])) {
            std::cout << "[GarnetAPI] device_paged_kv_write(handle, qkv_tensor, token_count, start_position) expected." << std::endl;
            retValue = NativeValue(Host(), X::Value(false));
            return retValue;
        }
        long long handle = params[0].ToLongLong();
        X::Tensor qkv(params[1]);
        ValidateDenseTensor(qkv);
        int tokenCount = CheckedInt(params[2], "argument");
        int startPosition = CheckedInt(params[3], "argument");
        if (qkv.Info().dtype != X3_TENSOR_FLOAT32 || qkv.Info().rank != 2) {
            std::cout << "[GarnetAPI] device_paged_kv_write requires a float32 2D qkv tensor." << std::endl;
            retValue = NativeValue(Host(), X::Value(false));
            return retValue;
        }
        DevicePagedKVFP32 cache;
        if (!GetDevicePagedKV(handle, cache)) {
            std::cout << "[GarnetAPI] device_paged_kv_write invalid handle: " << handle << std::endl;
            retValue = NativeValue(Host(), X::Value(false));
            return retValue;
        }
        int qWidth = cache.qHeads * cache.headDim;
        int kvWidth = cache.kvHeads * cache.headDim;
        int qkvStride = qWidth + 2 * kvWidth;
        if (tokenCount <= 0 || tokenCount > qkv.Info().shape[0] || qkv.Info().shape[1] < qkvStride ||
            startPosition < 0 || startPosition + tokenCount > cache.logicalPageCount * cache.pageSize) {
            std::cout << "[GarnetAPI] device_paged_kv_write invalid shape or range." << std::endl;
            retValue = NativeValue(Host(), X::Value(false));
            return retValue;
        }
        if (TensorHelper::EnsureGPUMemory(qkv) != TensorOpStatus::Success) {
            std::cout << "[GarnetAPI] device_paged_kv_write failed to ensure qkv GPU memory." << std::endl;
            retValue = NativeValue(Host(), X::Value(false));
            return retValue;
        }
        auto qkvUse = qkv.Acquire();
        float* dQKV = static_cast<float*>(TensorHelper::GetGPUMemory(qkv));
        if (!dQKV) {
            std::cout << "[GarnetAPI] device_paged_kv_write qkv tensor has no GPU memory." << std::endl;
            retValue = NativeValue(Host(), X::Value(false));
            return retValue;
        }
        cudaStream_t stream = nullptr;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err == cudaSuccess) {
            err = runTextPagedKVWriteFP32(
                dQKV, cache.keyPages, cache.valuePages, cache.pageTable,
                tokenCount, startPosition, cache.pageSize, cache.qHeads, cache.kvHeads, cache.headDim, stream);
        }
        if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
        if (stream) cudaStreamDestroy(stream);
        if (err != cudaSuccess) {
            std::cout << "[GarnetAPI] device_paged_kv_write failed: " << cudaGetErrorString(err) << std::endl;
            retValue = NativeValue(Host(), X::Value(false));
            return retValue;
        }
        retValue = NativeValue(Host(), X::Value(true));
        return retValue;
    }

    X::Value GarnetAPI::DevicePagedKVAttentionTensor(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() < 3 || !X::Tensor::IsTensor(params[1])) {
            std::cout << "[GarnetAPI] device_paged_kv_attention(handle, q_or_qkv_tensor, sequence_length) expected." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        long long handle = params[0].ToLongLong();
        X::Tensor q(params[1]);
        ValidateDenseTensor(q);
        int sequenceLength = CheckedInt(params[2], "argument");
        if (q.Info().dtype != X3_TENSOR_FLOAT32 || q.Info().rank != 2 || q.Info().shape[0] < 1) {
            std::cout << "[GarnetAPI] device_paged_kv_attention requires a float32 2D q/qkv tensor." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        DevicePagedKVFP32 cache;
        if (!GetDevicePagedKV(handle, cache)) {
            std::cout << "[GarnetAPI] device_paged_kv_attention invalid handle: " << handle << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        int qWidth = cache.qHeads * cache.headDim;
        if (q.Info().shape[1] < qWidth || sequenceLength <= 0 ||
            sequenceLength > cache.logicalPageCount * cache.pageSize) {
            std::cout << "[GarnetAPI] device_paged_kv_attention invalid shape or sequence length." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        if (TensorHelper::EnsureGPUMemory(q) != TensorOpStatus::Success) {
            std::cout << "[GarnetAPI] device_paged_kv_attention failed to ensure q GPU memory." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        auto qUse = q.Acquire();
        float* dQBase = static_cast<float*>(TensorHelper::GetGPUMemory(q));
        if (!dQBase) {
            std::cout << "[GarnetAPI] device_paged_kv_attention q tensor has no GPU memory." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        int qStride = q.Info().shape[1];
        float* dQ = dQBase + static_cast<size_t>(q.Info().shape[0] - 1) * static_cast<size_t>(qStride);
        size_t outputBytes = static_cast<size_t>(qWidth) * sizeof(float);
        float* dOut = nullptr;
        cudaStream_t stream = nullptr;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err == cudaSuccess) err = cudaMalloc(&dOut, outputBytes);
        if (err == cudaSuccess) {
            err = runTextPagedKVCachedAttentionFP32(
                dQ, cache.keyPages, cache.valuePages, cache.pageTable, dOut,
                sequenceLength, cache.pageSize, cache.qHeads, cache.kvHeads, cache.headDim, stream);
        }
        if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            if (dOut) cudaFree(dOut);
            if (stream) cudaStreamDestroy(stream);
            std::cout << "[GarnetAPI] device_paged_kv_attention failed: " << cudaGetErrorString(err) << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }

        if (stream) cudaStreamDestroy(stream);
        auto output = AdoptGpuOutput(Host(), X3_TENSOR_FLOAT32, {1, qWidth}, dOut);
        retValue = NativeValue(Host(), X::Value(output));
        return retValue;
    }

    X::Value GarnetAPI::TensorToCPU(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() < 1 || !X::Tensor::IsTensor(params[0])) {
            std::cout << "[GarnetAPI] tensor_to_cpu(tensor) expected." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }

        X::Tensor tensor(params[0]);
        retValue = NativeValue(Host(), TensorHelper::CopyToCPUTensor(tensor));
        if (!retValue.IsValid()) {
            std::cout << "[GarnetAPI] tensor_to_cpu failed." << std::endl;
            return retValue;
        }
        if (tensor.Info().dtype == X3_TENSOR_BFLOAT16) {
            X::Tensor raw(retValue);
            ValidateDenseTensor(raw);
            auto rawUse = raw.Acquire();
            std::vector<int64_t> shape;
            shape.reserve(raw.Info().rank);
            for (int dimension = 0; dimension < raw.Info().rank; ++dimension) {
                shape.push_back(raw.Info().shape[dimension]);
            }
            auto converted = X::Tensor::Create(Host(), X3_TENSOR_FLOAT32, shape);
            const auto* source = reinterpret_cast<const unsigned short*>(raw.Info().data);
            auto* destination = reinterpret_cast<float*>(converted.Info().data);
            for (long long index = 0; index < TensorCount(raw); ++index) {
                const unsigned int bits = static_cast<unsigned int>(source[index]) << 16;
                std::memcpy(destination + index, &bits, sizeof(float));
            }
            retValue = NativeValue(Host(), X::Value(converted));
        }
        return retValue;
    }

    X::Value GarnetAPI::TensorToGPU(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() < 1 || !X::Tensor::IsTensor(params[0])) {
            std::cout << "[GarnetAPI] tensor_to_gpu(tensor) expected." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        X::Tensor tensor(params[0]);
        if (TensorHelper::EnsureGPUMemory(tensor) != TensorOpStatus::Success) {
            std::cout << "[GarnetAPI] tensor_to_gpu failed." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        retValue = NativeValue(Host(), X::Value(tensor));
        return retValue;
    }

    X::Value GarnetAPI::TensorToBFloat16(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() < 1 || !X::Tensor::IsTensor(params[0])) {
            std::cout << "[GarnetAPI] tensor_to_bfloat16(tensor) expected." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        X::Tensor source(params[0]);
        ValidateDenseTensor(source);
        if (source.Info().dtype == X3_TENSOR_BFLOAT16) {
            retValue = NativeValue(Host(), X::Value(source));
            return retValue;
        }
        if (source.Info().dtype != X3_TENSOR_FLOAT32 ||
            TensorHelper::EnsureGPUMemory(source) != TensorOpStatus::Success) {
            std::cout << "[GarnetAPI] tensor_to_bfloat16 requires a GPU-capable FLOAT32 tensor." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }

        std::vector<int64_t> shape;
        shape.reserve(source.Info().rank);
        for (int dimension = 0; dimension < source.Info().rank; ++dimension) {
            shape.push_back(static_cast<int>(source.Info().shape[dimension]));
        }
        void* outputDevice = nullptr;
        const size_t outputBytes = static_cast<size_t>(TensorCount(source)) * sizeof(bfloat16);
        if (!outputBytes) return TensorHelper::CreateGPU(Host(), X3_TENSOR_BFLOAT16, shape);
        if (cudaMalloc(&outputDevice, outputBytes) != cudaSuccess) {
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        auto sourceUse = TensorHelper::AcquireGPU(source);
        cudaError_t status = runConvertFP32ToBF16Async(
            static_cast<const float*>(TensorHelper::GetGPUMemory(source)),
            static_cast<bfloat16*>(outputDevice),
            static_cast<int>(TensorCount(source)),
            cudaStreamPerThread);
        if (status == cudaSuccess) status = cudaStreamSynchronize(cudaStreamPerThread);
        if (status != cudaSuccess) {
            cudaFree(outputDevice);
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        retValue = AdoptGpuOutput(Host(), X3_TENSOR_BFLOAT16, shape, outputDevice);
        return retValue;
    }

    X::Value GarnetAPI::TensorFromBFloat16Bits(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() == 0 || !X::Tensor::IsTensor(params[0])) {
            std::cout << "[GarnetAPI] tensor_from_bfloat16_bits(uint16_tensor) expected." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        X::Tensor bits(params[0]);
        ValidateDenseTensor(bits);
        auto bitsUse = bits.Acquire();
        if (bits.Info().dtype != X3_TENSOR_UINT16 || bits.Info().device_type != 0 || !bits.Info().data) {
            std::cout << "[GarnetAPI] BF16 source must be a CPU uint16 tensor." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        std::vector<int64_t> shape;
        shape.reserve(bits.Info().rank);
        for (int dim = 0; dim < bits.Info().rank; ++dim) {
            shape.push_back(static_cast<int>(bits.Info().shape[dim]));
        }
        const std::string device = GetStringArg(
            params, kwParams, 1, "device", "cuda");
        if (device == "cpu") {
            retValue = X::Tensor::Create(Host(), X3_TENSOR_BFLOAT16, shape, bits.Info().data, bits.Info().byte_size);
            return retValue;
        }
        if (device != "cuda") {
            std::cout << "[GarnetAPI] tensor_from_bfloat16_bits device must be cpu or cuda." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        retValue = TensorHelper::CreateGPU(Host(), X3_TENSOR_BFLOAT16, shape, bits.Info().data);
        return retValue;
    }

    X::Value GarnetAPI::TensorFromHost(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() == 0 || !params[0].IsList()) {
            std::cout << "[GarnetAPI] tensor_from_host(values, dtype='int32', shape=[...]) expected." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        X::Value values(params[0]);
        const std::string dataTypeName = GetStringArg(params, kwParams, 1, "dtype", "float32");
        X3TensorDType dataType;
        size_t elementBytes = 0;
        if (dataTypeName == "int32") {
            dataType = X3_TENSOR_INT32;
            elementBytes = sizeof(int);
        }
        else if (dataTypeName == "int64") {
            dataType = X3_TENSOR_INT64;
            elementBytes = sizeof(long long);
        }
        else if (dataTypeName == "float32") {
            dataType = X3_TENSOR_FLOAT32;
            elementBytes = sizeof(float);
        }
        else {
            std::cout << "[GarnetAPI] tensor_from_host unsupported dtype: " << dataTypeName << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }

        std::vector<int> dimensions;
        X::Value shapeValue = GetKwarg(kwParams, "shape");
        if (shapeValue.IsList()) {
            X::Value shape(shapeValue);
            for (long long index = 0; index < shape.Size(); ++index) {
                dimensions.push_back(CheckedInt(shape.Get(index), "tensor dimension"));
            }
        }
        if (values.Size() > INT32_MAX) throw X::Error("tensor value count exceeds int32 limits");
        if (dimensions.empty()) dimensions.push_back(static_cast<int>(values.Size()));
        size_t expectedCount = 1;
        for (const int dimension : dimensions) {
            if (dimension < 0 || (dimension && expectedCount > INT32_MAX / static_cast<size_t>(dimension)))
                throw X::Error("tensor shape exceeds int32 kernel limits");
            expectedCount *= static_cast<size_t>(dimension);
        }
        if (expectedCount != static_cast<size_t>(values.Size())) {
            std::cout << "[GarnetAPI] tensor_from_host shape does not match value count." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }

        std::vector<char> bytes(expectedCount * elementBytes);
        for (size_t index = 0; index < expectedCount; ++index) {
            X::Value value = values.Get(static_cast<long long>(index));
            if (dataType == X3_TENSOR_INT32) {
                const int number = CheckedInt(value, "tensor value");
                std::memcpy(bytes.data() + index * sizeof(number), &number, sizeof(number));
            }
            else if (dataType == X3_TENSOR_INT64) {
                const int64_t number = CheckedInt64(value, "tensor value");
                std::memcpy(bytes.data() + index * sizeof(number), &number, sizeof(number));
            }
            else {
                const float number = static_cast<float>(value.ToDouble());
                std::memcpy(bytes.data() + index * sizeof(number), &number, sizeof(number));
            }
        }

        const std::vector<int64_t> shape(dimensions.begin(), dimensions.end());
        const std::string device = GetStringArg(params, kwParams, 3, "device", "cuda");
        if (device != "cuda" && device != "cpu") throw X::Error("tensor device must be cpu or cuda");
        retValue = device == "cuda" ? TensorHelper::CreateGPU(Host(), dataType, shape, bytes.data()) :
            X::Tensor::Create(Host(), dataType, shape, bytes.data(), bytes.size());
        return retValue;
    }

    X::Value GarnetAPI::TensorUpdateFromHost(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() < 2 || !X::Tensor::IsTensor(params[0]) ||
            !params[1].IsList()) {
            retValue = NativeValue(Host(), X::Value(false));
            return retValue;
        }
        X::Tensor tensor(params[0]);
        ValidateDenseTensor(tensor, true);
        X::Value values(params[1]);
        const size_t count = static_cast<size_t>(TensorCount(tensor));
        if (count != static_cast<size_t>(values.Size())) {
            retValue = NativeValue(Host(), X::Value(false));
            return retValue;
        }
        if (count == 0) return X::Value(true);

        size_t elementBytes = 0;
        switch (tensor.Info().dtype) {
        case X3_TENSOR_INT32:
            elementBytes = sizeof(int);
            break;
        case X3_TENSOR_INT64:
            elementBytes = sizeof(long long);
            break;
        case X3_TENSOR_FLOAT32:
            elementBytes = sizeof(float);
            break;
        default:
            retValue = NativeValue(Host(), X::Value(false));
            return retValue;
        }

        std::vector<char> bytes(count * elementBytes);
        for (size_t index = 0; index < count; ++index) {
            X::Value value = values.Get(static_cast<long long>(index));
            if (tensor.Info().dtype == X3_TENSOR_INT32) {
                const int number = CheckedInt(value, "tensor value");
                std::memcpy(bytes.data() + index * sizeof(number), &number, sizeof(number));
            }
            else if (tensor.Info().dtype == X3_TENSOR_INT64) {
                const int64_t number = CheckedInt64(value, "tensor value");
                std::memcpy(bytes.data() + index * sizeof(number), &number, sizeof(number));
            }
            else {
                const float number = static_cast<float>(value.ToDouble());
                std::memcpy(bytes.data() + index * sizeof(number), &number, sizeof(number));
            }
        }
        if (tensor.Info().device_type == 0) {
            auto use = tensor.Acquire(X3_TENSOR_WRITE);
            if (!bytes.empty()) std::memcpy(tensor.Info().data, bytes.data(), bytes.size());
            return X::Value(true);
        }
        auto tensorUse = TensorHelper::AcquireGPU(tensor, X3_TENSOR_WRITE);
        cudaError_t status = cudaMemcpyAsync(
            tensor.Info().data,
            bytes.data(),
            bytes.size(),
            cudaMemcpyHostToDevice,
            cudaStreamPerThread);
        const auto copyComplete = cudaStreamSynchronize(cudaStreamPerThread);
        if (status == cudaSuccess) status = copyComplete;
        retValue = NativeValue(Host(), X::Value(status == cudaSuccess));
        return retValue;
    }

    X::Value GarnetAPI::TensorAdd(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() < 2 || !X::Tensor::IsTensor(params[0]) || !X::Tensor::IsTensor(params[1])) {
            std::cout << "[GarnetAPI] tensor_add(lhs, rhs) expected." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        X::Tensor lhs(params[0]);
        ValidateDenseTensor(lhs);
        X::Tensor rhs(params[1]);
        ValidateDenseTensor(rhs);
        if (lhs.Info().dtype != X3_TENSOR_FLOAT32 ||
            rhs.Info().dtype != X3_TENSOR_FLOAT32 ||
            lhs.Info().rank != rhs.Info().rank ||
            TensorCount(lhs) != TensorCount(rhs)) {
            std::cout << "[GarnetAPI] tensor_add requires equal-shaped FLOAT32 tensors." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        for (int dim = 0; dim < lhs.Info().rank; ++dim) {
            if (lhs.Info().shape[dim] != rhs.Info().shape[dim]) {
                std::cout << "[GarnetAPI] tensor_add shape mismatch." << std::endl;
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }
        }
        if (TensorHelper::EnsureGPUMemory(lhs) != TensorOpStatus::Success ||
            TensorHelper::EnsureGPUMemory(rhs) != TensorOpStatus::Success) {
            std::cout << "[GarnetAPI] tensor_add failed to ensure GPU inputs." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }

        auto inputsUse = X::Tensor::AcquireMany({{lhs, X3_TENSOR_READ}, {rhs, X3_TENSOR_READ}});
        size_t bytes = static_cast<size_t>(lhs.Info().byte_size);
        float* outputDevice = nullptr;
        cudaStream_t stream = nullptr;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err == cudaSuccess) err = cudaMalloc(&outputDevice, bytes);
        if (err == cudaSuccess) {
            err = runTensorAddFP32(
                static_cast<const float*>(TensorHelper::GetGPUMemory(lhs)),
                static_cast<const float*>(TensorHelper::GetGPUMemory(rhs)),
                outputDevice,
                static_cast<int>(TensorCount(lhs)),
                stream);
        }
        if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            if (outputDevice) cudaFree(outputDevice);
            if (stream) cudaStreamDestroy(stream);
            std::cout << "[GarnetAPI] tensor_add failed: " << cudaGetErrorString(err) << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }

        std::vector<int64_t> shape; shape.reserve(lhs.Info().rank);
        for (int dim = 0; dim < lhs.Info().rank; ++dim) {
            shape.push_back(static_cast<int>(lhs.Info().shape[dim]));
        }
        cudaStreamDestroy(stream);
        auto output = AdoptGpuOutput(Host(), X3_TENSOR_FLOAT32, shape, outputDevice);
        retValue = NativeValue(Host(), X::Value(output));
        return retValue;
    }

    X::Value GarnetAPI::Embedding(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() < 2 || !X::Tensor::IsTensor(params[0]) || !X::Tensor::IsTensor(params[1])) {
            std::cout << "[GarnetAPI] embedding(weight, token_ids) expected." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        X::Tensor weight(params[0]);
        ValidateDenseTensor(weight);
        X::Tensor tokenIds(params[1]);
        ValidateDenseTensor(tokenIds);
        bool bf16Weight = weight.Info().dtype == X3_TENSOR_BFLOAT16;
        if ((!bf16Weight && weight.Info().dtype != X3_TENSOR_FLOAT32) || weight.Info().rank != 2 ||
            tokenIds.Info().dtype != X3_TENSOR_INT64 || tokenIds.Info().rank != 1) {
            std::cout << "[GarnetAPI] embedding requires FLOAT32/BF16 [vocab, hidden] and INT64 [tokens]." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        if (TensorHelper::EnsureGPUMemory(weight) != TensorOpStatus::Success ||
            TensorHelper::EnsureGPUMemory(tokenIds) != TensorOpStatus::Success) {
            std::cout << "[GarnetAPI] embedding failed to ensure GPU inputs." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        auto inputsUse = X::Tensor::AcquireMany({{weight, X3_TENSOR_READ}, {tokenIds, X3_TENSOR_READ}});
        int tokens = static_cast<int>(tokenIds.Info().shape[0]);
        int vocab = static_cast<int>(weight.Info().shape[0]);
        int hidden = static_cast<int>(weight.Info().shape[1]);
        if (hidden <= 0 || tokens < 0 || static_cast<int64_t>(tokens) * hidden > INT32_MAX - 1024)
            throw X::Error("embedding output exceeds kernel indexing limits");
        size_t bytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);
        float* outputDevice = nullptr;
        cudaStream_t stream = nullptr;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err == cudaSuccess) err = cudaMalloc(&outputDevice, bytes);
        if (err == cudaSuccess) {
            err = bf16Weight
                ? runEmbeddingGatherInt64BF16ToFP32(
                    static_cast<const bfloat16*>(TensorHelper::GetGPUMemory(weight)),
                    static_cast<const long long*>(TensorHelper::GetGPUMemory(tokenIds)),
                    outputDevice, tokens, vocab, hidden, stream)
                : runEmbeddingGatherInt64FP32(
                    static_cast<const float*>(TensorHelper::GetGPUMemory(weight)),
                    static_cast<const long long*>(TensorHelper::GetGPUMemory(tokenIds)),
                    outputDevice, tokens, vocab, hidden, stream);
        }
        if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            if (outputDevice) cudaFree(outputDevice);
            if (stream) cudaStreamDestroy(stream);
            std::cout << "[GarnetAPI] embedding failed: " << cudaGetErrorString(err) << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        std::vector<int64_t> shape; shape.reserve(2);
        shape.push_back(tokens);
        shape.push_back(hidden);
        cudaStreamDestroy(stream);
        auto output = AdoptGpuOutput(Host(), X3_TENSOR_FLOAT32, shape, outputDevice);
        retValue = NativeValue(Host(), X::Value(output));
        return retValue;
    }

    X::Value GarnetAPI::ReplaceRowsByMask(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() < 3 || !X::Tensor::IsTensor(params[0]) || !X::Tensor::IsTensor(params[1]) || !X::Tensor::IsTensor(params[2])) {
            std::cout << "[GarnetAPI] replace_rows_by_mask(base, mask, replacements, mask_value=1) expected." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        X::Tensor base(params[0]);
        ValidateDenseTensor(base);
        X::Tensor mask(params[1]);
        ValidateDenseTensor(mask);
        X::Tensor replacements(params[2]);
        ValidateDenseTensor(replacements);
        long long maskValue = params.size() >= 4 ? params[3].ToLongLong() : 1;
        if (base.Info().dtype != X3_TENSOR_FLOAT32 || base.Info().rank != 2 ||
            mask.Info().dtype != X3_TENSOR_INT64 || mask.Info().rank != 1 ||
            replacements.Info().dtype != X3_TENSOR_FLOAT32 || replacements.Info().rank != 2 ||
            mask.Info().shape[0] != base.Info().shape[0] ||
            replacements.Info().shape[1] != base.Info().shape[1]) {
            std::cout << "[GarnetAPI] replace_rows_by_mask shape or dtype mismatch." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        if (TensorHelper::EnsureGPUMemory(base) != TensorOpStatus::Success ||
            TensorHelper::EnsureGPUMemory(mask) != TensorOpStatus::Success ||
            TensorHelper::EnsureGPUMemory(replacements) != TensorOpStatus::Success) {
            std::cout << "[GarnetAPI] replace_rows_by_mask failed to ensure GPU inputs." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }

        auto inputsUse = X::Tensor::AcquireMany({{base, X3_TENSOR_READ}, {mask, X3_TENSOR_READ},
            {replacements, X3_TENSOR_READ}});
        size_t bytes = static_cast<size_t>(base.Info().byte_size);
        float* outputDevice = nullptr;
        cudaStream_t stream = nullptr;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err == cudaSuccess) err = cudaMalloc(&outputDevice, bytes);
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(outputDevice, TensorHelper::GetGPUMemory(base), bytes, cudaMemcpyDeviceToDevice, stream);
        }
        if (err == cudaSuccess) {
            err = runReplaceRowsByMaskInt64FP32(
                outputDevice,
                static_cast<const long long*>(TensorHelper::GetGPUMemory(mask)),
                static_cast<const float*>(TensorHelper::GetGPUMemory(replacements)),
                static_cast<int>(base.Info().shape[0]),
                static_cast<int>(replacements.Info().shape[0]),
                static_cast<int>(base.Info().shape[1]),
                maskValue,
                stream);
        }
        if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            if (outputDevice) cudaFree(outputDevice);
            if (stream) cudaStreamDestroy(stream);
            std::cout << "[GarnetAPI] replace_rows_by_mask failed: " << cudaGetErrorString(err) << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        std::vector<int64_t> shape; shape.reserve(2);
        shape.push_back(static_cast<int>(base.Info().shape[0]));
        shape.push_back(static_cast<int>(base.Info().shape[1]));
        cudaStreamDestroy(stream);
        auto output = AdoptGpuOutput(Host(), X3_TENSOR_FLOAT32, shape, outputDevice);
        retValue = NativeValue(Host(), X::Value(output));
        return retValue;
    }

    X::Value GarnetAPI::TensorLastRow(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() < 1 || !X::Tensor::IsTensor(params[0])) {
            std::cout << "[GarnetAPI] tensor_last_row(tensor) expected." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        X::Tensor input(params[0]);
        ValidateDenseTensor(input);
        if (input.Info().rank != 2 || input.Info().shape[0] <= 0 ||
            TensorHelper::EnsureGPUMemory(input) != TensorOpStatus::Success) {
            std::cout << "[GarnetAPI] tensor_last_row requires a non-empty 2D tensor." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        int rows = static_cast<int>(input.Info().shape[0]);
        int columns = static_cast<int>(input.Info().shape[1]);
        size_t rowBytes = static_cast<size_t>(columns) * TensorHelper::ItemSize(input.Info().dtype);
        auto inputUse = TensorHelper::AcquireGPU(input);
        auto* inputDevice = static_cast<const char*>(TensorHelper::GetGPUMemory(input));
        void* outputDevice = nullptr;
        cudaError_t err = cudaMalloc(&outputDevice, rowBytes);
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(
                outputDevice,
                inputDevice + static_cast<size_t>(rows - 1) * rowBytes,
                rowBytes,
                cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        }
        if (err == cudaSuccess) err = cudaStreamSynchronize(cudaStreamPerThread);
        if (err != cudaSuccess) {
            if (outputDevice) cudaFree(outputDevice);
            std::cout << "[GarnetAPI] tensor_last_row failed: " << cudaGetErrorString(err) << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        std::vector<int64_t> shape; shape.reserve(2);
        shape.push_back(1);
        shape.push_back(columns);
        auto output = AdoptGpuOutput(Host(), input.Info().dtype, shape, outputDevice);
        retValue = NativeValue(Host(), X::Value(output));
        return retValue;
    }

    X::Value GarnetAPI::GeluTanh(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() < 1 || !X::Tensor::IsTensor(params[0])) {
            std::cout << "[GarnetAPI] gelu_tanh(tensor) expected." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        X::Tensor input(params[0]);
        ValidateDenseTensor(input);
        if (input.Info().dtype != X3_TENSOR_FLOAT32 || TensorCount(input) <= 0 ||
            TensorHelper::EnsureGPUMemory(input) != TensorOpStatus::Success) {
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        auto inputUse = input.Acquire();
        size_t bytes = static_cast<size_t>(input.Info().byte_size);
        float* outputDevice = nullptr;
        cudaStream_t stream = nullptr;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err == cudaSuccess) err = cudaMalloc(&outputDevice, bytes);
        if (err == cudaSuccess) {
            err = runGeluTanhFP32(
                static_cast<const float*>(TensorHelper::GetGPUMemory(input)),
                outputDevice,
                static_cast<int>(TensorCount(input)),
                stream);
        }
        if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            if (outputDevice) cudaFree(outputDevice);
            if (stream) cudaStreamDestroy(stream);
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        std::vector<int64_t> shape; shape.reserve(input.Info().rank);
        for (int dim = 0; dim < input.Info().rank; ++dim) {
            shape.push_back(static_cast<int>(input.Info().shape[dim]));
        }
        cudaStreamDestroy(stream);
        auto output = AdoptGpuOutput(Host(), X3_TENSOR_FLOAT32, shape, outputDevice);
        retValue = NativeValue(Host(), X::Value(output));
        return retValue;
    }

    X::Value GarnetAPI::VisionRoPE(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() < 3 || !X::Tensor::IsTensor(params[0]) || !X::Tensor::IsTensor(params[1]) || !X::Tensor::IsTensor(params[2])) {
            std::cout << "[GarnetAPI] vision_rope(qkv, cos, sin, num_heads=16) expected." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        X::Tensor qkv(params[0]);
        ValidateDenseTensor(qkv);
        X::Tensor cos(params[1]);
        ValidateDenseTensor(cos);
        X::Tensor sin(params[2]);
        ValidateDenseTensor(sin);
        int numHeads = params.size() >= 4 ? CheckedInt(params[3], "argument") : 16;
        if (qkv.Info().dtype != X3_TENSOR_FLOAT32 || qkv.Info().rank != 2 ||
            cos.Info().dtype != X3_TENSOR_FLOAT32 || cos.Info().rank != 2 ||
            sin.Info().dtype != X3_TENSOR_FLOAT32 || sin.Info().rank != 2 ||
            qkv.Info().shape[0] != cos.Info().shape[0] || cos.Info().shape[0] != sin.Info().shape[0] ||
            cos.Info().shape[1] != sin.Info().shape[1] || numHeads <= 0) {
            std::cout << "[GarnetAPI] vision_rope shape or dtype mismatch." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        int tokens = static_cast<int>(qkv.Info().shape[0]);
        int headDim = static_cast<int>(cos.Info().shape[1]);
        if (headDim <= 0 || qkv.Info().shape[1] != int64_t{3} * numHeads * headDim ||
            TensorHelper::EnsureGPUMemory(qkv) != TensorOpStatus::Success ||
            TensorHelper::EnsureGPUMemory(cos) != TensorOpStatus::Success ||
            TensorHelper::EnsureGPUMemory(sin) != TensorOpStatus::Success) {
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        auto inputsUse = X::Tensor::AcquireMany({{qkv, X3_TENSOR_READ}, {cos, X3_TENSOR_READ}, {sin, X3_TENSOR_READ}});
        size_t bytes = static_cast<size_t>(qkv.Info().byte_size);
        float* outputDevice = nullptr;
        cudaStream_t stream = nullptr;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err == cudaSuccess) err = cudaMalloc(&outputDevice, bytes);
        if (err == cudaSuccess) {
            err = runVisionRoPEFP32(
                static_cast<const float*>(TensorHelper::GetGPUMemory(qkv)),
                static_cast<const float*>(TensorHelper::GetGPUMemory(cos)),
                static_cast<const float*>(TensorHelper::GetGPUMemory(sin)),
                outputDevice, tokens, numHeads, headDim, stream);
        }
        if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            if (outputDevice) cudaFree(outputDevice);
            if (stream) cudaStreamDestroy(stream);
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        std::vector<int64_t> shape; shape.reserve(2);
        shape.push_back(tokens);
        shape.push_back(static_cast<int>(qkv.Info().shape[1]));
        cudaStreamDestroy(stream);
        auto output = AdoptGpuOutput(Host(), X3_TENSOR_FLOAT32, shape, outputDevice);
        retValue = NativeValue(Host(), X::Value(output));
        return retValue;
    }

    X::Value GarnetAPI::LoadModelEx(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() == 0) return retValue;
        std::string modelPath = params[0].ToString();
        std::string runtimeMode;
        std::string weightsLocation;
        std::string compiledCacheDirectory;
        std::string entryFunction;
        std::string compiledFrontend;
        std::string compiledBackend = "tensorrt";
        std::string compiledPrecision;
        std::vector<std::vector<int>> compiledInputShapes;
        std::vector<std::string> compiledInputDataTypes;
        FusionPartitionOptions compiledPartitionOptions;
        for (auto& item : kwParams) {
            const std::string key(item.first);
            if (key == "runtime_mode") runtimeMode = item.second.ToString();
            else if (key == "weights" && item.second.IsString()) weightsLocation = item.second.ToString();
            else if (key == "cache_dir") compiledCacheDirectory = item.second.ToString();
            else if (key == "entry_function") entryFunction = item.second.ToString();
            else if (key == "frontend") compiledFrontend = item.second.ToString();
            else if (key == "backend") compiledBackend = item.second.ToString();
            else if (key == "precision") compiledPrecision = item.second.ToString();
            else if (key == "input_shapes" && item.second.IsList()) {
                X::Value shapes(item.second);
                for (long long inputIndex = 0; inputIndex < shapes.Size(); ++inputIndex) {
                    X::Value dimensionsValue = shapes.Get(inputIndex);
                    if (!dimensionsValue.IsList()) {
                        compiledInputShapes.clear();
                        break;
                    }
                    X::Value dimensions(dimensionsValue);
                    std::vector<int> shape;
                    for (long long dimension = 0; dimension < dimensions.Size(); ++dimension) {
                        shape.push_back(static_cast<int>(dimensions.Get(dimension).ToLongLong()));
                    }
                    compiledInputShapes.push_back(std::move(shape));
                }
            }
            else if (key == "input_dtypes" && item.second.IsList()) {
                X::Value dataTypes(item.second);
                for (long long index = 0; index < dataTypes.Size(); ++index) {
                    compiledInputDataTypes.push_back(dataTypes.Get(index).ToString());
                }
            }
            else if (key == "compile" && item.second.IsDict()) {
                X::Value compileOptions(item.second);
                X::Value workspaceMb = FindField(compileOptions, "builder_workspace_mb");
                if (workspaceMb.IsValid()) {
                    const unsigned long long megabytes = static_cast<unsigned long long>(
                        (std::max)(64LL, workspaceMb.ToLongLong()));
                    compiledPartitionOptions.builderWorkspaceBytes = megabytes << 20;
                }
                X::Value optimizationLevel = FindField(compileOptions, "builder_optimization_level");
                if (optimizationLevel.IsValid()) {
                    compiledPartitionOptions.builderOptimizationLevel = (std::max)(
                        0, (std::min)(5, static_cast<int>(optimizationLevel.ToLongLong())));
                }
                X::Value partitionValue = FindField(compileOptions, "partition");
                if (partitionValue.IsDict()) {
                    X::Value partition(partitionValue);
                    X::Value preferredEnabled = FindField(partition, "enable_preferred_boundaries");
                    X::Value preferredMin = FindField(partition, "preferred_min_operations");
                    X::Value maxAtomic = FindField(partition, "max_atomic_regions_per_partition");
                    if (preferredEnabled.IsValid()) {
                        compiledPartitionOptions.enablePreferredBoundaries =
                            preferredEnabled.ToLongLong() != 0;
                    }
                    if (preferredMin.IsValid()) {
                        compiledPartitionOptions.preferredMinOperations =
                            (std::max)(1, static_cast<int>(preferredMin.ToLongLong()));
                    }
                    if (maxAtomic.IsValid()) {
                        compiledPartitionOptions.maxAtomicRegionsPerPartition =
                            (std::max)(0, static_cast<int>(maxAtomic.ToLongLong()));
                    }
                }
            }
        }

        namespace fs = std::filesystem;
        fs::path path(modelPath);
        if (path.extension() == ".x") {
            auto pythonPath = path;
            pythonPath.replace_extension(".py");
            if (fs::is_regular_file(pythonPath)) {
                path = std::move(pythonPath);
                modelPath = path.string();
            }
        }

        if (runtimeMode == "compiled_xmodel") {
            auto varModel = CallChecked(__xlang3_package_->GetValue("model"));
            Model& model = *varModel.NativeData<Model>();
            auto emptyWeights = X::Value::Dict(Host());
            std::string directory = path.parent_path().string();
            std::string emptyString;
            model.SetInfo(directory, emptyString, emptyString, emptyWeights);
            if (compiledCacheDirectory.empty()) {
                compiledCacheDirectory = (path.parent_path() / "compiled_cache").string();
            }
            model.InitializeCompiledRuntime(
                modelPath,
                compiledCacheDirectory,
                weightsLocation,
                entryFunction,
                compiledFrontend,
                compiledInputShapes,
                compiledInputDataTypes,
                compiledPartitionOptions,
                compiledBackend,
                compiledPrecision);
            retValue = NativeValue(Host(), varModel);
            return retValue;
        }

        X::Value modelVal;

        if (!fs::is_directory(path) && (path.extension() == ".py" || path.extension() == ".x")) {
            // Load an empty model object
            auto varModel = CallChecked(__xlang3_package_->GetValue("model"));
            Model& model = *varModel.NativeData<Model>();
            auto dictModel = X::Value::Dict(Host());
            std::string dir = path.parent_path().string();
            std::string emptyStr = "";
            model.SetInfo(dir, emptyStr, emptyStr, dictModel);
            modelVal = varModel;

            {
                X::Value weightsDict;
                X::Value inputShapes;
                X::Value weightShape;
                std::string subgraph;
                std::string cacheDir = (path.parent_path() / "cache").string();
                for (auto& it : kwParams) {
                    if (std::string(it.first) == "weights") {
                        weightsDict = it.second;
                    }
                    else if (std::string(it.first) == "input_shapes") {
                        inputShapes = it.second;
                    }
                    else if (std::string(it.first) == "weight_shape") {
                        weightShape = it.second;
                    }
                    else if (std::string(it.first) == "cache_dir") {
                        cacheDir = it.second.ToString();
                    }
                    else if (std::string(it.first) == "subgraph") {
                        subgraph = it.second.ToString();
                    }
                }

                model.SetInfo(dir, emptyStr, emptyStr, weightsDict);
                model.SetSubgraph(subgraph);

                // Keep compilation context on this runtime's package instance.
                SetCurrentWeights(weightsDict);
                SetCompiledEngine(X::Value());

                X3Value evaluated = x3_value_invalid();
                const auto evalStatus = x3_runtime_eval_file(Host()->runtime, modelPath.c_str(), &evaluated);
                X::Value evaluationResult(Host(), evaluated, false);
                if (evalStatus != X3_STATUS_OK) throw X::Error(Host()->runtime_last_error(Host()->runtime));

                if (subgraph == "qwen3_text_mlp" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::Value shapeList(inputShapes);
                    if (shapeList.Size() > 0) {
                        X::Value weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList.Get(uint64_t{0}));
                        std::vector<int> gateShape = TensorShape(FindField(weights, "language_model.layers.0.mlp.gate_proj.weight"));
                        std::vector<int> upShape = TensorShape(FindField(weights, "language_model.layers.0.mlp.up_proj.weight"));
                        std::vector<int> downShape = TensorShape(FindField(weights, "language_model.layers.0.mlp.down_proj.weight"));
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        bool useCudaTextMlp = inputShape.size() == 2 && inputShape[0] > 64;
                        if (useCudaTextMlp) {
                            model.SetEngine(X::Value::String(Host(), "cuda_text_mlp"));
                        }
                        else if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder(Host());
                            builder.ExportTextMLPEngine(enginePath.string(), inputShape, gateShape, upShape, downShape);
                        }
                        if (!useCudaTextMlp && std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value::String(Host(), enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "text_qkv_proj" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::Value shapeList(inputShapes);
                    if (shapeList.Size() > 0) {
                        X::Value weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList.Get(uint64_t{0}));
                        std::vector<int> qShape = TensorShape(FindField(weights, "language_model.layers.0.self_attn.q_proj.weight"));
                        std::vector<int> kShape = TensorShape(FindField(weights, "language_model.layers.0.self_attn.k_proj.weight"));
                        std::vector<int> vShape = TensorShape(FindField(weights, "language_model.layers.0.self_attn.v_proj.weight"));
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder(Host());
                            builder.ExportTextQKVEngine(enginePath.string(), inputShape, qShape, kShape, vShape);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value::String(Host(), enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "text_qkv_head_norm" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::Value shapeList(inputShapes);
                    if (shapeList.Size() > 0) {
                        X::Value weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList.Get(uint64_t{0}));
                        std::vector<int> qShape = TensorShape(FindField(weights, "language_model.layers.0.self_attn.q_proj.weight"));
                        std::vector<int> kShape = TensorShape(FindField(weights, "language_model.layers.0.self_attn.k_proj.weight"));
                        std::vector<int> vShape = TensorShape(FindField(weights, "language_model.layers.0.self_attn.v_proj.weight"));
                        std::vector<int> qNormShape = TensorShape(FindField(weights, "language_model.layers.0.self_attn.q_norm.weight"));
                        std::vector<int> kNormShape = TensorShape(FindField(weights, "language_model.layers.0.self_attn.k_norm.weight"));
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder(Host());
                            builder.ExportTextQKVHeadNormEngine(enginePath.string(), inputShape, qShape, kShape, vShape, qNormShape, kNormShape, 1.0e-6f);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value::String(Host(), enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "text_rope_apply" && inputShapes.IsList()) {
                    X::Value shapeList(inputShapes);
                    if (shapeList.Size() >= 3) {
                        std::vector<int> qkvShape = ReadIntList(shapeList.Get(uint64_t{0}));
                        std::vector<int> cosShape = ReadIntList(shapeList.Get(1));
                        std::vector<int> sinShape = ReadIntList(shapeList.Get(2));
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder(Host());
                            builder.ExportTextRoPEEngine(enginePath.string(), qkvShape, cosShape, sinShape, 16, 8, 128);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value::String(Host(), enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "text_attention_core" && inputShapes.IsList()) {
                    X::Value shapeList(inputShapes);
                    if (shapeList.Size() > 0) {
                        std::vector<int> qkvShape = ReadIntList(shapeList.Get(uint64_t{0}));
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder(Host());
                            builder.ExportTextAttentionEngine(enginePath.string(), qkvShape, 16, 8, 128);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value::String(Host(), enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "vision_attention_core" && inputShapes.IsList()) {
                    X::Value shapeList(inputShapes);
                    if (shapeList.Size() > 0) {
                        std::vector<int> qkvShape = ReadIntList(shapeList.Get(uint64_t{0}));
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (qkvShape.size() == 2 && qkvShape[0] > 512) {
                            model.SetEngine(X::Value::String(Host(), "cuda_exact_vision_attention"));
                        }
                        else if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder(Host());
                            builder.ExportVisionAttentionEngine(enginePath.string(), qkvShape, 16, 64);
                        }
                        if (qkvShape.size() == 2 && qkvShape[0] > 512) {
                            model.SetEngine(X::Value::String(Host(), "cuda_exact_vision_attention"));
                        }
                        else if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value::String(Host(), enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "text_o_proj" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::Value shapeList(inputShapes);
                    if (shapeList.Size() > 0) {
                        X::Value weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList.Get(uint64_t{0}));
                        std::vector<int> oShape = TensorShape(FindField(weights, "language_model.layers.0.self_attn.o_proj.weight"));
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder(Host());
                            builder.ExportLinearTransposeEngine(enginePath.string(), inputShape, oShape);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value::String(Host(), enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "text_lm_head" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::Value shapeList(inputShapes);
                    if (shapeList.Size() > 0) {
                        X::Value weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList.Get(uint64_t{0}));
                        std::vector<int> embedShape = TensorShape(FindField(weights, "language_model.embed_tokens.weight"));
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        bool useCudaLinear = inputShape.size() == 2
                            && embedShape.size() == 2
                            && (embedShape[0] > 65536 || (static_cast<long long>(inputShape[0]) * static_cast<long long>(embedShape[0]) > 8LL * 1024LL * 1024LL));
                        if (useCudaLinear) {
                            model.SetEngine(X::Value::String(Host(), "cuda_linear_transpose"));
                        }
                        else if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder(Host());
                            builder.ExportLinearTransposeEngine(enginePath.string(), inputShape, embedShape);
                        }
                        if (!useCudaLinear && std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value::String(Host(), enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "vision_patch_embed" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::Value shapeList(inputShapes);
                    if (shapeList.Size() > 0) {
                        X::Value weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList.Get(uint64_t{0}));
                        std::vector<int> weightShape = TensorShape(FindField(weights, "visual.patch_embed.proj.weight"));
                        std::vector<int> biasShape = TensorShape(FindField(weights, "visual.patch_embed.proj.bias"));
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder(Host());
                            builder.ExportLinearBiasTransposeEngine(enginePath.string(), inputShape, weightShape, biasShape);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value::String(Host(), enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "linear_bias" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::Value shapeList(inputShapes);
                    if (shapeList.Size() > 0) {
                        X::Value weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList.Get(uint64_t{0}));
                        std::vector<int> weightShape = TensorShape(FindField(weights, "W"));
                        std::vector<int> biasShape = TensorShape(FindField(weights, "B"));
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        bool useCudaLinear = inputShape.size() == 2 && weightShape.size() == 2
                            && (inputShape[0] > 2048 || (static_cast<long long>(inputShape[0]) * static_cast<long long>(weightShape[0]) > 8LL * 1024LL * 1024LL));
                        if (useCudaLinear) {
                            model.SetEngine(X::Value::String(Host(), "cuda_linear_bias_transpose"));
                        }
                        else if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder(Host());
                            builder.ExportLinearBiasTransposeEngine(enginePath.string(), inputShape, weightShape, biasShape);
                        }
                        if (useCudaLinear) {
                            model.SetEngine(X::Value::String(Host(), "cuda_linear_bias_transpose"));
                        }
                        else if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value::String(Host(), enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "vision_mlp" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::Value shapeList(inputShapes);
                    if (shapeList.Size() > 0) {
                        X::Value weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList.Get(uint64_t{0}));
                        std::vector<int> fc1Shape = TensorShape(FindField(weights, "visual.blocks.0.mlp.linear_fc1.weight"));
                        std::vector<int> fc2Shape = TensorShape(FindField(weights, "visual.blocks.0.mlp.linear_fc2.weight"));
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder(Host());
                            builder.ExportVisionMLPEngine(enginePath.string(), inputShape, fc1Shape, fc2Shape);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value::String(Host(), enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "rms_norm" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::Value shapeList(inputShapes);
                    if (shapeList.Size() > 0) {
                        X::Value weights(weightsDict);
                        model.SetRMSNormWeight(FindField(weights, "language_model.layers.0.input_layernorm.weight"));
                        std::vector<int> inputShape = ReadIntList(shapeList.Get(uint64_t{0}));
                        std::vector<int> weightShape = TensorShape(FindField(weights, "language_model.layers.0.input_layernorm.weight"));
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder(Host());
                            builder.ExportRMSNormEngine(enginePath.string(), inputShape, weightShape, 1.0e-6f);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value::String(Host(), enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "text_post_attention_rms_norm" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::Value shapeList(inputShapes);
                    if (shapeList.Size() > 0) {
                        X::Value weights(weightsDict);
                        model.SetRMSNormWeight(FindField(weights, "language_model.layers.0.post_attention_layernorm.weight"));
                        std::vector<int> inputShape = ReadIntList(shapeList.Get(uint64_t{0}));
                        std::vector<int> weightShape = TensorShape(FindField(weights, "language_model.layers.0.post_attention_layernorm.weight"));
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder(Host());
                            builder.ExportRMSNormEngine(enginePath.string(), inputShape, weightShape, 1.0e-6f);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value::String(Host(), enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "layer_norm" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::Value shapeList(inputShapes);
                    if (shapeList.Size() > 0) {
                        X::Value weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList.Get(uint64_t{0}));
                        std::vector<int> weightShape = TensorShape(FindField(weights, "visual.blocks.0.norm1.weight"));
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder(Host());
                            builder.ExportLayerNormEngine(enginePath.string(), inputShape, weightShape, 1.0e-6f);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value::String(Host(), enginePath.string()));
                        }
                    }
                }
                else if (inputShapes.IsList() && weightShape.IsList()) {
                    X::Value shapeList(inputShapes);
                    if (shapeList.Size() > 0) {
                        std::vector<int> inputShape = ReadIntList(shapeList.Get(uint64_t{0}));
                        std::vector<int> wShape = ReadIntList(weightShape);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder(Host());
                            builder.ExportMatmulEngine(enginePath.string(), inputShape, wShape);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value::String(Host(), enginePath.string()));
                        }
                    }
                }

                // Extract compiled engine that was set during script execution
                X::Value compiledEngine = GetCompiledEngine();
                if (!model.GetEngine().IsValid() && compiledEngine.IsValid()) {
                    model.SetEngine(compiledEngine);
                } else if (!model.GetEngine().IsValid()) {
                    std::cout << "[Garnet] Warning: Script finished but no engine was compiled!" << std::endl;
                }
            }
        }
        else {
            modelVal = LoadModel(modelPath);
        }

        retValue = NativeValue(Host(), modelVal);
        return retValue;
    }

    X::Value GarnetAPI::QwenVLSmartResize(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        try {
            int height = GetIntArg(params, kwParams, 0, "height", 0);
            int width = GetIntArg(params, kwParams, 1, "width", 0);
            int patchSize = GetIntArg(params, kwParams, 2, "patch_size", 16);
            int mergeSize = GetIntArg(params, kwParams, 3, "merge_size", 2);
            int minPixels = GetIntArg(params, kwParams, 4, "min_pixels", 65536);
            int maxPixels = GetIntArg(params, kwParams, 5, "max_pixels", 65536);
            auto resized = Image::QwenVL::SmartResize(
                height,
                width,
                patchSize * mergeSize,
                minPixels,
                maxPixels);

            auto result = X::Value::Dict(Host());
            int gridH = resized.height / patchSize;
            int gridW = resized.width / patchSize;
            result.SetItem("height", X::Value(resized.height));
            result.SetItem("width", X::Value(resized.width));
            result.SetItem("patch_size", X::Value(patchSize));
            result.SetItem("merge_size", X::Value(mergeSize));
            result.SetItem("grid_h", X::Value(gridH));
            result.SetItem("grid_w", X::Value(gridW));
            result.SetItem("visual_tokens", X::Value(gridH * gridW / (mergeSize * mergeSize)));
            retValue = NativeValue(Host(), result);
        }
        catch (const std::exception& exc) {
            std::cout << "[GarnetAPI] qwen_vl_smart_resize failed: " << exc.what() << std::endl;
            retValue = NativeValue(Host(), X::Value());
        }
        return retValue;
    }

    X::Value GarnetAPI::QwenVLCreateRequest(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        auto totalStart = std::chrono::steady_clock::now();
        try {
            std::string modelDir = GetStringArg(params, kwParams, 0, "model_dir", "");
            std::string imagePath = GetStringArg(params, kwParams, 1, "image_path", "");
            std::string prompt = GetStringArg(params, kwParams, 2, "prompt", "");
            int minPixels = GetIntArg(params, kwParams, 3, "min_pixels", 65536);
            int maxPixels = GetIntArg(params, kwParams, 4, "max_pixels", 65536);
            if (modelDir.empty() || imagePath.empty() || prompt.empty()) {
                std::cout << "[GarnetAPI] qwen_vl_create_request requires model_dir, image_path, and prompt." << std::endl;
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }

            constexpr int patchSize = 16;
            constexpr int temporalPatchSize = 2;
            constexpr int mergeSize = 2;
            int featureDim = 3 * temporalPatchSize * patchSize * patchSize;

            auto imageStart = std::chrono::steady_clock::now();
            auto imageResult = Image::QwenVL::PreprocessJpegFileToTensor(Host(), imagePath, minPixels, maxPixels);
            double imageMs = MsSince(imageStart);
            X::Tensor pixelValues(imageResult.pixelValues);
            X::Tensor imageGridTensor(imageResult.imageGridTHW);
            auto* gridData = reinterpret_cast<long long*>(imageGridTensor.Info().data);
            long long grid[3] = { gridData[0], gridData[1], gridData[2] };

            int patchCount = (imageResult.resizedHeight / patchSize) * (imageResult.resizedWidth / patchSize);
            if (patchCount <= 0) {
                std::cout << "[GarnetAPI] qwen_vl_create_request invalid patch count: " << patchCount << std::endl;
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }

            auto tokenStart = std::chrono::steady_clock::now();
            std::string tokenError;
            auto tokenizer = Tokenization::GetCachedQwenTokenizer(modelDir, &tokenError);
            if (!tokenizer) {
                std::cout << "[GarnetAPI] qwen_vl_create_request tokenizer load failed: " << tokenError << std::endl;
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }
            std::vector<int64_t> promptIds = Tokenization::QwenVLPromptBuilder::BuildSingleImagePromptIds(
                *tokenizer,
                prompt,
                grid,
                mergeSize);
            int64_t imagePadId = tokenizer->TokenId("<|image_pad|>");
            if (imagePadId < 0 ||
                tokenizer->TokenId("<|vision_start|>") < 0 ||
                tokenizer->TokenId("<|vision_end|>") < 0 ||
                tokenizer->TokenId("<|im_start|>") < 0 ||
                tokenizer->TokenId("<|im_end|>") < 0) {
                std::cout << "[GarnetAPI] qwen_vl_create_request tokenizer missing required Qwen-VL special tokens." << std::endl;
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }

            std::vector<int64_t> inputIds;
            std::vector<int64_t> mmTypes;
            inputIds.reserve(promptIds.size());
            mmTypes.reserve(promptIds.size());
            int visualTokenCount = 0;
            for (int64_t id : promptIds) {
                inputIds.push_back(static_cast<long long>(id));
                long long mmType = id == imagePadId ? 1LL : 0LL;
                mmTypes.push_back(mmType);
                if (mmType == 1) {
                    ++visualTokenCount;
                }
            }
            int expectedVisualTokenCount = static_cast<int>((grid[0] * grid[1] * grid[2]) / (mergeSize * mergeSize));
            if (visualTokenCount != expectedVisualTokenCount) {
                std::cout << "[GarnetAPI] qwen_vl_create_request visual token mismatch: prompt="
                    << visualTokenCount << ", grid=" << expectedVisualTokenCount << std::endl;
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }
            double tokenizeMs = MsSince(tokenStart);

            auto uploadStart = std::chrono::steady_clock::now();
            X::Value inputIdsTensor = MakeInt64Tensor(Host(), inputIds, true);
            X::Value mmTypesTensor = MakeInt64Tensor(Host(), mmTypes, true);
            double uploadMs = MsSince(uploadStart);
            if (!X::Tensor::IsTensor(inputIdsTensor) || !X::Tensor::IsTensor(mmTypesTensor) ||
                TensorHelper::GetGPUMemory(pixelValues) == nullptr) {
                std::cout << "[GarnetAPI] qwen_vl_create_request failed to create GPU tensors." << std::endl;
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }

            auto requestValue = CallChecked(__xlang3_package_->GetValue("QwenVLRequestContext"));
            QwenVLRequestContext& request = *requestValue.NativeData<QwenVLRequestContext>();
            request.inputIds = inputIdsTensor;
            request.mmTokenTypeIds = mmTypesTensor;
            request.pixelValues = imageResult.pixelValues;
            request.imageGridTHW = imageResult.imageGridTHW;
            request.modelDir = modelDir;
            request.imagePath = imagePath;
            request.prompt = prompt;
            request.sourceHeight = imageResult.sourceHeight;
            request.sourceWidth = imageResult.sourceWidth;
            request.resizedHeight = imageResult.resizedHeight;
            request.resizedWidth = imageResult.resizedWidth;
            request.promptTokenCount = static_cast<int>(inputIds.size());
            request.visualTokenCount = visualTokenCount;
            request.pixelValueCount = patchCount * featureDim;
            request.patchSize = patchSize;
            request.temporalPatchSize = temporalPatchSize;
            request.mergeSize = mergeSize;
            request.imagePreprocessUs = static_cast<long long>(imageMs * 1000.0);
            request.tokenizeUs = static_cast<long long>(tokenizeMs * 1000.0);
            request.tensorUploadUs = static_cast<long long>(uploadMs * 1000.0);
            request.totalUs = static_cast<long long>(MsSince(totalStart) * 1000.0);
            retValue = NativeValue(Host(), requestValue);
        }
        catch (const std::exception& exc) {
            std::cout << "[GarnetAPI] qwen_vl_create_request failed: " << exc.what() << std::endl;
            retValue = NativeValue(Host(), X::Value());
        }
        return retValue;
    }

    X::Value GarnetAPI::QwenVLPrepareRequest(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        auto totalStart = std::chrono::steady_clock::now();
        try {
            std::string modelDir = GetStringArg(params, kwParams, 0, "model_dir", "");
            std::string imagePath = GetStringArg(params, kwParams, 1, "image_path", "");
            std::string prompt = GetStringArg(params, kwParams, 2, "prompt", "");
            int minPixels = GetIntArg(params, kwParams, 3, "min_pixels", 65536);
            int maxPixels = GetIntArg(params, kwParams, 4, "max_pixels", 65536);
            if (modelDir.empty() || imagePath.empty() || prompt.empty()) {
                std::cout << "[GarnetAPI] qwen_vl_prepare_request requires model_dir, image_path, and prompt." << std::endl;
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }

            constexpr int patchSize = 16;
            constexpr int temporalPatchSize = 2;
            constexpr int mergeSize = 2;
            int featureDim = 3 * temporalPatchSize * patchSize * patchSize;

            auto imageStart = std::chrono::steady_clock::now();
            auto imageResult = Image::QwenVL::PreprocessJpegFileToTensor(Host(), imagePath, minPixels, maxPixels);
            double imageMs = MsSince(imageStart);
            int patchCount = (imageResult.resizedHeight / patchSize) * (imageResult.resizedWidth / patchSize);
            if (patchCount <= 0) {
                std::cout << "[GarnetAPI] qwen_vl_prepare_request invalid patch count: " << patchCount << std::endl;
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }
            X::Tensor pixelValues(imageResult.pixelValues);
            X::Tensor imageGridTensor(imageResult.imageGridTHW);
            auto* gridData = reinterpret_cast<long long*>(imageGridTensor.Info().data);
            long long grid[3] = { gridData[0], gridData[1], gridData[2] };

            auto tokenStart = std::chrono::steady_clock::now();
            std::string tokenError;
            auto tokenizer = Tokenization::GetCachedQwenTokenizer(modelDir, &tokenError);
            if (!tokenizer) {
                std::cout << "[GarnetAPI] qwen_vl_prepare_request tokenizer load failed: " << tokenError << std::endl;
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }
            std::vector<int64_t> promptIds = Tokenization::QwenVLPromptBuilder::BuildSingleImagePromptIds(
                *tokenizer,
                prompt,
                grid,
                mergeSize);
            int64_t imagePadId = tokenizer->TokenId("<|image_pad|>");
            if (imagePadId < 0 ||
                tokenizer->TokenId("<|vision_start|>") < 0 ||
                tokenizer->TokenId("<|vision_end|>") < 0 ||
                tokenizer->TokenId("<|im_start|>") < 0 ||
                tokenizer->TokenId("<|im_end|>") < 0) {
                std::cout << "[GarnetAPI] qwen_vl_prepare_request tokenizer missing required Qwen-VL special tokens." << std::endl;
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }
            double tokenizeMs = MsSince(tokenStart);

            std::vector<int64_t> inputIds;
            std::vector<int64_t> mmTypes;
            inputIds.reserve(promptIds.size());
            mmTypes.reserve(promptIds.size());
            int visualTokenCount = 0;
            for (int64_t id : promptIds) {
                inputIds.push_back(static_cast<long long>(id));
                long long mmType = id == imagePadId ? 1LL : 0LL;
                mmTypes.push_back(mmType);
                if (mmType == 1) {
                    ++visualTokenCount;
                }
            }

            int expectedVisualTokenCount = static_cast<int>((grid[0] * grid[1] * grid[2]) / (mergeSize * mergeSize));
            if (visualTokenCount != expectedVisualTokenCount) {
                std::cout << "[GarnetAPI] qwen_vl_prepare_request visual token mismatch: prompt="
                    << visualTokenCount << ", grid=" << expectedVisualTokenCount << std::endl;
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }

            std::vector<int64_t> gridVector = { grid[0], grid[1], grid[2] };
            const auto mropeMetadata = Tokenization::QwenVLPromptBuilder::BuildSingleImageMRoPEMetadata(
                mmTypes,
                grid,
                mergeSize);
            auto dict = X::Value::Dict(Host());
            dict.SetItem("input_ids", MakeInt64List(Host(), inputIds));
            dict.SetItem("mm_token_type_ids", MakeInt64List(Host(), mmTypes));
            X::Value inputIdsTensorValue = MakeInt64Tensor(Host(), inputIds, true);
            X::Value mmTypesTensorValue = MakeInt64Tensor(Host(), mmTypes, true);
            dict.SetItem("input_ids_tensor", inputIdsTensorValue);
            dict.SetItem("mm_token_type_ids_tensor", mmTypesTensorValue);
            dict.SetItem("pixel_values", imageResult.pixelValues);
            dict.SetItem("vision_bilinear_indices", imageResult.bilinearIndices);
            dict.SetItem("vision_bilinear_weights", imageResult.bilinearWeights);
            dict.SetItem("vision_position_ids", imageResult.visionPositionIds);
            dict.SetItem("vision_cu_seqlens", imageResult.visionCuSeqlens);
            dict.SetItem("position_ids", MakeInt64Tensor3DGpu(Host(),
                mropeMetadata.positionIds,
                3,
                1,
                static_cast<int>(inputIds.size())));
            dict.SetItem("mrope_position_deltas", MakeInt64Tensor2D(Host(),
                { mropeMetadata.positionDelta }, 1, 1, true));
            dict.SetItem("pixel_values_shape", MakeInt64List(Host(), {
                static_cast<long long>(patchCount),
                static_cast<long long>(featureDim),
            }));
            dict.SetItem("pixel_value_count", X::Value(patchCount * featureDim));
            dict.SetItem("image_grid_thw", MakeInt64List(Host(), gridVector));
            dict.SetItem("pixel_values_gpu", X::Value(TensorHelper::GetGPUMemory(pixelValues) != nullptr));
            bool inputIdsGpu = false;
            bool mmTypesGpu = false;
            if (X::Tensor::IsTensor(inputIdsTensorValue)) {
                X::Tensor inputIdsTensor(inputIdsTensorValue);
                inputIdsGpu = TensorHelper::GetGPUMemory(inputIdsTensor) != nullptr;
            }
            if (X::Tensor::IsTensor(mmTypesTensorValue)) {
                X::Tensor mmTypesTensor(mmTypesTensorValue);
                mmTypesGpu = TensorHelper::GetGPUMemory(mmTypesTensor) != nullptr;
            }
            dict.SetItem("input_ids_gpu", X::Value(inputIdsGpu));
            dict.SetItem("mm_token_type_ids_gpu", X::Value(mmTypesGpu));

            auto timings = X::Value::Dict(Host());
            timings.SetItem("image_preprocess_us", X::Value(static_cast<long long>(imageMs * 1000.0)));
            timings.SetItem("tokenize_us", X::Value(static_cast<long long>(tokenizeMs * 1000.0)));
            timings.SetItem("total_us", X::Value(static_cast<long long>(MsSince(totalStart) * 1000.0)));
            dict.SetItem("prompt_token_count", X::Value(static_cast<int>(inputIds.size())));
            dict.SetItem("visual_token_count", X::Value(visualTokenCount));
            dict.SetItem("source_height", X::Value(imageResult.sourceHeight));
            dict.SetItem("source_width", X::Value(imageResult.sourceWidth));
            dict.SetItem("height", X::Value(imageResult.resizedHeight));
            dict.SetItem("width", X::Value(imageResult.resizedWidth));
            dict.SetItem("patch_size", X::Value(patchSize));
            dict.SetItem("temporal_patch_size", X::Value(temporalPatchSize));
            dict.SetItem("merge_size", X::Value(mergeSize));
            dict.SetItem("backend", X::Value::String(Host(), "qwen_vl_request_native_tokenizer_nvjpeg_cuda_gpu_xtensor"));
            dict.SetItem("timings", timings);
            retValue = NativeValue(Host(), dict);
        }
        catch (const std::exception& exc) {
            std::cout << "[GarnetAPI] qwen_vl_prepare_request failed: " << exc.what() << std::endl;
            retValue = NativeValue(Host(), X::Value());
        }
        return retValue;
    }

    X::Value GarnetAPI::QwenVLPreprocessImage(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        try {
            X::Value image = GetKwarg(kwParams, "image");
            if (!image.IsValid() && params.size() > 0) {
                image = params[0];
            }
            if (!image.IsValid()) {
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }

            Image::QwenVL::QwenVLImagePreprocessConfig config;
            int height = GetIntArg(params, kwParams, 1, "height", 0);
            int width = GetIntArg(params, kwParams, 2, "width", 0);
            config.patchSize = GetIntArg(params, kwParams, 3, "patch_size", 16);
            config.temporalPatchSize = GetIntArg(params, kwParams, 4, "temporal_patch_size", 2);
            config.mergeSize = GetIntArg(params, kwParams, 5, "merge_size", 2);
            config.inputScale = static_cast<float>(GetDoubleArg(params, kwParams, 6, "input_scale", 255.0));
            std::string inputFormat = GetStringArg(params, kwParams, 7, "format", "rgb");
            config.pixelFormat = Image::PixelFormatFromString(inputFormat);

            auto result = Image::QwenVL::PreprocessRawImageTensor(image, height, width, config);
            auto dict = X::Value::Dict(Host());
            dict.SetItem("pixel_values", result.pixelValues);
            dict.SetItem("image_grid_thw", result.imageGridTHW);
            dict.SetItem("vision_bilinear_indices", result.bilinearIndices);
            dict.SetItem("vision_bilinear_weights", result.bilinearWeights);
            dict.SetItem("vision_position_ids", result.visionPositionIds);
            dict.SetItem("vision_cu_seqlens", result.visionCuSeqlens);
            dict.SetItem("source_height", X::Value(result.sourceHeight));
            dict.SetItem("source_width", X::Value(result.sourceWidth));
            dict.SetItem("height", X::Value(result.resizedHeight));
            dict.SetItem("width", X::Value(result.resizedWidth));
            dict.SetItem("patch_size", X::Value(result.patchSize));
            dict.SetItem("temporal_patch_size", X::Value(result.temporalPatchSize));
            dict.SetItem("merge_size", X::Value(result.mergeSize));
            dict.SetItem("input_format", X::Value::String(Host(), inputFormat));
            dict.SetItem("backend", X::Value::String(Host(), "cuda_raw_tensor"));
            retValue = NativeValue(Host(), dict);
        }
        catch (const std::exception& exc) {
            std::cout << "[GarnetAPI] qwen_vl_preprocess_image failed: " << exc.what() << std::endl;
            retValue = NativeValue(Host(), X::Value());
        }
        return retValue;
    }

    X::Value GarnetAPI::QwenVLPreprocessJpegFile(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        try {
            std::string path = GetStringArg(params, kwParams, 0, "path", "");
            int minPixels = GetIntArg(params, kwParams, 1, "min_pixels", 65536);
            int maxPixels = GetIntArg(params, kwParams, 2, "max_pixels", 65536);
            if (path.empty() || minPixels <= 0 || maxPixels <= 0) {
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }

            auto result = Image::QwenVL::PreprocessJpegFileToTensor(Host(), path, minPixels, maxPixels);
            X::Tensor pixelValues(result.pixelValues);
            X::Tensor imageGrid(result.imageGridTHW);

            auto dict = X::Value::Dict(Host());
            dict.SetItem("pixel_values", result.pixelValues);
            dict.SetItem("image_grid_thw", result.imageGridTHW);
            dict.SetItem("vision_bilinear_indices", result.bilinearIndices);
            dict.SetItem("vision_bilinear_weights", result.bilinearWeights);
            dict.SetItem("vision_position_ids", result.visionPositionIds);
            dict.SetItem("vision_cu_seqlens", result.visionCuSeqlens);
            dict.SetItem("height", X::Value(result.resizedHeight));
            dict.SetItem("width", X::Value(result.resizedWidth));
            dict.SetItem("patch_size", X::Value(result.patchSize));
            dict.SetItem("temporal_patch_size", X::Value(result.temporalPatchSize));
            dict.SetItem("merge_size", X::Value(result.mergeSize));
            dict.SetItem("pixel_values_gpu", X::Value(TensorHelper::GetGPUMemory(pixelValues) != nullptr));
            dict.SetItem("image_grid_gpu", X::Value(TensorHelper::GetGPUMemory(imageGrid) != nullptr));
            dict.SetItem("backend", X::Value::String(Host(), "cuda_nvjpeg_to_gpu_xtensor"));
            retValue = NativeValue(Host(), dict);
        }
        catch (const std::exception& exc) {
            std::cout << "[GarnetAPI] qwen_vl_preprocess_jpeg_file failed: " << exc.what() << std::endl;
            retValue = NativeValue(Host(), X::Value());
        }
        return retValue;
    }

    X::Value GarnetAPI::RunTest(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        return retValue;
    }

}
