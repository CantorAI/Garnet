#include "garnet.h"
#include "../trt/trt_builder.h"
#include "../image/qwen_vl/qwen_vl_image_preprocessor.h"
#include "../tokenizer/qwen_tokenizer.h"
#include "../cuda/cuda_lib.h"
#include "../tensor/tensor_helper.h"
#include "xpackage.h"
#include "xlang.h"
#include <fstream> 
#include <numeric> 
#include <filesystem>
#include <regex>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <cuda_runtime.h>

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

        Garnet::Tokenization::QwenTokenizer tokenizer;
        std::string tokenizerError;
        if (!tokenizer.LoadFromFolder(modelDir, &tokenizerError)) {
            setError(tokenizerError);
            return 1;
        }
        std::vector<int64_t> promptIds = Garnet::Tokenization::QwenVLPromptBuilder::BuildSingleImagePromptIds(
            tokenizer,
            prompt,
            grid,
            mergeSize);
        int64_t imagePadId = tokenizer.TokenId("<|image_pad|>");
        if (imagePadId < 0 ||
            tokenizer.TokenId("<|vision_start|>") < 0 ||
            tokenizer.TokenId("<|vision_end|>") < 0 ||
            tokenizer.TokenId("<|im_start|>") < 0 ||
            tokenizer.TokenId("<|im_end|>") < 0) {
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
    namespace
    {
        std::vector<int> ReadIntList(X::Value value)
        {
            std::vector<int> result;
            if (!value.IsList()) return result;
            X::List list(value);
            long long size = list->Size();
            result.reserve(static_cast<size_t>(size));
            for (long long i = 0; i < size; ++i) {
                result.push_back(static_cast<int>(list->Get(i).ToLongLong()));
            }
            return result;
        }

        std::vector<int> TensorShape(X::Value value)
        {
            std::vector<int> result;
            if (!value.IsTensor()) return result;
            X::Tensor tensor(value);
            int dimCount = tensor->GetDimCount();
            result.reserve(static_cast<size_t>(dimCount));
            for (int i = 0; i < dimCount; ++i) {
                result.push_back(static_cast<int>(tensor->GetDimSize(i)));
            }
            return result;
        }

        X::Value GetKwarg(X::KWARGS& kwParams, const char* name)
        {
            if (kwParams.Has(name)) {
                auto it = kwParams.find(name);
                return it->val;
            }
            return X::Value();
        }

        int GetIntArg(X::ARGS& params, X::KWARGS& kwParams, size_t index, const char* name, int defaultValue)
        {
            X::Value value = GetKwarg(kwParams, name);
            if (value.IsValid()) {
                return static_cast<int>(value.ToLongLong());
            }
            if (params.size() > index) {
                return static_cast<int>(params[index].ToLongLong());
            }
            return defaultValue;
        }

        double GetDoubleArg(X::ARGS& params, X::KWARGS& kwParams, size_t index, const char* name, double defaultValue)
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

        std::string GetStringArg(X::ARGS& params, X::KWARGS& kwParams, size_t index, const char* name, const std::string& defaultValue)
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

        X::Value MakeInt64Tensor(const std::vector<long long>& values, bool ensureGpu = false)
        {
            X::Tensor tensor;
            X::Port::vector<int> shape(1);
            shape.push_back(static_cast<int>(values.size()));
            tensor->SetDataType(X::TensorDataType::INT64);
            tensor->SetShape(shape);
            X::Value init;
            bool created = tensor->Create(init);
            if (!created || tensor->GetData() == nullptr) {
                std::cout << "[GarnetAPI] MakeInt64Tensor failed to allocate count=" << values.size() << std::endl;
                return X::Value();
            }
            if (!values.empty()) {
                std::memcpy(tensor->GetData(), values.data(), values.size() * sizeof(long long));
            }
            if (ensureGpu && TensorHelper::EnsureGPUMemory(tensor) != TensorOpStatus::Success) {
                std::cout << "[GarnetAPI] MakeInt64Tensor failed to move tensor to GPU count=" << values.size() << std::endl;
                return X::Value();
            }
            return X::Value(tensor);
        }

        X::Value MakeInt64List(const std::vector<long long>& values)
        {
            X::V<X::XList> list;
            for (long long value : values) {
                X::Value item(value);
                list->AddItem(item);
            }
            return list;
        }

        X::Value MakeInt64Tensor2D(const std::vector<long long>& values, int rows, int cols)
        {
            X::Tensor tensor;
            X::Port::vector<int> shape(2);
            shape.push_back(rows);
            shape.push_back(cols);
            tensor->SetDataType(X::TensorDataType::INT64);
            tensor->SetShape(shape);
            X::Value init;
            bool created = tensor->Create(init);
            if (!created || tensor->GetData() == nullptr) {
                std::cout << "[GarnetAPI] MakeInt64Tensor2D failed to allocate count=" << values.size() << std::endl;
                return X::Value();
            }
            if (!values.empty()) {
                std::memcpy(tensor->GetData(), values.data(), values.size() * sizeof(long long));
            }
            return X::Value(tensor);
        }

        X::Value MakeFloatTensor2D(const float* data, int rows, int cols)
        {
            X::Tensor tensor;
            X::Port::vector<int> shape(2);
            shape.push_back(rows);
            shape.push_back(cols);
            tensor->SetDataType(X::TensorDataType::FLOAT32);
            tensor->SetShape(shape);
            X::Value init;
            bool created = tensor->Create(init);
            if (!created || tensor->GetData() == nullptr) {
                std::cout << "[GarnetAPI] MakeFloatTensor2D failed to allocate rows=" << rows << " cols=" << cols << std::endl;
                return X::Value();
            }
            if (data != nullptr && rows > 0 && cols > 0) {
                std::memcpy(tensor->GetData(), data, static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float));
            }
            return X::Value(tensor);
        }

        double MsSince(std::chrono::steady_clock::time_point start)
        {
            auto elapsed = std::chrono::steady_clock::now() - start;
            return std::chrono::duration<double, std::milli>(elapsed).count();
        }
    }

    void QwenVLRequestContext::Stats(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        auto tensorGpu = [](X::Value value) {
            if (!value.IsTensor()) {
                return false;
            }
            X::Tensor tensor(value);
            return TensorHelper::GetGPUMemory(tensor) != nullptr;
        };

        X::Dict stats;
        stats->Set("source_height", X::Value(sourceHeight));
        stats->Set("source_width", X::Value(sourceWidth));
        stats->Set("height", X::Value(resizedHeight));
        stats->Set("width", X::Value(resizedWidth));
        stats->Set("prompt_token_count", X::Value(promptTokenCount));
        stats->Set("visual_token_count", X::Value(visualTokenCount));
        stats->Set("pixel_value_count", X::Value(pixelValueCount));
        stats->Set("patch_size", X::Value(patchSize));
        stats->Set("temporal_patch_size", X::Value(temporalPatchSize));
        stats->Set("merge_size", X::Value(mergeSize));
        stats->Set("input_ids_gpu", X::Value(tensorGpu(inputIds)));
        stats->Set("mm_token_type_ids_gpu", X::Value(tensorGpu(mmTokenTypeIds)));
        stats->Set("pixel_values_gpu", X::Value(tensorGpu(pixelValues)));
        stats->Set("kv_allocated", X::Value(kvHandle > 0));
        stats->Set("kv_handle", X::Value(kvHandle));
        stats->Set("kv_max_tokens", X::Value(kvMaxTokens));
        stats->Set("kv_logical_length", X::Value(kvLogicalLength));
        stats->Set("kv_page_size", X::Value(kvPageSize));
        stats->Set("kv_logical_pages", X::Value(kvLogicalPages));
        stats->Set("kv_physical_pages", X::Value(kvPhysicalPages));
        stats->Set("kv_q_heads", X::Value(kvQHeads));
        stats->Set("kv_heads", X::Value(kvHeads));
        stats->Set("kv_head_dim", X::Value(kvHeadDim));
        stats->Set("image_preprocess_us", X::Value(imagePreprocessUs));
        stats->Set("tokenize_us", X::Value(tokenizeUs));
        stats->Set("tensor_upload_us", X::Value(tensorUploadUs));
        stats->Set("total_us", X::Value(totalUs));
        retValue = stats;
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
        X::V<X::XList> retList;
        for (int page : pages) {
            X::Value pageValue(page);
            retList->AddItem(pageValue);
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

    void KVCacheManager::Allocate(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
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
            retValue = X::Value();
            return;
        }
        retValue = MakePageList(m_sequences[seqId].pages);
    }

    void KVCacheManager::Append(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        long long seqId = params.size() > 0 ? params[0].ToLongLong() : 0;
        X::Value appendValue = GetKwarg(kwParams, "tokens");
        long long appendTokens = appendValue.IsValid()
            ? appendValue.ToLongLong()
            : (params.size() > 1 ? params[1].ToLongLong() : 1);
        if (appendTokens < 0) appendTokens = 0;
        auto existing = m_sequences.find(seqId);
        if (existing == m_sequences.end()) {
            if (!EnsurePages(seqId, appendTokens)) {
                retValue = X::Value();
                return;
            }
        }
        else {
            long long newLength = existing->second.logicalLength + appendTokens;
            if (!EnsurePages(seqId, newLength)) {
                retValue = X::Value();
                return;
            }
        }

        const auto& state = m_sequences[seqId];
        X::Dict dict;
        dict->Set("sequence_id", X::Value(seqId));
        dict->Set("logical_length", X::Value(state.logicalLength));
        dict->Set("page_count", X::Value(static_cast<int>(state.pages.size())));
        dict->Set("pages", MakePageList(state.pages));
        retValue = dict;
    }

    void KVCacheManager::Free(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        long long seqId = params.size() > 0 ? params[0].ToLongLong() : 0;
        auto existing = m_sequences.find(seqId);
        bool released = existing != m_sequences.end();
        if (released) {
            for (int page : existing->second.pages) {
                m_freePages.push_back(page);
            }
            m_sequences.erase(existing);
        }
        retValue = X::Value(released);
    }

    void KVCacheManager::Stats(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        X::Dict stats;
        stats->Set("max_num_pages", X::Value(m_maxNumPages));
        stats->Set("free_pages", X::Value(static_cast<int>(m_freePages.size())));
        stats->Set("used_pages", X::Value(m_maxNumPages - static_cast<int>(m_freePages.size())));
        stats->Set("page_size", X::Value(m_pageSize));
        stats->Set("head_dim", X::Value(m_headDim));
        stats->Set("num_kv_heads", X::Value(m_numKVHeads));
        stats->Set("num_layers", X::Value(m_numLayers));
        stats->Set("dtype_bytes", X::Value(m_dtypeBytes));
        stats->Set("device_id", X::Value(m_deviceId));
        stats->Set("bytes_per_page_per_layer", X::Value(static_cast<long long>(m_bytesPerPagePerLayer)));
        stats->Set("total_key_bytes", X::Value(static_cast<long long>(m_totalBytes)));
        stats->Set("total_value_bytes", X::Value(static_cast<long long>(m_totalBytes)));
        stats->Set("gpu_allocated", X::Value(m_keyArena != nullptr && m_valueArena != nullptr));
        stats->Set("sequence_count", X::Value(static_cast<int>(m_sequences.size())));
        retValue = stats;
    }

    bool GarnetAPI::LoadModelFromFile(std::string modelPath, X::Dict& model)
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
            X::TensorDataType tensor_data_type = X::TensorDataType::FLOAT32;  // Default to float32
            if (dtype == "torch.float32") {
                tensor_data_type = X::TensorDataType::FLOAT32;
            }
            else if (dtype == "torch.float64") {
                tensor_data_type = X::TensorDataType::FLOAT64;
            }
            else if (dtype == "torch.int32") {
                tensor_data_type = X::TensorDataType::INT32;
            }
            else if (dtype == "torch.int64") {
                tensor_data_type = X::TensorDataType::INT64;
            }
            else if (dtype == "torch.bfloat16") {
                tensor_data_type = X::TensorDataType::BFLOAT16;
            }
            else if (dtype.find("torch.float8_e4m3fn") != std::string::npos)
            {
                tensor_data_type = X::TensorDataType::FLOAT8_E4M3FN;
            }
            else if (dtype.find("torch.float8_e4m3fnuz") != std::string::npos)
            {
                tensor_data_type = X::TensorDataType::FLOAT8_E4M3FNUZ;
            }
            else if (dtype.find("torch.float8_e5m2") != std::string::npos)
            {
                tensor_data_type = X::TensorDataType::FLOAT8_E5M2;
            }
            else if (dtype.find("torch.float8_e5m2fnuz") != std::string::npos)
            {
                tensor_data_type = X::TensorDataType::FLOAT8_E5M2FNUZ;
            }
            // Read shape
            uint64_t num_dims;
            file.read(reinterpret_cast<char*>(&num_dims), sizeof(uint64_t));
            X::Port::vector<int> shape(static_cast<int>(num_dims));
            for (uint64_t i = 0; i < num_dims;i++) {
                int64_t d;
                file.read((char*)&d, sizeof(int64_t));
				shape.push_back((int)d);
            }
            X::Tensor tensor;
            tensor->SetShape(shape);
            tensor->SetDataType(tensor_data_type);

            int64_t num_elements = std::accumulate(shape.begin(), shape.end(),
                int64_t{1}, std::multiplies<int64_t>());
            int64_t num_bytes = num_elements * tensor->GetItemSize();
            std::vector<char> buffer(num_bytes);
            file.read(buffer.data(), num_bytes);

            // Create and populate X::Tensor
            X::Value dummy;
            tensor->Create(dummy);

            // Copy data into tensor
            memcpy(tensor->GetData(), buffer.data(), num_bytes);
            if (TensorHelper::EnsureGPUMemory(tensor) != TensorOpStatus::Success) {
                LOG << "LoadModelFromFile failed to move tensor to GPU: " << key << LINE_END;
                return false;
            }

            // Store in dictionary
            model->Set(key, tensor);
        }

        file.close();
        return true;
    }


    X::Value GarnetAPI::LoadModel(std::string modelPath)
    {
        X::Dict dictModel;
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

        X::XPackageValue<Model> varModel;
        Model& model = *varModel;
		model.SetInfo(strModelPath, tokenizerJsonPath, tokenizerConfigJsonPath, dictModel);
        return varModel;
    }

    void GarnetAPI::CreateKVCacheManager(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        auto getInt = [&](const char* name, size_t pos, int defaultValue) -> int {
            X::Value value = GetKwarg(kwParams, name);
            if (value.IsValid()) return static_cast<int>(value.ToLongLong());
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
        X::XPackageValue<KVCacheManager> manager;
        (*manager).Configure(maxNumPages, pageSize, headDim, numKVHeads, numLayers, dtypeBytes, deviceId);
        retValue = manager;
    }

    void GarnetAPI::CreateQwenTextRunner(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        X::XPackageValue<QwenTextRunner> runner;
        if (params.size() > 0) {
            if (!params[0].IsList()) {
                std::cout << "[GarnetAPI] QwenTextRunner(layer_bundles) requires a list." << std::endl;
                retValue = X::Value();
                return;
            }
            runner->SetLayerBundles(params[0]);
        }
        retValue = X::Value(runner);
    }

    void GarnetAPI::DevicePagedKVWriteTensor(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (params.size() < 4 || !params[1].IsTensor()) {
            std::cout << "[GarnetAPI] device_paged_kv_write(handle, qkv_tensor, token_count, start_position) expected." << std::endl;
            retValue = X::Value(false);
            return;
        }
        long long handle = params[0].ToLongLong();
        X::Tensor qkv(params[1]);
        int tokenCount = static_cast<int>(params[2].ToLongLong());
        int startPosition = static_cast<int>(params[3].ToLongLong());
        if (qkv->GetDataType() != X::TensorDataType::FLOAT32 || qkv->GetDimCount() != 2) {
            std::cout << "[GarnetAPI] device_paged_kv_write requires a float32 2D qkv tensor." << std::endl;
            retValue = X::Value(false);
            return;
        }
        DevicePagedKVFP32 cache;
        if (!GetDevicePagedKV(handle, cache)) {
            std::cout << "[GarnetAPI] device_paged_kv_write invalid handle: " << handle << std::endl;
            retValue = X::Value(false);
            return;
        }
        int qWidth = cache.qHeads * cache.headDim;
        int kvWidth = cache.kvHeads * cache.headDim;
        int qkvStride = qWidth + 2 * kvWidth;
        if (tokenCount <= 0 || tokenCount > qkv->GetDimSize(0) || qkv->GetDimSize(1) < qkvStride ||
            startPosition < 0 || startPosition + tokenCount > cache.logicalPageCount * cache.pageSize) {
            std::cout << "[GarnetAPI] device_paged_kv_write invalid shape or range." << std::endl;
            retValue = X::Value(false);
            return;
        }
        if (TensorHelper::EnsureGPUMemory(qkv) != TensorOpStatus::Success) {
            std::cout << "[GarnetAPI] device_paged_kv_write failed to ensure qkv GPU memory." << std::endl;
            retValue = X::Value(false);
            return;
        }
        float* dQKV = static_cast<float*>(TensorHelper::GetGPUMemory(qkv));
        if (!dQKV) {
            std::cout << "[GarnetAPI] device_paged_kv_write qkv tensor has no GPU memory." << std::endl;
            retValue = X::Value(false);
            return;
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
            retValue = X::Value(false);
            return;
        }
        retValue = X::Value(true);
    }

    void GarnetAPI::DevicePagedKVAttentionTensor(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (params.size() < 3 || !params[1].IsTensor()) {
            std::cout << "[GarnetAPI] device_paged_kv_attention(handle, q_or_qkv_tensor, sequence_length) expected." << std::endl;
            retValue = X::Value();
            return;
        }
        long long handle = params[0].ToLongLong();
        X::Tensor q(params[1]);
        int sequenceLength = static_cast<int>(params[2].ToLongLong());
        if (q->GetDataType() != X::TensorDataType::FLOAT32 || q->GetDimCount() != 2 || q->GetDimSize(0) < 1) {
            std::cout << "[GarnetAPI] device_paged_kv_attention requires a float32 2D q/qkv tensor." << std::endl;
            retValue = X::Value();
            return;
        }
        DevicePagedKVFP32 cache;
        if (!GetDevicePagedKV(handle, cache)) {
            std::cout << "[GarnetAPI] device_paged_kv_attention invalid handle: " << handle << std::endl;
            retValue = X::Value();
            return;
        }
        int qWidth = cache.qHeads * cache.headDim;
        if (q->GetDimSize(1) < qWidth || sequenceLength <= 0 ||
            sequenceLength > cache.logicalPageCount * cache.pageSize) {
            std::cout << "[GarnetAPI] device_paged_kv_attention invalid shape or sequence length." << std::endl;
            retValue = X::Value();
            return;
        }
        if (TensorHelper::EnsureGPUMemory(q) != TensorOpStatus::Success) {
            std::cout << "[GarnetAPI] device_paged_kv_attention failed to ensure q GPU memory." << std::endl;
            retValue = X::Value();
            return;
        }
        float* dQBase = static_cast<float*>(TensorHelper::GetGPUMemory(q));
        if (!dQBase) {
            std::cout << "[GarnetAPI] device_paged_kv_attention q tensor has no GPU memory." << std::endl;
            retValue = X::Value();
            return;
        }
        int qStride = q->GetDimSize(1);
        float* dQ = dQBase + static_cast<size_t>(q->GetDimSize(0) - 1) * static_cast<size_t>(qStride);
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
            retValue = X::Value();
            return;
        }

        X::Tensor output;
        output->SetDataType(X::TensorDataType::FLOAT32);
        X::Port::vector<int> outputShape(2);
        outputShape.push_back(1);
        outputShape.push_back(qWidth);
        output->SetShape(outputShape);
        X::Value initData;
        if (!output->Create(initData) || output->GetData() == nullptr ||
            TensorHelper::AttachGPUMemory(output, dOut) != TensorOpStatus::Success) {
            cudaFree(dOut);
            if (stream) cudaStreamDestroy(stream);
            retValue = X::Value();
            return;
        }
        const char* syncEnv = std::getenv("GARNET_TRT_SYNC_CPU_OUTPUTS");
        bool syncCPU = !syncEnv || !(syncEnv[0] == '0' && syncEnv[1] == '\0');
        if (syncCPU) {
            err = cudaMemcpyAsync(output->GetData(), dOut, outputBytes, cudaMemcpyDeviceToHost, stream);
            if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) {
                cudaFree(dOut);
                if (stream) cudaStreamDestroy(stream);
                std::cout << "[GarnetAPI] device_paged_kv_attention CPU sync failed: " << cudaGetErrorString(err) << std::endl;
                retValue = X::Value();
                return;
            }
        }
        if (stream) cudaStreamDestroy(stream);
        retValue = X::Value(output);
    }

    void GarnetAPI::TensorToCPU(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (params.size() < 1 || !params[0].IsTensor()) {
            std::cout << "[GarnetAPI] tensor_to_cpu(tensor) expected." << std::endl;
            retValue = X::Value();
            return;
        }

        X::Tensor tensor(params[0]);
        retValue = TensorHelper::CopyToCPUTensor(tensor);
        if (!retValue.IsValid()) {
            std::cout << "[GarnetAPI] tensor_to_cpu failed." << std::endl;
        }
    }

    void GarnetAPI::TensorToGPU(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (params.size() < 1 || !params[0].IsTensor()) {
            std::cout << "[GarnetAPI] tensor_to_gpu(tensor) expected." << std::endl;
            retValue = X::Value();
            return;
        }
        X::Tensor tensor(params[0]);
        if (TensorHelper::EnsureGPUMemory(tensor) != TensorOpStatus::Success) {
            std::cout << "[GarnetAPI] tensor_to_gpu failed." << std::endl;
            retValue = X::Value();
            return;
        }
        retValue = X::Value(tensor);
    }

    void GarnetAPI::TensorAdd(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (params.size() < 2 || !params[0].IsTensor() || !params[1].IsTensor()) {
            std::cout << "[GarnetAPI] tensor_add(lhs, rhs) expected." << std::endl;
            retValue = X::Value();
            return;
        }
        X::Tensor lhs(params[0]);
        X::Tensor rhs(params[1]);
        if (lhs->GetDataType() != X::TensorDataType::FLOAT32 ||
            rhs->GetDataType() != X::TensorDataType::FLOAT32 ||
            lhs->GetDimCount() != rhs->GetDimCount() ||
            lhs->GetCount() != rhs->GetCount()) {
            std::cout << "[GarnetAPI] tensor_add requires equal-shaped FLOAT32 tensors." << std::endl;
            retValue = X::Value();
            return;
        }
        for (int dim = 0; dim < lhs->GetDimCount(); ++dim) {
            if (lhs->GetDimSize(dim) != rhs->GetDimSize(dim)) {
                std::cout << "[GarnetAPI] tensor_add shape mismatch." << std::endl;
                retValue = X::Value();
                return;
            }
        }
        if (TensorHelper::EnsureGPUMemory(lhs) != TensorOpStatus::Success ||
            TensorHelper::EnsureGPUMemory(rhs) != TensorOpStatus::Success) {
            std::cout << "[GarnetAPI] tensor_add failed to ensure GPU inputs." << std::endl;
            retValue = X::Value();
            return;
        }

        size_t bytes = static_cast<size_t>(lhs->GetDataSize());
        float* outputDevice = nullptr;
        cudaStream_t stream = nullptr;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err == cudaSuccess) err = cudaMalloc(&outputDevice, bytes);
        if (err == cudaSuccess) {
            err = runTensorAddFP32(
                static_cast<const float*>(TensorHelper::GetGPUMemory(lhs)),
                static_cast<const float*>(TensorHelper::GetGPUMemory(rhs)),
                outputDevice,
                static_cast<int>(lhs->GetCount()),
                stream);
        }
        if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            if (outputDevice) cudaFree(outputDevice);
            if (stream) cudaStreamDestroy(stream);
            std::cout << "[GarnetAPI] tensor_add failed: " << cudaGetErrorString(err) << std::endl;
            retValue = X::Value();
            return;
        }

        X::Tensor output = X::g_pXHost->CreateTensor();
        X::Port::vector<int> shape(lhs->GetDimCount());
        for (int dim = 0; dim < lhs->GetDimCount(); ++dim) {
            shape.push_back(static_cast<int>(lhs->GetDimSize(dim)));
        }
        output->SetDataType(X::TensorDataType::FLOAT32);
        output->SetShape(shape);
        if (TensorHelper::AttachGPUMemory(output, outputDevice) != TensorOpStatus::Success) {
            cudaFree(outputDevice);
            cudaStreamDestroy(stream);
            retValue = X::Value();
            return;
        }
        cudaStreamDestroy(stream);
        retValue = X::Value(output);
    }

    void GarnetAPI::Embedding(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (params.size() < 2 || !params[0].IsTensor() || !params[1].IsTensor()) {
            std::cout << "[GarnetAPI] embedding(weight, token_ids) expected." << std::endl;
            retValue = X::Value();
            return;
        }
        X::Tensor weight(params[0]);
        X::Tensor tokenIds(params[1]);
        if (weight->GetDataType() != X::TensorDataType::FLOAT32 || weight->GetDimCount() != 2 ||
            tokenIds->GetDataType() != X::TensorDataType::LONGLONG || tokenIds->GetDimCount() != 1) {
            std::cout << "[GarnetAPI] embedding requires FLOAT32 [vocab, hidden] and INT64 [tokens]." << std::endl;
            retValue = X::Value();
            return;
        }
        if (TensorHelper::EnsureGPUMemory(weight) != TensorOpStatus::Success ||
            TensorHelper::EnsureGPUMemory(tokenIds) != TensorOpStatus::Success) {
            std::cout << "[GarnetAPI] embedding failed to ensure GPU inputs." << std::endl;
            retValue = X::Value();
            return;
        }
        int tokens = static_cast<int>(tokenIds->GetDimSize(0));
        int vocab = static_cast<int>(weight->GetDimSize(0));
        int hidden = static_cast<int>(weight->GetDimSize(1));
        size_t bytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);
        float* outputDevice = nullptr;
        cudaStream_t stream = nullptr;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err == cudaSuccess) err = cudaMalloc(&outputDevice, bytes);
        if (err == cudaSuccess) {
            err = runEmbeddingGatherInt64FP32(
                static_cast<const float*>(TensorHelper::GetGPUMemory(weight)),
                static_cast<const long long*>(TensorHelper::GetGPUMemory(tokenIds)),
                outputDevice, tokens, vocab, hidden, stream);
        }
        if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            if (outputDevice) cudaFree(outputDevice);
            if (stream) cudaStreamDestroy(stream);
            std::cout << "[GarnetAPI] embedding failed: " << cudaGetErrorString(err) << std::endl;
            retValue = X::Value();
            return;
        }
        X::Tensor output = X::g_pXHost->CreateTensor();
        X::Port::vector<int> shape(2);
        shape.push_back(tokens);
        shape.push_back(hidden);
        output->SetDataType(X::TensorDataType::FLOAT32);
        output->SetShape(shape);
        if (TensorHelper::AttachGPUMemory(output, outputDevice) != TensorOpStatus::Success) {
            cudaFree(outputDevice);
            cudaStreamDestroy(stream);
            retValue = X::Value();
            return;
        }
        cudaStreamDestroy(stream);
        retValue = X::Value(output);
    }

    void GarnetAPI::ReplaceRowsByMask(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (params.size() < 3 || !params[0].IsTensor() || !params[1].IsTensor() || !params[2].IsTensor()) {
            std::cout << "[GarnetAPI] replace_rows_by_mask(base, mask, replacements, mask_value=1) expected." << std::endl;
            retValue = X::Value();
            return;
        }
        X::Tensor base(params[0]);
        X::Tensor mask(params[1]);
        X::Tensor replacements(params[2]);
        long long maskValue = params.size() >= 4 ? params[3].ToLongLong() : 1;
        if (base->GetDataType() != X::TensorDataType::FLOAT32 || base->GetDimCount() != 2 ||
            mask->GetDataType() != X::TensorDataType::LONGLONG || mask->GetDimCount() != 1 ||
            replacements->GetDataType() != X::TensorDataType::FLOAT32 || replacements->GetDimCount() != 2 ||
            mask->GetDimSize(0) != base->GetDimSize(0) ||
            replacements->GetDimSize(1) != base->GetDimSize(1)) {
            std::cout << "[GarnetAPI] replace_rows_by_mask shape or dtype mismatch." << std::endl;
            retValue = X::Value();
            return;
        }
        if (TensorHelper::EnsureGPUMemory(base) != TensorOpStatus::Success ||
            TensorHelper::EnsureGPUMemory(mask) != TensorOpStatus::Success ||
            TensorHelper::EnsureGPUMemory(replacements) != TensorOpStatus::Success) {
            std::cout << "[GarnetAPI] replace_rows_by_mask failed to ensure GPU inputs." << std::endl;
            retValue = X::Value();
            return;
        }

        size_t bytes = static_cast<size_t>(base->GetDataSize());
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
                static_cast<int>(base->GetDimSize(0)),
                static_cast<int>(replacements->GetDimSize(0)),
                static_cast<int>(base->GetDimSize(1)),
                maskValue,
                stream);
        }
        if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            if (outputDevice) cudaFree(outputDevice);
            if (stream) cudaStreamDestroy(stream);
            std::cout << "[GarnetAPI] replace_rows_by_mask failed: " << cudaGetErrorString(err) << std::endl;
            retValue = X::Value();
            return;
        }
        X::Tensor output = X::g_pXHost->CreateTensor();
        X::Port::vector<int> shape(2);
        shape.push_back(static_cast<int>(base->GetDimSize(0)));
        shape.push_back(static_cast<int>(base->GetDimSize(1)));
        output->SetDataType(X::TensorDataType::FLOAT32);
        output->SetShape(shape);
        if (TensorHelper::AttachGPUMemory(output, outputDevice) != TensorOpStatus::Success) {
            cudaFree(outputDevice);
            cudaStreamDestroy(stream);
            retValue = X::Value();
            return;
        }
        cudaStreamDestroy(stream);
        retValue = X::Value(output);
    }

    void GarnetAPI::TensorLastRow(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (params.size() < 1 || !params[0].IsTensor()) {
            std::cout << "[GarnetAPI] tensor_last_row(tensor) expected." << std::endl;
            retValue = X::Value();
            return;
        }
        X::Tensor input(params[0]);
        if (input->GetDimCount() != 2 || input->GetDimSize(0) <= 0 ||
            TensorHelper::EnsureGPUMemory(input) != TensorOpStatus::Success) {
            std::cout << "[GarnetAPI] tensor_last_row requires a non-empty 2D tensor." << std::endl;
            retValue = X::Value();
            return;
        }
        int rows = static_cast<int>(input->GetDimSize(0));
        int columns = static_cast<int>(input->GetDimSize(1));
        size_t rowBytes = static_cast<size_t>(columns) * static_cast<size_t>(input->GetItemSize());
        auto* inputDevice = static_cast<const char*>(TensorHelper::GetGPUMemory(input));
        void* outputDevice = nullptr;
        cudaError_t err = cudaMalloc(&outputDevice, rowBytes);
        if (err == cudaSuccess) {
            err = cudaMemcpy(
                outputDevice,
                inputDevice + static_cast<size_t>(rows - 1) * rowBytes,
                rowBytes,
                cudaMemcpyDeviceToDevice);
        }
        if (err != cudaSuccess) {
            if (outputDevice) cudaFree(outputDevice);
            std::cout << "[GarnetAPI] tensor_last_row failed: " << cudaGetErrorString(err) << std::endl;
            retValue = X::Value();
            return;
        }
        X::Tensor output = X::g_pXHost->CreateTensor();
        X::Port::vector<int> shape(2);
        shape.push_back(1);
        shape.push_back(columns);
        output->SetDataType(input->GetDataType());
        output->SetShape(shape);
        if (TensorHelper::AttachGPUMemory(output, outputDevice) != TensorOpStatus::Success) {
            cudaFree(outputDevice);
            retValue = X::Value();
            return;
        }
        retValue = X::Value(output);
    }

    void GarnetAPI::LoadModelEx(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (params.size() == 0) return;
        std::string modelPath = params[0].ToString();
        
        namespace fs = std::filesystem;
        fs::path path(modelPath);
        
        X::Value modelVal;
        
        // Check if the path is a .x file
        if (!fs::is_directory(path) && path.extension() == ".x") {
            // Load an empty model object
            X::XPackageValue<Model> varModel;
            Model& model = *varModel;
            X::Dict dictModel;
            std::string dir = path.parent_path().string();
            std::string emptyStr = "";
            model.SetInfo(dir, emptyStr, emptyStr, dictModel);
            modelVal = varModel;
            
            // Read the script file
            std::cout << "[Garnet] Loading .x module from " << modelPath << std::endl;
            std::ifstream file(modelPath);
            std::string code((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            if (file.fail() && code.empty()) {
                std::cout << "[Garnet] Failed to read file " << modelPath << std::endl;
            }

            X::Value moduleVal;
            bool bOK = X::g_pXHost->LoadModule(modelPath.c_str(), code.c_str(), (int)code.size(), moduleVal);
            std::cout << "[Garnet] LoadModule returned " << bOK << std::endl;
            if (bOK && moduleVal.IsObject()) {
                X::Value weightsDict;
                X::Value inputShapes;
                X::Value weightShape;
                std::string subgraph;
                std::string cacheDir = (path.parent_path() / "cache").string();
                for (auto& it : kwParams) {
                    if (std::string(it.key) == "weights") {
                        weightsDict = it.val;
                    }
                    else if (std::string(it.key) == "input_shapes") {
                        inputShapes = it.val;
                    }
                    else if (std::string(it.key) == "weight_shape") {
                        weightShape = it.val;
                    }
                    else if (std::string(it.key) == "cache_dir") {
                        cacheDir = it.val.ToString();
                    }
                    else if (std::string(it.key) == "subgraph") {
                        subgraph = it.val.ToString();
                    }
                }

                model.SetInfo(dir, emptyStr, emptyStr, weightsDict);
                model.SetSubgraph(subgraph);
                
                // Store weights in GarnetAPI singleton before running script
                GarnetAPI::I().SetCurrentWeights(weightsDict);

                X::Value retVal;
                X::g_pXHost->RunModule(moduleVal, retVal, true);

                if (subgraph == "qwen3_text_mlp" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> gateShape = TensorShape(weights["language_model.layers.0.mlp.gate_proj.weight"]);
                        std::vector<int> upShape = TensorShape(weights["language_model.layers.0.mlp.up_proj.weight"]);
                        std::vector<int> downShape = TensorShape(weights["language_model.layers.0.mlp.down_proj.weight"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        bool useCudaTextMlp = inputShape.size() == 2 && inputShape[0] > 64;
                        if (useCudaTextMlp) {
                            model.SetEngine(X::Value("cuda_text_mlp"));
                        }
                        else if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportTextMLPEngine(enginePath.string(), inputShape, gateShape, upShape, downShape);
                        }
                        if (!useCudaTextMlp && std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "text_qkv_proj" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> qShape = TensorShape(weights["language_model.layers.0.self_attn.q_proj.weight"]);
                        std::vector<int> kShape = TensorShape(weights["language_model.layers.0.self_attn.k_proj.weight"]);
                        std::vector<int> vShape = TensorShape(weights["language_model.layers.0.self_attn.v_proj.weight"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportTextQKVEngine(enginePath.string(), inputShape, qShape, kShape, vShape);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "text_qkv_head_norm" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> qShape = TensorShape(weights["language_model.layers.0.self_attn.q_proj.weight"]);
                        std::vector<int> kShape = TensorShape(weights["language_model.layers.0.self_attn.k_proj.weight"]);
                        std::vector<int> vShape = TensorShape(weights["language_model.layers.0.self_attn.v_proj.weight"]);
                        std::vector<int> qNormShape = TensorShape(weights["language_model.layers.0.self_attn.q_norm.weight"]);
                        std::vector<int> kNormShape = TensorShape(weights["language_model.layers.0.self_attn.k_norm.weight"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportTextQKVHeadNormEngine(enginePath.string(), inputShape, qShape, kShape, vShape, qNormShape, kNormShape, 1.0e-6f);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "text_rope_apply" && inputShapes.IsList()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() >= 3) {
                        std::vector<int> qkvShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> cosShape = ReadIntList(shapeList->Get(1));
                        std::vector<int> sinShape = ReadIntList(shapeList->Get(2));
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportTextRoPEEngine(enginePath.string(), qkvShape, cosShape, sinShape, 16, 8, 128);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "text_attention_core" && inputShapes.IsList()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        std::vector<int> qkvShape = ReadIntList(shapeList->Get(0));
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportTextAttentionEngine(enginePath.string(), qkvShape, 16, 8, 128);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "vision_attention_core" && inputShapes.IsList()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        std::vector<int> qkvShape = ReadIntList(shapeList->Get(0));
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (qkvShape.size() == 2 && qkvShape[0] > 512) {
                            model.SetEngine(X::Value("cuda_exact_vision_attention"));
                        }
                        else if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportVisionAttentionEngine(enginePath.string(), qkvShape, 16, 64);
                        }
                        if (qkvShape.size() == 2 && qkvShape[0] > 512) {
                            model.SetEngine(X::Value("cuda_exact_vision_attention"));
                        }
                        else if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "text_o_proj" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> oShape = TensorShape(weights["language_model.layers.0.self_attn.o_proj.weight"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportLinearTransposeEngine(enginePath.string(), inputShape, oShape);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "text_lm_head" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> embedShape = TensorShape(weights["language_model.embed_tokens.weight"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        bool useCudaLinear = inputShape.size() == 2
                            && embedShape.size() == 2
                            && (embedShape[0] > 65536 || (static_cast<long long>(inputShape[0]) * static_cast<long long>(embedShape[0]) > 8LL * 1024LL * 1024LL));
                        if (useCudaLinear) {
                            model.SetEngine(X::Value("cuda_linear_transpose"));
                        }
                        else if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportLinearTransposeEngine(enginePath.string(), inputShape, embedShape);
                        }
                        if (!useCudaLinear && std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "vision_patch_embed" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> weightShape = TensorShape(weights["visual.patch_embed.proj.weight"]);
                        std::vector<int> biasShape = TensorShape(weights["visual.patch_embed.proj.bias"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportLinearBiasTransposeEngine(enginePath.string(), inputShape, weightShape, biasShape);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "linear_bias" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> weightShape = TensorShape(weights["W"]);
                        std::vector<int> biasShape = TensorShape(weights["B"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        bool useCudaLinear = inputShape.size() == 2 && weightShape.size() == 2
                            && (inputShape[0] > 2048 || (static_cast<long long>(inputShape[0]) * static_cast<long long>(weightShape[0]) > 8LL * 1024LL * 1024LL));
                        if (useCudaLinear) {
                            model.SetEngine(X::Value("cuda_linear_bias_transpose"));
                        }
                        else if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportLinearBiasTransposeEngine(enginePath.string(), inputShape, weightShape, biasShape);
                        }
                        if (useCudaLinear) {
                            model.SetEngine(X::Value("cuda_linear_bias_transpose"));
                        }
                        else if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "vision_mlp" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> fc1Shape = TensorShape(weights["visual.blocks.0.mlp.linear_fc1.weight"]);
                        std::vector<int> fc2Shape = TensorShape(weights["visual.blocks.0.mlp.linear_fc2.weight"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportVisionMLPEngine(enginePath.string(), inputShape, fc1Shape, fc2Shape);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "rms_norm" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        model.SetRMSNormWeight(weights["language_model.layers.0.input_layernorm.weight"]);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> weightShape = TensorShape(weights["language_model.layers.0.input_layernorm.weight"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportRMSNormEngine(enginePath.string(), inputShape, weightShape, 1.0e-6f);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "text_post_attention_rms_norm" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        model.SetRMSNormWeight(weights["language_model.layers.0.post_attention_layernorm.weight"]);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> weightShape = TensorShape(weights["language_model.layers.0.post_attention_layernorm.weight"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportRMSNormEngine(enginePath.string(), inputShape, weightShape, 1.0e-6f);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "layer_norm" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> weightShape = TensorShape(weights["visual.blocks.0.norm1.weight"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportLayerNormEngine(enginePath.string(), inputShape, weightShape, 1.0e-6f);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (inputShapes.IsList() && weightShape.IsList()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> wShape = ReadIntList(weightShape);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportMatmulEngine(enginePath.string(), inputShape, wShape);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }

                // Extract compiled engine that was set during script execution
                X::Value compiledEngine = GarnetAPI::I().GetCompiledEngine();
                if (!model.m_engine.IsValid() && compiledEngine.IsValid()) {
                    model.SetEngine(compiledEngine);
                } else if (!model.m_engine.IsValid()) {
                    std::cout << "[Garnet] Warning: Script finished but no engine was compiled!" << std::endl;
                }
            }
        }
        else {
            modelVal = LoadModel(modelPath);
        }
        
        retValue = modelVal;
    }

    void GarnetAPI::QwenVLSmartResize(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
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

            X::Dict result;
            int gridH = resized.height / patchSize;
            int gridW = resized.width / patchSize;
            result->Set("height", X::Value(resized.height));
            result->Set("width", X::Value(resized.width));
            result->Set("patch_size", X::Value(patchSize));
            result->Set("merge_size", X::Value(mergeSize));
            result->Set("grid_h", X::Value(gridH));
            result->Set("grid_w", X::Value(gridW));
            result->Set("visual_tokens", X::Value(gridH * gridW / (mergeSize * mergeSize)));
            retValue = result;
        }
        catch (const std::exception& exc) {
            std::cout << "[GarnetAPI] qwen_vl_smart_resize failed: " << exc.what() << std::endl;
            retValue = X::Value();
        }
    }

    void GarnetAPI::QwenVLCreateRequest(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        auto totalStart = std::chrono::steady_clock::now();
        try {
            std::string modelDir = GetStringArg(params, kwParams, 0, "model_dir", "");
            std::string imagePath = GetStringArg(params, kwParams, 1, "image_path", "");
            std::string prompt = GetStringArg(params, kwParams, 2, "prompt", "");
            int minPixels = GetIntArg(params, kwParams, 3, "min_pixels", 65536);
            int maxPixels = GetIntArg(params, kwParams, 4, "max_pixels", 65536);
            if (modelDir.empty() || imagePath.empty() || prompt.empty()) {
                std::cout << "[GarnetAPI] qwen_vl_create_request requires model_dir, image_path, and prompt." << std::endl;
                retValue = X::Value();
                return;
            }

            constexpr int patchSize = 16;
            constexpr int temporalPatchSize = 2;
            constexpr int mergeSize = 2;
            int featureDim = 3 * temporalPatchSize * patchSize * patchSize;

            auto imageStart = std::chrono::steady_clock::now();
            auto imageResult = Image::QwenVL::PreprocessJpegFileToTensor(imagePath, minPixels, maxPixels);
            double imageMs = MsSince(imageStart);
            X::Tensor pixelValues(imageResult.pixelValues);
            X::Tensor imageGridTensor(imageResult.imageGridTHW);
            auto* gridData = reinterpret_cast<long long*>(imageGridTensor->GetData());
            long long grid[3] = { gridData[0], gridData[1], gridData[2] };

            int patchCount = (imageResult.resizedHeight / patchSize) * (imageResult.resizedWidth / patchSize);
            if (patchCount <= 0) {
                std::cout << "[GarnetAPI] qwen_vl_create_request invalid patch count: " << patchCount << std::endl;
                retValue = X::Value();
                return;
            }

            auto tokenStart = std::chrono::steady_clock::now();
            std::string tokenError;
            auto tokenizer = Tokenization::GetCachedQwenTokenizer(modelDir, &tokenError);
            if (!tokenizer) {
                std::cout << "[GarnetAPI] qwen_vl_create_request tokenizer load failed: " << tokenError << std::endl;
                retValue = X::Value();
                return;
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
                retValue = X::Value();
                return;
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
                if (mmType == 1) {
                    ++visualTokenCount;
                }
            }
            int expectedVisualTokenCount = static_cast<int>((grid[0] * grid[1] * grid[2]) / (mergeSize * mergeSize));
            if (visualTokenCount != expectedVisualTokenCount) {
                std::cout << "[GarnetAPI] qwen_vl_create_request visual token mismatch: prompt="
                    << visualTokenCount << ", grid=" << expectedVisualTokenCount << std::endl;
                retValue = X::Value();
                return;
            }
            double tokenizeMs = MsSince(tokenStart);

            auto uploadStart = std::chrono::steady_clock::now();
            X::Value inputIdsTensor = MakeInt64Tensor(inputIds, true);
            X::Value mmTypesTensor = MakeInt64Tensor(mmTypes, true);
            double uploadMs = MsSince(uploadStart);
            if (!inputIdsTensor.IsTensor() || !mmTypesTensor.IsTensor() ||
                TensorHelper::GetGPUMemory(pixelValues) == nullptr) {
                std::cout << "[GarnetAPI] qwen_vl_create_request failed to create GPU tensors." << std::endl;
                retValue = X::Value();
                return;
            }

            X::XPackageValue<QwenVLRequestContext> requestValue;
            QwenVLRequestContext& request = *requestValue;
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
            retValue = requestValue;
        }
        catch (const std::exception& exc) {
            std::cout << "[GarnetAPI] qwen_vl_create_request failed: " << exc.what() << std::endl;
            retValue = X::Value();
        }
    }

    void GarnetAPI::QwenVLPrepareRequest(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        auto totalStart = std::chrono::steady_clock::now();
        try {
            std::string modelDir = GetStringArg(params, kwParams, 0, "model_dir", "");
            std::string imagePath = GetStringArg(params, kwParams, 1, "image_path", "");
            std::string prompt = GetStringArg(params, kwParams, 2, "prompt", "");
            int minPixels = GetIntArg(params, kwParams, 3, "min_pixels", 65536);
            int maxPixels = GetIntArg(params, kwParams, 4, "max_pixels", 65536);
            if (modelDir.empty() || imagePath.empty() || prompt.empty()) {
                std::cout << "[GarnetAPI] qwen_vl_prepare_request requires model_dir, image_path, and prompt." << std::endl;
                retValue = X::Value();
                return;
            }

            constexpr int patchSize = 16;
            constexpr int temporalPatchSize = 2;
            constexpr int mergeSize = 2;
            int featureDim = 3 * temporalPatchSize * patchSize * patchSize;

            auto imageStart = std::chrono::steady_clock::now();
            auto imageResult = Image::QwenVL::PreprocessJpegFileToTensor(imagePath, minPixels, maxPixels);
            double imageMs = MsSince(imageStart);
            int patchCount = (imageResult.resizedHeight / patchSize) * (imageResult.resizedWidth / patchSize);
            if (patchCount <= 0) {
                std::cout << "[GarnetAPI] qwen_vl_prepare_request invalid patch count: " << patchCount << std::endl;
                retValue = X::Value();
                return;
            }
            X::Tensor pixelValues(imageResult.pixelValues);
            X::Tensor imageGridTensor(imageResult.imageGridTHW);
            auto* gridData = reinterpret_cast<long long*>(imageGridTensor->GetData());
            long long grid[3] = { gridData[0], gridData[1], gridData[2] };

            auto tokenStart = std::chrono::steady_clock::now();
            std::string tokenError;
            auto tokenizer = Tokenization::GetCachedQwenTokenizer(modelDir, &tokenError);
            if (!tokenizer) {
                std::cout << "[GarnetAPI] qwen_vl_prepare_request tokenizer load failed: " << tokenError << std::endl;
                retValue = X::Value();
                return;
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
                retValue = X::Value();
                return;
            }
            double tokenizeMs = MsSince(tokenStart);

            std::vector<long long> inputIds;
            std::vector<long long> mmTypes;
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
                retValue = X::Value();
                return;
            }

            std::vector<long long> gridVector = { grid[0], grid[1], grid[2] };
            X::Dict dict;
            dict->Set("input_ids", MakeInt64List(inputIds));
            dict->Set("mm_token_type_ids", MakeInt64List(mmTypes));
            X::Value inputIdsTensorValue = MakeInt64Tensor(inputIds, true);
            X::Value mmTypesTensorValue = MakeInt64Tensor(mmTypes, true);
            dict->Set("input_ids_tensor", inputIdsTensorValue);
            dict->Set("mm_token_type_ids_tensor", mmTypesTensorValue);
            dict->Set("pixel_values", imageResult.pixelValues);
            dict->Set("pixel_values_shape", MakeInt64List({
                static_cast<long long>(patchCount),
                static_cast<long long>(featureDim),
            }));
            dict->Set("pixel_value_count", X::Value(patchCount * featureDim));
            dict->Set("image_grid_thw", MakeInt64List(gridVector));
            dict->Set("pixel_values_gpu", X::Value(TensorHelper::GetGPUMemory(pixelValues) != nullptr));
            bool inputIdsGpu = false;
            bool mmTypesGpu = false;
            if (inputIdsTensorValue.IsTensor()) {
                X::Tensor inputIdsTensor(inputIdsTensorValue);
                inputIdsGpu = TensorHelper::GetGPUMemory(inputIdsTensor) != nullptr;
            }
            if (mmTypesTensorValue.IsTensor()) {
                X::Tensor mmTypesTensor(mmTypesTensorValue);
                mmTypesGpu = TensorHelper::GetGPUMemory(mmTypesTensor) != nullptr;
            }
            dict->Set("input_ids_gpu", X::Value(inputIdsGpu));
            dict->Set("mm_token_type_ids_gpu", X::Value(mmTypesGpu));

            X::Dict timings;
            timings->Set("image_preprocess_us", X::Value(static_cast<long long>(imageMs * 1000.0)));
            timings->Set("tokenize_us", X::Value(static_cast<long long>(tokenizeMs * 1000.0)));
            timings->Set("total_us", X::Value(static_cast<long long>(MsSince(totalStart) * 1000.0)));
            dict->Set("prompt_token_count", X::Value(static_cast<int>(inputIds.size())));
            dict->Set("visual_token_count", X::Value(visualTokenCount));
            dict->Set("source_height", X::Value(imageResult.sourceHeight));
            dict->Set("source_width", X::Value(imageResult.sourceWidth));
            dict->Set("height", X::Value(imageResult.resizedHeight));
            dict->Set("width", X::Value(imageResult.resizedWidth));
            dict->Set("patch_size", X::Value(patchSize));
            dict->Set("temporal_patch_size", X::Value(temporalPatchSize));
            dict->Set("merge_size", X::Value(mergeSize));
            dict->Set("backend", X::Value("qwen_vl_request_native_tokenizer_nvjpeg_cuda_gpu_xtensor"));
            dict->Set("timings", timings);
            retValue = dict;
        }
        catch (const std::exception& exc) {
            std::cout << "[GarnetAPI] qwen_vl_prepare_request failed: " << exc.what() << std::endl;
            retValue = X::Value();
        }
    }

    void GarnetAPI::QwenVLPreprocessImage(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        try {
            X::Value image = GetKwarg(kwParams, "image");
            if (!image.IsValid() && params.size() > 0) {
                image = params[0];
            }
            if (!image.IsValid()) {
                retValue = X::Value();
                return;
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
            X::Dict dict;
            dict->Set("pixel_values", result.pixelValues);
            dict->Set("image_grid_thw", result.imageGridTHW);
            dict->Set("source_height", X::Value(result.sourceHeight));
            dict->Set("source_width", X::Value(result.sourceWidth));
            dict->Set("height", X::Value(result.resizedHeight));
            dict->Set("width", X::Value(result.resizedWidth));
            dict->Set("patch_size", X::Value(result.patchSize));
            dict->Set("temporal_patch_size", X::Value(result.temporalPatchSize));
            dict->Set("merge_size", X::Value(result.mergeSize));
            dict->Set("input_format", X::Value(inputFormat));
            dict->Set("backend", X::Value("cuda_raw_tensor"));
            retValue = dict;
        }
        catch (const std::exception& exc) {
            std::cout << "[GarnetAPI] qwen_vl_preprocess_image failed: " << exc.what() << std::endl;
            retValue = X::Value();
        }
    }

    void GarnetAPI::QwenVLPreprocessJpegFile(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        try {
            std::string path = GetStringArg(params, kwParams, 0, "path", "");
            int minPixels = GetIntArg(params, kwParams, 1, "min_pixels", 65536);
            int maxPixels = GetIntArg(params, kwParams, 2, "max_pixels", 65536);
            if (path.empty() || minPixels <= 0 || maxPixels <= 0) {
                retValue = X::Value();
                return;
            }

            auto result = Image::QwenVL::PreprocessJpegFileToTensor(path, minPixels, maxPixels);
            X::Tensor pixelValues(result.pixelValues);
            X::Tensor imageGrid(result.imageGridTHW);

            X::Dict dict;
            dict->Set("pixel_values", result.pixelValues);
            dict->Set("image_grid_thw", result.imageGridTHW);
            dict->Set("height", X::Value(result.resizedHeight));
            dict->Set("width", X::Value(result.resizedWidth));
            dict->Set("patch_size", X::Value(result.patchSize));
            dict->Set("temporal_patch_size", X::Value(result.temporalPatchSize));
            dict->Set("merge_size", X::Value(result.mergeSize));
            dict->Set("pixel_values_gpu", X::Value(TensorHelper::GetGPUMemory(pixelValues) != nullptr));
            dict->Set("image_grid_gpu", X::Value(TensorHelper::GetGPUMemory(imageGrid) != nullptr));
            dict->Set("backend", X::Value("cuda_nvjpeg_to_gpu_xtensor"));
            retValue = dict;
        }
        catch (const std::exception& exc) {
            std::cout << "[GarnetAPI] qwen_vl_preprocess_jpeg_file failed: " << exc.what() << std::endl;
            retValue = X::Value();
        }
    }

    void GarnetAPI::RunTest(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
    }

}
