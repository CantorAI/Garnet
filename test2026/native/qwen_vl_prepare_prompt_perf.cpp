// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "native_library.h"

#include <chrono>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using CreateDeviceRequestFn = int (*)(
    const char* modelDir,
    const char* jpegPath,
    const char* prompt,
    int minPixels,
    int maxPixels,
    long long* outputHandle,
    char* errorMessage,
    int errorMessageCapacity);

using DestroyDeviceRequestFn = int (*)(
    long long handle,
    char* errorMessage,
    int errorMessageCapacity);

using GetDeviceRequestInfoFn = int (*)(
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
    int errorMessageCapacity);

namespace
{
    std::string DefaultModelDir()
    {
        return {};
    }

    std::string DefaultImagePath()
    {
        return "data/Dataset.1980Love/imgs/frame_0.jpg";
    }

    std::string DefaultPrompt()
    {
        return "Describe the image and list visible objects with short coordinates.";
    }

    std::string DefaultDllPath()
    {
        return "garnet.dll";
    }

    double ToMilliseconds(
        const std::chrono::steady_clock::time_point& start,
        const std::chrono::steady_clock::time_point& end)
    {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
}

int main(int argc, char** argv)
{
    std::string modelDir = argc > 1 ? argv[1] : DefaultModelDir();
    std::string imagePath = argc > 2 ? argv[2] : DefaultImagePath();
    std::string prompt = argc > 3 ? argv[3] : DefaultPrompt();
    int iterations = argc > 4 ? std::max(1, std::atoi(argv[4])) : 20;
    int minPixels = argc > 5 ? std::atoi(argv[5]) : 65536;
    int maxPixels = argc > 6 ? std::atoi(argv[6]) : 1003520;
    std::string dllPath = argc > 7 ? argv[7] : DefaultDllPath();
    int warmup = argc > 8 ? std::max(0, std::atoi(argv[8])) : 3;

    if (modelDir.empty()) {
        std::cerr << "usage: qwen_vl_prepare_prompt_perf <model-directory> [image] [prompt] "
                     "[iterations] [min-pixels] [max-pixels] [garnet-library] [warmup]\n";
        return 64;
    }

    NativeLibraryHandle dll = OpenNativeLibrary(dllPath.c_str());
    if (!dll) {
        std::cerr << "failed to load " << dllPath << ", error=" << NativeLibraryError() << "\n";
        return 1;
    }

    auto createRequest = reinterpret_cast<CreateDeviceRequestFn>(
        NativeLibrarySymbol(dll, "GarnetCreateQwenVLDeviceRequest"));
    auto destroyRequest = reinterpret_cast<DestroyDeviceRequestFn>(
        NativeLibrarySymbol(dll, "GarnetDestroyQwenVLDeviceRequest"));
    auto getRequestInfo = reinterpret_cast<GetDeviceRequestInfoFn>(
        NativeLibrarySymbol(dll, "GarnetGetQwenVLDeviceRequestInfo"));
    if (!createRequest || !destroyRequest || !getRequestInfo) {
        std::cerr << "missing Garnet prepare/request exports\n";
        CloseNativeLibrary(dll);
        return 2;
    }

    std::vector<double> wallTimings;
    std::vector<double> imageTimings;
    std::vector<double> tokenTimings;
    std::vector<double> uploadTimings;
    std::vector<double> apiTimings;
    wallTimings.reserve(static_cast<size_t>(iterations));
    imageTimings.reserve(static_cast<size_t>(iterations));
    tokenTimings.reserve(static_cast<size_t>(iterations));
    uploadTimings.reserve(static_cast<size_t>(iterations));
    apiTimings.reserve(static_cast<size_t>(iterations));

    int inputIdCount = 0;
    int pixelValueCount = 0;
    size_t pixelBytes = 0;
    size_t inputIdsBytes = 0;
    size_t mmTypesBytes = 0;
    long long grid[3] = { 0, 0, 0 };
    int sourceH = 0;
    int sourceW = 0;
    int resizedH = 0;
    int resizedW = 0;
    char error[2048] = {};

    for (int i = 0; i < warmup + iterations; ++i) {
        void* pixelDevice = nullptr;
        void* inputIdsDevice = nullptr;
        void* mmTypesDevice = nullptr;
        long long imageUs = 0;
        long long tokenUs = 0;
        long long uploadUs = 0;
        long long totalUs = 0;
        long long requestHandle = 0;
        auto start = std::chrono::steady_clock::now();
        int rc = createRequest(
            modelDir.c_str(),
            imagePath.c_str(),
            prompt.c_str(),
            minPixels,
            maxPixels,
            &requestHandle,
            error,
            static_cast<int>(sizeof(error)));
        if (rc == 0) {
            rc = getRequestInfo(
                requestHandle,
                &inputIdsDevice,
                &inputIdsBytes,
                &inputIdCount,
                &mmTypesDevice,
                &mmTypesBytes,
                &pixelDevice,
                &pixelBytes,
                &pixelValueCount,
                grid,
                &sourceH,
                &sourceW,
                &resizedH,
                &resizedW,
                &imageUs,
                &tokenUs,
                &uploadUs,
                &totalUs,
                error,
                static_cast<int>(sizeof(error)));
        }
        auto end = std::chrono::steady_clock::now();
        if (rc != 0) {
            if (requestHandle != 0) {
                destroyRequest(requestHandle, error, static_cast<int>(sizeof(error)));
            }
            std::cerr << "request prepare failed rc=" << rc << " error=" << error << "\n";
            CloseNativeLibrary(dll);
            return 3;
        }
        if (!inputIdsDevice || !mmTypesDevice || !pixelDevice) {
            destroyRequest(requestHandle, error, static_cast<int>(sizeof(error)));
            std::cerr << "request returned a null device tensor pointer\n";
            CloseNativeLibrary(dll);
            return 5;
        }
        int destroyRc = destroyRequest(requestHandle, error, static_cast<int>(sizeof(error)));
        if (destroyRc != 0) {
            std::cerr << "failed to destroy request handle: " << error << "\n";
            CloseNativeLibrary(dll);
            return 4;
        }

        if (i >= warmup) {
            wallTimings.push_back(ToMilliseconds(start, end));
            imageTimings.push_back(static_cast<double>(imageUs) / 1000.0);
            tokenTimings.push_back(static_cast<double>(tokenUs) / 1000.0);
            uploadTimings.push_back(static_cast<double>(uploadUs) / 1000.0);
            apiTimings.push_back(static_cast<double>(totalUs) / 1000.0);
        }
    }

    auto mean = [](const std::vector<double>& values) {
        double sum = 0.0;
        for (double value : values) sum += value;
        return values.empty() ? 0.0 : sum / static_cast<double>(values.size());
    };
    auto minValue = [](const std::vector<double>& values) {
        return values.empty() ? 0.0 : *std::min_element(values.begin(), values.end());
    };
    auto maxValue = [](const std::vector<double>& values) {
        return values.empty() ? 0.0 : *std::max_element(values.begin(), values.end());
    };

    int visualTokens = static_cast<int>((grid[0] * grid[1] * grid[2]) / 4);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Garnet Qwen-VL GPU prepare prompt perf\n";
    std::cout << "model_dir: " << modelDir << "\n";
    std::cout << "image: " << imagePath << "\n";
    std::cout << "source: " << sourceW << "x" << sourceH << "\n";
    std::cout << "resized: " << resizedW << "x" << resizedH << "\n";
    std::cout << "grid_thw: [" << grid[0] << ", " << grid[1] << ", " << grid[2] << "]\n";
    std::cout << "prompt_tokens: " << inputIdCount << "\n";
    std::cout << "visual_tokens: " << visualTokens << "\n";
    std::cout << "input_ids_bytes_device: " << inputIdsBytes << "\n";
    std::cout << "mm_token_type_ids_bytes_device: " << mmTypesBytes << "\n";
    std::cout << "pixel_values_count: " << pixelValueCount << "\n";
    std::cout << "pixel_values_bytes_device: " << pixelBytes << "\n";
    std::cout << "iterations: " << iterations << "\n";
    std::cout << "warmup_discarded: " << warmup << "\n";
    std::cout << "wall_mean_ms: " << mean(wallTimings) << "\n";
    std::cout << "wall_min_ms: " << minValue(wallTimings) << "\n";
    std::cout << "wall_max_ms: " << maxValue(wallTimings) << "\n";
    std::cout << "api_total_mean_ms: " << mean(apiTimings) << "\n";
    std::cout << "image_mean_ms: " << mean(imageTimings) << "\n";
    std::cout << "tokenize_mean_ms: " << mean(tokenTimings) << "\n";
    std::cout << "tensor_upload_mean_ms: " << mean(uploadTimings) << "\n";

    CloseNativeLibrary(dll);
    return 0;
}
