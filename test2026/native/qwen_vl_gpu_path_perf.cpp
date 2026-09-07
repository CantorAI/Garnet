#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "native_library.h"

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using PreprocessJpegFileDeviceFn = int (*)(
    const char* jpegPath,
    int minPixels,
    int maxPixels,
    void** pixelValuesDevice,
    size_t* outputBytes,
    long long* imageGridTHW,
    int* sourceHeight,
    int* sourceWidth,
    int* resizedHeight,
    int* resizedWidth,
    int* patchCount,
    int* featureDim,
    char* errorMessage,
    int errorMessageCapacity);

using FreeDeviceBufferFn = int (*)(void* deviceBuffer);

namespace
{
    std::string DefaultImagePath()
    {
        return "D:\\CantorAI\\Garnet\\data\\Dataset.1980Love\\imgs\\frame_0.jpg";
    }

    std::string DefaultDllPath()
    {
        return "D:\\CantorAI\\Garnet\\out\\build\\x64-Debug\\bin\\garnet.dll";
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
    std::string imagePath = argc > 1 ? argv[1] : DefaultImagePath();
    int iterations = argc > 2 ? std::max(1, std::atoi(argv[2])) : 10;
    int minPixels = argc > 3 ? std::atoi(argv[3]) : 65536;
    int maxPixels = argc > 4 ? std::atoi(argv[4]) : 65536;
    std::string dllPath = argc > 5 ? argv[5] : DefaultDllPath();
    int warmup = argc > 6 ? std::max(0, std::atoi(argv[6])) : 3;

    NativeLibraryHandle dll = OpenNativeLibrary(dllPath.c_str());
    if (!dll) {
        std::cerr << "failed to load " << dllPath << ", error=" << NativeLibraryError() << "\n";
        return 1;
    }

    auto preprocess = reinterpret_cast<PreprocessJpegFileDeviceFn>(
        NativeLibrarySymbol(dll, "GarnetQwenVLPreprocessJpegFileDevice"));
    auto freeDeviceBuffer = reinterpret_cast<FreeDeviceBufferFn>(
        NativeLibrarySymbol(dll, "GarnetFreeDeviceBuffer"));
    if (!preprocess || !freeDeviceBuffer) {
        std::cerr << "missing Garnet device preprocess exports\n";
        CloseNativeLibrary(dll);
        return 2;
    }

    std::vector<double> timings;
    timings.reserve(static_cast<size_t>(iterations));

    long long grid[3] = { 0, 0, 0 };
    int sourceH = 0;
    int sourceW = 0;
    int resizedH = 0;
    int resizedW = 0;
    int patchCount = 0;
    int featureDim = 0;
    size_t outputBytes = 0;
    char error[1024] = {};

    for (int i = 0; i < warmup + iterations; ++i) {
        void* devicePtr = nullptr;
        auto start = std::chrono::steady_clock::now();
        int rc = preprocess(
            imagePath.c_str(),
            minPixels,
            maxPixels,
            &devicePtr,
            &outputBytes,
            grid,
            &sourceH,
            &sourceW,
            &resizedH,
            &resizedW,
            &patchCount,
            &featureDim,
            error,
            static_cast<int>(sizeof(error)));
        auto end = std::chrono::steady_clock::now();
        if (rc != 0) {
            std::cerr << "preprocess failed rc=" << rc << " error=" << error << "\n";
            CloseNativeLibrary(dll);
            return 3;
        }
        if (i >= warmup) {
            timings.push_back(ToMilliseconds(start, end));
        }
        int freeRc = freeDeviceBuffer(devicePtr);
        if (freeRc != 0) {
            std::cerr << "failed to free device output\n";
            CloseNativeLibrary(dll);
            return 4;
        }
    }

    double sum = 0.0;
    double minValue = timings.empty() ? 0.0 : timings[0];
    double maxValue = timings.empty() ? 0.0 : timings[0];
    for (double value : timings) {
        sum += value;
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
    }
    double mean = timings.empty() ? 0.0 : sum / static_cast<double>(timings.size());

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Garnet Qwen-VL GPU image path perf\n";
    std::cout << "image: " << imagePath << "\n";
    std::cout << "source: " << sourceW << "x" << sourceH << "\n";
    std::cout << "resized: " << resizedW << "x" << resizedH << "\n";
    std::cout << "grid_thw: [" << grid[0] << ", " << grid[1] << ", " << grid[2] << "]\n";
    std::cout << "pixel_values: [" << patchCount << ", " << featureDim << "] "
              << outputBytes << " bytes device output\n";
    std::cout << "iterations: " << iterations << "\n";
    std::cout << "warmup_discarded: " << warmup << "\n";
    std::cout << "mean_ms: " << mean << "\n";
    std::cout << "min_ms: " << minValue << "\n";
    std::cout << "max_ms: " << maxValue << "\n";

    CloseNativeLibrary(dll);
    return 0;
}
