#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "native_library.h"
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using DebugSampleLogitsTop1Fn = int (*)(
    const float* deviceLogits,
    int rows,
    int vocabSize,
    long long* deviceOutputTokenId,
    float* deviceOutputTokenValue,
    char* errorMessage,
    int errorMessageCapacity);

namespace
{
    std::string DefaultDllPath()
    {
        return "D:\\CantorAI\\Garnet\\out\\build\\x64-Debug\\bin\\garnet.dll";
    }

    bool CheckCuda(cudaError_t err, const char* op)
    {
        if (err == cudaSuccess) {
            return true;
        }
        std::cerr << op << " failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }
}

int main(int argc, char** argv)
{
    std::string dllPath = argc > 1 ? argv[1] : DefaultDllPath();
    NativeLibraryHandle dll = OpenNativeLibrary(dllPath.c_str());
    if (!dll) {
        std::cerr << "failed to load " << dllPath << ", error=" << NativeLibraryError() << "\n";
        return 1;
    }

    auto sample = reinterpret_cast<DebugSampleLogitsTop1Fn>(
        NativeLibrarySymbol(dll, "GarnetDebugSampleLogitsTop1FP32"));
    if (!sample) {
        std::cerr << "missing GarnetDebugSampleLogitsTop1FP32 export\n";
        CloseNativeLibrary(dll);
        return 2;
    }

    constexpr int rows = 3;
    constexpr int vocab = 11;
    std::vector<float> logits(static_cast<size_t>(rows) * static_cast<size_t>(vocab), -10.0f);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < vocab; ++c) {
            logits[static_cast<size_t>(r) * vocab + c] = static_cast<float>(r * 0.1 + c * 0.01);
        }
    }
    logits[static_cast<size_t>(rows - 1) * vocab + 7] = 42.5f;
    logits[static_cast<size_t>(rows - 1) * vocab + 9] = 41.5f;

    float* dLogits = nullptr;
    long long* dTokenId = nullptr;
    float* dTokenValue = nullptr;
    bool cudaOk =
        CheckCuda(cudaMalloc(&dLogits, logits.size() * sizeof(float)), "cudaMalloc logits") &&
        CheckCuda(cudaMalloc(&dTokenId, sizeof(long long)), "cudaMalloc token id") &&
        CheckCuda(cudaMalloc(&dTokenValue, sizeof(float)), "cudaMalloc token value") &&
        CheckCuda(cudaMemcpy(dLogits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy logits");
    if (!cudaOk) {
        if (dLogits) cudaFree(dLogits);
        if (dTokenId) cudaFree(dTokenId);
        if (dTokenValue) cudaFree(dTokenValue);
        CloseNativeLibrary(dll);
        return 3;
    }

    char error[1024] = {};
    int rc = sample(dLogits, rows, vocab, dTokenId, dTokenValue, error, static_cast<int>(sizeof(error)));
    if (rc != 0) {
        std::cerr << "debug logits sample failed rc=" << rc << " error=" << error << "\n";
        cudaFree(dLogits);
        cudaFree(dTokenId);
        cudaFree(dTokenValue);
        CloseNativeLibrary(dll);
        return 4;
    }

    long long tokenId = -1;
    float tokenValue = 0.0f;
    cudaOk =
        CheckCuda(cudaMemcpy(&tokenId, dTokenId, sizeof(long long), cudaMemcpyDeviceToHost), "cudaMemcpy token id") &&
        CheckCuda(cudaMemcpy(&tokenValue, dTokenValue, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy token value");
    cudaFree(dLogits);
    cudaFree(dTokenId);
    cudaFree(dTokenValue);
    CloseNativeLibrary(dll);
    if (!cudaOk) {
        return 5;
    }
    if (tokenId != 7 || std::fabs(tokenValue - 42.5f) > 0.0001f) {
        std::cerr << "unexpected sample token_id=" << tokenId << " token_value=" << tokenValue << "\n";
        return 6;
    }

    std::cout << "Garnet debug logits sampler native smoke passed\n";
    std::cout << "rows: " << rows << "\n";
    std::cout << "vocab_size: " << vocab << "\n";
    std::cout << "token_id: " << tokenId << "\n";
    std::cout << "token_value: " << tokenValue << "\n";
    return 0;
}
