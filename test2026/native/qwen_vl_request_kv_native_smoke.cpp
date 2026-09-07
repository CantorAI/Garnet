#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "native_library.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
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

using AllocateRequestKVFn = int (*)(
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
    int errorMessageCapacity);

using GetRequestKVInfoFn = int (*)(
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
    int errorMessageCapacity);

using GetRequestKVStateFn = int (*)(
    long long requestHandle,
    long long* outputKVHandle,
    int* outputMaxTokens,
    int* outputLogicalLength,
    int* outputPageSize,
    int* outputLogicalPages,
    int* outputPhysicalPages,
    char* errorMessage,
    int errorMessageCapacity);

using WriteDeviceKVDeviceFn = int (*)(
    long long requestHandle,
    const float* deviceQKV,
    int tokenCount,
    int startPosition,
    char* errorMessage,
    int errorMessageCapacity);

using AttentionDeviceKVDeviceFn = int (*)(
    long long requestHandle,
    const float* deviceQ,
    float* deviceOutput,
    int sequenceLength,
    char* errorMessage,
    int errorMessageCapacity);

namespace
{
    std::string DefaultModelDir()
    {
        return "D:\\CantorAI\\xWorld\\models\\hf_models\\qwen2.5-0.5b";
    }

    std::string DefaultImagePath()
    {
        return "D:\\CantorAI\\Garnet\\data\\Dataset.1980Love\\imgs\\frame_0.jpg";
    }

    std::string DefaultPrompt()
    {
        return "Describe the image and list visible objects with short coordinates.";
    }

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

    float MakeValue(int index)
    {
        return std::sin(static_cast<float>(index) * 0.017f) * 0.125f;
    }

    double AbsSum(const std::vector<float>& values, float* maxAbs)
    {
        double sum = 0.0;
        float localMax = 0.0f;
        for (float value : values) {
            float absValue = std::fabs(value);
            sum += absValue;
            localMax = std::max(localMax, absValue);
        }
        if (maxAbs) {
            *maxAbs = localMax;
        }
        return sum;
    }
}

int main(int argc, char** argv)
{
    std::string modelDir = argc > 1 ? argv[1] : DefaultModelDir();
    std::string imagePath = argc > 2 ? argv[2] : DefaultImagePath();
    std::string prompt = argc > 3 ? argv[3] : DefaultPrompt();
    int minPixels = argc > 4 ? std::atoi(argv[4]) : 65536;
    int maxPixels = argc > 5 ? std::atoi(argv[5]) : 1003520;
    std::string dllPath = argc > 6 ? argv[6] : DefaultDllPath();

    NativeLibraryHandle dll = OpenNativeLibrary(dllPath.c_str());
    if (!dll) {
        std::cerr << "failed to load " << dllPath << ", error=" << NativeLibraryError() << "\n";
        return 1;
    }

    auto createRequest = reinterpret_cast<CreateDeviceRequestFn>(NativeLibrarySymbol(dll, "GarnetCreateQwenVLDeviceRequest"));
    auto destroyRequest = reinterpret_cast<DestroyDeviceRequestFn>(NativeLibrarySymbol(dll, "GarnetDestroyQwenVLDeviceRequest"));
    auto getRequestInfo = reinterpret_cast<GetDeviceRequestInfoFn>(NativeLibrarySymbol(dll, "GarnetGetQwenVLDeviceRequestInfo"));
    auto allocateRequestKV = reinterpret_cast<AllocateRequestKVFn>(NativeLibrarySymbol(dll, "GarnetAllocateQwenVLDeviceRequestKV"));
    auto getRequestKVInfo = reinterpret_cast<GetRequestKVInfoFn>(NativeLibrarySymbol(dll, "GarnetGetQwenVLDeviceRequestKVInfo"));
    auto getRequestKVState = reinterpret_cast<GetRequestKVStateFn>(NativeLibrarySymbol(dll, "GarnetGetQwenVLDeviceRequestKVState"));
    auto writeKVDevice = reinterpret_cast<WriteDeviceKVDeviceFn>(NativeLibrarySymbol(dll, "GarnetQwenVLDeviceRequestKVWriteDevice"));
    auto attentionKVDevice = reinterpret_cast<AttentionDeviceKVDeviceFn>(NativeLibrarySymbol(dll, "GarnetQwenVLDeviceRequestKVAttentionDevice"));
    if (!createRequest || !destroyRequest || !getRequestInfo || !allocateRequestKV || !getRequestKVInfo || !getRequestKVState || !writeKVDevice || !attentionKVDevice) {
        std::cerr << "missing request/KV exports\n";
        CloseNativeLibrary(dll);
        return 2;
    }

    char error[2048] = {};
    long long requestHandle = 0;
    int rc = createRequest(
        modelDir.c_str(),
        imagePath.c_str(),
        prompt.c_str(),
        minPixels,
        maxPixels,
        &requestHandle,
        error,
        static_cast<int>(sizeof(error)));
    if (rc != 0) {
        std::cerr << "create Qwen-VL GPU request failed rc=" << rc << " error=" << error << "\n";
        CloseNativeLibrary(dll);
        return 3;
    }

    void* inputIdsDevice = nullptr;
    void* mmTypesDevice = nullptr;
    void* pixelValuesDevice = nullptr;
    size_t inputIdsBytes = 0;
    size_t mmTypesBytes = 0;
    size_t pixelValuesBytes = 0;
    int inputIdCount = 0;
    int pixelValueCount = 0;
    long long grid[3] = { 0, 0, 0 };
    int sourceH = 0;
    int sourceW = 0;
    int resizedH = 0;
    int resizedW = 0;
    long long imageUs = 0;
    long long tokenizeUs = 0;
    long long uploadUs = 0;
    long long totalUs = 0;
    rc = getRequestInfo(
        requestHandle,
        &inputIdsDevice,
        &inputIdsBytes,
        &inputIdCount,
        &mmTypesDevice,
        &mmTypesBytes,
        &pixelValuesDevice,
        &pixelValuesBytes,
        &pixelValueCount,
        grid,
        &sourceH,
        &sourceW,
        &resizedH,
        &resizedW,
        &imageUs,
        &tokenizeUs,
        &uploadUs,
        &totalUs,
        error,
        static_cast<int>(sizeof(error)));
    if (rc != 0 || !inputIdsDevice || !mmTypesDevice || !pixelValuesDevice || inputIdCount <= 0 || pixelValueCount <= 0) {
        std::cerr << "Qwen-VL GPU request info invalid rc=" << rc << " error=" << error << "\n";
        destroyRequest(requestHandle, error, static_cast<int>(sizeof(error)));
        CloseNativeLibrary(dll);
        return 4;
    }

    constexpr int pageSize = 16;
    constexpr int qHeads = 4;
    constexpr int kvHeads = 2;
    constexpr int headDim = 16;
    constexpr int qWidth = qHeads * headDim;
    constexpr int kvWidth = kvHeads * headDim;
    constexpr int qkvStride = qWidth + 2 * kvWidth;
    int sequenceLength = std::min(inputIdCount, 64);
    int maxNewTokens = 32;

    long long kvHandle = 0;
    int kvMaxTokens = 0;
    int logicalPages = 0;
    int physicalPages = 0;
    rc = allocateRequestKV(
        requestHandle,
        maxNewTokens,
        pageSize,
        qHeads,
        kvHeads,
        headDim,
        0,
        &kvHandle,
        &kvMaxTokens,
        &logicalPages,
        &physicalPages,
        error,
        static_cast<int>(sizeof(error)));
    if (rc != 0) {
        std::cerr << "allocate request-owned device KV failed rc=" << rc << " error=" << error << "\n";
        destroyRequest(requestHandle, error, static_cast<int>(sizeof(error)));
        CloseNativeLibrary(dll);
        return 5;
    }
    long long kvInfoHandle = 0;
    int kvInfoMaxTokens = 0;
    int kvInfoPageSize = 0;
    int kvInfoLogicalPages = 0;
    int kvInfoPhysicalPages = 0;
    int kvInfoQHeads = 0;
    int kvInfoKVHeads = 0;
    int kvInfoHeadDim = 0;
    rc = getRequestKVInfo(
        requestHandle,
        &kvInfoHandle,
        &kvInfoMaxTokens,
        &kvInfoPageSize,
        &kvInfoLogicalPages,
        &kvInfoPhysicalPages,
        &kvInfoQHeads,
        &kvInfoKVHeads,
        &kvInfoHeadDim,
        error,
        static_cast<int>(sizeof(error)));
    if (rc != 0 || kvInfoHandle != kvHandle || kvInfoPageSize != pageSize || kvInfoQHeads != qHeads ||
        kvInfoKVHeads != kvHeads || kvInfoHeadDim != headDim) {
        std::cerr << "request KV info mismatch rc=" << rc << " error=" << error << "\n";
        destroyRequest(requestHandle, error, static_cast<int>(sizeof(error)));
        CloseNativeLibrary(dll);
        return 5;
    }

    int kvLogicalLengthBeforeWrite = -1;
    rc = getRequestKVState(
        requestHandle,
        nullptr,
        nullptr,
        &kvLogicalLengthBeforeWrite,
        nullptr,
        nullptr,
        nullptr,
        error,
        static_cast<int>(sizeof(error)));
    if (rc != 0 || kvLogicalLengthBeforeWrite != 0) {
        std::cerr << "request KV initial logical length mismatch rc=" << rc
            << " logical_length=" << kvLogicalLengthBeforeWrite << " error=" << error << "\n";
        destroyRequest(requestHandle, error, static_cast<int>(sizeof(error)));
        CloseNativeLibrary(dll);
        return 5;
    }

    std::vector<float> qkv(static_cast<size_t>(sequenceLength) * static_cast<size_t>(qkvStride));
    std::vector<float> q(qWidth);
    for (size_t i = 0; i < qkv.size(); ++i) {
        qkv[i] = MakeValue(static_cast<int>(i));
    }
    for (int i = 0; i < qWidth; ++i) {
        q[static_cast<size_t>(i)] = MakeValue(i + 4096);
    }

    float* dQKV = nullptr;
    float* dQ = nullptr;
    float* dOutput = nullptr;
    std::vector<float> output(qWidth, 0.0f);
    bool cudaOk =
        CheckCuda(cudaMalloc(&dQKV, qkv.size() * sizeof(float)), "cudaMalloc dQKV") &&
        CheckCuda(cudaMalloc(&dQ, q.size() * sizeof(float)), "cudaMalloc dQ") &&
        CheckCuda(cudaMalloc(&dOutput, output.size() * sizeof(float)), "cudaMalloc dOutput") &&
        CheckCuda(cudaMemcpy(dQKV, qkv.data(), qkv.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy qkv H2D") &&
        CheckCuda(cudaMemcpy(dQ, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy q H2D") &&
        CheckCuda(cudaMemset(dOutput, 0, output.size() * sizeof(float)), "cudaMemset output");
    if (!cudaOk) {
        if (dQKV) cudaFree(dQKV);
        if (dQ) cudaFree(dQ);
        if (dOutput) cudaFree(dOutput);
        destroyRequest(requestHandle, error, static_cast<int>(sizeof(error)));
        CloseNativeLibrary(dll);
        return 6;
    }

    rc = writeKVDevice(requestHandle, dQKV, sequenceLength, 0, error, static_cast<int>(sizeof(error)));
    if (rc != 0) {
        std::cerr << "device KV write failed rc=" << rc << " error=" << error << "\n";
        cudaFree(dQKV);
        cudaFree(dQ);
        cudaFree(dOutput);
        destroyRequest(requestHandle, error, static_cast<int>(sizeof(error)));
        CloseNativeLibrary(dll);
        return 7;
    }

    int kvLogicalLengthAfterWrite = -1;
    rc = getRequestKVState(
        requestHandle,
        nullptr,
        nullptr,
        &kvLogicalLengthAfterWrite,
        nullptr,
        nullptr,
        nullptr,
        error,
        static_cast<int>(sizeof(error)));
    if (rc != 0 || kvLogicalLengthAfterWrite != sequenceLength) {
        std::cerr << "request KV logical length after write mismatch rc=" << rc
            << " logical_length=" << kvLogicalLengthAfterWrite << " error=" << error << "\n";
        cudaFree(dQKV);
        cudaFree(dQ);
        cudaFree(dOutput);
        destroyRequest(requestHandle, error, static_cast<int>(sizeof(error)));
        CloseNativeLibrary(dll);
        return 7;
    }

    int overReadRc = attentionKVDevice(requestHandle, dQ, dOutput, sequenceLength + 1, error, static_cast<int>(sizeof(error)));
    if (overReadRc == 0) {
        std::cerr << "request KV attention unexpectedly allowed read past logical length\n";
        cudaFree(dQKV);
        cudaFree(dQ);
        cudaFree(dOutput);
        destroyRequest(requestHandle, error, static_cast<int>(sizeof(error)));
        CloseNativeLibrary(dll);
        return 8;
    }

    rc = attentionKVDevice(requestHandle, dQ, dOutput, sequenceLength, error, static_cast<int>(sizeof(error)));
    if (rc != 0) {
        std::cerr << "device KV attention failed rc=" << rc << " error=" << error << "\n";
        cudaFree(dQKV);
        cudaFree(dQ);
        cudaFree(dOutput);
        destroyRequest(requestHandle, error, static_cast<int>(sizeof(error)));
        CloseNativeLibrary(dll);
        return 8;
    }
    if (!CheckCuda(cudaMemcpy(output.data(), dOutput, output.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy output D2H")) {
        cudaFree(dQKV);
        cudaFree(dQ);
        cudaFree(dOutput);
        destroyRequest(requestHandle, error, static_cast<int>(sizeof(error)));
        CloseNativeLibrary(dll);
        return 9;
    }

    float maxAbs = 0.0f;
    double absSum = AbsSum(output, &maxAbs);
    if (!(absSum > 0.0) || !std::isfinite(absSum)) {
        std::cerr << "KV output invalid\n";
        cudaFree(dQKV);
        cudaFree(dQ);
        cudaFree(dOutput);
        destroyRequest(requestHandle, error, static_cast<int>(sizeof(error)));
        CloseNativeLibrary(dll);
        return 10;
    }

    cudaFree(dQKV);
    cudaFree(dQ);
    cudaFree(dOutput);
    int destroyReqRc = destroyRequest(requestHandle, error, static_cast<int>(sizeof(error)));
    if (destroyReqRc != 0) {
        std::cerr << "destroy request failed request_rc=" << destroyReqRc << " error=" << error << "\n";
        CloseNativeLibrary(dll);
        return 11;
    }

    int visualTokens = static_cast<int>((grid[0] * grid[1] * grid[2]) / 4);
    std::cout << "Garnet Qwen-VL request + device KV native smoke passed\n";
    std::cout << "source: " << sourceW << "x" << sourceH << "\n";
    std::cout << "resized: " << resizedW << "x" << resizedH << "\n";
    std::cout << "grid_thw: [" << grid[0] << ", " << grid[1] << ", " << grid[2] << "]\n";
    std::cout << "prompt_tokens: " << inputIdCount << "\n";
    std::cout << "visual_tokens: " << visualTokens << "\n";
    std::cout << "pixel_values_bytes_device: " << pixelValuesBytes << "\n";
    std::cout << "kv_sequence_length_tested: " << sequenceLength << "\n";
    std::cout << "kv_request_max_tokens: " << kvMaxTokens << "\n";
    std::cout << "kv_logical_length_after_write: " << kvLogicalLengthAfterWrite << "\n";
    std::cout << "kv_logical_pages: " << logicalPages << "\n";
    std::cout << "kv_physical_pages: " << physicalPages << "\n";
    std::cout << "kv_output_abs_sum: " << absSum << "\n";
    std::cout << "kv_output_max_abs: " << maxAbs << "\n";
    std::cout << "image_preprocess_ms: " << static_cast<double>(imageUs) / 1000.0 << "\n";
    std::cout << "tokenize_ms: " << static_cast<double>(tokenizeUs) / 1000.0 << "\n";
    std::cout << "tensor_upload_ms: " << static_cast<double>(uploadUs) / 1000.0 << "\n";
    std::cout << "request_total_ms: " << static_cast<double>(totalUs) / 1000.0 << "\n";

    CloseNativeLibrary(dll);
    return 0;
}
