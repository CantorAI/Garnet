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

using CreateDeviceKVFn = int (*)(
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

using DestroyDeviceKVFn = int (*)(
    long long handle,
    char* errorMessage,
    int errorMessageCapacity);

using WriteDeviceKVFn = int (*)(
    long long handle,
    const float* qkv,
    int tokenCount,
    int startPosition,
    char* errorMessage,
    int errorMessageCapacity);

using AttentionDeviceKVFn = int (*)(
    long long handle,
    const float* q,
    float* output,
    int sequenceLength,
    char* errorMessage,
    int errorMessageCapacity);

using WriteDeviceKVDeviceFn = int (*)(
    long long handle,
    const float* deviceQKV,
    int tokenCount,
    int startPosition,
    char* errorMessage,
    int errorMessageCapacity);

using AttentionDeviceKVDeviceFn = int (*)(
    long long handle,
    const float* deviceQ,
    float* deviceOutput,
    int sequenceLength,
    char* errorMessage,
    int errorMessageCapacity);

namespace
{
    std::string DefaultDllPath()
    {
        return "D:\\CantorAI\\Garnet\\out\\build\\x64-Debug\\bin\\garnet.dll";
    }

    float MakeValue(int index)
    {
        return std::sin(static_cast<float>(index) * 0.013f) * 0.25f;
    }

    bool CheckCuda(cudaError_t err, const char* op)
    {
        if (err == cudaSuccess) {
            return true;
        }
        std::cerr << op << " failed: " << cudaGetErrorString(err) << "\n";
        return false;
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
    std::string dllPath = argc > 1 ? argv[1] : DefaultDllPath();
    NativeLibraryHandle dll = OpenNativeLibrary(dllPath.c_str());
    if (!dll) {
        std::cerr << "failed to load " << dllPath << ", error=" << NativeLibraryError() << "\n";
        return 1;
    }

    auto createKV = reinterpret_cast<CreateDeviceKVFn>(NativeLibrarySymbol(dll, "GarnetCreateDevicePagedKVFP32"));
    auto destroyKV = reinterpret_cast<DestroyDeviceKVFn>(NativeLibrarySymbol(dll, "GarnetDestroyDevicePagedKVFP32"));
    auto writeKV = reinterpret_cast<WriteDeviceKVFn>(NativeLibrarySymbol(dll, "GarnetDevicePagedKVWriteFP32"));
    auto attentionKV = reinterpret_cast<AttentionDeviceKVFn>(NativeLibrarySymbol(dll, "GarnetDevicePagedKVAttentionFP32"));
    auto writeKVDevice = reinterpret_cast<WriteDeviceKVDeviceFn>(NativeLibrarySymbol(dll, "GarnetDevicePagedKVWriteDeviceFP32"));
    auto attentionKVDevice = reinterpret_cast<AttentionDeviceKVDeviceFn>(NativeLibrarySymbol(dll, "GarnetDevicePagedKVAttentionDeviceFP32"));
    if (!createKV || !destroyKV || !writeKV || !attentionKV || !writeKVDevice || !attentionKVDevice) {
        std::cerr << "missing device KV exports\n";
        CloseNativeLibrary(dll);
        return 2;
    }

    constexpr int pageSize = 8;
    constexpr int sequenceLength = 17;
    constexpr int logicalPages = (sequenceLength + pageSize - 1) / pageSize;
    constexpr int physicalPages = logicalPages;
    constexpr int qHeads = 4;
    constexpr int kvHeads = 2;
    constexpr int headDim = 16;
    constexpr int qWidth = qHeads * headDim;
    constexpr int kvWidth = kvHeads * headDim;
    constexpr int qkvStride = qWidth + 2 * kvWidth;

    std::vector<int> pageTable(logicalPages);
    for (int i = 0; i < logicalPages; ++i) {
        pageTable[static_cast<size_t>(i)] = i;
    }

    char error[2048] = {};
    long long handle = 0;
    int rc = createKV(
        physicalPages,
        pageSize,
        logicalPages,
        qHeads,
        kvHeads,
        headDim,
        pageTable.data(),
        &handle,
        error,
        static_cast<int>(sizeof(error)));
    if (rc != 0) {
        std::cerr << "create device KV failed rc=" << rc << " error=" << error << "\n";
        CloseNativeLibrary(dll);
        return 3;
    }

    std::vector<float> qkv(static_cast<size_t>(sequenceLength) * static_cast<size_t>(qkvStride));
    for (size_t i = 0; i < qkv.size(); ++i) {
        qkv[i] = MakeValue(static_cast<int>(i));
    }

    rc = writeKV(handle, qkv.data(), sequenceLength, 0, error, static_cast<int>(sizeof(error)));
    if (rc != 0) {
        std::cerr << "write device KV failed rc=" << rc << " error=" << error << "\n";
        destroyKV(handle, error, static_cast<int>(sizeof(error)));
        CloseNativeLibrary(dll);
        return 4;
    }

    std::vector<float> q(qWidth);
    for (int i = 0; i < qWidth; ++i) {
        q[static_cast<size_t>(i)] = MakeValue(i + 1000);
    }
    std::vector<float> output(qWidth, 0.0f);
    rc = attentionKV(handle, q.data(), output.data(), sequenceLength, error, static_cast<int>(sizeof(error)));
    if (rc != 0) {
        std::cerr << "attention device KV failed rc=" << rc << " error=" << error << "\n";
        destroyKV(handle, error, static_cast<int>(sizeof(error)));
        CloseNativeLibrary(dll);
        return 5;
    }

    float maxAbs = 0.0f;
    double absSum = AbsSum(output, &maxAbs);
    if (!(absSum > 0.0) || !std::isfinite(absSum)) {
        std::cerr << "device KV attention output is invalid\n";
        destroyKV(handle, error, static_cast<int>(sizeof(error)));
        CloseNativeLibrary(dll);
        return 6;
    }

    rc = destroyKV(handle, error, static_cast<int>(sizeof(error)));
    if (rc != 0) {
        std::cerr << "destroy device KV failed rc=" << rc << " error=" << error << "\n";
        CloseNativeLibrary(dll);
        return 7;
    }

    long long deviceHandle = 0;
    rc = createKV(
        physicalPages,
        pageSize,
        logicalPages,
        qHeads,
        kvHeads,
        headDim,
        pageTable.data(),
        &deviceHandle,
        error,
        static_cast<int>(sizeof(error)));
    if (rc != 0) {
        std::cerr << "create device-pointer KV failed rc=" << rc << " error=" << error << "\n";
        CloseNativeLibrary(dll);
        return 8;
    }

    float* dQKV = nullptr;
    float* dQ = nullptr;
    float* dOutput = nullptr;
    std::vector<float> deviceOutput(qWidth, 0.0f);
    bool cudaOk =
        CheckCuda(cudaMalloc(&dQKV, qkv.size() * sizeof(float)), "cudaMalloc dQKV") &&
        CheckCuda(cudaMalloc(&dQ, q.size() * sizeof(float)), "cudaMalloc dQ") &&
        CheckCuda(cudaMalloc(&dOutput, output.size() * sizeof(float)), "cudaMalloc dOutput") &&
        CheckCuda(cudaMemcpy(dQKV, qkv.data(), qkv.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy qkv H2D") &&
        CheckCuda(cudaMemcpy(dQ, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy q H2D") &&
        CheckCuda(cudaMemset(dOutput, 0, output.size() * sizeof(float)), "cudaMemset dOutput");
    if (!cudaOk) {
        if (dQKV) cudaFree(dQKV);
        if (dQ) cudaFree(dQ);
        if (dOutput) cudaFree(dOutput);
        destroyKV(deviceHandle, error, static_cast<int>(sizeof(error)));
        CloseNativeLibrary(dll);
        return 9;
    }

    rc = writeKVDevice(deviceHandle, dQKV, sequenceLength, 0, error, static_cast<int>(sizeof(error)));
    if (rc != 0) {
        std::cerr << "device-pointer write KV failed rc=" << rc << " error=" << error << "\n";
        cudaFree(dQKV);
        cudaFree(dQ);
        cudaFree(dOutput);
        destroyKV(deviceHandle, error, static_cast<int>(sizeof(error)));
        CloseNativeLibrary(dll);
        return 10;
    }

    rc = attentionKVDevice(deviceHandle, dQ, dOutput, sequenceLength, error, static_cast<int>(sizeof(error)));
    if (rc != 0) {
        std::cerr << "device-pointer attention KV failed rc=" << rc << " error=" << error << "\n";
        cudaFree(dQKV);
        cudaFree(dQ);
        cudaFree(dOutput);
        destroyKV(deviceHandle, error, static_cast<int>(sizeof(error)));
        CloseNativeLibrary(dll);
        return 11;
    }
    if (!CheckCuda(cudaMemcpy(deviceOutput.data(), dOutput, deviceOutput.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy device output D2H")) {
        cudaFree(dQKV);
        cudaFree(dQ);
        cudaFree(dOutput);
        destroyKV(deviceHandle, error, static_cast<int>(sizeof(error)));
        CloseNativeLibrary(dll);
        return 12;
    }

    float deviceMaxAbs = 0.0f;
    double deviceAbsSum = AbsSum(deviceOutput, &deviceMaxAbs);
    double maxDiff = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        maxDiff = std::max(maxDiff, static_cast<double>(std::fabs(output[i] - deviceOutput[i])));
    }
    if (!(deviceAbsSum > 0.0) || !std::isfinite(deviceAbsSum) || maxDiff > 1.0e-6) {
        std::cerr << "device-pointer KV output mismatch or invalid output, max_diff=" << maxDiff << "\n";
        cudaFree(dQKV);
        cudaFree(dQ);
        cudaFree(dOutput);
        destroyKV(deviceHandle, error, static_cast<int>(sizeof(error)));
        CloseNativeLibrary(dll);
        return 13;
    }

    cudaFree(dQKV);
    cudaFree(dQ);
    cudaFree(dOutput);
    rc = destroyKV(deviceHandle, error, static_cast<int>(sizeof(error)));
    if (rc != 0) {
        std::cerr << "destroy device-pointer KV failed rc=" << rc << " error=" << error << "\n";
        CloseNativeLibrary(dll);
        return 14;
    }

    std::cout << "Garnet device KV native smoke passed\n";
    std::cout << "sequence_length: " << sequenceLength << "\n";
    std::cout << "page_size: " << pageSize << "\n";
    std::cout << "logical_pages: " << logicalPages << "\n";
    std::cout << "q_heads: " << qHeads << "\n";
    std::cout << "kv_heads: " << kvHeads << "\n";
    std::cout << "head_dim: " << headDim << "\n";
    std::cout << "output_abs_sum: " << absSum << "\n";
    std::cout << "output_max_abs: " << maxAbs << "\n";
    std::cout << "device_pointer_output_abs_sum: " << deviceAbsSum << "\n";
    std::cout << "device_pointer_output_max_abs: " << deviceMaxAbs << "\n";
    std::cout << "device_pointer_max_diff_vs_host_abi: " << maxDiff << "\n";

    CloseNativeLibrary(dll);
    return 0;
}
