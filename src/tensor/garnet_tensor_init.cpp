#include "garnet_tensor.h"
#include <cuda_runtime.h>
#include "tensor_helper.h"
#include <cuda_fp16.h>    // For __half and __float2half
#include <cuda_bf16.h>    // For __nv_bfloat16 and __float2bfloat16
#include <cuda_fp8.h>     // For __nv_fp8_e4m3 and __nv_fp8_e5m2
#include <cstring>

extern "C" {
    // Zeros
    void runInitZerosFloat(float* output, int numel);
    void runInitZerosFP16(__half* output, int numel);
    void runInitZerosBF16(__nv_bfloat16* output, int numel);

    // Ones
    void runInitOnesFloat(float* output, int numel);
    void runInitOnesFP16(__half* output, int numel);
    void runInitOnesBF16(__nv_bfloat16* output, int numel);

    // Full (constant value)
    void runInitFullFloat(float* output, int numel, float value);
    void runInitFullFP16(__half* output, int numel, __half value);
    void runInitFullBF16(__nv_bfloat16* output, int numel, __nv_bfloat16 value);

    // Rand (uniform [0,1))
    void runInitRandFloat(float* output, int numel, unsigned int seed);
    void runInitRandFP16(__half* output, int numel, unsigned int seed);
    void runInitRandBF16(__nv_bfloat16* output, int numel, unsigned int seed);

    // Randn (normal distribution, mean=0, std=1)
    void runInitRandnFloat(float* output, int numel, unsigned int seed);
    void runInitRandnFP16(__half* output, int numel, unsigned int seed);
    void runInitRandnBF16(__nv_bfloat16* output, int numel, unsigned int seed);

    // Uniform distribution in [low, high)
    void runInitUniformFloat(float* output, int numel, float low, float high, unsigned int seed);
    void runInitUniformFP16(__half* output, int numel, __half low, __half high, unsigned int seed);
    void runInitUniformBF16(__nv_bfloat16* output, int numel, __nv_bfloat16 low, __nv_bfloat16 high, unsigned int seed);

    // Normal distribution with specified mean and std
    void runInitNormalFloat(float* output, int numel, float mean, float std, unsigned int seed);
    void runInitNormalFP16(__half* output, int numel, float mean, float std, unsigned int seed);
    void runInitNormalBF16(__nv_bfloat16* output, int numel, float mean, float std, unsigned int seed);

    // Truncated Normal distribution
    void runInitTruncNormalFloat(float* output, int numel, float mean, float std, float a, float b, unsigned int seed);
    void runInitTruncNormalFP16(__half* output, int numel, float mean, float std, float a, float b, unsigned int seed);
    void runInitTruncNormalBF16(__nv_bfloat16* output, int numel, float mean, float std, float a, float b, unsigned int seed);
}

namespace Garnet {

    X::Value GarnetTensor::InitZeros(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& output)
    {
        if (!input.IsTensor()) {
            return X::Value();
        }
        X::Tensor tensor(input);
        int numel = tensor->GetCount();
        if (TensorHelper::EnsureGPUMemory(tensor) != TensorOpStatus::Success) {
            return X::Value();
        }
        void* gpuData = TensorHelper::GetGPUMemory(tensor);
        X::TensorDataType dtype = tensor->GetDataType();
        switch (dtype)
        {
        case X::TensorDataType::FLOAT32:
            runInitZerosFloat(reinterpret_cast<float*>(gpuData), numel);
            break;
        case X::TensorDataType::FLOAT16:
            runInitZerosFP16(reinterpret_cast<__half*>(gpuData), numel);
            break;
        case X::TensorDataType::BFLOAT16:
            runInitZerosBF16(reinterpret_cast<__nv_bfloat16*>(gpuData), numel);
            break;
        default:
            output = X::Value();
            return X::Value();
        }
        TensorHelper::CopyResultFromGPU(tensor);
        output =  X::Value(tensor);
		return output;
    }

    X::Value GarnetTensor::InitOnes(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& output)
    {
        if (!input.IsTensor()) {
            output =  X::Value();
            return X::Value();
        }
        X::Tensor tensor(input);
        int numel = tensor->GetCount();
        if (TensorHelper::EnsureGPUMemory(tensor) != TensorOpStatus::Success) {
            output =  X::Value();
            return X::Value();
        }
        void* gpuData = TensorHelper::GetGPUMemory(tensor);
        X::TensorDataType dtype = tensor->GetDataType();
        switch (dtype)
        {
        case X::TensorDataType::FLOAT32:
            runInitOnesFloat(reinterpret_cast<float*>(gpuData), numel);
            break;
        case X::TensorDataType::FLOAT16:
            runInitOnesFP16(reinterpret_cast<__half*>(gpuData), numel);
            break;
        case X::TensorDataType::BFLOAT16:
            runInitOnesBF16(reinterpret_cast<__nv_bfloat16*>(gpuData), numel);
            break;
        default:
            output =  X::Value();
            return X::Value();
        }
        TensorHelper::CopyResultFromGPU(tensor);
        output =  X::Value(tensor);
        return output;
    }

    X::Value GarnetTensor::InitFull(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& output)
    {
        // Expect extra parameter "value"
        if (!input.IsTensor() || !kwParams.Has("value")) {
            output =  X::Value();
            return output;
        }
        X::Tensor tensor(input);
        int numel = tensor->GetCount();
        auto* pItem = kwParams.find("value");
        float val = (float)pItem->val.ToDouble();
        if (TensorHelper::EnsureGPUMemory(tensor) != TensorOpStatus::Success) {
            output =  X::Value();
            return output;
        }
        void* gpuData = TensorHelper::GetGPUMemory(tensor);
        X::TensorDataType dtype = tensor->GetDataType();
        switch (dtype)
        {
        case X::TensorDataType::FLOAT32:
            runInitFullFloat(reinterpret_cast<float*>(gpuData), numel, val);
            break;
        case X::TensorDataType::FLOAT16:
            runInitFullFP16(reinterpret_cast<__half*>(gpuData), numel, __float2half(val));
            break;
        case X::TensorDataType::BFLOAT16:
            runInitFullBF16(reinterpret_cast<__nv_bfloat16*>(gpuData), numel, __float2bfloat16(val));
            break;
        default:
            output =  X::Value();
            return X::Value();
        }
        TensorHelper::CopyResultFromGPU(tensor);
        output =  X::Value(tensor);
        return output;
    }

    X::Value GarnetTensor::InitRand(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& output)
    {
        if (!input.IsTensor()) {
            output =  X::Value();
            return output;
        }
        X::Tensor tensor(input);
        int numel = tensor->GetCount();
        unsigned int seed = kwParams.Has("seed") ? (unsigned int)kwParams.find("seed")->val.ToDouble() : 1234;
        if (TensorHelper::EnsureGPUMemory(tensor) != TensorOpStatus::Success) {
            output =  X::Value();
            return output;
        }
        void* gpuData = TensorHelper::GetGPUMemory(tensor);
        X::TensorDataType dtype = tensor->GetDataType();
        switch (dtype)
        {
        case X::TensorDataType::FLOAT32:
            runInitRandFloat(reinterpret_cast<float*>(gpuData), numel, seed);
            break;
        case X::TensorDataType::FLOAT16:
            runInitRandFP16(reinterpret_cast<__half*>(gpuData), numel, seed);
            break;
        case X::TensorDataType::BFLOAT16:
            runInitRandBF16(reinterpret_cast<__nv_bfloat16*>(gpuData), numel, seed);
            break;
        default:
            output =  X::Value();
            return output;
        }
        TensorHelper::CopyResultFromGPU(tensor);
        output =  X::Value(tensor);
        return output;
    }

    X::Value GarnetTensor::InitRandn(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& output)
    {
        if (!input.IsTensor()) {
            output =  X::Value();
            return output;
        }
        X::Tensor tensor(input);
        int numel = tensor->GetCount();
        unsigned int seed = kwParams.Has("seed") ? (unsigned int)kwParams.find("seed")->val.ToDouble() : 1234;
        if (TensorHelper::EnsureGPUMemory(tensor) != TensorOpStatus::Success) {
            output =  X::Value();
            return output;
        }
        void* gpuData = TensorHelper::GetGPUMemory(tensor);
        X::TensorDataType dtype = tensor->GetDataType();
        switch (dtype)
        {
        case X::TensorDataType::FLOAT32:
            runInitRandnFloat(reinterpret_cast<float*>(gpuData), numel, seed);
            break;
        case X::TensorDataType::FLOAT16:
            runInitRandnFP16(reinterpret_cast<__half*>(gpuData), numel, seed);
            break;
        case X::TensorDataType::BFLOAT16:
            runInitRandnBF16(reinterpret_cast<__nv_bfloat16*>(gpuData), numel, seed);
            break;
        default:
            output =  X::Value();
            return output;
        }
        TensorHelper::CopyResultFromGPU(tensor);
        output =  X::Value(tensor);
    }

    X::Value GarnetTensor::InitUniform(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& output)
    {
        // Expect extra parameters "low" and "high"
        if (!input.IsTensor() || !kwParams.Has("low") || !kwParams.Has("high")) {
            output =  X::Value();
            return output;
        }
        X::Tensor tensor(input);
        int numel = tensor->GetCount();
        float fLow = (float)kwParams.find("low")->val.ToDouble();
        float fHigh = (float)kwParams.find("high")->val.ToDouble();
        unsigned int seed = kwParams.Has("seed") ? (unsigned int)kwParams.find("seed")->val.ToDouble() : 1234;
        if (TensorHelper::EnsureGPUMemory(tensor) != TensorOpStatus::Success) {
            output =  X::Value();
            return output;
        }
        void* gpuData = TensorHelper::GetGPUMemory(tensor);
        X::TensorDataType dtype = tensor->GetDataType();
        switch (dtype)
        {
        case X::TensorDataType::FLOAT32:
            runInitUniformFloat(reinterpret_cast<float*>(gpuData), numel, fLow, fHigh, seed);
            break;
        case X::TensorDataType::FLOAT16:
            runInitUniformFP16(reinterpret_cast<__half*>(gpuData), numel, __float2half(fLow), __float2half(fHigh), seed);
            break;
        case X::TensorDataType::BFLOAT16:
            runInitUniformBF16(reinterpret_cast<__nv_bfloat16*>(gpuData), numel, __float2bfloat16(fLow), __float2bfloat16(fHigh), seed);
            break;
        default:
            output =  X::Value();
            return output;
        }
        TensorHelper::CopyResultFromGPU(tensor);
        output =  X::Value(tensor);
        return output;
    }

    X::Value GarnetTensor::InitNormal(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& output)
    {
        // Expect extra parameters "mean" and "std"
        if (!input.IsTensor() || !kwParams.Has("mean") || !kwParams.Has("std")) {
            output =  X::Value();
            return output;
        }
        X::Tensor tensor(input);
        int numel = tensor->GetCount();
        float fMean = (float)kwParams.find("mean")->val.ToDouble();
        float fStd = (float)kwParams.find("std")->val.ToDouble();
        unsigned int seed = kwParams.Has("seed") ? (unsigned int)kwParams.find("seed")->val.ToDouble() : 1234;
        if (TensorHelper::EnsureGPUMemory(tensor) != TensorOpStatus::Success) {
            output =  X::Value();
            return output;
        }
        void* gpuData = TensorHelper::GetGPUMemory(tensor);
        X::TensorDataType dtype = tensor->GetDataType();
        switch (dtype)
        {
        case X::TensorDataType::FLOAT32:
            runInitNormalFloat(reinterpret_cast<float*>(gpuData), numel, fMean, fStd, seed);
            break;
        case X::TensorDataType::FLOAT16:
            runInitNormalFP16(reinterpret_cast<__half*>(gpuData), numel, fMean, fStd, seed);
            break;
        case X::TensorDataType::BFLOAT16:
            runInitNormalBF16(reinterpret_cast<__nv_bfloat16*>(gpuData), numel, fMean, fStd, seed);
            break;
        default:
            output =  X::Value();
            return output;
        }
        TensorHelper::CopyResultFromGPU(tensor);
        output =  X::Value(tensor);
        return output;
    }

    X::Value GarnetTensor::InitTruncNormal(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& output)
    {
        // Expect extra parameters "mean", "std", "a", and "b"
        if (!input.IsTensor() || !kwParams.Has("mean") || !kwParams.Has("std") ||
            !kwParams.Has("a") || !kwParams.Has("b"))
        {
            output =  X::Value();
            return output;
        }
        X::Tensor tensor(input);
        int numel = tensor->GetCount();
        float fMean = (float)kwParams.find("mean")->val.ToDouble();
        float fStd = (float)kwParams.find("std")->val.ToDouble();
        float fA = (float)kwParams.find("a")->val.ToDouble();
        float fB = (float)kwParams.find("b")->val.ToDouble();
        unsigned int seed = kwParams.Has("seed") ? (unsigned int)kwParams.find("seed")->val.ToDouble() : 1234;
        if (TensorHelper::EnsureGPUMemory(tensor) != TensorOpStatus::Success) {
            output =  X::Value();
            return output;
        }
        void* gpuData = TensorHelper::GetGPUMemory(tensor);
        X::TensorDataType dtype = tensor->GetDataType();
        switch (dtype)
        {
        case X::TensorDataType::FLOAT32:
            runInitTruncNormalFloat(reinterpret_cast<float*>(gpuData), numel, fMean, fStd, fA, fB, seed);
            break;
        case X::TensorDataType::FLOAT16:
            runInitTruncNormalFP16(reinterpret_cast<__half*>(gpuData), numel, fMean, fStd, fA, fB, seed);
            break;
        case X::TensorDataType::BFLOAT16:
            runInitTruncNormalBF16(reinterpret_cast<__nv_bfloat16*>(gpuData), numel, fMean, fStd, fA, fB, seed);
            break;
        default:
            output =  X::Value();
            return output;
        }
        TensorHelper::CopyResultFromGPU(tensor);
        output =  X::Value(tensor);
        return output;
    }

} // namespace Garnet
