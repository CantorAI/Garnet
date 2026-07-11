#include "qwen_vision_runner.h"

#include "../cuda/cuda_lib.h"
#include "../tensor/garnet_tensor.h"
#include "../tensor/tensor_helper.h"

#include <cuda_runtime.h>
#include <iostream>

namespace
{
    cudaError_t CreateVisionExecutionStream(cudaStream_t* stream)
    {
        if (!stream) return cudaErrorInvalidValue;
        *stream = cudaStreamPerThread;
        return cudaSuccess;
    }

    cudaError_t DestroyVisionExecutionStream(cudaStream_t)
    {
        return cudaSuccess;
    }
}

#define cudaStreamCreate CreateVisionExecutionStream
#define cudaStreamDestroy DestroyVisionExecutionStream

namespace Garnet
{
    namespace
    {
        X::Value CallForward(X::Value model, std::initializer_list<X::Value> values)
        {
            X::Value forward = model["forward"];
            if (!forward.IsObject()) return X::Value();
            X::ARGS args(static_cast<int>(values.size()));
            for (const auto& value : values) args.push_back(value);
            return forward.ObjCall(args);
        }

        X::Value WrapGPU2D(void* deviceMemory, int rows, int columns)
        {
            if (!deviceMemory || rows <= 0 || columns <= 0) return X::Value();
            X::Tensor output = X::g_pXHost->CreateTensor();
            X::Port::vector<int> shape(2);
            shape.push_back(rows);
            shape.push_back(columns);
            output->SetDataType(X::TensorDataType::FLOAT32);
            output->SetShape(shape);
            if (TensorHelper::AttachGPUMemory(output, deviceMemory) != TensorOpStatus::Success) {
                return X::Value();
            }
            return X::Value(output);
        }

        X::Value AddGPU(X::Value lhsValue, X::Value rhsValue)
        {
            if (!lhsValue.IsTensor() || !rhsValue.IsTensor()) return X::Value();
            X::Tensor lhs(lhsValue);
            X::Tensor rhs(rhsValue);
            if (lhs->GetDataType() != X::TensorDataType::FLOAT32 ||
                rhs->GetDataType() != X::TensorDataType::FLOAT32 ||
                lhs->GetDimCount() != 2 || rhs->GetDimCount() != 2 ||
                lhs->GetDimSize(0) != rhs->GetDimSize(0) || lhs->GetDimSize(1) != rhs->GetDimSize(1) ||
                TensorHelper::EnsureGPUMemory(lhs) != TensorOpStatus::Success ||
                TensorHelper::EnsureGPUMemory(rhs) != TensorOpStatus::Success) return X::Value();
            size_t bytes = static_cast<size_t>(lhs->GetDataSize());
            float* output = nullptr;
            cudaStream_t stream = nullptr;
            cudaError_t err = cudaStreamCreate(&stream);
            if (err == cudaSuccess) err = cudaMalloc(&output, bytes);
            if (err == cudaSuccess) err = runTensorAddFP32(
                static_cast<const float*>(TensorHelper::GetGPUMemory(lhs)),
                static_cast<const float*>(TensorHelper::GetGPUMemory(rhs)), output,
                static_cast<int>(lhs->GetCount()), stream);
            if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
            if (stream) cudaStreamDestroy(stream);
            if (err != cudaSuccess) {
                if (output) cudaFree(output);
                return X::Value();
            }
            X::Value wrapped = WrapGPU2D(output, static_cast<int>(lhs->GetDimSize(0)), static_cast<int>(lhs->GetDimSize(1)));
            if (!wrapped.IsValid()) cudaFree(output);
            return wrapped;
        }

        X::Value GeluGPU(X::Value inputValue)
        {
            if (!inputValue.IsTensor()) return X::Value();
            X::Tensor input(inputValue);
            if (input->GetDataType() != X::TensorDataType::FLOAT32 || input->GetDimCount() != 2 ||
                TensorHelper::EnsureGPUMemory(input) != TensorOpStatus::Success) return X::Value();
            size_t bytes = static_cast<size_t>(input->GetDataSize());
            float* output = nullptr;
            cudaStream_t stream = nullptr;
            cudaError_t err = cudaStreamCreate(&stream);
            if (err == cudaSuccess) err = cudaMalloc(&output, bytes);
            if (err == cudaSuccess) err = runGeluTanhFP32(
                static_cast<const float*>(TensorHelper::GetGPUMemory(input)), output,
                static_cast<int>(input->GetCount()), stream);
            if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
            if (stream) cudaStreamDestroy(stream);
            if (err != cudaSuccess) {
                if (output) cudaFree(output);
                return X::Value();
            }
            X::Value wrapped = WrapGPU2D(output, static_cast<int>(input->GetDimSize(0)), static_cast<int>(input->GetDimSize(1)));
            if (!wrapped.IsValid()) cudaFree(output);
            return wrapped;
        }

        X::Value VisionRoPEGPU(X::Value qkvValue, X::Value cosValue, X::Value sinValue)
        {
            if (!qkvValue.IsTensor() || !cosValue.IsTensor() || !sinValue.IsTensor()) return X::Value();
            X::Tensor qkv(qkvValue);
            X::Tensor cos(cosValue);
            X::Tensor sin(sinValue);
            int tokens = static_cast<int>(qkv->GetDimSize(0));
            int headDim = static_cast<int>(cos->GetDimSize(1));
            int numHeads = static_cast<int>(qkv->GetDimSize(1)) / (3 * headDim);
            if (tokens <= 0 || headDim <= 0 || numHeads <= 0 ||
                qkv->GetDimSize(1) != 3 * numHeads * headDim ||
                TensorHelper::EnsureGPUMemory(qkv) != TensorOpStatus::Success ||
                TensorHelper::EnsureGPUMemory(cos) != TensorOpStatus::Success ||
                TensorHelper::EnsureGPUMemory(sin) != TensorOpStatus::Success) return X::Value();
            size_t bytes = static_cast<size_t>(qkv->GetDataSize());
            float* output = nullptr;
            cudaStream_t stream = nullptr;
            cudaError_t err = cudaStreamCreate(&stream);
            if (err == cudaSuccess) err = cudaMalloc(&output, bytes);
            if (err == cudaSuccess) err = runVisionRoPEFP32(
                static_cast<const float*>(TensorHelper::GetGPUMemory(qkv)),
                static_cast<const float*>(TensorHelper::GetGPUMemory(cos)),
                static_cast<const float*>(TensorHelper::GetGPUMemory(sin)),
                output, tokens, numHeads, headDim, stream);
            if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
            if (stream) cudaStreamDestroy(stream);
            if (err != cudaSuccess) {
                if (output) cudaFree(output);
                return X::Value();
            }
            X::Value wrapped = WrapGPU2D(output, tokens, static_cast<int>(qkv->GetDimSize(1)));
            if (!wrapped.IsValid()) cudaFree(output);
            return wrapped;
        }

        X::Value ReshapeCopyGPU(X::Value inputValue, int rows, int columns)
        {
            if (!inputValue.IsTensor()) return X::Value();
            X::Tensor input(inputValue);
            if (input->GetDataType() != X::TensorDataType::FLOAT32 ||
                input->GetCount() != static_cast<long long>(rows) * columns ||
                TensorHelper::EnsureGPUMemory(input) != TensorOpStatus::Success) return X::Value();
            size_t bytes = static_cast<size_t>(input->GetDataSize());
            void* output = nullptr;
            cudaError_t err = cudaMalloc(&output, bytes);
            if (err == cudaSuccess) err = cudaMemcpy(output, TensorHelper::GetGPUMemory(input), bytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                if (output) cudaFree(output);
                return X::Value();
            }
            X::Value wrapped = WrapGPU2D(output, rows, columns);
            if (!wrapped.IsValid()) cudaFree(output);
            return wrapped;
        }

        X::Value RunMerger(X::Value bundle, X::Value hidden, bool postShuffleNorm)
        {
            if (!bundle.IsObject() || !hidden.IsTensor()) return X::Value();
            X::Tensor hiddenTensor(hidden);
            int patchCount = static_cast<int>(hiddenTensor->GetDimSize(0));
            int hiddenSize = static_cast<int>(hiddenTensor->GetDimSize(1));
            if ((patchCount % 4) != 0) return X::Value();

            X::Value mergedInput;
            if (postShuffleNorm) {
                X::Value reshaped = ReshapeCopyGPU(hidden, patchCount / 4, hiddenSize * 4);
                mergedInput = CallForward(bundle["norm"], { reshaped });
            }
            else {
                X::Value normed = CallForward(bundle["norm"], { hidden });
                mergedInput = ReshapeCopyGPU(normed, patchCount / 4, hiddenSize * 4);
            }
            X::Value fc1 = CallForward(bundle["fc1"], { mergedInput });
            X::Value activated = GeluGPU(fc1);
            return CallForward(bundle["fc2"], { activated });
        }
    }

    void QwenVisionRunner::Forward(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (params.size() < 3 || !params[0].IsTensor() || !params[1].IsTensor() || !params[2].IsTensor() ||
            !mLayerBundles.IsList() || !mMergerBundle.IsObject()) {
            std::cout << "[QwenVisionRunner] forward(hidden, cos, sin) invalid inputs." << std::endl;
            retValue = X::Value();
            return;
        }
        X::Value hidden = params[0];
        X::Value cos = params[1];
        X::Value sin = params[2];
        X::List layers(mLayerBundles);
        X::List deepstackFeatures;
        for (long long layerIndex = 0; layerIndex < layers->Size(); ++layerIndex) {
            X::Value bundle = layers->Get(layerIndex);
            X::Value norm1 = CallForward(bundle["norm1"], { hidden });
            X::Value qkv = CallForward(bundle["qkv"], { norm1 });
            X::Value rope = VisionRoPEGPU(qkv, cos, sin);
            X::Value attention = CallForward(bundle["attention"], { rope });
            X::Value projected = CallForward(bundle["proj"], { attention });
            X::Value residual = AddGPU(hidden, projected);
            X::Value norm2 = CallForward(bundle["norm2"], { residual });
            X::Value mlpLinear = CallForward(bundle["mlp_fc1"], { norm2 });
            X::Value mlpHidden = GeluGPU(mlpLinear);
            X::Value mlp = CallForward(bundle["mlp_fc2"], { mlpHidden });
            hidden = AddGPU(residual, mlp);
            if (!hidden.IsTensor()) {
                std::cout << "[QwenVisionRunner] failed at layer " << layerIndex << std::endl;
                retValue = X::Value();
                return;
            }
            X::Value deepstackBundle = bundle["deepstack"];
            if (deepstackBundle.IsObject()) {
                X::Value feature = RunMerger(deepstackBundle, hidden, true);
                if (!feature.IsTensor()) {
                    std::cout << "[QwenVisionRunner] deepstack merger failed at layer " << layerIndex << std::endl;
                    retValue = X::Value();
                    return;
                }
                deepstackFeatures->AddItem(feature);
            }
        }
        X::Value poolerOutput = RunMerger(mMergerBundle, hidden, false);
        X::Dict result;
        result->Set("last_hidden_state", hidden);
        result->Set("pooler_output", poolerOutput);
        result->Set("deepstack_features", X::Value(deepstackFeatures));
        retValue = result;
    }

    void QwenVisionRunner::Stats(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        X::Dict stats;
        stats->Set("layer_count", X::Value(mLayerBundles.IsList() ? static_cast<int>(X::List(mLayerBundles)->Size()) : 0));
        stats->Set("merger_configured", X::Value(mMergerBundle.IsObject()));
        stats->Set("deepstack_supported", X::Value(true));
        stats->Set("device_tensor_chain", X::Value(true));
        retValue = stats;
    }
}
