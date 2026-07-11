#include "qwen_text_runner.h"

#include "../cuda/cuda_lib.h"
#include "../tensor/garnet_tensor.h"
#include "../tensor/tensor_helper.h"

#include <cuda_runtime.h>
#include <iostream>

namespace Garnet
{
    namespace
    {
        X::Value CallForward(X::Value model, std::initializer_list<X::Value> values)
        {
            if (!model.IsObject()) {
                return X::Value();
            }
            X::Value forward = model["forward"];
            if (!forward.IsObject()) {
                return X::Value();
            }
            X::ARGS args(static_cast<int>(values.size()));
            for (const auto& value : values) {
                args.push_back(value);
            }
            return forward.ObjCall(args);
        }

        X::Value AddGPU(X::Value lhsValue, X::Value rhsValue)
        {
            if (!lhsValue.IsTensor() || !rhsValue.IsTensor()) {
                return X::Value();
            }
            X::Tensor lhs(lhsValue);
            X::Tensor rhs(rhsValue);
            if (lhs->GetDataType() != X::TensorDataType::FLOAT32 ||
                rhs->GetDataType() != X::TensorDataType::FLOAT32 ||
                lhs->GetDimCount() != rhs->GetDimCount() ||
                lhs->GetCount() != rhs->GetCount()) {
                return X::Value();
            }
            for (int dim = 0; dim < lhs->GetDimCount(); ++dim) {
                if (lhs->GetDimSize(dim) != rhs->GetDimSize(dim)) {
                    return X::Value();
                }
            }
            if (TensorHelper::EnsureGPUMemory(lhs) != TensorOpStatus::Success ||
                TensorHelper::EnsureGPUMemory(rhs) != TensorOpStatus::Success) {
                return X::Value();
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
                return X::Value();
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
                return X::Value();
            }
            cudaStreamDestroy(stream);
            return X::Value(output);
        }
    }

    X::Value QwenTextRunner::RunLayers(
        X::Value hidden,
        X::Value cos,
        X::Value sin,
        X::Value kvHandles,
        int startPosition,
        int sequenceLength,
        bool prefill)
    {
        if (!hidden.IsTensor() || !cos.IsTensor() || !sin.IsTensor() ||
            !mLayerBundles.IsList() || !kvHandles.IsList()) {
            std::cout << "[QwenTextRunner] invalid tensor/list inputs." << std::endl;
            return X::Value();
        }
        X::List layers(mLayerBundles);
        X::List handles(kvHandles);
        if (layers->Size() <= 0 || layers->Size() != handles->Size()) {
            std::cout << "[QwenTextRunner] layer and KV handle counts differ." << std::endl;
            return X::Value();
        }
        X::Tensor cosTensor(cos);
        int tokenCount = static_cast<int>(cosTensor->GetDimSize(0));
        if (tokenCount <= 0 || startPosition < 0 || sequenceLength <= 0) {
            return X::Value();
        }

        for (long long layerIndex = 0; layerIndex < layers->Size(); ++layerIndex) {
            X::Value bundle = layers->Get(layerIndex);
            long long kvHandle = handles->Get(layerIndex).ToLongLong();
            X::Value normed = CallForward(bundle["input_norm"], { hidden });
            X::Value qkv = CallForward(bundle["qkv"], { normed });
            X::Value ropeResult = CallForward(bundle["rope"], {
                qkv, cos, sin, X::Value(kvHandle), X::Value(tokenCount), X::Value(startPosition)
            });
            X::Value rope = ropeResult["output_tensor"];
            if (!rope.IsTensor()) {
                std::cout << "[QwenTextRunner] RoPE/KV failed at layer " << layerIndex << std::endl;
                return X::Value();
            }

            X::Value attention;
            if (prefill) {
                attention = CallForward(bundle["attention"], { rope });
            }
            else {
                X::Value attentionResult = CallForward(bundle["attention"], {
                    rope, X::Value(kvHandle), X::Value(sequenceLength), X::Value(2048)
                });
                attention = attentionResult["output_tensor"];
            }
            X::Value projected = CallForward(bundle["o_proj"], { attention });
            X::Value residual = AddGPU(hidden, projected);
            X::Value postNorm = CallForward(bundle["post_norm"], { residual });
            X::Value mlp = CallForward(bundle["mlp"], { postNorm });
            hidden = AddGPU(residual, mlp);
            if (!hidden.IsTensor()) {
                std::cout << "[QwenTextRunner] layer output failed at layer " << layerIndex << std::endl;
                return X::Value();
            }
        }
        return hidden;
    }

    void QwenTextRunner::Configure(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (params.size() < 1 || !params[0].IsList()) {
            retValue = X::Value(false);
            return;
        }
        mLayerBundles = params[0];
        retValue = X::Value(true);
    }

    void QwenTextRunner::Prefill(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (params.size() < 6) {
            std::cout << "[QwenTextRunner] prefill(hidden, cos, sin, kv_handles, start, sequence_length) expected." << std::endl;
            retValue = X::Value();
            return;
        }
        retValue = RunLayers(
            params[0], params[1], params[2], params[3],
            static_cast<int>(params[4].ToLongLong()),
            static_cast<int>(params[5].ToLongLong()),
            true);
    }

    void QwenTextRunner::Decode(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (params.size() < 6) {
            std::cout << "[QwenTextRunner] decode(hidden, cos, sin, kv_handles, start, sequence_length) expected." << std::endl;
            retValue = X::Value();
            return;
        }
        retValue = RunLayers(
            params[0], params[1], params[2], params[3],
            static_cast<int>(params[4].ToLongLong()),
            static_cast<int>(params[5].ToLongLong()),
            false);
    }

    void QwenTextRunner::Stats(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        X::Dict stats;
        stats->Set("configured", X::Value(mLayerBundles.IsList()));
        stats->Set("layer_count", X::Value(mLayerBundles.IsList() ? static_cast<int>(X::List(mLayerBundles)->Size()) : 0));
        stats->Set("device_tensor_chain", X::Value(true));
        retValue = stats;
    }
}
