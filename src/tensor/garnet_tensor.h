#pragma once

#include "compiled_graph_capture.h"
#include "lowering_context.h"
#include "xlang.h"
#include "xpackage.h"

#include <string>

namespace Garnet
{
    // Compatibility descriptor for XLang tensors backed by device memory.
    // Ownership remains on XLang Tensor; this object only describes the binding.
    class TensorDescriptor
    {
        friend class TensorHelper;

        std::string mDeviceName;
        void* gpuMemory = nullptr;

    public:
        BEGIN_PACKAGE(TensorDescriptor)
            APISET().AddPropWithType<std::string>(
                "DeviceName", &TensorDescriptor::mDeviceName);
        END_PACKAGE
    };

    class Fusionist
    {
        X::Value mFunc;
        X::Value mTensorProvider;
        FusionAnnotation mAnnotation;

    public:
        Fusionist() = default;
        explicit Fusionist(X::Value& func) : mFunc(func) {}

        BEGIN_PACKAGE(Fusionist)
            APISET().SetCallHandler(&Fusionist::Call);
        END_PACKAGE

        void SetFunction(X::Value function) { mFunc = std::move(function); }
        void SetTensorProvider(X::Value provider)
        {
            mTensorProvider = std::move(provider);
        }
        void SetAnnotation(FusionAnnotation annotation)
        {
            mAnnotation = std::move(annotation);
        }

        bool Call(
            X::XRuntime* runtime,
            X::ARGS& params,
            X::KWARGS& keywordParams,
            X::Value& output);
    };

    // XLang TensorGraph capture adapter. It performs no eager execution and
    // owns no backend. Concrete backends are supplied through ILoweringContext
    // only while a captured graph is compiled.
    class GarnetTensor
    {
        std::string ProcessCondition(X::Value& astNode);

        X::Value IntrinsicBinary(
            const char* opName,
            X::Value& graph,
            X::ARGS& params,
            X::KWARGS& keywordParams,
            X::Value input1,
            X::Value input2,
            X::Value& output);

    public:
        BEGIN_PACKAGE(GarnetTensor)
            APISET().AddClass<0, Fusionist>("fusionist");
            APISET().AddVarFuncEx("fusion", &GarnetTensor::Fusion);

            APISET().AddTensorStructuralOps("header", &GarnetTensor::Header);
            APISET().AddTensorStructuralOps("trailer", &GarnetTensor::Trailer);
            APISET().AddTensorStructuralOps("branchBegin", &GarnetTensor::BranchBegin);
            APISET().AddTensorStructuralOps("branchEnd", &GarnetTensor::BranchEnd);

            // XLang uses these intrinsic hooks to represent expression syntax.
            // They only forward graph nodes; there is no eager CUDA path.
            APISET().AddTensorBinaryOp("add", &GarnetTensor::Add);
            APISET().AddTensorBinaryOp("minus", &GarnetTensor::Minus);
            APISET().AddTensorBinaryOp("mul", &GarnetTensor::Multiply);
            APISET().AddTensorBinaryOp("matmul", &GarnetTensor::Matmul);

            APISET().AddTensorBinaryOp("binary_op", &GarnetTensor::BinaryOp);
            APISET().AddTensorUnaryOp("unary_op", &GarnetTensor::UnaryOp);
        END_PACKAGE

        void Fusion(
            X::XRuntime* runtime,
            X::XObj* self,
            X::XObj* context,
            X::ARGS& params,
            X::KWARGS& keywordParams,
            X::Value& decoratedFunction,
            X::Value& output);

        X::Value Header(X::Value& graph, X::ARGS& params);
        X::Value Trailer(X::Value& graph, X::ARGS& params);
        X::Value BranchBegin(X::Value& graph, X::ARGS& params);
        X::Value BranchEnd(X::Value& graph, X::ARGS& params);

        X::Value Add(
            X::Value& graph, X::ARGS& params, X::KWARGS& keywordParams,
            X::Value input1, X::Value input2, X::Value& output);
        X::Value Minus(
            X::Value& graph, X::ARGS& params, X::KWARGS& keywordParams,
            X::Value input1, X::Value input2, X::Value& output);
        X::Value Multiply(
            X::Value& graph, X::ARGS& params, X::KWARGS& keywordParams,
            X::Value input1, X::Value input2, X::Value& output);
        X::Value Matmul(
            X::Value& graph, X::ARGS& params, X::KWARGS& keywordParams,
            X::Value input1, X::Value input2, X::Value& output);

        X::Value BinaryOp(
            X::Value& graph, X::ARGS& params, X::KWARGS& keywordParams,
            X::Value input1, X::Value input2, X::Value& output);
        X::Value UnaryOp(
            X::Value& graph, X::ARGS& params, X::KWARGS& keywordParams,
            X::Value input, X::Value& output);
    };
}
