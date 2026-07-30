#include "garnet_tensor.h"

#include <memory>
#include <set>

namespace Garnet
{
    namespace
    {
        void CopyTensorMetadata(const X::Value& source, X::Value& output)
        {
            if (!source.IsTensor() || !output.IsTensor()) return;
            X::Tensor sourceTensor(source);
            X::Tensor outputTensor(output);
            X::Port::vector<int> dimensions(sourceTensor->GetDimCount());
            for (int index = 0; index < sourceTensor->GetDimCount(); ++index) {
                dimensions.push_back(sourceTensor->GetDimSize(index));
            }
            outputTensor->SetDataType(sourceTensor->GetDataType());
            outputTensor->SetShape(dimensions);
        }
    }

    bool Fusionist::Call(
        X::XRuntime*,
        X::ARGS& params,
        X::KWARGS& keywordParams,
        X::Value& output)
    {
        if (!mFunc.IsValid()) return false;
        if (!IsCompiledGraphCaptureActive()) {
            output = mFunc.ObjCall(params, keywordParams);
            return output.IsValid();
        }

        ScopedCompiledFusionRegion region(mAnnotation);
        if (!region.IsValid()) return false;
        region.CaptureInputs(params);

        if (IsCompiledFusionCaptureRootActive()) {
            output = mFunc.ObjCall(params, keywordParams);
            region.CaptureResult(output);
            return output.IsValid();
        }

        ScopedCompiledFusionCaptureRoot root;
        X::Value expression = mFunc.ObjCall(params, keywordParams);
        region.CaptureResult(expression);

        X::ARGS graphOutputs;
        if (expression.IsList()) {
            X::List values(expression);
            graphOutputs.resize(values->Size());
            for (auto& value : *values) graphOutputs.push_back(value);
        }
        else if (expression.IsDict()) {
            X::Dict values(expression);
            graphOutputs.resize(values->Size());
            for (auto& value : *values) graphOutputs.push_back(value.second());
        }
        else {
            graphOutputs.resize(1);
            graphOutputs.push_back(expression);
        }
        graphOutputs.Close();

        X::KWARGS graphOptions;
        auto* graph = X::g_pXHost->CreateTensorGraph();
        graph->Create(mTensorProvider.GetObj(), graphOutputs, graphOptions);
        output = X::Value(graph);
        return true;
    }

    void GarnetTensor::Fusion(
        X::XRuntime*,
        X::XObj* context,
        X::XObj*,
        X::ARGS& params,
        X::KWARGS& keywordParams,
        X::Value& decoratedFunction,
        X::Value& output)
    {
        if (!decoratedFunction.IsObject() ||
            decoratedFunction.GetObj()->GetType() != X::ObjType::Function) {
            return;
        }

        X::Func function(decoratedFunction);
        function->ChangeStatmentsIntoTranslateMode(true, false);
        const std::string functionName = function->GetName().ToString();

        FusionAnnotation annotation;
        annotation.name = functionName;
        annotation.functionName = functionName;
        const std::set<std::string> supported{
            "name", "role", "boundary", "atomic", "cuda_graph"};
        std::set<std::string> seen;

        auto apply = [&](const std::string& key, X::Value value) {
            if (supported.find(key) == supported.end()) {
                annotation.validationError =
                    "T.fusion for '" + functionName +
                    "' has unsupported parameter '" + key + "'";
                return;
            }
            if (!seen.insert(key).second) {
                annotation.validationError =
                    "T.fusion for '" + functionName +
                    "' repeats parameter '" + key + "'";
                return;
            }
            if ((key == "name" || key == "role" || key == "boundary") &&
                !value.IsString()) {
                annotation.validationError =
                    "T.fusion parameter '" + key + "' for '" + functionName +
                    "' must be a string";
                return;
            }
            if ((key == "atomic" || key == "cuda_graph") && !value.IsBool()) {
                annotation.validationError =
                    "T.fusion parameter '" + key + "' for '" + functionName +
                    "' must be a boolean";
                return;
            }
            if (key == "name") annotation.name = value.ToString();
            else if (key == "role") annotation.role = value.ToString();
            else if (key == "boundary") annotation.boundary = value.ToString();
            else if (key == "atomic") annotation.atomic = value.ToInt() != 0;
            else if (key == "cuda_graph") annotation.cudaGraph = value.ToInt() != 0;
        };

        for (size_t index = 0;
             index < params.size() && annotation.validationError.empty();
             ++index) {
            auto* expression = params[index].IsObject()
                ? dynamic_cast<X::XExpr*>(params[index].GetObj())
                : nullptr;
            X::Value keyValue = expression ? expression->ToKV() : X::Value();
            if (!keyValue.IsDict()) {
                annotation.validationError =
                    "T.fusion for '" + functionName +
                    "' accepts key=value arguments only";
                break;
            }
            X::Dict dictionary(keyValue);
            for (auto& item : *dictionary) {
                apply(item.first().ToString(), item.second());
            }
        }
        for (auto& item : keywordParams) {
            if (!annotation.validationError.empty()) break;
            apply(std::string(item.key), item.val);
        }
        if (annotation.validationError.empty() && annotation.name.empty()) {
            annotation.validationError =
                "T.fusion for '" + functionName + "' requires a non-empty name";
        }
        if (annotation.validationError.empty() &&
            annotation.boundary != "none" &&
            annotation.boundary != "preferred" &&
            annotation.boundary != "required") {
            annotation.validationError =
                "T.fusion for '" + functionName +
                "' has invalid boundary '" + annotation.boundary + "'";
        }

        X::XPackageValue<Fusionist> fusion;
        fusion->SetFunction(decoratedFunction);
        fusion->SetTensorProvider(X::Value(context));
        fusion->SetAnnotation(std::move(annotation));
        output = fusion;
    }

    X::Value GarnetTensor::Header(X::Value&, X::ARGS&)
    {
        return X::Value("");
    }

    X::Value GarnetTensor::Trailer(X::Value&, X::ARGS&)
    {
        return X::Value("");
    }

    X::Value GarnetTensor::IntrinsicBinary(
        const char* opName,
        X::Value& graph,
        X::ARGS& params,
        X::KWARGS& keywordParams,
        X::Value input1,
        X::Value input2,
        X::Value& output)
    {
        if (g_loweringContext) {
            return g_loweringContext->HandleBinaryOp(
                opName, graph, params, keywordParams, input1, input2, output);
        }
        CopyTensorMetadata(input1.IsTensor() ? input1 : input2, output);
        return X::Value(opName);
    }

    X::Value GarnetTensor::Add(
        X::Value& graph, X::ARGS& params, X::KWARGS& keywordParams,
        X::Value input1, X::Value input2, X::Value& output)
    {
        return IntrinsicBinary(
            "add", graph, params, keywordParams, input1, input2, output);
    }

    X::Value GarnetTensor::Minus(
        X::Value& graph, X::ARGS& params, X::KWARGS& keywordParams,
        X::Value input1, X::Value input2, X::Value& output)
    {
        return IntrinsicBinary(
            "minus", graph, params, keywordParams, input1, input2, output);
    }

    X::Value GarnetTensor::Multiply(
        X::Value& graph, X::ARGS& params, X::KWARGS& keywordParams,
        X::Value input1, X::Value input2, X::Value& output)
    {
        return IntrinsicBinary(
            "mul", graph, params, keywordParams, input1, input2, output);
    }

    X::Value GarnetTensor::Matmul(
        X::Value& graph, X::ARGS& params, X::KWARGS& keywordParams,
        X::Value input1, X::Value input2, X::Value& output)
    {
        return IntrinsicBinary(
            "matmul", graph, params, keywordParams, input1, input2, output);
    }

    X::Value GarnetTensor::BinaryOp(
        X::Value& graph,
        X::ARGS& params,
        X::KWARGS& keywordParams,
        X::Value input1,
        X::Value input2,
        X::Value& output)
    {
        const std::string opName =
            params.size() == 0 ? std::string() : params[0].ToString();
        if (!g_loweringContext) {
            CopyTensorMetadata(input1.IsTensor() ? input1 : input2, output);
            return X::Value(opName);
        }
        return g_loweringContext->HandleBinaryOp(
            opName, graph, params, keywordParams, input1, input2, output);
    }

    X::Value GarnetTensor::UnaryOp(
        X::Value& graph,
        X::ARGS& params,
        X::KWARGS& keywordParams,
        X::Value input,
        X::Value& output)
    {
        const std::string opName =
            params.size() == 0 ? std::string() : params[0].ToString();
        if (!g_loweringContext) {
            CopyTensorMetadata(input, output);
            return X::Value(opName);
        }
        return g_loweringContext->HandleUnaryOp(
            opName, graph, params, keywordParams, input, output);
    }

    X::Value GarnetTensor::BranchBegin(X::Value&, X::ARGS& params)
    {
        if (!g_loweringContext) return X::Value("");
        return g_loweringContext->HandleBranchBegin(
            ProcessCondition(params[0]),
            params[1].ToInt(),
            static_cast<unsigned long long>(params[2].ToLongLong()),
            params[3].ToInt());
    }

    X::Value GarnetTensor::BranchEnd(X::Value&, X::ARGS&)
    {
        return g_loweringContext
            ? g_loweringContext->HandleBranchEnd()
            : X::Value("");
    }

    std::string GarnetTensor::ProcessCondition(X::Value& node)
    {
        const std::string type = node["type"]().ToString();
        if (type == "BinaryOp") {
            X::Value children = node["children"]();
            std::string operation = node["OperatorType"]().ToString();
            if (operation == "and") operation = "&&";
            else if (operation == "or") operation = "||";
            return ProcessCondition(children[0]) + " " + operation + " " +
                ProcessCondition(children[1]);
        }
        if (type == "UnaryOp") {
            X::Value children = node["children"]();
            std::string operation = node["OperatorType"]().ToString();
            if (operation == "not") operation = "!";
            return operation + ProcessCondition(children[0]);
        }
        if (type == "Var" || type == "Str" ||
            type == "Number" || type == "Double") {
            return node["name"]().ToString();
        }
        return node.ToString();
    }
}
