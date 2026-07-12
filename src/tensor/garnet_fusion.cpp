#include "garnet_tensor.h"
#include "md5.h"
#include "garnet.h"
#include "compiled_graph_capture.h"
#include "trt_builder.h"

#include <memory>
#include <set>


namespace Garnet
{
 
    bool Fusionist::Call(X::XRuntime* rt, X::ARGS& params,
        X::KWARGS& kwParams, X::Value& retValue)
    {
        const bool captureOnly = IsCompiledGraphCaptureActive();
        std::unique_ptr<ScopedCompiledFusionRegion> captureRegion;
        if (captureOnly) {
            captureRegion = std::make_unique<ScopedCompiledFusionRegion>(mAnnotation);
            if (!captureRegion->IsValid()) return false;
            captureRegion->CaptureInputs(params);
        }
        if (captureOnly && IsCompiledFusionCaptureRootActive())
        {
            // A compiled model has one fusion boundary at its root. This path
            // remains for compatibility, but model helper functions should be
            // plain functions so xlang naturally inlines them into that graph.
            retValue = mFunc.ObjCall(params, kwParams);
            captureRegion->CaptureResult(retValue);
            return retValue.IsValid();
        }
        std::unique_ptr<ScopedCompiledFusionCaptureRoot> captureRoot;
        if (captureOnly)
        {
            captureRoot = std::make_unique<ScopedCompiledFusionCaptureRoot>();
        }
        X::Func func(mFunc);
        X::Value valParamNames = func->GetParameterNameList();
        X::List nameList(valParamNames);

        X::XPackageValue<GarnetTensor> varTensor(mVarTensor);
        GarnetTensor& gt = *varTensor;
        auto& compiler = gt.GetCompiler();

        auto gen_compile_proc = [&]() 
        {
            // Convert positional parameters.
            for (int i = 0; i < (int)params.size(); i++)
            {
                auto& v = params[i];
                if (!v.IsTensor() && !v.IsDict() && !v.IsNone())
                {
                    X::Tensor tensor;
                    tensor->Create(v);
                    if (i < nameList.size())
                    {
                        auto& name = nameList[i];
                        if (name.IsString())
                        {
                            tensor->SetName(name);
                        }
                    }
                    v = tensor;
                }
            }
            // Convert keyword parameters.
            for (auto& it : kwParams)
            {
                if (!it.val.IsTensor() && !it.val.IsDict() && !it.val.IsNone())
                {
                    X::Tensor tensor;
                    tensor->Create(it.val);
                    it.val = tensor;
                }
            }

            // Call the original function.
            X::Value t = mFunc.ObjCall(params, kwParams);
            if (captureRegion) captureRegion->CaptureResult(t);
            X::ARGS params_t;
            if (t.IsList())
            {
                X::List list(t);
                params_t.resize(list->Size());
                for (auto& it : *list)
                {
                    params_t.push_back(it);
                }
                params_t.Close();
            }
            else if (t.IsDict())
            {
                X::Dict dict(t);
                params_t.resize(dict->Size());
                for (auto& it : *dict)
                {
                    params_t.push_back(it.second());
                }
                params_t.Close();
            }
            else
            {
                params_t.resize(1);
                params_t.push_back(t);
                params_t.Close();
            }

            // Create and run the tensor graph.
            X::KWARGS kwParams_t;
            auto* pTensorGraph = X::g_pXHost->CreateTensorGraph();
            pTensorGraph->Create(mVarTensor.GetObj(), params_t, kwParams_t);
            mTensorGraph = X::Value(pTensorGraph);
            if (captureOnly)
            {
                retValue = mTensorGraph;
                return true;
            }
            X::KWARGS kwArgs;
            kwArgs.Add("Func", mFunc);
            pTensorGraph->Run(params, kwArgs);

            // Retrieve the generated code.
            X::Value varCode = pTensorGraph->GetCodeGenerated();
            std::string code = varCode.ToString();

            compiler.add_option("-D__CUDA_ARCH__=860");
            compiler.add_option("-D__CUDACC_RTC__");
            compiler.add_option("--gpu-architecture=compute_89");

            bool bOK = compiler.compile_or_load(code, mFuncName, mFuncCodeHash);
            return bOK;
        };//end gen_compile_proc

        if (mNeedGenAndCompile)
        {
            bool bOK = gen_compile_proc();
            if (!bOK)
            {
				return false; // Indicate failure
            }
            mNeedGenAndCompile = false;
        }
        if (captureOnly)
        {
            return true;
        }
        if (!mHasKernel)
        {
            m_kernel = compiler.get_kernel(mFuncName, mFuncName);
            mHasKernel = true;
        }

        // --- Build kernel arguments using inline lambdas ---
        // Define a union to hold any scalar value.
        union ScalarValue {
            int   i;
            float f;
            double d;
        };

        // Single storage vector for all scalar values (both arguments and dimensions).
        std::vector<ScalarValue> scalarStorage;
        // The kernel argument list will hold either tensor data pointers or addresses in scalarStorage.
        std::vector<void*> kernelArgs;

        // Lambda to store a scalar value into scalarStorage and push its address into kernelArgs.
        auto pushScalarValue = [&](auto value)
            {
                using T = decltype(value);
                ScalarValue scalar;
                if constexpr (std::is_same_v<T, int>) {
                    scalar.i = value;
                    scalarStorage.push_back(scalar);
                    kernelArgs.push_back(reinterpret_cast<void*>(&scalarStorage.back().i));
                }
                else if constexpr (std::is_same_v<T, float>) {
                    scalar.f = value;
                    scalarStorage.push_back(scalar);
                    kernelArgs.push_back(reinterpret_cast<void*>(&scalarStorage.back().f));
                }
                else if constexpr (std::is_same_v<T, double>) {
                    scalar.d = value;
                    scalarStorage.push_back(scalar);
                    kernelArgs.push_back(reinterpret_cast<void*>(&scalarStorage.back().d));
                }
            };

        // Helper lambda for processing a non-tensor scalar.
        auto processScalarNonTensor = [&](X::Value& v)
            {
                if (v.IsLong())
                    pushScalarValue((int)v);
                else if (v.IsDouble())
                    pushScalarValue(v.ToDouble());
            };

        // Lambda to build the complete kernel arguments list.
        auto buildKernelArgs = [&]()
            {
                for (size_t i = 0; i < nameList->Size(); i++)
                {
                    X::Value argValue = params[i];
                    if (argValue.IsTensor())
                    {
                        X::Tensor tensor(argValue);
                        if (tensor->GetCount() == 1)
                        {
                            X::Value firstVal;
							tensor->GetIndexValue(0, firstVal);
                            processScalarNonTensor(firstVal);
                        }
                        else
                        {
                            // For multi-element tensors, first push the data pointer.
                            void* dataPtr = tensor->GetData();
                            kernelArgs.push_back(dataPtr);
                            // Then push each shape dimension.
                            X::Value shapesVal = tensor->Shapes();
                            X::List shapeList(shapesVal);
                            for (int j = 0; j < (int)shapeList->Size(); j++)
                            {
                                int dim = (int)shapeList[j];
                                pushScalarValue(dim);
                            }
                        }
                    }
                    else
                    {
                        // Process non-tensor as a scalar.
                        processScalarNonTensor(argValue);
                    }
                }
            };

        // Build the kernel argument list.
        buildKernelArgs();

        // --- Launch the kernel ---
        dim3 block_dim(1);
        dim3 grid_dim(1);
        compiler.launch_kernel(m_kernel, grid_dim, block_dim, kernelArgs.data(), 0, nullptr);

        return true;
    }



    GarnetTensor::GarnetTensor()
    {
        // JIT Compiler initialization is deferred to lazy initialization
        // when Fusion is actually requested, preventing unconditional CUDA/NVRTC crashes
        // for users using the TensorRT backend.
    }
    void GarnetTensor::Fusion(X::XRuntime* rt, X::XObj* pThis,
		X::XObj* pContext, X::ARGS& params, X::KWARGS& kwParams, 
		X::Value& trailer, X::Value& retValue)
	{
		if (trailer.IsObject() 
			&& trailer.GetObj()->GetType() == X::ObjType::Function)
		{
			X::Func func(trailer);
			func->ChangeStatmentsIntoTranslateMode(true, false);
		}

        X::Func func(trailer);
		X::Value varFuncName = func->GetName();
		std::string funcName = varFuncName.ToString();
        X::Value funcCode = func->GetCode(true);
		std::string code = funcCode.ToString();
        MD5 md5 = MD5(code);
        std::string codeHash = md5.hexdigest();
		bool bHasSameAndNoChange = mCompiler.check_module_hash(funcName, codeHash);
		X::XPackageValue<Fusionist> varFusion;
		Fusionist& f = *varFusion;
        FusionAnnotation annotation;
        annotation.name = funcName;
        annotation.functionName = funcName;
        const std::set<std::string> supportedKeys{
            "name", "role", "boundary", "atomic", "cuda_graph"};
        std::set<std::string> seenKeys;
        auto applyAnnotationValue = [&](const std::string& key, X::Value value) {
            if (supportedKeys.find(key) == supportedKeys.end()) {
                annotation.validationError =
                    "T.fusion for '" + funcName + "' has unsupported parameter '" + key + "'";
                return;
            }
            if (!seenKeys.insert(key).second) {
                annotation.validationError =
                    "T.fusion for '" + funcName + "' repeats parameter '" + key + "'";
                return;
            }
            if ((key == "name" || key == "role" || key == "boundary") &&
                !value.IsString()) {
                annotation.validationError =
                    "T.fusion parameter '" + key + "' for '" + funcName +
                    "' must be a string";
                return;
            }
            if ((key == "atomic" || key == "cuda_graph") && !value.IsBool()) {
                annotation.validationError =
                    "T.fusion parameter '" + key + "' for '" + funcName +
                    "' must be a boolean";
                return;
            }
            if (key == "name") annotation.name = value.ToString();
            else if (key == "role") annotation.role = value.ToString();
            else if (key == "boundary") annotation.boundary = value.ToString();
            else if (key == "atomic") annotation.atomic = value.ToInt() != 0;
            else if (key == "cuda_graph") annotation.cudaGraph = value.ToInt() != 0;
        };
        // xlang decorators expose each `key=value` argument as an Expr. ToKV()
        // evaluates the constant right side and preserves the assignment name.
        for (size_t index = 0;
             index < params.size() && annotation.validationError.empty();
             ++index) {
            if (!params[index].IsObject()) {
                annotation.validationError =
                    "T.fusion for '" + funcName + "' accepts named arguments only";
                break;
            }
            auto* expression = dynamic_cast<X::XExpr*>(params[index].GetObj());
            X::Value keyValue = expression ? expression->ToKV() : X::Value();
            if (!keyValue.IsDict()) {
                annotation.validationError =
                    "T.fusion for '" + funcName + "' accepts key=value arguments only";
                break;
            }
            X::Dict dictionary(keyValue);
            for (auto& entry : *dictionary) {
                applyAnnotationValue(entry.first().ToString(), entry.second());
            }
        }
        for (auto& item : kwParams) {
            if (!annotation.validationError.empty()) break;
            applyAnnotationValue(std::string(item.key), item.val);
        }
        if (annotation.validationError.empty() && annotation.name.empty()) {
            annotation.validationError =
                "T.fusion for '" + funcName + "' requires a non-empty name";
        }
        if (annotation.validationError.empty() &&
            annotation.boundary != "none" &&
            annotation.boundary != "preferred" &&
            annotation.boundary != "required") {
            annotation.validationError =
                "T.fusion for '" + funcName +
                "' has invalid boundary '" + annotation.boundary + "'";
        }
        f.SetAnnotation(std::move(annotation));
        //if not existed or changed
        f.SetNeedGenAndCompile(!bHasSameAndNoChange);
        X::Value varGarnetTensor(pContext);
		f.SetParent(varGarnetTensor);
		f.SetFunc(trailer, funcName, codeHash);
		retValue = varFusion;
	}

	// Implementation of BranchBegin function
    X::Value GarnetTensor::BranchBegin(X::Value& graph, X::ARGS& params)
    {
        if (Garnet::g_trtContext) {
            const std::string condition = ProcessCondition(params[0]);
            return Garnet::g_trtContext->HandleBranchBegin(
                condition,
                params[1].ToInt(),
                static_cast<unsigned long long>(params[2].ToLongLong()),
                params[3].ToInt());
        }
        std::string code = "    // Begin branch\n";

        // Extract parameters
        X::Value condExpr = params[0];
        std::string branchType = params[1].ToString();
        int flowId = params[2].ToInt();
        int branchId = params[3].ToInt();

        // Process branch type based on branchId
        if (branchId == 0) {
            code += "    if (" + ProcessCondition(condExpr) + ") {\n";
        }
        else if (branchId > 0) {
            code += "    else if (" + ProcessCondition(condExpr) + ") {\n";
        }
        else if (branchId == -1) {
            code += "    else {\n";
        }

        return X::Value(code);
    }

    std::string GarnetTensor::ProcessCondition(X::Value& astNode)
    {
        std::string nodeType = astNode["type"]().ToString();

        if (nodeType == "BinaryOp") {
            X::Value children = astNode["children"]();
            std::string opType = astNode["OperatorType"]().ToString();

            // Translate XLang operators to CUDA syntax
            if (opType == "and") opType = "&&";
            else if (opType == "or") opType = "||";

            std::string leftExpr = ProcessCondition(children[0]);
            std::string rightExpr = ProcessCondition(children[1]);

            // Only add parentheses when needed for logical operators
            if (opType == "&&" || opType == "||") {
                // Check if left/right are binary ops that need parentheses
                if (children[0]["type"]().ToString() == "BinaryOp") {
                    leftExpr = "(" + leftExpr + ")";
                }
                if (children[1]["type"]().ToString() == "BinaryOp") {
                    rightExpr = "(" + rightExpr + ")";
                }
                return leftExpr + " " + opType + " " + rightExpr;
            }
            else {
                return leftExpr + " " + opType + " " + rightExpr;
            }
        }
        else if (nodeType == "UnaryOp") {
            X::Value children = astNode["children"]();
            std::string opType = astNode["OperatorType"]().ToString();

            if (opType == "not") opType = "!";

            return opType + ProcessCondition(children[0]);
        }
        else if (nodeType == "Var") {
            return astNode["name"]().ToString();
        }
        else if (nodeType == "Str") {
            return "\"" + astNode["name"]().ToString() + "\"";
        }
        else if (nodeType == "Number" || nodeType == "Double") {
            return astNode["name"]().ToString();
        }
        else {
            return astNode.ToString();
        }
    }
	// Implementation of BranchEnd function
	X::Value GarnetTensor::BranchEnd(X::Value& graph, X::ARGS& params)
	{
		if (Garnet::g_trtContext) {
			return Garnet::g_trtContext->HandleBranchEnd();
		}
		std::string code = "    }\n";  // Close the scope for the branch
		return X::Value(code);
	}
}
