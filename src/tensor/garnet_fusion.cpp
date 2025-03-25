#include "garnet_tensor.h"
#include "md5.h"
#include "garnet.h"

namespace Garnet
{
	bool Fusionist::Call(X::XRuntime* rt, X::ARGS& params, 
		X::KWARGS& kwParams, X::Value& retValue)
	{
        if (!mNeedGenAndCompile)
        {
            //TODO: call
            return true;
        }
        //Need to gen code and compile

		X::Func func(mFunc);
		X::Value valParamNames = func->GetParameterNameList();
		X::List nameList(valParamNames);
		//all parameters change to tensor
		for (int i=0;i<(int)params.size();i++)
		{
			auto& v = params[i];
			if (!v.IsTensor())
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
		for (auto& it : kwParams)
		{
			if (!it.val.IsTensor())
			{
				X::Tensor tensor;
				tensor->Create(it.val);
				it.val = tensor;
			}
		}
        X::Value t = mFunc.ObjCall(params, kwParams);
        X::ARGS params_t;
        if (t.IsList())
        {
            X::List list(t);
            params_t.resize(list->Size());
            for (auto& it : *list)
            {
                params_t.push_back(it);
            }
        }
        else if (t.IsDict())
        {
            X::Dict dict(t);
            params_t.resize(dict->Size());
            for (auto& it : *dict)
            {
                params_t.push_back(it.second());
            }
        }
        else
        {
            params_t.resize(1);
            params_t.push_back(t);
        }
        X::KWARGS kwParams_t;
        auto* pTensorGraph = X::g_pXHost->CreateTensorGraph();
        pTensorGraph->Create(mVarTensor.GetObj(), params_t, kwParams_t);
		mTensorGraph = X::Value(pTensorGraph);
		X::KWARGS kwArgs;
        kwArgs.Add("Func", mFunc);
		pTensorGraph->Run(params,kwArgs);
        X::Value varCode = pTensorGraph->GetCodeGenerated();
        std::string code = varCode.ToString();
		X::XPackageValue<GarnetTensor> varTensor(mVarTensor);
        GarnetTensor& gt = *varTensor;
        gt.GetCompiler().add_option("-D__CUDA_ARCH__=860");
        gt.GetCompiler().add_option("-D__CUDACC_RTC__");

        //gt.GetCompiler().add_option("-arch=sm_89");
        //gt.GetCompiler().add_option("--gpu-architecture=compute_80");
        gt.GetCompiler().add_option("--gpu-architecture=compute_89");

		bool bOK = gt.GetCompiler().compile_or_load(code, mFuncName,mFuncCodeHash);
		return true;
	}
    GarnetTensor::GarnetTensor()
    {
        std::string baseFolder = GarnetAPI::I().GetBaseFolder();
		mCompiler.Init(baseFolder);
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
		std::string code = "    }\n";  // Close the scope for the branch
		return X::Value(code);
	}
}