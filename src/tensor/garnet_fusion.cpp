#include "garnet_tensor.h"

namespace Garnet
{
	bool Fusionist::Call(X::XRuntime* rt, X::ARGS& params, 
		X::KWARGS& kwParams, X::Value& retValue)
	{
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

		retValue = mFunc.ObjCall(params, kwParams);
		return true;
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
		X::XPackageValue<Fusionist> varFusion;
		Fusionist& f = *varFusion;
		f.SetParent(this);
		f.SetFunc(trailer);
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