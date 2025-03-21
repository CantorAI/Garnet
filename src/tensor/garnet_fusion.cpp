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
        std::string code;

        // Extract parameters
        X::Value condExpr = params[0];        // Condition expression (AST node)
        std::string branchType = params[1].ToString();
        int flowId = params[2].ToInt();
        int branchId = params[3].ToInt();     // 0=if, 1=elif1, -1=else

        // Process branch type based on branchId
        if (branchId == 0) {
            // If branch
            code += "    if (" + ProcessCondition(condExpr) + ") {\n";
        }
        else if (branchId > 0) {
            // Else if branch
            code += "    else if (" + ProcessCondition(condExpr) + ") {\n";
        }
        else if (branchId == -1) {
            // Else branch (no condition needed)
            code += "    else {\n";
        }

        return X::Value(code);
    }

    // Helper function to process condition AST nodes
    std::string GarnetTensor::ProcessCondition(X::Value& astNode)
    {
        std::string nodeType = astNode["type"]().ToString();

        if (nodeType == "BinaryOp") {
            // Handle binary operations (like x > 9)
            X::Value children = astNode["children"]();
            std::string opType = astNode["OperatorType"]().ToString();

            // Process left and right operands
            X::Value leftNode = children[0];
            X::Value rightNode = children[1];

            return ProcessCondition(leftNode) + " " + opType + " " + ProcessCondition(rightNode);
        }
        else if (nodeType == "UnaryOp") {
            // Handle unary operations (like !x)
            X::Value children = astNode["children"]();
            std::string opType = astNode["OperatorType"]().ToString();

            return opType + ProcessCondition(children[0]);
        }
        else if (nodeType == "Var") {
            // Handle variable references
            return astNode["name"]().ToString();
        }
        else if (nodeType == "Str") {
            // Handle string literals - add quotes
            return "\"" + astNode["name"]().ToString() + "\"";
        }
        else if (nodeType == "Number") {
            // Handle integer literals
            return astNode["name"]().ToString();
        }
        else if (nodeType == "Double") {
            // Handle floating point literals
            return astNode["name"]().ToString();
        }
        else {
            // Fallback for other node types
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