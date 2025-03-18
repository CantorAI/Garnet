#include "garnet_tensor.h"

namespace Garnet
{
	bool Fusionist::Call(X::XRuntime* rt, X::ARGS& params, 
		X::KWARGS& kwParams, X::Value& retValue)
	{
		//all parameters change to tensor
		for (auto& v : params)
		{
			if (!v.IsTensor())
			{
				X::Tensor tensor;
				tensor->Create(v);
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
		code += "    {\n";  // Open a new scope for the branch
		return X::Value(code);
	}

	// Implementation of BranchEnd function
	X::Value GarnetTensor::BranchEnd(X::Value& graph, X::ARGS& params)
	{
		std::string code = "    // End branch\n";
		code += "    }\n";  // Close the scope for the branch
		return X::Value(code);
	}
}