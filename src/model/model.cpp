#include "model.h"
namespace Garnet
{
    X::Value Model::Access(X::Port::vector<X::Value>& IdxAry)
    {
		return mModel.GetObjectValue(IdxAry);
    }
    void Model::Tokenizer(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
		if (!mTokenizer.IsValid())
		{
            X::Runtime rt0(rt);
            //from transformers import AutoTokenizer
            X::PyObject pyAutoTokenizer(rt0, "AutoTokenizer", "transformers", "");
            mTokenizer = pyAutoTokenizer["from_pretrained"](mModelPath);
        }
        X::Value pyResults = mTokenizer.ObjCall(params, kwParams);
        X::Value py_input_ids = pyResults["input_ids"]["tolist"]();
        X::PyObject pyObjIds(py_input_ids);
        X::Value input_ids = pyObjIds->ToXlang();
        X::Value py_attention_mask = pyResults["attention_mask"]["tolist"]();
        X::PyObject pyObjAttentionMask(py_attention_mask);
		X::Value attention_mask = pyObjAttentionMask->ToXlang();
        X::Tensor tensorIds;
		tensorIds->SetDataType(X::TensorDataType::INT);
        tensorIds->Create(input_ids);
        X::Tensor tensorAttentionMask;
        tensorAttentionMask->SetDataType(X::TensorDataType::INT);
        tensorAttentionMask->Create(attention_mask);
        X::Dict dictInputs;
        dictInputs->Set("input_ids", tensorIds);
        dictInputs->Set("attention_mask", tensorAttentionMask);
        retValue = dictInputs;
    }
}