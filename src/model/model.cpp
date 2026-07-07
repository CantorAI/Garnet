#include "model.h"
#include "../trt/trt_builder.h"
#include <iostream>
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

    void Model::BuildTRTEngine(X::Value forwardFunc, X::Value inputShapes)
    {
        TRTBuilder builder;
        // In Phase 1, we just invoke TRTBuilder
        // mTRTEngine = builder.BuildEngine(forwardFunc, inputShapes, mModel);
        // For now, we print a message.
        std::cout << "[Model] Building TRT Engine with shapes..." << std::endl;
        builder.BuildEngine(forwardFunc, inputShapes, mModel);
    }

    void Model::Forward(X::XRuntime* rt, X::XObj* pContext, X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        std::cout << "[Model] Executing forward pass..." << std::endl;
        if (!m_engine.IsValid()) {
            std::cout << "[Model] No compiled engine attached." << std::endl;
            retValue = X::Value();
            return;
        }
        if (params.size() < 1) {
            std::cout << "[Model] Forward requires an input tensor." << std::endl;
            retValue = X::Value();
            return;
        }
        if (!mModel.IsObject() || mModel.GetObj()->GetType() != X::ObjType::Dict) {
            std::cout << "[Model] Loaded weights are not a dictionary." << std::endl;
            retValue = X::Value();
            return;
        }
        X::Dict weights(mModel);
        X::Value weight = weights["W"];
        if (!weight.IsValid()) {
            std::cout << "[Model] Missing loaded weight: W." << std::endl;
            retValue = X::Value();
            return;
        }

        TRTBuilder builder;
        retValue = builder.RunMatmulEngine(m_engine.ToString(), params[0], weight);
    }
}
