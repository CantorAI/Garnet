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
        TRTBuilder builder;
        X::Value gate = weights["language_model.layers.0.mlp.gate_proj.weight"];
        X::Value up = weights["language_model.layers.0.mlp.up_proj.weight"];
        X::Value down = weights["language_model.layers.0.mlp.down_proj.weight"];
        if ((mSubgraph == "qwen3_text_mlp" || mSubgraph.empty()) && gate.IsValid() && up.IsValid() && down.IsValid()) {
            retValue = builder.RunTextMLPEngine(m_engine.ToString(), params[0], gate, up, down);
            return;
        }

        X::Value qProj = weights["language_model.layers.0.self_attn.q_proj.weight"];
        X::Value kProj = weights["language_model.layers.0.self_attn.k_proj.weight"];
        X::Value vProj = weights["language_model.layers.0.self_attn.v_proj.weight"];
        X::Value qNorm = weights["language_model.layers.0.self_attn.q_norm.weight"];
        X::Value kNorm = weights["language_model.layers.0.self_attn.k_norm.weight"];
        if (mSubgraph == "text_rope_apply" && params.size() >= 3) {
            retValue = builder.RunTextRoPEEngine(m_engine.ToString(), params[0], params[1], params[2]);
            return;
        }
        if (mSubgraph == "text_attention_core") {
            retValue = builder.RunTextAttentionEngine(m_engine.ToString(), params[0]);
            return;
        }
        if (mSubgraph == "vision_attention_core") {
            retValue = builder.RunVisionAttentionEngine(m_engine.ToString(), params[0]);
            return;
        }
        if ((mSubgraph == "text_qkv_head_norm" || mSubgraph.empty()) && qProj.IsValid() && kProj.IsValid() && vProj.IsValid() && qNorm.IsValid() && kNorm.IsValid()) {
            retValue = builder.RunTextQKVHeadNormEngine(m_engine.ToString(), params[0], qProj, kProj, vProj, qNorm, kNorm);
            return;
        }
        if ((mSubgraph == "text_qkv_proj" || mSubgraph.empty()) && qProj.IsValid() && kProj.IsValid() && vProj.IsValid()) {
            retValue = builder.RunTextQKVEngine(m_engine.ToString(), params[0], qProj, kProj, vProj);
            return;
        }

        X::Value oProj = weights["language_model.layers.0.self_attn.o_proj.weight"];
        if ((mSubgraph == "text_o_proj" || mSubgraph.empty()) && oProj.IsValid()) {
            retValue = builder.RunLinearTransposeEngine(m_engine.ToString(), params[0], oProj);
            return;
        }
        X::Value embedTokens = weights["language_model.embed_tokens.weight"];
        if ((mSubgraph == "text_lm_head" || mSubgraph.empty()) && embedTokens.IsValid()) {
            retValue = builder.RunLinearTransposeEngine(m_engine.ToString(), params[0], embedTokens);
            return;
        }

        X::Value fc1 = weights["visual.blocks.0.mlp.linear_fc1.weight"];
        X::Value fc1Bias = weights["visual.blocks.0.mlp.linear_fc1.bias"];
        X::Value fc2 = weights["visual.blocks.0.mlp.linear_fc2.weight"];
        X::Value fc2Bias = weights["visual.blocks.0.mlp.linear_fc2.bias"];
        X::Value patchWeight = weights["visual.patch_embed.proj.weight"];
        X::Value patchBias = weights["visual.patch_embed.proj.bias"];
        if ((mSubgraph == "vision_patch_embed" || mSubgraph.empty()) && patchWeight.IsValid() && patchBias.IsValid()) {
            retValue = builder.RunLinearBiasTransposeEngine(m_engine.ToString(), params[0], patchWeight, patchBias);
            return;
        }
        X::Value genericWeight = weights["W"];
        X::Value genericBias = weights["B"];
        if ((mSubgraph == "linear_bias" || mSubgraph.empty()) && genericWeight.IsValid() && genericBias.IsValid()) {
            retValue = builder.RunLinearBiasTransposeEngine(m_engine.ToString(), params[0], genericWeight, genericBias);
            return;
        }
        if ((mSubgraph == "vision_mlp" || mSubgraph.empty()) && fc1.IsValid() && fc1Bias.IsValid() && fc2.IsValid() && fc2Bias.IsValid()) {
            retValue = builder.RunVisionMLPEngine(m_engine.ToString(), params[0], fc1, fc1Bias, fc2, fc2Bias);
            return;
        }

        X::Value rmsWeight = mRmsNormWeight.IsValid()
            ? mRmsNormWeight
            : weights["language_model.layers.0.input_layernorm.weight"];
        if ((mSubgraph == "rms_norm" || mSubgraph.empty()) && rmsWeight.IsValid()) {
            retValue = builder.RunRMSNormEngine(m_engine.ToString(), params[0], rmsWeight);
            return;
        }
        X::Value postAttentionRmsWeight = mRmsNormWeight.IsValid()
            ? mRmsNormWeight
            : weights["language_model.layers.0.post_attention_layernorm.weight"];
        if ((mSubgraph == "text_post_attention_rms_norm" || mSubgraph.empty()) && postAttentionRmsWeight.IsValid()) {
            retValue = builder.RunRMSNormEngine(m_engine.ToString(), params[0], postAttentionRmsWeight);
            return;
        }

        X::Value lnWeight = weights["visual.blocks.0.norm1.weight"];
        X::Value lnBias = weights["visual.blocks.0.norm1.bias"];
        if ((mSubgraph == "layer_norm" || mSubgraph.empty()) && lnWeight.IsValid() && lnBias.IsValid()) {
            retValue = builder.RunLayerNormEngine(m_engine.ToString(), params[0], lnWeight, lnBias);
            return;
        }

        X::Value weight = weights["W"];
        if ((mSubgraph == "matmul" || mSubgraph.empty()) && weight.IsValid()) {
            retValue = builder.RunMatmulEngine(m_engine.ToString(), params[0], weight);
            return;
        }

        std::cout << "[Model] Missing supported loaded weights for subgraph: " << mSubgraph << std::endl;
        retValue = X::Value();
    }
}
