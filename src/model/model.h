#pragma once

#include "xpackage.h"
#include "xlang.h"
#include "compiled_model_runtime.h"

#include <memory>

namespace Garnet
{
	class Model
	{
		X::Value mModel;//a dictionary to store {key:tersor}
		std::string mModelPath;
		std::string mTokenizerJsonPath;
		std::string mTokenizerConfigJsonPath;
		std::string mSubgraph;
		X::Value mRmsNormWeight;

		X::Value mTokenizer;
        std::shared_ptr<CompiledModelRuntime> mCompiledRuntime;
	public:
		X::Value m_engine;
		void SetEngine(X::Value engine) { m_engine = engine; }
		void SetSubgraph(const std::string& subgraph) { mSubgraph = subgraph; }
		void SetRMSNormWeight(X::Value weight) { mRmsNormWeight = weight; }
		BEGIN_PACKAGE(Model)
            APISET().SetAccessor(&Model::Access);
            APISET().AddVarFunc("tokenizer", &Model::Tokenizer);
            APISET().AddVarFunc("detokenizer", &Model::Detokenizer);
            APISET().AddVarFunc("debug_probe", &Model::DebugProbe);
            APISET().AddVarFunc("runtime_status", &Model::RuntimeStatus);
            APISET().AddVarFunc("release_runtime", &Model::ReleaseRuntime);
            APISET().AddVarFunc("forward", &Model::Forward);
            APISET().AddVarFunc("forward_request", &Model::ForwardRequest);
            APISET().AddVarFunc("create_device_kv_cache", &Model::CreateDeviceKVCache);
            APISET().AddVarFunc("destroy_device_kv_cache", &Model::DestroyDeviceKVCache);
            APISET().AddVarFunc("write_device_kv_cache", &Model::WriteDeviceKVCache);
            APISET().AddVarFunc("attention_device_kv_cache", &Model::AttentionDeviceKVCache);
			APISET().AddProp0("weights", &Model::mModel);
			APISET().AddProp0("engine", &Model::m_engine);
			APISET().AddPropWithType<std::string>("modelPath", &Model::mModelPath);
		END_PACKAGE

		inline void SetInfo(
				std::string& modelPath,
				std::string& tokenizerJsonPath,
				std::string& tokenizerConfigJsonPath,
				X::Value& model)
		{
			mModel = model;
			mModelPath = modelPath;
			mTokenizerJsonPath = tokenizerJsonPath;
			mTokenizerConfigJsonPath = tokenizerConfigJsonPath;
		}
		X::Value Access(X::Port::vector<X::Value>& IdxAry);
		void Tokenizer(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
        void Detokenizer(X::XRuntime* rt, X::XObj* pContext,
            X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
        void DebugProbe(X::XRuntime* rt, X::XObj* pContext,
            X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
        void SampleLogits(X::XRuntime* rt, X::XObj* pContext,
            X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
        bool InitializeCompiledRuntime(
            const std::string& rootXModel,
            const std::string& cacheDirectory,
            const std::string& weightsLocation,
            const std::string& entryFunction,
            const std::string& frontend,
            const std::vector<std::vector<int>>& inputShapes,
            const std::vector<std::string>& inputDataTypes,
            const FusionPartitionOptions& partitionOptions = {},
            const std::string& backend = "tensorrt",
            const std::string& precision = "");
        X::Value CompiledRuntimeStatus()
        {
            return mCompiledRuntime ? mCompiledRuntime->Status() : X::Value();
        }
        void RuntimeStatus(X::XRuntime* rt, X::XObj* pContext,
            X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
        void ReleaseRuntime(X::XRuntime* rt, X::XObj* pContext,
            X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);

        void Forward(X::XRuntime* rt, X::XObj* pContext, X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
        void ForwardRequest(X::XRuntime* rt, X::XObj* pContext, X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
        void CreateDeviceKVCache(X::XRuntime* rt, X::XObj* pContext, X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
        void DestroyDeviceKVCache(X::XRuntime* rt, X::XObj* pContext, X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
        void WriteDeviceKVCache(X::XRuntime* rt, X::XObj* pContext, X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
        void AttentionDeviceKVCache(X::XRuntime* rt, X::XObj* pContext, X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
	};
}
