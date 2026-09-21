// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "xlang3/xlang3.h"
#include "compiled_model_runtime.h"

#include <memory>
#include <mutex>

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
        bool mCompiledMode = false;
        X::Value m_engine;
        struct FixtureExecution {
            std::string path;
            std::shared_ptr<void> owner;
        };
        mutable std::mutex mFixtureMutex;
        FixtureExecution mFixtureExecution;
        FixtureExecution RetainFixtureExecution();
        void ReleaseFixtureExecution();
	public:
        void SetEngine(X::Value engine);
        X::Value GetEngine() const;
		void SetSubgraph(const std::string& subgraph) { mSubgraph = subgraph; }
		void SetRMSNormWeight(X::Value weight) { mRmsNormWeight = weight; }
		BEGIN_PACKAGE(Model)
            APISET().AddFunc<1>("__getitem__", &Model::Access);
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
            APISET().AddPropL("engine",
                [](auto* model, X::Value value) { model->SetEngine(std::move(value)); },
                [](auto* model) { return model->GetEngine(); });
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
		X::Value Access(X::Value index);
		X::Value Tokenizer(const X::ARGS& params, const X::KWARGS& kwParams);
        X::Value Detokenizer(const X::ARGS& params, const X::KWARGS& kwParams);
        X::Value DebugProbe(const X::ARGS& params, const X::KWARGS& kwParams);
        X::Value SampleLogits(const X::ARGS& params, const X::KWARGS& kwParams);
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
        X::Value RuntimeStatus(const X::ARGS& params, const X::KWARGS& kwParams);
        X::Value ReleaseRuntime(const X::ARGS& params, const X::KWARGS& kwParams);

        X::Value Forward(const X::ARGS& params, const X::KWARGS& kwParams);
        X::Value ForwardRequest(const X::ARGS& params, const X::KWARGS& kwParams);
        X::Value CreateDeviceKVCache(const X::ARGS& params, const X::KWARGS& kwParams);
        X::Value DestroyDeviceKVCache(const X::ARGS& params, const X::KWARGS& kwParams);
        X::Value WriteDeviceKVCache(const X::ARGS& params, const X::KWARGS& kwParams);
        X::Value AttentionDeviceKVCache(const X::ARGS& params, const X::KWARGS& kwParams);
	};
}
