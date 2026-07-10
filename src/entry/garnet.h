#pragma once
#include "singleton.h"
#include "xpackage.h"
#include "xlang.h"
#include "garnet_tensor.h"
#include "model.h"
#include "log.h"
#include <string>
#include <deque>
#include <unordered_map>
#include <vector>

namespace Garnet
{
	class KVCacheManager
	{
		struct SequenceState
		{
			std::vector<int> pages;
			long long logicalLength = 0;
		};
		int m_maxNumPages = 0;
		int m_pageSize = 0;
		int m_headDim = 0;
		int m_numKVHeads = 0;
		int m_numLayers = 0;
		int m_dtypeBytes = 2;
		int m_deviceId = 0;
		size_t m_bytesPerPagePerLayer = 0;
		size_t m_totalBytes = 0;
		void* m_keyArena = nullptr;
		void* m_valueArena = nullptr;
		std::deque<int> m_freePages;
		std::unordered_map<long long, SequenceState> m_sequences;

		int PagesForTokens(long long tokenCount) const;
		bool EnsurePages(long long sequenceId, long long tokenCount);
		X::Value MakePageList(const std::vector<int>& pages) const;
	public:
		~KVCacheManager();
		BEGIN_PACKAGE(KVCacheManager)
			APISET().AddVarFunc("allocate", &KVCacheManager::Allocate);
			APISET().AddVarFunc("append", &KVCacheManager::Append);
			APISET().AddVarFunc("free", &KVCacheManager::Free);
			APISET().AddVarFunc("stats", &KVCacheManager::Stats);
		END_PACKAGE

		void Configure(int maxNumPages, int pageSize, int headDim, int numKVHeads,
			int numLayers = 1, int dtypeBytes = 2, int deviceId = 0);
		void Allocate(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void Append(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void Free(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void Stats(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
	};

	class GarnetAPI :
		public Singleton<GarnetAPI>
	{
		X::Value m_cantor;
		X::Value m_log;
		X::Value m_curModule;
		std::string m_baseFolder;//like cantor's folder
		X::Value m_current_weights;
		X::Value m_compiledEngine;
		bool LoadModelFromFile(std::string modelPath, X::Dict& model);
	public:
		BEGIN_PACKAGE(GarnetAPI)
			APISET().AddPropL("cantor",
				[](auto* pThis, X::Value v)
				{
					pThis->SetCantor(v);
				},
				[](auto* pThis) {return pThis->m_cantor; });
			APISET().AddVarFunc("load_model", &GarnetAPI::LoadModelEx);
			APISET().AddVarFunc("qwen_vl_smart_resize", &GarnetAPI::QwenVLSmartResize);
			APISET().AddVarFunc("qwen_vl_prepare_request", &GarnetAPI::QwenVLPrepareRequest);
			APISET().AddVarFunc("qwen_vl_preprocess_image", &GarnetAPI::QwenVLPreprocessImage);
			APISET().AddVarFunc("qwen_vl_preprocess_jpeg_file", &GarnetAPI::QwenVLPreprocessJpegFile);
			APISET().AddVarFunc("KVCacheManager", &GarnetAPI::CreateKVCacheManager);
			APISET().AddVarFunc("runTest", &GarnetAPI::RunTest);
			APISET().AddClass<0, KVCacheManager>("KVCacheManagerClass");
			APISET().AddClass<0, Model>("model");
			APISET().AddClass<0, GarnetTensor>("tensor");
		END_PACKAGE

		void SetBaseFolder(std::string folder)
		{
			m_baseFolder = folder;
		}
		std::string GetBaseFolder()
		{
			return m_baseFolder;
		}
		void SetModule(X::Value curModule)
		{
			m_curModule = curModule;
		}

		void SetCurrentWeights(X::Value weights) { m_current_weights = weights; }
		X::Value GetCurrentWeights() { return m_current_weights; }

		void SetCompiledEngine(X::Value engine) { m_compiledEngine = engine; }
		X::Value GetCompiledEngine() { return m_compiledEngine; }
		bool SetCantor(X::Value cantor)
		{
			if (m_cantor.IsValid())
			{
				return true;
			}
			m_cantor = cantor;
			m_log = cantor["log_nolineend"];
			InitLog(m_log);

			return true;
		}
		X::Value LoadModel(std::string modelPath);
		void CreateKVCacheManager(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void LoadModelEx(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void QwenVLSmartResize(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void QwenVLPrepareRequest(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void QwenVLPreprocessImage(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void QwenVLPreprocessJpegFile(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void RunTest(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
	};
}
