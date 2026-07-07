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
		int m_maxNumPages = 0;
		int m_pageSize = 0;
		int m_headDim = 0;
		int m_numKVHeads = 0;
		std::deque<int> m_freePages;
		std::unordered_map<long long, std::vector<int>> m_sequencePages;
	public:
		BEGIN_PACKAGE(KVCacheManager)
			APISET().AddVarFunc("allocate", &KVCacheManager::Allocate);
			APISET().AddVarFunc("free", &KVCacheManager::Free);
			APISET().AddVarFunc("stats", &KVCacheManager::Stats);
		END_PACKAGE

		void Configure(int maxNumPages, int pageSize, int headDim, int numKVHeads);
		void Allocate(X::XRuntime* rt, X::XObj* pContext,
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
		void RunTest(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
	};
}
