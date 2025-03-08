#pragma once
#include "singleton.h"
#include "xpackage.h"
#include "xlang.h"
#include "garnet_tensor.h"
#include "model.h"
#include "log.h"

namespace Garnet
{
	class GarnetAPI :
		public Singleton<GarnetAPI>
	{
		X::Value m_cantor;
		X::Value m_log;
		X::Value m_curModule;
		bool LoadModelFromFile(std::string modelPath, X::Dict& model);
	public:
		BEGIN_PACKAGE(GarnetAPI)
			APISET().AddPropL("cantor",
				[](auto* pThis, X::Value v)
				{
					pThis->SetCantor(v);
				},
				[](auto* pThis) {return pThis->m_cantor; });
			APISET().AddFunc<1>("loadModel", &GarnetAPI::LoadModel);
			APISET().AddVarFunc("runTest", &GarnetAPI::RunTest);
			APISET().AddClass<0, Model>("model");
			APISET().AddClass<0, GarnetTensor>("tensor");
		END_PACKAGE

		void SetModule(X::Value curModule)
		{
			m_curModule = curModule;
		}
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
		void RunTest(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
	};
}
