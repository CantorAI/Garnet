#pragma once
#include "singleton.h"
#include "xpackage.h"
#include "xlang.h"
#include "garnet_tensor.h"

namespace Garnet
{
	class GarnetAPI :
		public Singleton<GarnetAPI>
	{
		X::Value m_curModule;
		bool LoadModelFromFile(std::string modelPath, X::Dict& model);
	public:
		BEGIN_PACKAGE(GarnetAPI)
			APISET().AddFunc<1>("loadModel", &GarnetAPI::LoadModel);
			APISET().AddFunc<0>("runTest", &GarnetAPI::RunTest);
			APISET().AddClass<0, GarnetTensor>("tensor");
		END_PACKAGE

		void SetModule(X::Value curModule)
		{
			m_curModule = curModule;
		}
		X::Value LoadModel(std::string modelPath);
		void RunTest();
	};
}
