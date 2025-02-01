#pragma once
#include "singleton.h"
#include "xpackage.h"
#include "xlang.h"

namespace Garnet
{
	class GarnetAPI :
		public Singleton<GarnetAPI>
	{
		X::Value m_curModule;
	public:
		BEGIN_PACKAGE(GarnetAPI)
			APISET().AddFunc<1>("loadModel", &GarnetAPI::LoadModel);
		END_PACKAGE

		void SetModule(X::Value curModule)
		{
			m_curModule = curModule;
		}
		X::Value LoadModel(std::string modelPath);
	};
}
