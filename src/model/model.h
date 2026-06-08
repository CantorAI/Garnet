#pragma once

#include "xpackage.h"
#include "xlang.h"

namespace Garnet
{
	class Model
	{
		X::Value mModel;//a dictionary to store {key:tersor}
		std::string mModelPath;
		std::string mTokenizerJsonPath;
		std::string mTokenizerConfigJsonPath;

		X::Value mTokenizer;
	public:
		X::Value m_engine;
		void SetEngine(X::Value engine) { m_engine = engine; }
		BEGIN_PACKAGE(Model)
			APISET().SetAccessor(&Model::Access);
			APISET().AddVarFunc("tokenizer", &Model::Tokenizer);
            APISET().AddVarFunc("forward", &Model::Forward);
			APISET().AddProp0("weights", &Model::mModel);
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

        void BuildTRTEngine(X::Value forwardFunc, X::Value inputShapes);
        void Forward(X::XRuntime* rt, X::XObj* pContext, X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
	};
}
