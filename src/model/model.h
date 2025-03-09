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
		BEGIN_PACKAGE(Model)
			APISET().SetAccessor(&Model::Access);
			APISET().AddVarFunc("tokenizer", &Model::Tokenizer);
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
	};
}