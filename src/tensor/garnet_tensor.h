#pragma once
#include "xpackage.h"
#include "xlang.h"


namespace Garnet
{
	class GarnetTensor
	{
	public:
		BEGIN_PACKAGE(GarnetTensor)
			APISET().AddTensorBinaryOp("add", &GarnetTensor::Add);
			APISET().AddTensorBinaryOp("minus", &GarnetTensor::Minus);
			APISET().AddTensorBinaryOp("mul", &GarnetTensor::Multiply);
			APISET().AddTensorUnaryOp("permute", &GarnetTensor::Permute);
		END_PACKAGE

		void Add(X::ARGS& params, X::KWARGS& kwParams,
				X::Value input1, X::Value input2, X::Value& retVal);
		void Minus(X::ARGS& params, X::KWARGS& kwParams,
			X::Value input1, X::Value input2, X::Value& retVal);
		void Multiply(X::ARGS& params, X::KWARGS& kwParams,
			X::Value input1, X::Value input2, X::Value& retVal);
		void Permute(X::ARGS& params, X::KWARGS& kwParams,
				X::Value input, X::Value& retVal);
	};
}