#pragma once
#include "xpackage.h"
#include "xlang.h"


namespace Garnet
{
	class TensorDescriptor
	{
		friend class GarnetTensor;
		friend class TensorHelper;
		std::string mDeviceName;//such as cuda device name
		void* gpuMemory = nullptr;
	public:
		BEGIN_PACKAGE(TensorDescriptor)
			APISET().AddPropWithType<std::string>("DeviceName", &TensorDescriptor::mDeviceName);
		END_PACKAGE
	};
	class GarnetTensor
	{
	public:
		BEGIN_PACKAGE(GarnetTensor)
			APISET().AddTensorBinaryOp("add", &GarnetTensor::Add);
			APISET().AddTensorBinaryOp("minus", &GarnetTensor::Minus);
			APISET().AddTensorBinaryOp("mul", &GarnetTensor::Multiply);
			APISET().AddTensorBinaryOp("matmul", &GarnetTensor::Matmul);
			APISET().AddTensorUnaryOp("permute", &GarnetTensor::Permute);
			APISET().AddTensorBinaryOp("gather", &GarnetTensor::Gather);
		END_PACKAGE

		void Add(X::ARGS& params, X::KWARGS& kwParams,
				X::Value input1, X::Value input2, X::Value& retVal);
		void Minus(X::ARGS& params, X::KWARGS& kwParams,
			X::Value input1, X::Value input2, X::Value& retVal);
		void Multiply(X::ARGS& params, X::KWARGS& kwParams,
			X::Value input1, X::Value input2, X::Value& retVal);
		void Matmul(X::ARGS& params, X::KWARGS& kwParams,
			X::Value input1, X::Value input2, X::Value& retVal);
		void Permute(X::ARGS& params, X::KWARGS& kwParams,
				X::Value input, X::Value& retVal);
		void Gather(X::ARGS& params, X::KWARGS& kwParams,
			X::Value input1, X::Value input2, X::Value& retVal);
	};
}