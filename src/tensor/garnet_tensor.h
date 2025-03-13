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
			APISET().AddTensorUnaryOp("convert", &GarnetTensor::Convert);

			APISET().AddTensorUnaryOp("zeros", &GarnetTensor::InitZeros);
			APISET().AddTensorUnaryOp("ones", &GarnetTensor::InitOnes);
			APISET().AddTensorUnaryOp("full", &GarnetTensor::InitFull);
			APISET().AddTensorUnaryOp("rand", &GarnetTensor::InitRand);
			APISET().AddTensorUnaryOp("randn", &GarnetTensor::InitRandn);
			APISET().AddTensorUnaryOp("uniform", &GarnetTensor::InitUniform);
			APISET().AddTensorUnaryOp("normal", &GarnetTensor::InitNormal);
			APISET().AddTensorUnaryOp("trunc_normal", &GarnetTensor::InitTruncNormal);

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
		void Convert(X::ARGS& params, X::KWARGS& kwParams,
			X::Value input, X::Value& retVal);

		void InitZeros(X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& retVal);
		void InitOnes(X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& retVal);
		void InitFull(X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& retVal);
		void InitRand(X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& retVal);
		void InitRandn(X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& retVal);
		void InitUniform(X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& retVal);
		void InitNormal(X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& retVal);
		void InitTruncNormal(X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& retVal);
	};
}