#include "garnet_tensor.h"

namespace Garnet
{
	void GarnetTensor::Add(X::ARGS& params, X::KWARGS& kwParams, 
		X::Value input1, X::Value input2, X::Value& retVal)
	{
	}
	void GarnetTensor::Minus(X::ARGS& params, X::KWARGS& kwParams, 
		X::Value input1, X::Value input2, X::Value& retVal)
	{
	}
	void GarnetTensor::Multiply(X::ARGS& params, X::KWARGS& kwParams, 
		X::Value input1, X::Value input2, X::Value& retVal)
	{
		bool isTensor1 = input1.IsTensor();
		bool isTensor2 = input2.IsTensor();
		if (isTensor1 && isTensor2)
		{
			// Multiply two tensors.
			X::Tensor tensor1(input1);
			X::Tensor tensor2(input2);
		}
		else if (isTensor1)
		{
			// Multiply tensor and scalar.
		}
		else if (isTensor2)
		{
			// Multiply scalar and tensor.
		}
		else
		{
			// Multiply two scalars.
		}
	}
	void GarnetTensor::Permute(X::ARGS& params, X::KWARGS& kwParams, 
		X::Value input, X::Value& retVal)
	{
	}
}