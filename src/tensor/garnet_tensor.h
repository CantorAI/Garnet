#pragma once
#include "xpackage.h"
#include "xlang.h"
#include <set>
#include <string>
#include <sstream>
#include "cuda_jit_compiler.h"

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
	class GarnetTensor;
	class Fusionist
	{
		X::Value mVarTensor;//bind to a tensor
		X::Value mFunc;//the Func with this Fusion dectoration
		std::string mFuncName;
		std::string mFuncCodeHash;
		X::Value mTensorGraph;
		X::Value mReturnValue;//the xlang return value, then if kernl run ok, it will be set to this value
		bool mNeedGenAndCompile = false;
		CUfunction m_kernel;
		bool mHasKernel = false;

		BEGIN_PACKAGE(Fusionist)
			APISET().SetCallHandler(&Fusionist::Call);
		END_PACKAGE

	public:
		Fusionist() {}
		Fusionist(X::Value& func) :
			mFunc(func)
		{
		}
		inline void SetNeedGenAndCompile(bool b)
		{
			mNeedGenAndCompile = b;
		}
		inline void SetParent(X::Value& t) { mVarTensor = t; }
		bool Call(X::XRuntime* rt, X::ARGS& params, X::KWARGS& kwParams, X::Value& outputue);
		inline void SetFunc(X::Value& func,std::string& funcName,std::string& funcCodeHash)
		{
			mFunc = func;
			mFuncName = funcName;
			mFuncCodeHash = funcCodeHash;
		}
	};
	// Helper functions for CUDA code generation
	class CudaCodeGen {
		friend class GarnetTensor;

		std::set<std::string> declaredVariables;
		// Function to reset the declared variables set
		void ResetDeclaredVariables() {
			declaredVariables.clear();
		}

		std::string GetTensorName(X::Tensor& tensor)
		{
			std::string strName = tensor->GetName().ToString();
			if (strName.empty())
			{
				strName = "tensor_" + std::to_string(tensor->GetID());
			}
			return strName;
		}
		// Convert tensor data type to CUDA type string
		std::string GetTypeString(X::TensorDataType type) {
			switch (type) {
			case X::TensorDataType::FLOAT32:
				return "float";
			case X::TensorDataType::FLOAT16:
				return "__half";
			case X::TensorDataType::BFLOAT16:
				return "__nv_bfloat16";
			case X::TensorDataType::FLOAT8_E4M3FN:
				return "__nv_fp8_e4m3";
			case X::TensorDataType::FLOAT8_E5M2:
				return "__nv_fp8_e5m2";
			default:
				return "float"; // Default to float
			}
		}

		// Generate pointer variable declaration and assignment
			// Updated to avoid duplicate declarations
		std::string GenerateVariableDeclaration(const std::string& varName,
			X::TensorDataType type,
			void* memoryPtr) {
			// Check if variable is already declared
			if (declaredVariables.find(varName) != declaredVariables.end()) {
				return ""; // Variable already declared, return empty string
			}

			// Add variable to declared set
			declaredVariables.insert(varName);

			// Generate declaration as before
			std::string typeStr = GetTypeString(type);
			std::stringstream ss;
			ss << std::hex << reinterpret_cast<uintptr_t>(memoryPtr);
			std::string addrStr = "0x" + ss.str();
			return typeStr + "* " + varName + " = (" + typeStr + "*)" + addrStr + ";\n";
		}

		// Generate operation comment header
		std::string GenerateCommentHeader(const std::string& opName,
			X::Tensor& tensor1,
			X::Tensor& tensor2) {
			std::string comment = "// " + opName + " operation for tensor " +
				GetTensorName(tensor1);
			if (tensor2) {
				comment += " and tensor " + GetTensorName(tensor2);
			}
			comment += "\n";
			return comment;
		}

		std::string GenerateCommentHeader(const std::string& opName, X::Tensor& tensor) {
			std::string comment = "// " + opName + " operation for tensor " +
				GetTensorName(tensor);
			comment += "\n";
			return comment;
		}
	};

	class GarnetTensor
	{
		CudaCodeGen mCodeGen;
		CudaJitCompiler mCompiler;
		std::string ProcessCondition(X::Value& astNode);
	public:
		BEGIN_PACKAGE(GarnetTensor)
			APISET().AddClass<0, Fusionist>("fusionist");
			APISET().AddVarFuncEx("fusion", &GarnetTensor::Fusion);
			//APISET().AddVarFuncEx("compile", &GarnetTensor::Fusion);

			APISET().AddTensorStructuralOps("header", &GarnetTensor::Header);
			APISET().AddTensorStructuralOps("trailer", &GarnetTensor::Trailer);
			APISET().AddTensorStructuralOps("branchBegin", &GarnetTensor::BranchBegin);
			APISET().AddTensorStructuralOps("branchEnd", &GarnetTensor::BranchEnd);

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
	public:
		GarnetTensor();
		CudaJitCompiler& GetCompiler() { return mCompiler; }
		// Fusion function
		void Fusion(X::XRuntime* rt, X::XObj* pThis, X::XObj* pContext,
					X::ARGS& params, X::KWARGS& kwParams, X::Value& trailer, X::Value& outputue);

		X::Value Header(X::Value& graph,X::ARGS& params);
		X::Value Trailer(X::Value& graph, X::ARGS& params);
		X::Value BranchBegin(X::Value& graph, X::ARGS& params);
		X::Value BranchEnd(X::Value& graph, X::ARGS& params);

		X::Value Add(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams,
				X::Value input1, X::Value input2, X::Value& output);
		X::Value Minus(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams,
			X::Value input1, X::Value input2, X::Value& output);
		X::Value Multiply(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams,
			X::Value input1, X::Value input2, X::Value& output);
		X::Value Matmul(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams,
			X::Value input1, X::Value input2, X::Value& output);
		X::Value Permute(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams,
				X::Value input, X::Value& output);
		X::Value Gather(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams,
			X::Value input1, X::Value input2, X::Value& output);
		X::Value Convert(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams,
			X::Value input, X::Value& output);

		X::Value InitZeros(X::Value& graph,X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& output);
		X::Value InitOnes(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& output);
		X::Value InitFull(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& output);
		X::Value InitRand(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& output);
		X::Value InitRandn(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& output);
		X::Value InitUniform(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& output);
		X::Value InitNormal(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& output);
		X::Value InitTruncNormal(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value& output);
	};
}