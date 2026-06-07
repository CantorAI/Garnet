#include "garnet_tensor.h"

#if __USE_DIRECT_RUN

#include <cuda_runtime.h>
#include "tensor_helper.h"
#include <cuda_fp16.h>    // For __half and __float2half
#include <cuda_bf16.h>    // For __nv_bfloat16
#include <cuda_fp8.h>     // For __nv_fp8_e4m3 and __nv_fp8_e5m2


// Forward declarations for the CUDA functions from ptxGemm_kernel.cu
extern "C" {
    void runGemmFP16(const __half* d_A, const __half* d_B, float* d_C, int M, int N, int K);
    void runGemmBF16(const __nv_bfloat16* d_A, const __nv_bfloat16* d_B, float* d_C, int M, int N, int K);
    void runGemmFP8E4M3(const __nv_fp8_e4m3* d_A, const __nv_fp8_e4m3* d_B, float* d_C, int M, int N, int K);
    void runGemmFP8E5M2(const __nv_fp8_e5m2* d_A, const __nv_fp8_e5m2* d_B, float* d_C, int M, int N, int K);
    void runGemmFP32(const float* d_A, const float* d_B, float* d_C, int M, int N, int K);
}

extern "C" {
    void runGatherKernelFloat(const float* input, const int* indices, float* output, int M, int N, int numIndices);
    void runGatherKernelFP16(const __half* input, const int* indices, __half* output, int M, int N, int numIndices);
    void runGatherKernelBF16(const __nv_bfloat16* input, const int* indices, __nv_bfloat16* output, int M, int N, int numIndices);
    void runGatherKernelFP8E4M3(const __nv_fp8_e4m3* input, const int* indices, __nv_fp8_e4m3* output, int M, int N, int numIndices);
    void runGatherKernelFP8E5M2(const __nv_fp8_e5m2* input, const int* indices, __nv_fp8_e5m2* output, int M, int N, int numIndices);
}

namespace Garnet
{
    void GarnetTensor::Multiply(X::ARGS& params, X::KWARGS& kwParams,
        X::Value input1, X::Value input2, X::Value& retVal)
    {
        bool isTensor1 = input1.IsTensor();
        bool isTensor2 = input2.IsTensor();

        if (isTensor1 && isTensor2)
        {
            // Tensor-tensor multiplication
            X::Tensor tensor1(input1);
            X::Tensor tensor2(input2);

            // Get the data types
            auto tensor1_type = tensor1->GetDataType();
            auto tensor2_type = tensor2->GetDataType();

            // Validate dimensions for matrix multiplication
            int dimCount1 = tensor1->GetDimCount();
            int dimCount2 = tensor2->GetDimCount();

            if (dimCount1 > 2 || dimCount2 > 2)
            {
                retVal = X::Value();
                return;
            }

            int m = tensor1->GetDimSize(0);
            int n = (dimCount1 > 1) ? tensor1->GetDimSize(1) : 1;
            int k = (dimCount2 > 1) ? tensor2->GetDimSize(1) : 1;

            // Check if dimensions match for matrix multiplication
            if (dimCount2 == 1)
            {
                // Vector case: n must match
                if (n != tensor2->GetDimSize(0))
                {
                    retVal = X::Value();
                    return;
                }
            }
            else if (n != tensor2->GetDimSize(0))
            {
                // Matrix case: inner dimensions must match
                retVal = X::Value();
                return;
            }

            // Set the result dimensions
            int resultD = 1;
            if (tensor2->GetDimCount() > 1)
            {
                resultD = 2;
            }
            X::Port::vector<int> resultDims(resultD);
            resultDims.push_back(m);
            if (tensor2->GetDimCount() > 1)
            {
                resultDims.push_back(k);
            }

            // Ensure GPU memory is allocated for both tensors
            TensorOpStatus status = TensorHelper::EnsureGPUMemory(tensor1);
            if (status != TensorOpStatus::Success)
            {
                retVal = X::Value();
                return;
            }

            status = TensorHelper::EnsureGPUMemory(tensor2);
            if (status != TensorOpStatus::Success)
            {
                retVal = X::Value();
                return;
            }

            // Check device compatibility
            std::string device1 = TensorHelper::GetDeviceName(tensor1);
            std::string device2 = TensorHelper::GetDeviceName(tensor2);

            if (!device1.empty() && !device2.empty() && device1 != device2)
            {
                retVal = X::Value();
                return;
            }

            // Create result tensor
            std::string deviceName = !device1.empty() ? device1 : (!device2.empty() ? device2 : "cuda");

            // Create new descriptor for result tensor
            X::XPackageValue<TensorDescriptor> resultDescValue;
            TensorDescriptor& resultDesc = *resultDescValue;
            resultDesc.mDeviceName = deviceName;

            // don't create new tensor, that will not deliver to  the result tensor
            X::XTensor* pRetTensor = dynamic_cast<X::XTensor*>(retVal.GetObj());
            pRetTensor->SetDataType(tensor1_type);
            pRetTensor->SetShape(resultDims);
            X::Value initData;
            pRetTensor->Create(initData);
			X::Tensor resultTensor(pRetTensor);
            status = TensorHelper::EnsureGPUMemory(resultTensor);
            if (status != TensorOpStatus::Success)
            {
                retVal = X::Value();
                return;
            }

            void* gpuResultData = TensorHelper::GetGPUMemory(resultTensor);


            // Get GPU memory pointers
            void* gpuData1 = TensorHelper::GetGPUMemory(tensor1);
            void* gpuData2 = TensorHelper::GetGPUMemory(tensor2);

            // Perform matrix multiplication based on data type
            if (tensor1_type == X::TensorDataType::FLOAT32 
                && tensor2_type == X::TensorDataType::FLOAT32)
            {
                runGemmFP32(
                    reinterpret_cast<float*>(gpuData1),
                    reinterpret_cast<float*>(gpuData2),
                    reinterpret_cast<float*>(gpuResultData), m, k, n);
            }
            else if (tensor1_type == X::TensorDataType::FLOAT16 
                && tensor2_type == X::TensorDataType::FLOAT16)
            {
                runGemmFP16(
                    reinterpret_cast<__half*>(gpuData1),
                    reinterpret_cast<__half*>(gpuData2),
                    reinterpret_cast<float*>(gpuResultData), m, k, n);
            }
            else if (tensor1_type == X::TensorDataType::BFLOAT16 && tensor2_type == X::TensorDataType::BFLOAT16)
            {
                runGemmBF16(
                    reinterpret_cast<__nv_bfloat16*>(gpuData1),
                    reinterpret_cast<__nv_bfloat16*>(gpuData2),
                    reinterpret_cast<float*>(gpuResultData), m, k, n);
            }
            else if (tensor1_type == X::TensorDataType::FLOAT8_E4M3FN
                && tensor2_type == X::TensorDataType::FLOAT8_E4M3FN)
            {
                runGemmFP8E4M3(
                    reinterpret_cast<__nv_fp8_e4m3*>(gpuData1),
                    reinterpret_cast<__nv_fp8_e4m3*>(gpuData2),
                    reinterpret_cast<float*>(gpuResultData), m, k, n);
			}
			else if (tensor1_type == X::TensorDataType::FLOAT8_E5M2
				&& tensor2_type == X::TensorDataType::FLOAT8_E5M2)
			{
				runGemmFP8E5M2(
					reinterpret_cast<__nv_fp8_e5m2*>(gpuData1),
					reinterpret_cast<__nv_fp8_e5m2*>(gpuData2),
					reinterpret_cast<float*>(gpuResultData), m, k, n);
			}
            else
            {
                // Unsupported data type combination
                cudaFree(gpuResultData);
                retVal = X::Value();
                return;
            }

            // Copy result from GPU to CPU
            status = TensorHelper::CopyResultFromGPU(resultTensor);
            if (status != TensorOpStatus::Success)
            {
                cudaFree(gpuResultData);
                retVal = X::Value();
                return;
            }

            retVal = X::Value(resultTensor);
        }
        else if (isTensor1)
        {
            // Tensor-scalar multiplication
            X::Tensor tensor(input1);
            auto tensorType = tensor->GetDataType();
            float scalar = (float)input2.ToDouble();

            // We only support 1D or 2D tensors
            int dimCount = tensor->GetDimCount();
            if (dimCount > 2)
            {
                retVal = X::Value();
                return;
            }

            // Copy dimensions for result tensor
            X::Port::vector<int> dims;
            for (int i = 0; i < dimCount; i++)
            {
                dims.push_back(tensor->GetDimSize(i));
            }

            // Calculate total elements
            long long totalElements = 1;
            for (int i = 0; i < dimCount; i++)
            {
                totalElements *= tensor->GetDimSize(i);
            }

            // Ensure GPU memory is allocated
            TensorOpStatus status = TensorHelper::EnsureGPUMemory(tensor);
            if (status != TensorOpStatus::Success)
            {
                retVal = X::Value();
                return;
            }

            // Create result tensor
            std::string deviceName = TensorHelper::GetDeviceName(tensor);
            if (deviceName.empty()) {
                deviceName = "cuda";
            }

            // Create new descriptor for result tensor
            X::XPackageValue<TensorDescriptor> resultDescValue;
            TensorDescriptor& resultDesc = *resultDescValue;
            resultDesc.mDeviceName = deviceName;

            // Create the result tensor
            X::Value resultTensorDesc = X::Value(resultDescValue);
            X::Tensor resultTensor(resultTensorDesc);
            resultTensor->SetDataType(tensorType);
            resultTensor->SetShape(dims);

            // Allocate GPU memory for result
            long long resultSize = totalElements * TensorHelper::GetItemSizeForType(tensorType);
            void* gpuResultData = nullptr;
            cudaError_t err = cudaMalloc(&gpuResultData, resultSize);
            if (err != cudaSuccess)
            {
                retVal = X::Value();
                return;
            }
            resultDesc.gpuMemory = gpuResultData;
            resultTensor->SetDesc(X::Value(resultDescValue));

            // Get GPU memory pointers
            void* gpuData = TensorHelper::GetGPUMemory(tensor);

            // Perform scalar multiplication based on data type
            if (tensorType == X::TensorDataType::FLOAT32)
            {
                // Define kernel dimensions
                dim3 blockSize(256);
                dim3 gridSize((totalElements + blockSize.x - 1) / blockSize.x);

                // Launch your custom CUDA kernel for scalar multiplication
                // scalarMultiplyKernelFP32<<<gridSize, blockSize>>>(
                //     reinterpret_cast<float*>(gpuData),
                //     reinterpret_cast<float*>(gpuResultData),
                //     scalar, totalElements);

                cudaDeviceSynchronize();
            }
            else if (tensorType == X::TensorDataType::FLOAT16)
            {
                // Define kernel dimensions
                dim3 blockSize(256);
                dim3 gridSize((totalElements + blockSize.x - 1) / blockSize.x);

                // Launch your custom CUDA kernel for scalar multiplication
                // scalarMultiplyKernelFP16<<<gridSize, blockSize>>>(
                //     reinterpret_cast<__half*>(gpuData),
                //     reinterpret_cast<__half*>(gpuResultData),
                //     scalar, totalElements);

                cudaDeviceSynchronize();
            }
            else if (tensorType == X::TensorDataType::BFLOAT16)
            {
                // Define kernel dimensions
                dim3 blockSize(256);
                dim3 gridSize((totalElements + blockSize.x - 1) / blockSize.x);

                // Launch your custom CUDA kernel for scalar multiplication
                // scalarMultiplyKernelBF16<<<gridSize, blockSize>>>(
                //     reinterpret_cast<__nv_bfloat16*>(gpuData),
                //     reinterpret_cast<__nv_bfloat16*>(gpuResultData),
                //     scalar, totalElements);

                cudaDeviceSynchronize();
            }
            else if (tensorType == X::TensorDataType::FLOAT8)
            {
                // Define kernel dimensions
                dim3 blockSize(256);
                dim3 gridSize((totalElements + blockSize.x - 1) / blockSize.x);

                // Launch your custom CUDA kernel for scalar multiplication
                // scalarMultiplyKernelFP8<<<gridSize, blockSize>>>(
                //     reinterpret_cast<__nv_fp8_e4m3*>(gpuData),
                //     reinterpret_cast<__nv_fp8_e4m3*>(gpuResultData),
                //     scalar, totalElements);

                cudaDeviceSynchronize();
            }
            else
            {
                // Unsupported data type
                cudaFree(gpuResultData);
                retVal = X::Value();
                return;
            }

            // Copy result from GPU to CPU
            status = TensorHelper::CopyResultFromGPU(resultTensor);
            if (status != TensorOpStatus::Success)
            {
                cudaFree(gpuResultData);
                retVal = X::Value();
                return;
            }

            retVal = X::Value(resultTensor);
        }
        else if (isTensor2)
        {
            // Scalar-tensor multiplication is commutative
            X::Value result;
            Multiply(params, kwParams, input2, input1, result);
            retVal = result;
        }
        else
        {
            // Scalar-scalar multiplication
            float val1 = (float)input1;
            float val2 = (float)input2;
            retVal = X::Value(val1 * val2);
        }
    }



    void GarnetTensor::Add(X::ARGS& params, X::KWARGS& kwParams,
		X::Value input1, X::Value input2, X::Value& retVal)
	{
	}
	void GarnetTensor::Minus(X::ARGS& params, X::KWARGS& kwParams, 
		X::Value input1, X::Value input2, X::Value& retVal)
	{
	}
	void GarnetTensor::Matmul(X::ARGS& params, X::KWARGS& kwParams, 
		X::Value input1, X::Value input2, X::Value& retVal)
	{
		bool isTensor1 = input1.IsTensor();
		bool isTensor2 = input2.IsTensor();
		if (!isTensor1 || !isTensor2)
		{
			retVal = X::Value();
		}

	}
	void GarnetTensor::Permute(X::ARGS& params, X::KWARGS& kwParams, 
		X::Value input, X::Value& retVal)
	{
	}
    void GarnetTensor::Gather(X::ARGS& params, X::KWARGS& kwParams,
        X::Value input1, X::Value input2, X::Value& retVal)
    {
        // Ensure both inputs are tensors.
        bool isTensor1 = input1.IsTensor();
        bool isTensor2 = input2.IsTensor();
        if (!isTensor1 || !isTensor2)
        {
            retVal = X::Value();
            return;
        }

        // input1 is the embedding matrix and must be 2D.
        X::Tensor tensor = input1;
        if (tensor->GetDimCount() != 2)
        {
            retVal = X::Value();
            return;
        }

        // input2 is the indices tensor and we require it be 1D.
        X::Tensor indicesTensor = input2;
        if (indicesTensor->GetDimCount() != 1)
        {
            retVal = X::Value();
            return;
        }

        // Let N be the number of rows (vocabulary size) and M be the embedding dimension.
        int N = tensor->GetDimSize(0);
        int M = tensor->GetDimSize(1);
        int numIndices = indicesTensor->GetDimSize(0);

        // The result of gathering will be a matrix of shape (numIndices, M)
        X::Port::vector<int> resultDims(2);
        resultDims.push_back(numIndices);
        resultDims.push_back(M);

        // Ensure GPU memory is allocated for both tensors.
        TensorOpStatus status = TensorHelper::EnsureGPUMemory(tensor);
        if (status != TensorOpStatus::Success)
        {
            retVal = X::Value();
            return;
        }
        status = TensorHelper::EnsureGPUMemory(indicesTensor);
        if (status != TensorOpStatus::Success)
        {
            retVal = X::Value();
            return;
        }

        // Check that both tensors are on the same device.
        std::string device1 = TensorHelper::GetDeviceName(tensor);
        std::string device2 = TensorHelper::GetDeviceName(indicesTensor);
        if (!device1.empty() && !device2.empty() && device1 != device2)
        {
            retVal = X::Value();
            return;
        }
        std::string deviceName = !device1.empty() ? device1 : (!device2.empty() ? device2 : "cuda");

        // Create the result tensor.
        X::XPackageValue<TensorDescriptor> resultDescValue;
        TensorDescriptor& resultDesc = *resultDescValue;
        resultDesc.mDeviceName = deviceName;

        // Use the provided result tensor (do not create a new one).
        X::XTensor* pRetTensor = dynamic_cast<X::XTensor*>(retVal.GetObj());
        if (pRetTensor == nullptr)
        {
            retVal = X::Value();
            return;
        }
        pRetTensor->SetDataType(tensor->GetDataType());
        pRetTensor->SetShape(resultDims);
        X::Value initData;
        pRetTensor->Create(initData);
        X::Tensor resultTensor(pRetTensor);
        status = TensorHelper::EnsureGPUMemory(resultTensor);
        if (status != TensorOpStatus::Success)
        {
            retVal = X::Value();
            return;
        }

        void* gpuResultData = TensorHelper::GetGPUMemory(resultTensor);
        void* gpuData = TensorHelper::GetGPUMemory(tensor);
        void* gpuIndices = TensorHelper::GetGPUMemory(indicesTensor);

        // Dispatch to the appropriate gather kernel based on the tensor's data type.
        X::TensorDataType tensorType = tensor->GetDataType();
        if (tensorType == X::TensorDataType::FLOAT32)
        {
            runGatherKernelFloat(
                reinterpret_cast<float*>(gpuData),
                reinterpret_cast<int*>(gpuIndices),
                reinterpret_cast<float*>(gpuResultData),
                M, N, numIndices);
        }
        else if (tensorType == X::TensorDataType::FLOAT16)
        {
            runGatherKernelFP16(
                reinterpret_cast<__half*>(gpuData),
                reinterpret_cast<int*>(gpuIndices),
                reinterpret_cast<__half*>(gpuResultData),
                M, N, numIndices);
        }
        else if (tensorType == X::TensorDataType::BFLOAT16)
        {
            runGatherKernelBF16(
                reinterpret_cast<__nv_bfloat16*>(gpuData),
                reinterpret_cast<int*>(gpuIndices),
                reinterpret_cast<__nv_bfloat16*>(gpuResultData),
                M, N, numIndices);
        }
        else if (tensorType == X::TensorDataType::FLOAT8_E4M3FN)
        {
            runGatherKernelFP8E4M3(
                reinterpret_cast<__nv_fp8_e4m3*>(gpuData),
                reinterpret_cast<int*>(gpuIndices),
                reinterpret_cast<__nv_fp8_e4m3*>(gpuResultData),
                M, N, numIndices);
        }
        else if (tensorType == X::TensorDataType::FLOAT8_E5M2)
        {
            runGatherKernelFP8E5M2(
                reinterpret_cast<__nv_fp8_e5m2*>(gpuData),
                reinterpret_cast<int*>(gpuIndices),
                reinterpret_cast<__nv_fp8_e5m2*>(gpuResultData),
                M, N, numIndices);
        }
        else
        {
            // Unsupported type.
            cudaFree(gpuResultData);
            retVal = X::Value();
            return;
        }

        // Copy the result from GPU to CPU.
        status = TensorHelper::CopyResultFromGPU(resultTensor);
        if (status != TensorOpStatus::Success)
        {
            cudaFree(gpuResultData);
            retVal = X::Value();
            return;
        }

        retVal = X::Value(resultTensor);
    }
    void GarnetTensor::Convert(X::ARGS& params, X::KWARGS& kwParams,
        X::Value input, X::Value& retVal)
    {

    }
}

#endif