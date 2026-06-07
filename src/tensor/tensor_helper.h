#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>    // For __half and __float2half
#include <cuda_bf16.h>    // For __nv_bfloat16
#include <cuda_fp8.h>     // For __nv_fp8_e4m3 and __nv_fp8_e5m2

namespace Garnet
{
    // Enum to track operation status
    enum class TensorOpStatus {
        Success,
        InvalidDescriptor,
        CudaAllocationError,
        CudaCopyError,
        GeneralError
    };

    // Helper class for tensor operations
    class TensorHelper
    {
    public:
        // Ensure tensor has GPU memory allocated
        static TensorOpStatus EnsureGPUMemory(X::Tensor& tensor)
        {
            X::Value tensorDesc = tensor->GetDesc();
            if (tensorDesc.IsObject())
            {
                X::XPackageValue<TensorDescriptor> varDesc(tensorDesc);
                TensorDescriptor& desc = *varDesc;

                // Allocate GPU memory if not already allocated
                if (!desc.gpuMemory)
                {
                    long long size = tensor->GetDataSize();
                    void* gpuMem = nullptr;

                    cudaError_t err = cudaMalloc(&gpuMem, size);
                    if (err != cudaSuccess)
                    {
                        return TensorOpStatus::CudaAllocationError;
                    }

                    err = cudaMemcpy(gpuMem, tensor->GetData(), size, cudaMemcpyHostToDevice);
                    if (err != cudaSuccess)
                    {
                        cudaFree(gpuMem);
                        return TensorOpStatus::CudaCopyError;
                    }

                    desc.gpuMemory = gpuMem;
                    tensor->SetDesc(X::Value(varDesc));
                }
            }
            else
            {
                // Create new descriptor
                X::XPackageValue<TensorDescriptor> varDesc;
                TensorDescriptor& desc = *varDesc;
                desc.mDeviceName = "cuda";

                // Allocate GPU memory
                long long size = tensor->GetDataSize();
                void* gpuMem = nullptr;

                cudaError_t err = cudaMalloc(&gpuMem, size);
                if (err != cudaSuccess)
                {
                    return TensorOpStatus::CudaAllocationError;
                }

                err = cudaMemcpy(gpuMem, tensor->GetData(), size, cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    cudaFree(gpuMem);
                    return TensorOpStatus::CudaCopyError;
                }

                desc.gpuMemory = gpuMem;
                tensor->SetDesc(X::Value(varDesc));
            }

            return TensorOpStatus::Success;
        }

        // Get GPU memory pointer from tensor
        static void* GetGPUMemory(X::Tensor& tensor)
        {
            X::Value tensorDesc = tensor->GetDesc();
            if (tensorDesc.IsObject())
            {
                X::XPackageValue<TensorDescriptor> varDesc(tensorDesc);
                TensorDescriptor& desc = *varDesc;
                return desc.gpuMemory;
            }
            return nullptr;
        }

        // Get device name from tensor
        static std::string GetDeviceName(X::Tensor& tensor)
        {
            X::Value tensorDesc = tensor->GetDesc();
            if (tensorDesc.IsObject())
            {
                X::XPackageValue<TensorDescriptor> varDesc(tensorDesc);
                TensorDescriptor& desc = *varDesc;
                return desc.mDeviceName;
            }
            return "";
        }

        // Copy result data from GPU to CPU
        static TensorOpStatus CopyResultFromGPU(X::Tensor& tensor)
        {
            X::Value tensorDesc = tensor->GetDesc();
            if (!tensorDesc.IsObject())
            {
                return TensorOpStatus::InvalidDescriptor;
            }

            X::XPackageValue<TensorDescriptor> varDesc(tensorDesc);
            TensorDescriptor& desc = *varDesc;

            if (!desc.gpuMemory)
            {
                return TensorOpStatus::GeneralError;
            }

            // Calculate total size
            long long totalSize = tensor->GetDataSize();

            // Allocate CPU memory
            char* cpuData = tensor->GetData();

            // Copy from GPU to CPU
            cudaError_t err = cudaMemcpy(cpuData, desc.gpuMemory, totalSize, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                return TensorOpStatus::CudaCopyError;
            }

            return TensorOpStatus::Success;
        }

        // Get item size for tensor data type
        static int GetItemSizeForType(X::TensorDataType type)
        {
            switch (type)
            {
            case X::TensorDataType::FLOAT32:
                return sizeof(float);
            case X::TensorDataType::FLOAT16:
                return sizeof(__half);
            case X::TensorDataType::BFLOAT16:
                return sizeof(__nv_bfloat16);
            case X::TensorDataType::FLOAT8_E4M3FN:
            case X::TensorDataType::FLOAT8_E4M3FNUZ:
            case X::TensorDataType::FLOAT8_E5M2:
            case X::TensorDataType::FLOAT8_E5M2FNUZ:
                return sizeof(__nv_fp8_e4m3);
            default:
                return 0;
            }
        }
    };
}