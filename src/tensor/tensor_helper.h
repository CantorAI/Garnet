#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>    // For __half and __float2half
#include <cuda_bf16.h>    // For __nv_bfloat16
#include <cuda_fp8.h>     // For __nv_fp8_e4m3 and __nv_fp8_e5m2
#include <cstdint>
#include <cstdlib>
#include <vector>

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
        static X::Value CudaDeviceOps()
        {
            static X::Value opsValue;
            if (!opsValue.IsObject()) {
                static X::U_FUNC freeCallback = [](X::XRuntime* rt, X::XObj* pThis, X::XObj* pContext,
                    X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue) -> bool
                {
                    if (params.size() > 0) {
                        void* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(params[0].ToLongLong()));
                        if (ptr) {
                            const char* syncCpu = std::getenv("GARNET_TRT_SYNC_CPU_OUTPUTS");
                            if (syncCpu && syncCpu[0] == '0' && syncCpu[1] == '\0') {
                                cudaError_t err = cudaFreeAsync(ptr, cudaStreamPerThread);
                                if (err == cudaErrorNotSupported) {
                                    cudaGetLastError();
                                    cudaFree(ptr);
                                }
                            }
                            else {
                                cudaFree(ptr);
                            }
                        }
                    }
                    retValue = X::Value(true);
                    return true;
                };
                X::XFunc* freeObj = X::g_pXHost->CreateFunction("free", freeCallback, nullptr);
                X::Value freeFunc(freeObj, false);
                X::Dict ops;
                ops->Set("free", freeFunc);
                opsValue = X::Value(ops);
            }
            return opsValue;
        }

    public:
        // Ensure tensor has GPU memory allocated
        static TensorOpStatus EnsureGPUMemory(X::Tensor& tensor)
        {
            if (tensor->GetDeviceType() == X::TensorDeviceType::GPU) {
                return tensor->GetData() != nullptr ? TensorOpStatus::Success : TensorOpStatus::GeneralError;
            }
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
                    tensor->DirectSetData(static_cast<char*>(gpuMem), size);
                    tensor->SetDeviceType(X::TensorDeviceType::GPU);
                    tensor->SetDeviceContext(X::Value(static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(gpuMem))));
                    tensor->SetDeviceOps(CudaDeviceOps());
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
                tensor->DirectSetData(static_cast<char*>(gpuMem), size);
                tensor->SetDeviceType(X::TensorDeviceType::GPU);
                tensor->SetDeviceContext(X::Value(static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(gpuMem))));
                tensor->SetDeviceOps(CudaDeviceOps());
                tensor->SetDesc(X::Value(varDesc));
            }

            return TensorOpStatus::Success;
        }

        // Get GPU memory pointer from tensor
        static void* GetGPUMemory(X::Tensor& tensor)
        {
            if (tensor->GetDeviceType() == X::TensorDeviceType::GPU) {
                return tensor->GetData();
            }
            X::Value tensorDesc = tensor->GetDesc();
            if (tensorDesc.IsObject())
            {
                X::XPackageValue<TensorDescriptor> varDesc(tensorDesc);
                TensorDescriptor& desc = *varDesc;
                return desc.gpuMemory;
            }
            return nullptr;
        }

        static TensorOpStatus AttachGPUMemory(X::Tensor& tensor, void* gpuMemory, const std::string& deviceName = "cuda")
        {
            if (!gpuMemory)
            {
                return TensorOpStatus::InvalidDescriptor;
            }

            X::Value tensorDesc = tensor->GetDesc();
            long long bytes = tensor->GetDataSize();
            tensor->DirectSetData(static_cast<char*>(gpuMemory), bytes);
            tensor->SetDeviceType(X::TensorDeviceType::GPU);
            tensor->SetDeviceContext(X::Value(static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(gpuMemory))));
            tensor->SetDeviceOps(CudaDeviceOps());
            if (tensorDesc.IsObject())
            {
                X::XPackageValue<TensorDescriptor> varDesc(tensorDesc);
                TensorDescriptor& desc = *varDesc;
                desc.mDeviceName = deviceName;
                desc.gpuMemory = gpuMemory;
                tensor->SetDesc(X::Value(varDesc));
                return TensorOpStatus::Success;
            }

            X::XPackageValue<TensorDescriptor> varDesc;
            TensorDescriptor& desc = *varDesc;
            desc.mDeviceName = deviceName;
            desc.gpuMemory = gpuMemory;
            tensor->SetDesc(X::Value(varDesc));
            return TensorOpStatus::Success;
        }

        static TensorOpStatus AttachGPUMemoryRaw(X::XTensor* tensor, void* gpuMemory, const std::string& deviceName = "cuda")
        {
            if (!tensor || !gpuMemory)
            {
                return TensorOpStatus::InvalidDescriptor;
            }

            X::Value tensorDesc = tensor->GetDesc();
            long long bytes = tensor->GetDataSize();
            tensor->DirectSetData(static_cast<char*>(gpuMemory), bytes);
            tensor->SetDeviceType(X::TensorDeviceType::GPU);
            tensor->SetDeviceContext(X::Value(static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(gpuMemory))));
            tensor->SetDeviceOps(CudaDeviceOps());
            if (tensorDesc.IsObject())
            {
                X::XPackageValue<TensorDescriptor> varDesc(tensorDesc);
                TensorDescriptor& desc = *varDesc;
                desc.mDeviceName = deviceName;
                desc.gpuMemory = gpuMemory;
                X::Value descValue(varDesc);
                tensor->SetDesc(descValue);
                return TensorOpStatus::Success;
            }

            X::XPackageValue<TensorDescriptor> varDesc;
            TensorDescriptor& desc = *varDesc;
            desc.mDeviceName = deviceName;
            desc.gpuMemory = gpuMemory;
            X::Value descValue(varDesc);
            tensor->SetDesc(descValue);
            return TensorOpStatus::Success;
        }

        // Bind memory owned by a longer-lived runtime object (for example the
        // global paged KV pool). The empty device-ops dictionary tells XLang
        // that this is device memory while deliberately providing no free
        // callback; the owner remains responsible for its lifetime.
        static TensorOpStatus AttachBorrowedGPUMemory(
            X::Tensor& tensor,
            void* gpuMemory)
        {
            if (!tensor || !gpuMemory) return TensorOpStatus::InvalidDescriptor;
            const long long bytes = tensor->GetDataSize();
            tensor->DirectSetData(static_cast<char*>(gpuMemory), bytes);
            tensor->SetDeviceType(X::TensorDeviceType::GPU);
            tensor->SetDeviceContext(X::Value(
                static_cast<unsigned long long>(
                    reinterpret_cast<uintptr_t>(gpuMemory))));
            X::Dict borrowedDeviceOps;
            tensor->SetDeviceOps(X::Value(borrowedDeviceOps));
            return TensorOpStatus::Success;
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
            if (tensor->GetDeviceType() == X::TensorDeviceType::GPU) {
                void* gpuMemory = tensor->GetData();
                if (!gpuMemory) {
                    return TensorOpStatus::GeneralError;
                }

                long long totalSize = tensor->GetDataSize();
                std::vector<char> host(static_cast<size_t>(totalSize));
                cudaError_t err = cudaMemcpy(host.data(), gpuMemory, totalSize, cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) {
                    return TensorOpStatus::CudaCopyError;
                }
                tensor->DirectSetData(nullptr, 0);
                tensor->SetDeviceType(X::TensorDeviceType::CPU);
                tensor->SetDeviceContext(X::Value());
                tensor->SetDeviceOps(X::Value());
                tensor->SetData(host.data(), totalSize);
                cudaFree(gpuMemory);
                return TensorOpStatus::Success;
            }
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

        static X::Value CopyToCPUTensor(X::Tensor& tensor)
        {
            if (!tensor)
            {
                return X::Value();
            }

            if (tensor->GetDeviceType() == X::TensorDeviceType::CPU)
            {
                return X::Value(tensor);
            }

            void* gpuMemory = tensor->GetData();
            if (!gpuMemory)
            {
                return X::Value();
            }

            long long totalSize = tensor->GetDataSize();
            std::vector<char> host(static_cast<size_t>(totalSize));
            cudaStreamSynchronize(cudaStreamPerThread);
            cudaError_t err = cudaMemcpy(host.data(), gpuMemory, totalSize, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                return X::Value();
            }

            X::Tensor cpuTensor = X::g_pXHost->CreateTensor();
            if (!cpuTensor)
            {
                return X::Value();
            }
            X::Port::vector<int> shape(tensor->GetDimCount());
            for (int i = 0; i < tensor->GetDimCount(); ++i)
            {
                shape.push_back(static_cast<int>(tensor->GetDimSize(i)));
            }
            cpuTensor->SetDataType(tensor->GetDataType());
            cpuTensor->SetShape(shape);
            cpuTensor->SetData(host.data(), totalSize);
            return X::Value(cpuTensor);
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
