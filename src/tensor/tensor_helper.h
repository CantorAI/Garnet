#pragma once
#include "xlang3/xlang3.h"
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include <thread>

namespace Garnet {

enum class TensorOpStatus {
    Success, InvalidDescriptor, CudaAllocationError, CudaCopyError, GeneralError
};

class CUDAUse {
    X::TensorUse use_;
    cudaStream_t stream_ = nullptr;
    int device_ = 0;
    std::thread::id thread_;
public:
    // Explicit CUDA streams may move between threads. The per-thread sentinel
    // must be finished on the acquiring/enqueueing thread.
    CUDAUse(const X::Tensor&, X3TensorAccess, cudaStream_t);
    CUDAUse(const std::vector<std::pair<X::Tensor, X3TensorAccess>>&, cudaStream_t);
    CUDAUse(const CUDAUse&) = delete;
    CUDAUse& operator=(const CUDAUse&) = delete;
    CUDAUse(CUDAUse&&) noexcept = default;
    CUDAUse& operator=(CUDAUse&&) = delete;
    ~CUDAUse();
    void Finish();
};

class TensorHelper {
public:
    static constexpr int CudaDevice = 1;
    static CUDAUse AcquireGPU(const X::Tensor& tensor, X3TensorAccess access = X3_TENSOR_READ,
        cudaStream_t stream = cudaStreamPerThread) { return CUDAUse(tensor, access, stream); }
    static CUDAUse AcquireGPU(const std::vector<std::pair<X::Tensor, X3TensorAccess>>& tensors,
        cudaStream_t stream = cudaStreamPerThread) { return CUDAUse(tensors, stream); }
    static uint64_t ItemSize(X3TensorDType dtype);
    static X::Tensor CreateGPU(X3PackageHost* host, X3TensorDType dtype,
        const std::vector<int64_t>& shape, const void* hostData = nullptr);
    // Ownership transfers only on success; the allocation must be cudaMalloc
    // memory on deviceId and cover the span described by info.
    static X::Tensor WrapGPU(X3PackageHost* host, X3TensorInfo info,
        void* allocation, int deviceId);
    // The owner must keep this allocation alive until the final tensor view dies.
    static X::Tensor WrapBorrowedGPU(X3PackageHost* host, X3TensorInfo info,
        std::shared_ptr<void> owner);
    static X::Tensor CopyToCPU(const X::Tensor& tensor);
    static X::Tensor CopyToGPU(const X::Tensor& tensor);
    static void* GetGPUMemory(const X::Tensor& tensor);
    static std::string GetDeviceName(const X::Tensor& tensor);
    static TensorOpStatus EnsureGPUMemory(X::Tensor& tensor);
    static TensorOpStatus CopyResultFromGPU(X::Tensor& tensor);
    static X::Value CopyToCPUTensor(const X::Tensor& tensor) { return CopyToCPU(tensor); }
    // Release this reference, never forcibly free storage retained elsewhere.
    static void ReleaseGPUMemory(X::Tensor& tensor) { tensor = X::Tensor(); }
};

}
