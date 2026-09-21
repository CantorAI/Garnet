// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "tensor_helper.h"
#include <cstring>
#include <limits>
#include <stdexcept>

namespace Garnet {
namespace {
void Check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
class DeviceScope {
    int previous_ = 0;
public:
    explicit DeviceScope(int device) {
        Check(cudaGetDevice(&previous_));
        if (previous_ != device) Check(cudaSetDevice(device));
    }
    ~DeviceScope() { cudaSetDevice(previous_); }
};
struct DeviceAllocation {
    void* pointer = nullptr;
    int device = 0;
    bool owned = false;
    ~DeviceAllocation() {
        if (!owned || !pointer) return;
        int previous = 0;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        if (previous != device && cudaSetDevice(device) != cudaSuccess) return;
        cudaFree(pointer);
        if (previous != device) cudaSetDevice(previous);
    }
};
void RequireStorage(const X3TensorInfo& info) {
    if (info.symbolic || info.rank == UINT32_MAX ||
        (info.byte_size && !info.data))
        throw std::invalid_argument("materialized tensor storage is required");
    if (info.byte_size > SIZE_MAX) throw std::overflow_error("tensor storage exceeds address space");
}
struct CUDACompletion {
    cudaEvent_t event = nullptr;
    int device = 0;
    static X3Status Wait(void* pointer, const X3TensorExecution* execution) noexcept {
        auto& self = *static_cast<CUDACompletion*>(pointer);
        try {
            if (execution->device_type == TensorHelper::CudaDevice) {
                DeviceScope scope(execution->device_id);
                Check(cudaStreamWaitEvent(static_cast<cudaStream_t>(execution->stream), self.event, 0));
            } else {
                DeviceScope scope(self.device);
                Check(cudaEventSynchronize(self.event));
            }
            return X3_STATUS_OK;
        } catch (...) { return X3_STATUS_ERROR; }
    }
    static int32_t Query(void* pointer) noexcept {
        auto& self = *static_cast<CUDACompletion*>(pointer);
        try {
            DeviceScope scope(self.device);
            auto result = cudaEventQuery(self.event);
            return result == cudaSuccess ? 1 : result == cudaErrorNotReady ? 0 : -1;
        } catch (...) { return -1; }
    }
    ~CUDACompletion() {
        try {
            DeviceScope scope(device);
            if (event) {
                if (cudaEventQuery(event) != cudaSuccess) cudaEventSynchronize(event);
                cudaEventDestroy(event);
            }
        } catch (...) {}
    }
};
}

CUDAUse::CUDAUse(const X::Tensor& tensor, X3TensorAccess access, cudaStream_t stream)
    : CUDAUse(std::vector<std::pair<X::Tensor, X3TensorAccess>>{{tensor, access}}, stream) {}
CUDAUse::CUDAUse(const std::vector<std::pair<X::Tensor, X3TensorAccess>>& tensors, cudaStream_t stream)
    : stream_(stream), thread_(std::this_thread::get_id()) {
    Check(cudaGetDevice(&device_));
    for (const auto& entry : tensors) {
        const auto info = entry.first.Info();
        if (info.device_type == TensorHelper::CudaDevice && info.device_id != device_)
            throw std::invalid_argument("tensor CUDA device differs from execution device");
    }
    X3TensorExecution execution{};
    execution.size = sizeof(execution); execution.device_type = TensorHelper::CudaDevice;
    execution.device_id = device_; execution.stream = stream;
    use_ = X::Tensor::AcquireMany(tensors, &execution);
}
void CUDAUse::Finish() {
    if (!use_) return;
    if (stream_ == cudaStreamPerThread && thread_ != std::this_thread::get_id())
        throw std::logic_error("per-thread CUDA use must finish on its enqueue thread; use an explicit stream for handoff");
    DeviceScope scope(device_);
    auto completion = std::make_unique<CUDACompletion>();
    completion->device = device_;
    Check(cudaEventCreateWithFlags(&completion->event, cudaEventDisableTiming));
    Check(cudaEventRecord(completion->event, stream_));
    X3TensorCompletion callback{}; callback.size = sizeof(callback);
    callback.context = completion.get(); callback.wait = CUDACompletion::Wait;
    callback.query = CUDACompletion::Query;
    callback.cleanup = [](void* pointer) { delete static_cast<CUDACompletion*>(pointer); };
    use_.Finish(&callback);
    completion.release();
}
CUDAUse::~CUDAUse() {
    if (!use_) return;
    try { Finish(); }
    catch (...) {
        // Recording can fail after work was queued. Retire that work before
        // releasing the synchronous lease.
        try {
            DeviceScope scope(device_);
            // Contract violation only: CUDA exposes no transferable handle for
            // another thread's sentinel stream. Do not retire storage early.
            if (stream_ == cudaStreamPerThread && thread_ != std::this_thread::get_id()) cudaDeviceSynchronize();
            else cudaStreamSynchronize(stream_);
        } catch (...) {}
    }
}

uint64_t TensorHelper::ItemSize(X3TensorDType dtype) {
    switch (dtype) {
        case X3_TENSOR_UINT8: return 1;
        case X3_TENSOR_FLOAT8_E4M3FN: case X3_TENSOR_FLOAT8_E4M3FNUZ:
        case X3_TENSOR_FLOAT8_E5M2: case X3_TENSOR_FLOAT8_E5M2FNUZ: return 1;
        case X3_TENSOR_FLOAT16: case X3_TENSOR_BFLOAT16: case X3_TENSOR_UINT16: return 2;
        case X3_TENSOR_FLOAT32: case X3_TENSOR_INT32: return 4;
        case X3_TENSOR_FLOAT64: case X3_TENSOR_INT64: return 8;
        default: throw std::invalid_argument("unsupported Garnet tensor dtype");
    }
}

X::Tensor TensorHelper::WrapGPU(X3PackageHost* host, X3TensorInfo info,
    void* allocation, int deviceId) {
    if (info.data != allocation)
        throw std::invalid_argument("owned CUDA storage must start at its allocation");
    auto owner = std::make_unique<DeviceAllocation>();
    owner->pointer = allocation;
    owner->device = deviceId;
    info.device_type = CudaDevice;
    info.device_id = deviceId;
    info.symbolic = 0;
    auto tensor = X::Tensor::Wrap(host, info, owner.get(),
        [](void* pointer) { delete static_cast<DeviceAllocation*>(pointer); });
    owner->owned = true;
    owner.release();
    return tensor;
}

X::Tensor TensorHelper::WrapBorrowedGPU(X3PackageHost* host, X3TensorInfo info,
    std::shared_ptr<void> owner) {
    if (!owner) throw std::invalid_argument("borrowed CUDA storage requires a lifetime owner");
    auto retained = std::make_unique<std::shared_ptr<void>>(std::move(owner));
    info.device_type = CudaDevice;
    info.symbolic = 0;
    auto tensor = X::Tensor::Wrap(host, info, retained.get(),
        [](void* pointer) { delete static_cast<std::shared_ptr<void>*>(pointer); });
    retained.release();
    return tensor;
}

X::Tensor TensorHelper::CreateGPU(X3PackageHost* host, X3TensorDType dtype,
    const std::vector<int64_t>& shape, const void* hostData) {
    if (shape.size() > 32) throw std::invalid_argument("tensor rank exceeds 32");
    uint64_t bytes = ItemSize(dtype);
    std::vector<int64_t> strides(shape.size());
    for (size_t index = shape.size(); index-- > 0;) {
        if (shape[index] < 0 || bytes > INT64_MAX ||
            (shape[index] && bytes > SIZE_MAX / static_cast<uint64_t>(shape[index])))
            throw std::overflow_error("invalid or overflowing tensor shape");
        strides[index] = static_cast<int64_t>(bytes);
        bytes *= static_cast<uint64_t>(shape[index]);
    }
    int device = 0;
    Check(cudaGetDevice(&device));
    void* memory = nullptr;
    if (bytes) Check(cudaMalloc(&memory, static_cast<size_t>(bytes)));
    try {
        if (bytes && hostData) {
            Check(cudaMemcpyAsync(memory, hostData, static_cast<size_t>(bytes), cudaMemcpyHostToDevice, cudaStreamPerThread));
            // The caller owns hostData and may release it immediately on return.
            Check(cudaStreamSynchronize(cudaStreamPerThread));
        }
        X3TensorInfo info{};
        info.size = sizeof(info); info.dtype = dtype;
        info.rank = static_cast<uint32_t>(shape.size());
        info.shape = shape.data(); info.strides = strides.data();
        info.data = memory; info.byte_size = bytes;
        auto tensor = WrapGPU(host, info, memory, device);
        memory = nullptr;
        if (bytes && !hostData) {
            auto use = AcquireGPU(tensor, X3_TENSOR_WRITE);
            Check(cudaMemsetAsync(info.data, 0, static_cast<size_t>(bytes), cudaStreamPerThread));
            use.Finish();
        }
        return tensor;
    } catch (...) {
        if (memory) cudaFree(memory);
        throw;
    }
}

X::Tensor TensorHelper::CopyToCPU(const X::Tensor& tensor) {
    auto info = tensor.Info();
    RequireStorage(info);
    auto use = tensor.Acquire();
    if (info.device_type == 0) return tensor;
    if (info.device_type != CudaDevice) throw std::invalid_argument("unknown tensor device");
    DeviceScope device(info.device_id);
    std::unique_ptr<unsigned char[]> memory(new unsigned char[static_cast<size_t>(info.byte_size)]);
    if (info.byte_size)
        Check(cudaMemcpy(memory.get(), info.data, static_cast<size_t>(info.byte_size), cudaMemcpyDeviceToHost));
    info.data = memory.get(); info.device_type = 0; info.device_id = 0;
    auto output = X::Tensor::Wrap(tensor.host(), info, memory.get(),
        [](void* pointer) { delete[] static_cast<unsigned char*>(pointer); });
    memory.release();
    return output;
}

X::Tensor TensorHelper::CopyToGPU(const X::Tensor& tensor) {
    auto info = tensor.Info();
    RequireStorage(info);
    if (info.device_type == CudaDevice) return tensor;
    if (info.device_type != 0) throw std::invalid_argument("unknown tensor device");
    auto use = tensor.Acquire();
    int device = 0;
    Check(cudaGetDevice(&device));
    void* memory = nullptr;
    if (info.byte_size) Check(cudaMalloc(&memory, static_cast<size_t>(info.byte_size)));
    try {
        if (info.byte_size) {
            Check(cudaMemcpyAsync(memory, info.data, static_cast<size_t>(info.byte_size), cudaMemcpyHostToDevice, cudaStreamPerThread));
            Check(cudaStreamSynchronize(cudaStreamPerThread));
        }
        info.data = memory;
        return WrapGPU(tensor.host(), info, memory, device);
    } catch (...) {
        if (memory) cudaFree(memory);
        throw;
    }
}

void* TensorHelper::GetGPUMemory(const X::Tensor& tensor) {
    auto info = tensor.Info();
    return info.device_type == CudaDevice ? info.data : nullptr;
}
std::string TensorHelper::GetDeviceName(const X::Tensor& tensor) {
    auto info = tensor.Info();
    if (info.device_type == 0) return "cpu";
    if (info.device_type == CudaDevice) return "cuda";
    throw std::invalid_argument("unknown tensor device");
}
TensorOpStatus TensorHelper::EnsureGPUMemory(X::Tensor& tensor) {
    try { tensor = CopyToGPU(tensor); return TensorOpStatus::Success; }
    catch (...) { return TensorOpStatus::GeneralError; }
}
TensorOpStatus TensorHelper::CopyResultFromGPU(X::Tensor& tensor) {
    try { tensor = CopyToCPU(tensor); return TensorOpStatus::Success; }
    catch (...) { return TensorOpStatus::GeneralError; }
}
}
