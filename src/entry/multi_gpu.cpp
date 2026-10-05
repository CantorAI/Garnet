// SPDX-License-Identifier: Apache-2.0
#include "garnet.h"
#include "native_values.h"
#include "tensor_helper.h"
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <limits>

namespace Garnet {
namespace {
void Check(cudaError_t status) {
    if (status != cudaSuccess) throw X::Error(cudaGetErrorString(status));
}
int Device(const X::Value& value) {
    const auto id = CheckedInt64(value, "device_id");
    int count = 0; Check(cudaGetDeviceCount(&count));
    if (id < 0 || id >= count) throw X::Error("CUDA device index out of range");
    return static_cast<int>(id);
}
struct DeviceScope {
    int previous;
    explicit DeviceScope(int id) { Check(cudaGetDevice(&previous)); Check(cudaSetDevice(id)); }
    ~DeviceScope() { cudaSetDevice(previous); }
};
}

X::Value GarnetAPI::CudaDevicesJson(const X::ARGS&, const X::KWARGS&) {
    int count = 0; Check(cudaGetDeviceCount(&count));
    nlohmann::json result = nlohmann::json::array();
    for (int id = 0; id < count; ++id) {
        DeviceScope scope(id);
        cudaDeviceProp p{}; Check(cudaGetDeviceProperties(&p, id));
        size_t free = 0, total = 0; Check(cudaMemGetInfo(&free, &total));
        nlohmann::json peers = nlohmann::json::array();
        for (int other = 0; other < count; ++other) {
            int capable = 0;
            if (other != id) Check(cudaDeviceCanAccessPeer(&capable, id, other));
            peers.push_back(capable != 0);
        }
        char pci[32]{}; Check(cudaDeviceGetPCIBusId(pci, sizeof(pci), id));
        result.push_back({{"id", id}, {"name", p.name}, {"total_bytes", total},
            {"free_bytes", free}, {"compute_major", p.major}, {"compute_minor", p.minor},
            {"pci_bus_id", pci}, {"peer_access", peers}});
    }
    return NativeValue(Host(), result.dump());
}
X::Value GarnetAPI::CudaSetDevice(const X::ARGS& args, const X::KWARGS&) {
    if (args.size() != 1) throw X::Error("cuda_set_device(device_id) expected");
    int previous = 0; Check(cudaGetDevice(&previous));
    Check(cudaSetDevice(Device(args[0])));
    return NativeValue(Host(), previous);
}
X::Value GarnetAPI::CudaSynchronize(const X::ARGS&, const X::KWARGS&) {
    // A stage completion fence; other devices and streams remain independent.
    Check(cudaStreamSynchronize(cudaStreamPerThread));
    return NativeValue(Host(), true);
}
X::Value GarnetAPI::TensorToDevice(const X::ARGS& args, const X::KWARGS&) {
    if (args.size() != 2 || !X::Tensor::IsTensor(args[0]))
        throw X::Error("tensor_to_device(tensor, device_id) expected");
    X::Tensor source(args[0]);
    const auto info = source.Info();
    if (info.device_type == TensorHelper::CudaDevice) {
        // Ordinary operator validation assumes the source device is selected;
        // a transfer deliberately has a different destination selected.
        DeviceScope sourceScope(info.device_id);
        ValidateDenseTensor(source);
    } else {
        ValidateDenseTensor(source);
    }
    const int destination = Device(args[1]);
    if (info.device_type == TensorHelper::CudaDevice && info.device_id == destination) return source;
    if (info.device_type != 0 && info.device_type != TensorHelper::CudaDevice)
        throw X::Error("unsupported source tensor device");
    // Acquire waits for the source producer's completion, including another GPU.
    auto read = source.Acquire();
    std::vector<int64_t> shape(info.shape, info.shape + info.rank);
    uint64_t expected = TensorHelper::ItemSize(info.dtype);
    for (size_t i = shape.size(); i-- > 0;) {
        if (shape[i] < 0 || (shape[i] && expected > SIZE_MAX / static_cast<uint64_t>(shape[i])))
            throw X::Error("invalid transfer shape");
        if (info.strides && info.strides[i] != static_cast<int64_t>(expected))
            throw X::Error("pipeline transfer requires a contiguous tensor");
        expected *= shape[i];
    }
    if (expected > info.byte_size) throw X::Error("transfer storage is too small");
    DeviceScope scope(destination);
    auto output = TensorHelper::CreateGPU(Host(), info.dtype, shape);
    auto write = output.Acquire(X3_TENSOR_WRITE);
    if (expected) {
        if (info.device_type == 0) {
            Check(cudaMemcpy(output.Info().data, info.data, expected, cudaMemcpyHostToDevice));
        } else {
            int peer = 0; Check(cudaDeviceCanAccessPeer(&peer, destination, info.device_id));
            if (peer) {
                auto enabled = cudaDeviceEnablePeerAccess(info.device_id, 0);
                if (enabled == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
                else Check(enabled);
                Check(cudaMemcpyPeer(output.Info().data, destination, info.data, info.device_id, expected));
            } else {
                // RTX configurations can lack P2P. Preserve correctness via host RAM.
                auto cpu = TensorHelper::CopyToCPU(source);
                Check(cudaMemcpy(output.Info().data, cpu.Info().data, expected, cudaMemcpyHostToDevice));
            }
        }
    }
    return output;
}
X::Value GarnetAPI::TensorZeros(const X::ARGS& args, const X::KWARGS&) {
    if (args.size() != 2 || !args[0].IsList()) throw X::Error("tensor_zeros(shape, dtype) expected");
    std::vector<int64_t> shape;
    for (long long i = 0; i < args[0].Size(); ++i)
        shape.push_back(CheckedInt64(args[0].Get(i), "tensor dimension"));
    const auto dtype = args[1].ToString();
    X3TensorDType type;
    if (dtype == "bfloat16") type = X3_TENSOR_BFLOAT16;
    else if (dtype == "float32") type = X3_TENSOR_FLOAT32;
    else if (dtype == "int32") type = X3_TENSOR_INT32;
    else if (dtype == "int64") type = X3_TENSOR_INT64;
    else throw X::Error("unsupported tensor_zeros dtype");
    return TensorHelper::CreateGPU(Host(), type, shape);
}
}
