#include "tensor_helper.h"
#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>

namespace {
void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { Check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
struct Gate {
    cudaStream_t stream;
    std::promise<void> started;
    std::promise<void> release;
    std::shared_future<void> ready = release.get_future().share();
    explicit Gate(cudaStream_t value) : stream(value) {}
    ~Gate() { try { release.set_value(); } catch (...) {} cudaStreamSynchronize(stream); }
    static void CUDART_CB Wait(void* pointer) {
        auto& gate = *static_cast<Gate*>(pointer);
        gate.started.set_value();
        gate.ready.wait();
    }
};
void Delayed(X::Runtime& runtime, bool gpuConsumer) {
    Stream producerStream, consumerStream;
    auto source = Garnet::TensorHelper::CreateGPU(runtime.host(), X3_TENSOR_INT32, {8});
    auto view = source.View({4}, {4}, 16);
    auto output = Garnet::TensorHelper::CreateGPU(runtime.host(), X3_TENSOR_INT32, {4});
    Gate gate(producerStream.value);
    void* pointer = source.Info().data;
    std::exception_ptr failure;
    std::thread producer([&] {
        try {
            Check(cudaSetDevice(0));
            auto write = Garnet::TensorHelper::AcquireGPU(source, X3_TENSOR_WRITE, producerStream.value);
            Check(cudaLaunchHostFunc(producerStream.value, Gate::Wait, &gate));
            Check(cudaMemsetAsync(pointer, 0x2a, 32, producerStream.value));
            write.Finish();
        } catch (...) { failure = std::current_exception(); }
    });
    producer.join();
    if (failure) { gate.release.set_value(); std::rethrow_exception(failure); }
    gate.started.get_future().wait();
    if (gpuConsumer) {
        auto use = Garnet::TensorHelper::AcquireGPU(
            {{view, X3_TENSOR_READ}, {source, X3_TENSOR_READ}, {output, X3_TENSOR_WRITE}}, consumerStream.value);
        Check(cudaMemcpyAsync(output.Info().data, view.Info().data, 16,
            cudaMemcpyDeviceToDevice, consumerStream.value));
        use.Finish();
        // Acquiring/ending GPU use must enqueue a wait, not synchronize producer.
        Require(cudaStreamQuery(consumerStream.value) == cudaErrorNotReady, "GPU use unexpectedly blocked producer");
    }
    std::thread release([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        gate.release.set_value();
    });
    X::Tensor cpu;
    try { cpu = Garnet::TensorHelper::CopyToCPU(gpuConsumer ? output : view); }
    catch (...) { failure = std::current_exception(); }
    release.join();
    if (failure) std::rethrow_exception(failure);
    auto read = cpu.Acquire();
    auto* values = static_cast<int32_t*>(cpu.Info().data);
    for (int i = 0; i != 4; ++i) Require(values[i] == 0x2a2a2a2a, "delayed producer result mismatch");
}
void Lifetime(X::Runtime& runtime) {
    Stream stream;
    std::atomic<bool> freed{false};
    void* memory = nullptr;
    Check(cudaMalloc(&memory, 16));
    auto owner = std::shared_ptr<void>(memory, [&](void* pointer) { cudaFree(pointer); freed = true; });
    int64_t shape = 4;
    X3TensorInfo info{}; info.size = sizeof(info); info.dtype = X3_TENSOR_INT32;
    info.rank = 1; info.shape = &shape; info.data = memory; info.byte_size = 16; info.device_id = 0;
    auto tensor = Garnet::TensorHelper::WrapBorrowedGPU(runtime.host(), info, owner);
    auto alias = tensor.View({2}, {4}, 4);
    auto other = Garnet::TensorHelper::CreateGPU(runtime.host(), X3_TENSOR_INT32, {4});
    Gate gate(stream.value);
    owner.reset();
    auto use = Garnet::TensorHelper::AcquireGPU({{tensor, X3_TENSOR_READ}, {alias, X3_TENSOR_WRITE},
        {other, X3_TENSOR_READ}}, stream.value);
    Check(cudaLaunchHostFunc(stream.value, Gate::Wait, &gate));
    Check(cudaMemsetAsync(memory, 0, 16, stream.value));
    use.Finish();
    gate.started.get_future().wait();
    tensor = X::Tensor();
    std::promise<void> dropping;
    auto droppingFuture = dropping.get_future();
    std::thread releaser([alias = std::move(alias), &dropping]() mutable {
        dropping.set_value(); alias = X::Tensor();
    });
    droppingFuture.wait();
    const bool freedEarly = freed;
    gate.release.set_value();
    releaser.join();
    Require(!freedEarly, "storage freed before pending alias use completes");
    Require(freed, "borrowed storage owner not released");
}
}
int main() {
    try {
        Check(cudaSetDevice(0));
        X::Runtime runtime;
        Delayed(runtime, false);
        Delayed(runtime, true);
        Lifetime(runtime);
        std::cout << "tensor-readiness-cuda-passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
