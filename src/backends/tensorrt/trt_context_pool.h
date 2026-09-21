// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace Garnet {
// Slots remain unavailable to a new invocation until all work using their
// execution context has completed. Pool saturation waits for one slot only.
class TRTContextPool {
    class DeviceScope {
        int previous_ = -1;
        bool changed_ = false;
    public:
        explicit DeviceScope(int device) noexcept {
            if (cudaGetDevice(&previous_) == cudaSuccess && previous_ != device)
                changed_ = cudaSetDevice(device) == cudaSuccess;
        }
        ~DeviceScope() { if (changed_) cudaSetDevice(previous_); }
    };
public:
    struct Slot {
        int device = -1;
        nvinfer1::IExecutionContext* context = nullptr;
        cudaEvent_t completion = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        std::vector<void*> bindings;
        bool warmed = false;
        bool busy = false;
        ~Slot() {
            DeviceScope deviceScope(device);
            if (completion) cudaEventSynchronize(completion);
            ResetGraph();
            delete context;
            if (completion) cudaEventDestroy(completion);
        }
        void ResetGraph() {
            DeviceScope deviceScope(device);
            if (executable) cudaGraphExecDestroy(executable);
            if (graph) cudaGraphDestroy(graph);
            executable = nullptr; graph = nullptr; warmed = false; bindings.clear();
        }
    };
    class Lease {
        TRTContextPool* pool_ = nullptr;
        Slot* slot_ = nullptr;
        cudaStream_t stream_ = nullptr;
        friend class TRTContextPool;
        Lease(TRTContextPool* pool, Slot* slot, cudaStream_t stream)
            : pool_(pool), slot_(slot), stream_(stream) {}
    public:
        Lease() = default;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept
            : pool_(other.pool_), slot_(other.slot_), stream_(other.stream_) { other.pool_ = nullptr; }
        ~Lease() {
            if (!pool_) return;
            DeviceScope deviceScope(slot_->device);
            if (cudaEventRecord(slot_->completion, stream_) != cudaSuccess)
                cudaStreamSynchronize(stream_);
            std::lock_guard<std::mutex> lock(pool_->mutex_);
            slot_->busy = false;
            pool_->available_.notify_one();
        }
        Slot* operator->() const { return slot_; }
        explicit operator bool() const { return slot_ != nullptr; }
    };
private:
    nvinfer1::ICudaEngine* engine_;
    const size_t capacity_;
    int device_ = -1;
    std::mutex mutex_;
    std::condition_variable available_;
    std::vector<std::unique_ptr<Slot>> slots_;
public:
    explicit TRTContextPool(nvinfer1::ICudaEngine* engine, size_t capacity = 8)
        : engine_(engine), capacity_(capacity) {
        if (!engine || !capacity) throw std::invalid_argument("invalid TensorRT context pool");
        if (cudaGetDevice(&device_) != cudaSuccess)
            throw std::runtime_error("TensorRT context pool device query failed");
    }
    Lease Acquire(cudaStream_t stream) {
        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess || device != device_)
            throw std::runtime_error("TensorRT context pool device mismatch");
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            Slot* pending = nullptr;
            for (auto& slot : slots_) {
                if (slot->busy) continue;
                const auto status = cudaEventQuery(slot->completion);
                if (status == cudaSuccess) {
                    slot->busy = true;
                    return Lease(this, slot.get(), stream);
                }
                if (status != cudaErrorNotReady) throw std::runtime_error("TensorRT completion query failed");
                pending = slot.get();
            }
            if (slots_.size() < capacity_) {
                auto slot = std::make_unique<Slot>();
                slot->device = device_;
                slot->context = engine_->createExecutionContext();
                if (!slot->context || cudaEventCreateWithFlags(&slot->completion,
                        cudaEventDisableTiming) != cudaSuccess)
                    throw std::runtime_error("TensorRT context slot creation failed");
                slot->busy = true;
                auto* result = slot.get();
                slots_.push_back(std::move(slot));
                return Lease(this, result, stream);
            }
            if (pending) {
                pending->busy = true;
                lock.unlock();
                if (cudaEventSynchronize(pending->completion) != cudaSuccess) {
                    lock.lock(); pending->busy = false; available_.notify_one();
                    throw std::runtime_error("TensorRT completion wait failed");
                }
                return Lease(this, pending, stream);
            }
            available_.wait(lock);
        }
    }
    size_t Size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return slots_.size();
    }
};
}
