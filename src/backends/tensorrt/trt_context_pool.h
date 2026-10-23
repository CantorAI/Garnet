// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <atomic>
#include <array>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <thread>
#include "trt_memory_profile.h"
#include "trt_sync_allocator.h"

namespace Garnet {
class OperatorExecutionOwner;
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
        // Diagnostic opt-in only. Keep this alive through context destruction.
        std::unique_ptr<TRTSyncAllocator> temporaryAllocator;
        nvinfer1::IExecutionContext* context = nullptr;
        cudaEvent_t completion = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        std::vector<void*> bindings;
        std::vector<int64_t> shapeSignature;
        int activeOptimizationProfile = -1;
        bool warmed = false;
        bool busy = false;
        cudaStream_t lastStream=nullptr;
        std::thread::id lastThread;
        std::atomic<uint64_t> graphCaptures{0},graphReplays{0},graphCaptureFailures{0};
        std::atomic<uint64_t> serialGraphReuses{0},pendingGraphReuses{0},warmupWaits{0};
        std::shared_ptr<OperatorExecutionOwner> operatorOwner;
        int operatorPhase=-1,operatorRank=-1;
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
            executable = nullptr; graph = nullptr; warmed = false; bindings.clear(); shapeSignature.clear();
            // Captured pointers retire before the last provider buffer owner.
            operatorOwner.reset();operatorPhase=-1;operatorRank=-1;
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
            slot_->lastStream=stream_;slot_->lastThread=std::this_thread::get_id();
            slot_->busy = false;
            pool_->available_.notify_one();
        }
        Slot* operator->() const { return slot_; }
        explicit operator bool() const { return slot_ != nullptr; }
    };
private:
    nvinfer1::ICudaEngine* engine_;
    const size_t capacity_;
    const std::string enginePath_;
    int device_ = -1;
    std::mutex mutex_;
    std::condition_variable available_;
    std::vector<std::unique_ptr<Slot>> slots_;
public:
    explicit TRTContextPool(nvinfer1::ICudaEngine* engine, size_t capacity = 8,
        std::string enginePath = {})
        : engine_(engine), capacity_(capacity), enginePath_(std::move(enginePath)) {
        if (!engine || !capacity) throw std::invalid_argument("invalid TensorRT context pool");
        if (cudaGetDevice(&device_) != cudaSuccess)
            throw std::runtime_error("TensorRT context pool device query failed");
    }
    Lease Acquire(cudaStream_t stream,const std::vector<void*>* serialGraphBindings=nullptr) {
        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess || device != device_)
            throw std::runtime_error("TensorRT context pool device mismatch");
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            Slot* pending = nullptr;
            Slot* warmup=nullptr;
            auto matches=[&](const Slot& slot){return serialGraphBindings&&!slot.busy&&!slot.operatorOwner&&slot.warmed&&slot.bindings==*serialGraphBindings&&
                slot.shapeSignature.empty()&&slot.lastStream==stream&&(stream!=cudaStreamPerThread||slot.lastThread==std::this_thread::get_id());};
            // Only an already-instantiated graph with identical bindings can
            // be queued before GPU retirement. No enqueueV3, graph update,
            // profile change or provider-owner change is permitted here.
            for(auto& slot:slots_)if(matches(*slot)&&slot->executable){
                auto status=cudaEventQuery(slot->completion);
                if(status!=cudaSuccess&&status!=cudaErrorNotReady)throw std::runtime_error("TensorRT serial graph completion query failed");
                ++slot->serialGraphReuses;if(status==cudaErrorNotReady)++slot->pendingGraphReuses;
                slot->busy=true;return Lease(this,slot.get(),stream);
            }
            for (auto& slot : slots_) {
                if (slot->busy) continue;
                const auto status = cudaEventQuery(slot->completion);
                if (status == cudaSuccess) {
                    slot->busy = true;
                    return Lease(this, slot.get(), stream);
                }
                if (status != cudaErrorNotReady) throw std::runtime_error("TensorRT completion query failed");
                pending = slot.get();
                if(matches(*slot)&&!slot->executable)warmup=slot.get();
            }
            // One initial warm enqueue must retire before capture. Waiting
            // for that one slot prevents an asynchronous prefill from growing
            // a pool of duplicate warm/capture contexts.
            if(warmup){
                warmup->busy=true;++warmup->warmupWaits;lock.unlock();
                if(cudaEventSynchronize(warmup->completion)!=cudaSuccess){lock.lock();warmup->busy=false;available_.notify_one();throw std::runtime_error("TensorRT serial graph warmup wait failed");}
                return Lease(this,warmup,stream);
            }
            if (slots_.size() < capacity_) {
                auto slot = std::make_unique<Slot>();
                slot->device = device_;
                LogTRTMemoryCheckpoint("before_context_create",enginePath_,
                    engine_->getDeviceMemorySizeV2(),slots_.size());
                slot->context = engine_->createExecutionContext();
                if (!slot->context || cudaEventCreateWithFlags(&slot->completion,
                        cudaEventDisableTiming) != cudaSuccess)
                    throw std::runtime_error("TensorRT context slot creation failed");
                if(const char* policy=std::getenv("GARNET_TRT_SYNC_TEMPORARY_ALLOCATOR");policy&&*policy=='1'){
                    slot->temporaryAllocator=std::make_unique<TRTSyncAllocator>(device_);
                    if(!slot->context->setTemporaryStorageAllocator(slot->temporaryAllocator.get()))
                        throw std::runtime_error("TensorRT temporary allocator diagnostic setup failed");
                }
                slot->busy = true;
                auto* result = slot.get();
                slots_.push_back(std::move(slot));
                LogTRTMemoryCheckpoint("after_context_create",enginePath_,
                    engine_->getDeviceMemorySizeV2(),slots_.size());
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
    std::array<uint64_t,6> GraphCounts() {
        std::lock_guard<std::mutex> lock(mutex_);std::array<uint64_t,6> counts{};
        for(auto& slot:slots_){counts[0]+=slot->graphCaptures.load();counts[1]+=slot->graphReplays.load();counts[2]+=slot->graphCaptureFailures.load();
            counts[3]+=slot->serialGraphReuses.load();counts[4]+=slot->pendingGraphReuses.load();counts[5]+=slot->warmupWaits.load();}
        return counts;
    }
};
}
