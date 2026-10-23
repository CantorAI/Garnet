// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>

namespace Garnet {
// A cold runtime allocation policy. The owning runtime/engines/contexts must
// die before this object. Captured enqueue never calls this allocator.
class TRTSyncAllocator final : public nvinfer1::IGpuAllocator {
    const int device_;
    std::atomic<size_t> outstanding_{0};
    mutable std::mutex accountingMutex_;
    std::unordered_map<void*,size_t> sizes_;
    size_t liveBytes_=0,peakBytes_=0;
    class DeviceScope {
        int previous_ = -1;
        bool changed_ = false;
        bool valid_ = false;
    public:
        explicit DeviceScope(int device) noexcept {
            if (device < 0 || cudaGetDevice(&previous_) != cudaSuccess) return;
            if (previous_ != device) {
                if (cudaSetDevice(device) != cudaSuccess) return;
                changed_ = true;
            }
            valid_ = true;
        }
        ~DeviceScope() { if (changed_) cudaSetDevice(previous_); }
        explicit operator bool() const { return valid_; }
    };
    void* Allocate(uint64_t size, uint64_t alignment,
                   nvinfer1::AllocatorFlags flags) noexcept {
        if (!size || size > std::numeric_limits<size_t>::max() || flags ||
            (alignment && (alignment & (alignment - 1)))) return nullptr;
        DeviceScope device(device_);
        if (!device) return nullptr;
        void* memory = nullptr;
        if (cudaMalloc(&memory, static_cast<size_t>(size)) != cudaSuccess) return nullptr;
        if (alignment && reinterpret_cast<uintptr_t>(memory) % alignment) {
            cudaFree(memory);
            return nullptr;
        }
        try {
            std::lock_guard<std::mutex> lock(accountingMutex_);
            sizes_.emplace(memory,size);liveBytes_+=size;
            if(liveBytes_>peakBytes_)peakBytes_=liveBytes_;
            ++outstanding_;
        }catch(...){cudaFree(memory);return nullptr;}
        if(const char* trace=std::getenv("GARNET_TRT_ALLOC_TRACE");trace&&*trace=='1')
            std::fprintf(stderr,"TRT_ALLOC_SYNC allocator=%p pointer=%p bytes=%llu\n",static_cast<void*>(this),memory,static_cast<unsigned long long>(size));
        return memory;
    }
    bool Free(void* memory, cudaStream_t stream, bool wait) noexcept {
        if (!memory) return true;
        DeviceScope device(device_);
        if (!device) return false;
        // TensorRT may use a private stream. Reach its current position before
        // retiring storage, even when it requests asynchronous deallocation.
        if (wait && cudaStreamSynchronize(stream) != cudaSuccess) return false;
        std::lock_guard<std::mutex> lock(accountingMutex_);
        auto found=sizes_.find(memory);if(found==sizes_.end())return false;
        if(const char* trace=std::getenv("GARNET_TRT_ALLOC_TRACE");trace&&*trace=='1')
            std::fprintf(stderr,"TRT_FREE_SYNC allocator=%p pointer=%p bytes=%llu\n",static_cast<void*>(this),memory,static_cast<unsigned long long>(found->second));
        if (cudaFree(memory) != cudaSuccess) return false;
        liveBytes_-=found->second;sizes_.erase(found);--outstanding_;
        return true;
    }
public:
    explicit TRTSyncAllocator(int device) noexcept : device_(device) {}
    size_t Outstanding() const noexcept { return outstanding_.load(); }
    size_t LiveBytes() const {std::lock_guard<std::mutex> lock(accountingMutex_);return liveBytes_;}
    size_t PeakBytes() const {std::lock_guard<std::mutex> lock(accountingMutex_);return peakBytes_;}
    void* allocate(uint64_t size, uint64_t alignment,
                   nvinfer1::AllocatorFlags flags) noexcept override {
        return Allocate(size, alignment, flags);
    }
    void* allocateAsync(uint64_t size, uint64_t alignment,
                        nvinfer1::AllocatorFlags flags, cudaStream_t) noexcept override {
        return Allocate(size, alignment, flags);
    }
    bool deallocate(void* memory) noexcept override { return Free(memory, nullptr, false); }
    bool deallocateAsync(void* memory, cudaStream_t stream) noexcept override {
        return Free(memory, stream, true);
    }
    // Refusing resize leaves the original allocation valid, as required by
    // IGpuAllocator. Flags are currently reserved and must be zero.
    void* reallocate(void*, uint64_t, uint64_t) noexcept override { return nullptr; }
};
}
