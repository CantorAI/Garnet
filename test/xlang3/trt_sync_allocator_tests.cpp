// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "trt_sync_allocator.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void Require(bool condition, const char* text) {
    if (!condition) throw std::runtime_error(text);
}
void CUDART_CB DelayedMarker(void* pointer) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    static_cast<std::atomic<bool>*>(pointer)->store(true);
}
}
int main() {
    try {
        int devices = 0;
        Check(cudaGetDeviceCount(&devices));
        Require(devices > 0, "no CUDA device");
        Garnet::TRTSyncAllocator allocator(0), invalid(-1);
        Require(!allocator.allocateAsync(0, 256, 0, nullptr), "zero allocation");
        Require(!allocator.allocateAsync(128, 3, 0, nullptr), "invalid alignment");
        Require(!allocator.allocateAsync(128, 256, 2, nullptr), "reserved flags");
        Require(!invalid.allocateAsync(128, 256, 0, nullptr), "invalid device");
        Require(allocator.deallocateAsync(nullptr, nullptr), "null free");
        std::atomic<int> failures{0};
        std::vector<std::thread> workers;
        for (int worker = 0; worker < 8; ++worker) workers.emplace_back([&, worker] {
            try {
                const int previous = devices > 1 ? 1 : 0;
                Check(cudaSetDevice(previous));
                for (int run = 0; run < 24; ++run) {
                    void* memory = allocator.allocateAsync(4096, run % 2 ? 256 : 0, 0, nullptr);
                    Require(memory != nullptr, "allocation failed");
                    int current = -1;
                    Check(cudaGetDevice(&current));
                    Require(current == previous, "allocation changed caller device");
                    cudaPointerAttributes attributes{};
                    Check(cudaPointerGetAttributes(&attributes, memory));
                    Require(attributes.device == 0, "allocation on wrong device");
                    Require(!allocator.reallocate(memory, 256, 8192), "resize must be refused");
                    Check(cudaSetDevice(0));
                    cudaStream_t stream = nullptr;
                    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
                    const int marker = (worker + run) % 255;
                    Check(cudaMemsetAsync(memory, marker, 4096, stream));
                    std::vector<unsigned char> observed(4096);
                    Check(cudaMemcpyAsync(observed.data(), memory, observed.size(), cudaMemcpyDeviceToHost, stream));
                    std::atomic<bool> retired{false};
                    Check(cudaLaunchHostFunc(stream, DelayedMarker, &retired));
                    Check(cudaSetDevice(previous));
                    Require(allocator.deallocateAsync(memory, stream), "stream retirement failed");
                    Require(retired.load(), "returned before stream position");
                    Check(cudaGetDevice(&current));
                    Require(current == previous, "free changed caller device");
                    for (auto value : observed) Require(value == marker, "resize/copy corrupted allocation");
                    Check(cudaSetDevice(0));
                    Check(cudaStreamDestroy(stream));
                }
            } catch (const std::exception& error) {
                ++failures;
                std::cerr << error.what() << '\n';
            }
        });
        for (auto& worker : workers) worker.join();
        Require(!failures && !allocator.Outstanding(), "allocator lifetime test failed");
        std::cout << "garnet-trt-sync-allocator-passed: 192 changing stream retirements, device restoration, guards\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
