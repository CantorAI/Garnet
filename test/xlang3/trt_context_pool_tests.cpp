// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "trt_context_pool.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

namespace {
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* text) noexcept override {
        if (severity <= Severity::kERROR) std::cerr << text << '\n';
    }
};
void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void CUDART_CB Delay(void*) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); }
}

int main() {
    try {
        Logger logger;
        std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
        if (!builder) throw std::runtime_error("cannot create TensorRT builder");
        std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(
            1u << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
        auto* input = network->addInput("x", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{1, 256});
        auto* layer = network->addElementWise(*input, *input, nvinfer1::ElementWiseOperation::kSUM);
        layer->getOutput(0)->setName("y");
        network->markOutput(*layer->getOutput(0));
        std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 64ull << 20);
        config->setBuilderOptimizationLevel(0);
        std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
        if (!plan) throw std::runtime_error("cannot build test engine");
        std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
        std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(plan->data(), plan->size()));
        if (!engine) throw std::runtime_error("cannot deserialize test engine");
        Garnet::TRTContextPool pool(engine.get(), 3);
        std::atomic<int> failures{0};
        std::vector<std::thread> workers;
        for (int worker = 0; worker < 12; ++worker) workers.emplace_back([&, worker] {
            cudaStream_t stream = nullptr;
            float* deviceInput = nullptr;
            float* deviceOutput = nullptr;
            try {
                Check(cudaSetDevice(0));
                Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
                Check(cudaMalloc(&deviceInput, 256 * sizeof(float)));
                Check(cudaMalloc(&deviceOutput, 256 * sizeof(float)));
                for (int run = 0; run < 40; ++run) {
                    const float marker = static_cast<float>(worker * 1000 + run);
                    std::vector<float> inputData(256, marker), outputData(256, -1);
                    Check(cudaMemcpyAsync(deviceInput, inputData.data(), 256 * sizeof(float), cudaMemcpyHostToDevice, stream));
                    {
                        auto lease = pool.Acquire(stream);
                        if (!lease->context->setTensorAddress("x", deviceInput)) throw std::runtime_error("input bind failed");
                        std::this_thread::yield();
                        if (!lease->context->setTensorAddress("y", deviceOutput)) throw std::runtime_error("output bind failed");
                        Check(cudaLaunchHostFunc(stream, Delay, nullptr));
                        if (!lease->context->enqueueV3(stream)) throw std::runtime_error("enqueue failed");
                    }
                    Check(cudaMemcpyAsync(outputData.data(), deviceOutput, 256 * sizeof(float), cudaMemcpyDeviceToHost, stream));
                    Check(cudaStreamSynchronize(stream));
                    for (float value : outputData)
                        if (value != marker * 2) throw std::runtime_error("concurrent TensorRT binding corruption");
                }
            } catch (const std::exception& error) {
                ++failures;
                std::cerr << error.what() << '\n';
            }
            if (stream) cudaStreamSynchronize(stream);
            if (deviceInput) cudaFree(deviceInput);
            if (deviceOutput) cudaFree(deviceOutput);
            if (stream) cudaStreamDestroy(stream);
        });
        for (auto& worker : workers) worker.join();
        if (failures || pool.Size() > 3) throw std::runtime_error("context pool test failed");
        std::cout << "garnet-trt-concurrent-contexts-passed: 480 calls, " << pool.Size() << " slots\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
