#include "trt_builder.h"
#include "paged_kv_plugin.h"
#include "cuda_lib.h"
#include "garnet_tensor.h"
#include "tensor_helper.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <memory>
#include <unordered_map>
#include <vector>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>

using namespace nvinfer1;

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;
#include "xpackage.h"

namespace Garnet {

    thread_local ITRTContext* g_trtContext = nullptr;

    namespace {
        class TensorRTFileStreamReader final : public nvinfer1::IStreamReaderV2 {
        public:
            explicit TensorRTFileStreamReader(const std::string& path)
                : stream(path, std::ios::binary) {}

            bool IsOpen() const { return stream.is_open(); }

            int64_t read(void* destination, int64_t byteCount, cudaStream_t cudaStream) noexcept override {
                if (!stream.is_open() || !destination || byteCount < 0) return -1;
                cudaPointerAttributes attributes{};
                const cudaError_t attributeStatus = cudaPointerGetAttributes(&attributes, destination);
                const bool isDevice = attributeStatus == cudaSuccess &&
                    attributes.type == cudaMemoryTypeDevice;
                if (attributeStatus != cudaSuccess) cudaGetLastError();
                if (!isDevice) {
                    stream.read(static_cast<char*>(destination), byteCount);
                    return static_cast<int64_t>(stream.gcount());
                }

                constexpr std::streamsize kChunkBytes = 8 * 1024 * 1024;
                std::vector<char> staging(static_cast<size_t>(std::min<int64_t>(byteCount, kChunkBytes)));
                int64_t copied = 0;
                while (copied < byteCount) {
                    const std::streamsize requested = static_cast<std::streamsize>(
                        std::min<int64_t>(byteCount - copied, static_cast<int64_t>(staging.size())));
                    stream.read(staging.data(), requested);
                    const std::streamsize received = stream.gcount();
                    if (received <= 0) break;
                    if (cudaMemcpyAsync(
                            static_cast<char*>(destination) + copied,
                            staging.data(),
                            static_cast<size_t>(received),
                            cudaMemcpyHostToDevice,
                            cudaStream) != cudaSuccess ||
                        cudaStreamSynchronize(cudaStream) != cudaSuccess) {
                        return -1;
                    }
                    copied += received;
                    if (received != requested) break;
                }
                return copied;
            }

            bool seek(int64_t offset, nvinfer1::SeekPosition where) noexcept override {
                if (!stream.is_open()) return false;
                std::ios_base::seekdir direction;
                if (where == nvinfer1::SeekPosition::kSET) direction = std::ios::beg;
                else if (where == nvinfer1::SeekPosition::kCUR) direction = std::ios::cur;
                else direction = std::ios::end;
                stream.clear();
                stream.seekg(offset, direction);
                return stream.good();
            }

        private:
            std::ifstream stream;
        };

        struct CachedTRTExecution {
            nvinfer1::IRuntime* runtime = nullptr;
            nvinfer1::ICudaEngine* engine = nullptr;
            nvinfer1::IExecutionContext* context = nullptr;
        };

        std::mutex g_trtExecutionCacheMutex;
        std::unordered_map<std::string, CachedTRTExecution> g_trtExecutionCache;

        nvinfer1::IExecutionContext* GetCachedTRTExecutionContext(
            const std::string& enginePath,
            const Garnet::SafeTensorsIndex* weightIndex = nullptr) {
            std::lock_guard<std::mutex> lock(g_trtExecutionCacheMutex);
            auto found = g_trtExecutionCache.find(enginePath);
            if (found != g_trtExecutionCache.end()) {
                return found->second.context;
            }

            TensorRTFileStreamReader engineStream(enginePath);
            if (!engineStream.IsOpen()) {
                std::cout << "[TRTBuilder] Failed to open engine for read: " << enginePath << std::endl;
                return nullptr;
            }

            CachedTRTExecution cached;
            cached.runtime = createInferRuntime(gLogger);
            if (!cached.runtime) {
                std::cout << "[TRTBuilder] createInferRuntime failed for cached engine: " << enginePath << std::endl;
                return nullptr;
            }
            cached.engine = cached.runtime->deserializeCudaEngine(engineStream);
            if (!cached.engine) {
                std::cout << "[TRTBuilder] deserializeCudaEngine failed for cached engine: " << enginePath << std::endl;
                return nullptr;
            }
            if (weightIndex && weightIndex->TensorCount() > 0) {
                std::unique_ptr<nvinfer1::IRefitter> refitter(
                    nvinfer1::createInferRefitter(*cached.engine, gLogger));
                if (!refitter) {
                    std::cout << "[TRTBuilder] createInferRefitter failed: " << enginePath << std::endl;
                    return nullptr;
                }
                const int32_t weightCount = refitter->getAllWeights(0, nullptr);
                if (weightCount > 0) {
                    std::vector<const char*> weightNames(static_cast<size_t>(weightCount));
                    refitter->getAllWeights(weightCount, weightNames.data());
                    Garnet::SafeTensorsMappedFile mappedWeights;
                    std::string mappingError;
                    if (!mappedWeights.Open(weightIndex->FilePath(), mappingError)) {
                        std::cout << "[TRTBuilder] refit mapping failed: " << mappingError << std::endl;
                        return nullptr;
                    }
                    for (const char* weightName : weightNames) {
                        const Garnet::SafeTensorMetadata* metadata = weightIndex->Find(weightName);
                        if (!metadata) {
                            std::cout << "[TRTBuilder] refit weight absent: " << weightName << std::endl;
                            return nullptr;
                        }
                        DataType dataType;
                        size_t elementBytes = 0;
                        if (metadata->dataType == "BF16") {
                            dataType = DataType::kBF16;
                            elementBytes = 2;
                        }
                        else if (metadata->dataType == "F16") {
                            dataType = DataType::kHALF;
                            elementBytes = 2;
                        }
                        else if (metadata->dataType == "F32") {
                            dataType = DataType::kFLOAT;
                            elementBytes = 4;
                        }
                        else {
                            std::cout << "[TRTBuilder] unsupported refit dtype: " << metadata->dataType << std::endl;
                            return nullptr;
                        }
                        const void* data = mappedWeights.DataAt(
                            metadata->dataOffset,
                            metadata->dataSize);
                        const int64_t elementCount = static_cast<int64_t>(metadata->dataSize / elementBytes);
                        if (!data || !refitter->setNamedWeights(
                                weightName,
                                Weights{dataType, data, elementCount})) {
                            std::cout << "[TRTBuilder] setNamedWeights failed: " << weightName << std::endl;
                            return nullptr;
                        }
                    }
                    if (!refitter->refitCudaEngine()) {
                        std::cout << "[TRTBuilder] refitCudaEngine failed: " << enginePath << std::endl;
                        return nullptr;
                    }
                }
            }
            cached.context = cached.engine->createExecutionContext();
            if (!cached.context) {
                std::cout << "[TRTBuilder] createExecutionContext failed for cached engine: " << enginePath << std::endl;
                return nullptr;
            }

            auto inserted = g_trtExecutionCache.emplace(enginePath, cached);
            return inserted.first->second.context;
        }

        bool GetCachedTRTExecutionObjects(
            const std::string& enginePath,
            nvinfer1::ICudaEngine*& engine,
            nvinfer1::IExecutionContext*& context,
            const Garnet::SafeTensorsIndex* weightIndex = nullptr) {
            context = GetCachedTRTExecutionContext(enginePath, weightIndex);
            if (!context) {
                engine = nullptr;
                return false;
            }
            std::lock_guard<std::mutex> lock(g_trtExecutionCacheMutex);
            auto found = g_trtExecutionCache.find(enginePath);
            if (found == g_trtExecutionCache.end()) {
                engine = nullptr;
                context = nullptr;
                return false;
            }
            engine = found->second.engine;
            context = found->second.context;
            return engine != nullptr && context != nullptr;
        }

        void InvalidateCachedTRTExecution(const std::string& enginePath) {
            std::lock_guard<std::mutex> lock(g_trtExecutionCacheMutex);
            auto found = g_trtExecutionCache.find(enginePath);
            if (found == g_trtExecutionCache.end()) {
                return;
            }
            delete found->second.context;
            delete found->second.engine;
            delete found->second.runtime;
            g_trtExecutionCache.erase(found);
        }

        bool ShouldSyncTRTOutputToCPU() {
            const char* value = std::getenv("GARNET_TRT_SYNC_CPU_OUTPUTS");
            if (!value) {
                return true;
            }
            return !(value[0] == '0' && value[1] == '\0');
        }

        // GPU-resident model execution is ordered on CUDA's per-thread stream.
        // The stream is synchronized only at a CPU observation boundary.
        cudaError_t CreateExecutionStream(cudaStream_t* stream) {
            if (!stream) return cudaErrorInvalidValue;
            *stream = cudaStreamPerThread;
            return cudaSuccess;
        }

        cudaError_t DestroyExecutionStream(cudaStream_t) {
            return cudaSuccess;
        }

        cudaError_t AllocateExecutionMemory(void** ptr, size_t bytes) {
            cudaError_t err = cudaMallocAsync(ptr, bytes, cudaStreamPerThread);
            if (err == cudaErrorNotSupported) {
                cudaGetLastError();
                return cudaMalloc(ptr, bytes);
            }
            return err;
        }

        cudaError_t FreeExecutionMemory(void* ptr) {
            if (!ptr) return cudaSuccess;
            cudaError_t err = cudaFreeAsync(ptr, cudaStreamPerThread);
            if (err == cudaErrorNotSupported) {
                cudaGetLastError();
                return cudaFree(ptr);
            }
            return err;
        }

#define cudaStreamCreate CreateExecutionStream
#define cudaStreamDestroy DestroyExecutionStream
#define cudaMalloc AllocateExecutionMemory
#define cudaFree FreeExecutionMemory

        struct TensorDeviceBinding {
            void* ptr = nullptr;
            bool owned = false;
        };

        bool BindTensorInput(X::Tensor& tensor, size_t bytes, cudaStream_t stream, TensorDeviceBinding& binding) {
            void* gpuPtr = TensorHelper::GetGPUMemory(tensor);
            if (gpuPtr) {
                binding.ptr = gpuPtr;
                binding.owned = false;
                return true;
            }

            if (TensorHelper::EnsureGPUMemory(tensor) != TensorOpStatus::Success) {
                return false;
            }
            gpuPtr = TensorHelper::GetGPUMemory(tensor);
            if (!gpuPtr || static_cast<size_t>(tensor->GetDataSize()) < bytes) {
                return false;
            }
            binding.ptr = gpuPtr;
            binding.owned = false;
            return true;
        }

        void FreeOwnedBinding(TensorDeviceBinding& binding) {
            if (binding.owned && binding.ptr) {
                cudaFree(binding.ptr);
            }
            binding.ptr = nullptr;
            binding.owned = false;
        }

        X::Value MakeGPUBackedTensor2D(
            int rows,
            int cols,
            void* gpuOutput,
            size_t outputBytes,
            cudaStream_t stream) {
            if (ShouldSyncTRTOutputToCPU()) {
                cudaError_t syncErr = cudaStreamSynchronize(stream);
                if (syncErr != cudaSuccess) {
                    std::cout << "[TRTBuilder] MakeGPUBackedTensor2D stream sync failed before tensor wrap: "
                        << cudaGetErrorString(syncErr) << std::endl;
                    return X::Value();
                }
            }
            cudaError_t lastErr = cudaGetLastError();
            if (lastErr != cudaSuccess) {
                std::cout << "[TRTBuilder] MakeGPUBackedTensor2D CUDA error before tensor wrap: "
                    << cudaGetErrorString(lastErr) << std::endl;
                return X::Value();
            }
            X::Port::vector<int> outputShape(2);
            outputShape.push_back(rows);
            outputShape.push_back(cols);
            X::Tensor output = X::g_pXHost->CreateTensor();
            if (!output) {
                std::cout << "[TRTBuilder] MakeGPUBackedTensor2D failed to create XTensor" << std::endl;
                return X::Value();
            }
            output->SetDataType(X::TensorDataType::FLOAT32);
            output->SetShape(outputShape);
            if (ShouldSyncTRTOutputToCPU()) {
                std::vector<char> host(outputBytes);
                cudaError_t err = cudaMemcpy(host.data(), gpuOutput, outputBytes, cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) {
                    std::cout << "[TRTBuilder] MakeGPUBackedTensor2D failed to sync CPU output" << std::endl;
                    return X::Value();
                }
                output->DirectSetData(nullptr, 0);
                output->SetDeviceType(X::TensorDeviceType::CPU);
                output->SetDeviceContext(X::Value());
                output->SetDeviceOps(X::Value());
                output->SetData(host.data(), outputBytes);
                cudaFree(gpuOutput);
            }
            else {
                if (TensorHelper::AttachGPUMemory(output, gpuOutput) != TensorOpStatus::Success) {
                    std::cout << "[TRTBuilder] MakeGPUBackedTensor2D failed to attach GPU memory" << std::endl;
                    return X::Value();
                }
            }
            return X::Value(output);
        }

        X::Value RebindExistingTensor2D(
            X::Tensor& tensor,
            int rows,
            int cols,
            void* gpuOutput,
            size_t outputBytes,
            cudaStream_t stream) {
            if (ShouldSyncTRTOutputToCPU()) {
                cudaError_t syncErr = cudaStreamSynchronize(stream);
                if (syncErr != cudaSuccess) {
                    std::cout << "[TRTBuilder] RebindExistingTensor2D stream sync failed: "
                        << cudaGetErrorString(syncErr) << std::endl;
                    return X::Value();
                }
            }

            X::Port::vector<int> outputShape(2);
            outputShape.push_back(rows);
            outputShape.push_back(cols);
            tensor->SetShape(outputShape);
            tensor->SetDataType(X::TensorDataType::FLOAT32);
            if (TensorHelper::AttachGPUMemory(tensor, gpuOutput) != TensorOpStatus::Success) {
                std::cout << "[TRTBuilder] RebindExistingTensor2D failed to attach GPU memory" << std::endl;
                return X::Value();
            }
            if (ShouldSyncTRTOutputToCPU()) {
                std::vector<char> host(outputBytes);
                cudaError_t err = cudaMemcpy(host.data(), gpuOutput, outputBytes, cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) {
                    std::cout << "[TRTBuilder] RebindExistingTensor2D failed to sync CPU output" << std::endl;
                    return X::Value();
                }
                tensor->DirectSetData(nullptr, 0);
                tensor->SetDeviceType(X::TensorDeviceType::CPU);
                tensor->SetDeviceContext(X::Value());
                tensor->SetDeviceOps(X::Value());
                tensor->SetData(host.data(), outputBytes);
                cudaFree(gpuOutput);
            }
            return X::Value(tensor);
        }
    }

    TRTBuilder::TRTBuilder() {
    }

    TRTBuilder::~TRTBuilder() {
    }

    X::Value TRTBuilder::ExportMatmulEngine(const std::string& enginePath, const std::vector<int>& inputShape, const std::vector<int>& weightShape) {
        std::cout << "[TRTBuilder] ExportMatmulEngine -> " << enginePath << std::endl;
        if (inputShape.size() != 2 || weightShape.size() != 2) {
            std::cout << "[TRTBuilder] Matmul export requires 2D input and weight shapes." << std::endl;
            return X::Value();
        }
        if (inputShape[1] != weightShape[0]) {
            std::cout << "[TRTBuilder] Shape mismatch: input K=" << inputShape[1]
                << ", weight K=" << weightShape[0] << std::endl;
            return X::Value();
        }

        auto builder = createInferBuilder(gLogger);
        if (!builder) {
            std::cout << "[TRTBuilder] createInferBuilder failed." << std::endl;
            return X::Value();
        }

        uint32_t flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(flags);
        if (!network) {
            std::cout << "[TRTBuilder] createNetworkV2 failed." << std::endl;
            return X::Value();
        }

        auto config = builder->createBuilderConfig();
        if (!config) {
            std::cout << "[TRTBuilder] createBuilderConfig failed." << std::endl;
            return X::Value();
        }
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 64ULL << 20);

        Dims inputDims{};
        inputDims.nbDims = 2;
        inputDims.d[0] = inputShape[0];
        inputDims.d[1] = inputShape[1];

        Dims weightDims{};
        weightDims.nbDims = 2;
        weightDims.d[0] = weightShape[0];
        weightDims.d[1] = weightShape[1];

        ITensor* input = network->addInput("a", DataType::kFLOAT, inputDims);
        ITensor* weight = network->addInput("W", DataType::kFLOAT, weightDims);
        if (!input || !weight) {
            std::cout << "[TRTBuilder] addInput failed." << std::endl;
            return X::Value();
        }

        auto matmul = network->addMatrixMultiply(*input, MatrixOperation::kNONE, *weight, MatrixOperation::kNONE);
        if (!matmul || !matmul->getOutput(0)) {
            std::cout << "[TRTBuilder] addMatrixMultiply failed." << std::endl;
            return X::Value();
        }
        matmul->getOutput(0)->setName("output");
        network->markOutput(*matmul->getOutput(0));

        auto serialized = builder->buildSerializedNetwork(*network, *config);
        if (!serialized) {
            std::cout << "[TRTBuilder] buildSerializedNetwork failed." << std::endl;
            return X::Value();
        }

        std::filesystem::path outputPath(enginePath);
        std::filesystem::create_directories(outputPath.parent_path());
        std::ofstream out(outputPath, std::ios::binary);
        if (!out.is_open()) {
            std::cout << "[TRTBuilder] Failed to open engine path for write." << std::endl;
            return X::Value();
        }
        out.write(static_cast<const char*>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
        out.close();

        std::cout << "[TRTBuilder] Serialized TensorRT engine bytes: " << serialized->size() << std::endl;
        return X::Value(enginePath);
    }

    X::Value TRTBuilder::RunMatmulEngine(const std::string& enginePath, X::Value inputValue, X::Value weightValue) {
        std::cout << "[TRTBuilder] RunMatmulEngine <- " << enginePath << std::endl;
        if (!inputValue.IsTensor() || !weightValue.IsTensor()) {
            std::cout << "[TRTBuilder] RunMatmulEngine requires tensor inputs." << std::endl;
            return X::Value();
        }

        X::Tensor input(inputValue);
        X::Tensor weight(weightValue);
        std::cout << "[TRTBuilder] Input dims=" << input->GetDimCount()
            << ", weight dims=" << weight->GetDimCount() << std::endl;
        if (input->GetDataType() != X::TensorDataType::FLOAT32 || weight->GetDataType() != X::TensorDataType::FLOAT32) {
            std::cout << "[TRTBuilder] RunMatmulEngine currently supports float32 tensors only." << std::endl;
            return X::Value();
        }
        if (input->GetDimCount() != 2 || weight->GetDimCount() != 2) {
            std::cout << "[TRTBuilder] RunMatmulEngine requires 2D tensors." << std::endl;
            return X::Value();
        }

        int m = input->GetDimSize(0);
        int k = input->GetDimSize(1);
        int wK = weight->GetDimSize(0);
        int n = weight->GetDimSize(1);
        std::cout << "[TRTBuilder] Matmul shapes: [" << m << ", " << k
            << "] x [" << wK << ", " << n << "]" << std::endl;
        if (k != wK) {
            std::cout << "[TRTBuilder] RunMatmulEngine shape mismatch." << std::endl;
            return X::Value();
        }

        std::ifstream in(enginePath, std::ios::binary | std::ios::ate);
        if (!in.is_open()) {
            std::cout << "[TRTBuilder] Failed to open engine for read: " << enginePath << std::endl;
            return X::Value();
        }
        std::streamsize size = in.tellg();
        std::cout << "[TRTBuilder] Engine bytes to load: " << size << std::endl;
        in.seekg(0, std::ios::beg);
        std::vector<char> engineBytes(static_cast<size_t>(size));
        if (!in.read(engineBytes.data(), size)) {
            std::cout << "[TRTBuilder] Failed to read engine bytes." << std::endl;
            return X::Value();
        }

        auto runtime = createInferRuntime(gLogger);
        std::cout << "[TRTBuilder] createInferRuntime returned " << (runtime ? "ok" : "null") << std::endl;
        if (!runtime) {
            std::cout << "[TRTBuilder] createInferRuntime failed." << std::endl;
            return X::Value();
        }
        auto engine = runtime->deserializeCudaEngine(engineBytes.data(), engineBytes.size());
        std::cout << "[TRTBuilder] deserializeCudaEngine returned " << (engine ? "ok" : "null") << std::endl;
        if (!engine) {
            std::cout << "[TRTBuilder] deserializeCudaEngine failed." << std::endl;
            return X::Value();
        }
        auto context = engine->createExecutionContext();
        std::cout << "[TRTBuilder] createExecutionContext returned " << (context ? "ok" : "null") << std::endl;
        if (!context) {
            std::cout << "[TRTBuilder] createExecutionContext failed." << std::endl;
            return X::Value();
        }

        size_t inputBytes = static_cast<size_t>(m) * static_cast<size_t>(k) * sizeof(float);
        size_t weightBytes = static_cast<size_t>(k) * static_cast<size_t>(n) * sizeof(float);
        size_t outputBytes = static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float);

        void* dInput = nullptr;
        void* dWeight = nullptr;
        void* dOutput = nullptr;
        cudaStream_t stream = nullptr;

        TensorDeviceBinding inputBinding;
        TensorDeviceBinding weightBinding;
        if (cudaStreamCreate(&stream) != cudaSuccess ||
            !BindTensorInput(input, inputBytes, stream, inputBinding) ||
            !BindTensorInput(weight, weightBytes, stream, weightBinding) ||
            cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
            std::cout << "[TRTBuilder] CUDA allocation failed." << std::endl;
            FreeOwnedBinding(inputBinding);
            FreeOwnedBinding(weightBinding);
            if (dOutput) cudaFree(dOutput);
            if (stream) cudaStreamDestroy(stream);
            return X::Value();
        }
        dInput = inputBinding.ptr;
        dWeight = weightBinding.ptr;
        std::cout << "[TRTBuilder] CUDA buffers bound." << std::endl;

        bool bound = context->setTensorAddress("a", dInput)
            && context->setTensorAddress("W", dWeight)
            && context->setTensorAddress("output", dOutput);
        std::cout << "[TRTBuilder] setTensorAddress returned " << (bound ? "ok" : "false") << std::endl;
        if (!bound) {
            std::cout << "[TRTBuilder] setTensorAddress failed." << std::endl;
            FreeOwnedBinding(inputBinding);
            FreeOwnedBinding(weightBinding);
            cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }

        bool ok = context->enqueueV3(stream);
        std::cout << "[TRTBuilder] enqueueV3 returned " << (ok ? "ok" : "false") << std::endl;
        if (!ok) {
            std::cout << "[TRTBuilder] enqueueV3 failed." << std::endl;
            FreeOwnedBinding(inputBinding);
            FreeOwnedBinding(weightBinding);
            cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }

        X::Value output = MakeGPUBackedTensor2D(m, n, dOutput, outputBytes, stream);
        if (!output.IsValid()) {
            FreeOwnedBinding(inputBinding);
            FreeOwnedBinding(weightBinding);
            cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }

        FreeOwnedBinding(inputBinding);
        FreeOwnedBinding(weightBinding);
        cudaStreamDestroy(stream);

        std::cout << "[TRTBuilder] RunMatmulEngine completed: [" << m << ", " << n << "]" << std::endl;
        return output;
    }

    X::Value TRTBuilder::ExportTextMLPEngine(const std::string& enginePath, const std::vector<int>& inputShape, const std::vector<int>& gateShape, const std::vector<int>& upShape, const std::vector<int>& downShape) {
        std::cout << "[TRTBuilder] ExportTextMLPEngine -> " << enginePath << std::endl;
        if (inputShape.size() != 2 || gateShape.size() != 2 || upShape.size() != 2 || downShape.size() != 2) {
            std::cout << "[TRTBuilder] TextMLP export requires 2D input and weight shapes." << std::endl;
            return X::Value();
        }
        int tokens = inputShape[0];
        int hidden = inputShape[1];
        int intermediate = gateShape[0];
        if (gateShape[1] != hidden || upShape[0] != intermediate || upShape[1] != hidden ||
            downShape[0] != hidden || downShape[1] != intermediate) {
            std::cout << "[TRTBuilder] TextMLP shape mismatch." << std::endl;
            return X::Value();
        }

        auto builder = createInferBuilder(gLogger);
        if (!builder) return X::Value();
        uint32_t flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(flags);
        if (!network) return X::Value();
        auto config = builder->createBuilderConfig();
        if (!config) return X::Value();
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 128ULL << 20);

        Dims xDims{};
        xDims.nbDims = 2;
        xDims.d[0] = tokens;
        xDims.d[1] = hidden;
        Dims projDims{};
        projDims.nbDims = 2;
        projDims.d[0] = intermediate;
        projDims.d[1] = hidden;
        Dims downDims{};
        downDims.nbDims = 2;
        downDims.d[0] = hidden;
        downDims.d[1] = intermediate;

        ITensor* x = network->addInput("x", DataType::kFLOAT, xDims);
        ITensor* wGate = network->addInput("W_gate", DataType::kFLOAT, projDims);
        ITensor* wUp = network->addInput("W_up", DataType::kFLOAT, projDims);
        ITensor* wDown = network->addInput("W_down", DataType::kFLOAT, downDims);
        if (!x || !wGate || !wUp || !wDown) {
            std::cout << "[TRTBuilder] TextMLP addInput failed." << std::endl;
            return X::Value();
        }

        auto gate = network->addMatrixMultiply(*x, MatrixOperation::kNONE, *wGate, MatrixOperation::kTRANSPOSE);
        auto up = network->addMatrixMultiply(*x, MatrixOperation::kNONE, *wUp, MatrixOperation::kTRANSPOSE);
        if (!gate || !up || !gate->getOutput(0) || !up->getOutput(0)) {
            std::cout << "[TRTBuilder] TextMLP projection matmul failed." << std::endl;
            return X::Value();
        }

        auto sigmoid = network->addActivation(*gate->getOutput(0), ActivationType::kSIGMOID);
        if (!sigmoid || !sigmoid->getOutput(0)) {
            std::cout << "[TRTBuilder] TextMLP sigmoid failed." << std::endl;
            return X::Value();
        }
        auto silu = network->addElementWise(*gate->getOutput(0), *sigmoid->getOutput(0), ElementWiseOperation::kPROD);
        if (!silu || !silu->getOutput(0)) {
            std::cout << "[TRTBuilder] TextMLP silu product failed." << std::endl;
            return X::Value();
        }
        auto hiddenAct = network->addElementWise(*silu->getOutput(0), *up->getOutput(0), ElementWiseOperation::kPROD);
        if (!hiddenAct || !hiddenAct->getOutput(0)) {
            std::cout << "[TRTBuilder] TextMLP gated product failed." << std::endl;
            return X::Value();
        }
        auto out = network->addMatrixMultiply(*hiddenAct->getOutput(0), MatrixOperation::kNONE, *wDown, MatrixOperation::kTRANSPOSE);
        if (!out || !out->getOutput(0)) {
            std::cout << "[TRTBuilder] TextMLP down projection failed." << std::endl;
            return X::Value();
        }
        out->getOutput(0)->setName("output");
        network->markOutput(*out->getOutput(0));

        auto serialized = builder->buildSerializedNetwork(*network, *config);
        if (!serialized) {
            std::cout << "[TRTBuilder] TextMLP buildSerializedNetwork failed." << std::endl;
            return X::Value();
        }

        std::filesystem::path outputPath(enginePath);
        std::filesystem::create_directories(outputPath.parent_path());
        std::ofstream outFile(outputPath, std::ios::binary);
        if (!outFile.is_open()) {
            std::cout << "[TRTBuilder] Failed to open TextMLP engine path for write." << std::endl;
            return X::Value();
        }
        outFile.write(static_cast<const char*>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
        outFile.close();

        std::cout << "[TRTBuilder] Serialized TextMLP TensorRT engine bytes: " << serialized->size() << std::endl;
        return X::Value(enginePath);
    }

    X::Value TRTBuilder::RunTextMLPEngine(const std::string& enginePath, X::Value inputValue, X::Value gateWeight, X::Value upWeight, X::Value downWeight) {
        std::cout << "[TRTBuilder] RunTextMLPEngine <- " << enginePath << std::endl;
        if (!inputValue.IsTensor() || !gateWeight.IsTensor() || !upWeight.IsTensor() || !downWeight.IsTensor()) {
            std::cout << "[TRTBuilder] RunTextMLPEngine requires tensor inputs." << std::endl;
            return X::Value();
        }

        X::Tensor input(inputValue);
        X::Tensor gate(gateWeight);
        X::Tensor up(upWeight);
        X::Tensor down(downWeight);
        bool bf16Weights = gate->GetDataType() == X::TensorDataType::BFLOAT16 &&
            up->GetDataType() == X::TensorDataType::BFLOAT16 &&
            down->GetDataType() == X::TensorDataType::BFLOAT16;
        if (input->GetDataType() != X::TensorDataType::FLOAT32 ||
            (!bf16Weights && (gate->GetDataType() != X::TensorDataType::FLOAT32 ||
                up->GetDataType() != X::TensorDataType::FLOAT32 ||
                down->GetDataType() != X::TensorDataType::FLOAT32))) {
            std::cout << "[TRTBuilder] RunTextMLPEngine requires FP32 activations and uniform FP32/BF16 weights." << std::endl;
            return X::Value();
        }
        if (input->GetDimCount() != 2 || gate->GetDimCount() != 2 || up->GetDimCount() != 2 || down->GetDimCount() != 2) {
            std::cout << "[TRTBuilder] RunTextMLPEngine requires 2D tensors." << std::endl;
            return X::Value();
        }

        int tokens = input->GetDimSize(0);
        int hidden = input->GetDimSize(1);
        int intermediate = gate->GetDimSize(0);
        if (gate->GetDimSize(1) != hidden || up->GetDimSize(0) != intermediate ||
            up->GetDimSize(1) != hidden || down->GetDimSize(0) != hidden ||
            down->GetDimSize(1) != intermediate) {
            std::cout << "[TRTBuilder] RunTextMLPEngine shape mismatch." << std::endl;
            return X::Value();
        }

        if (enginePath == "cuda_text_mlp" || bf16Weights) {
            std::cout << "[TRTBuilder] Running CUDA TextMLP: tokens=" << tokens
                << ", hidden=" << hidden << ", intermediate=" << intermediate << std::endl;
            size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);
            size_t projBytes = static_cast<size_t>(intermediate) * static_cast<size_t>(hidden) * gate->GetItemSize();
            size_t downBytes = static_cast<size_t>(hidden) * static_cast<size_t>(intermediate) * down->GetItemSize();
            size_t intermediateBytes = static_cast<size_t>(tokens) * static_cast<size_t>(intermediate) * sizeof(float);
            size_t outputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);

            void* dInput = nullptr;
            void* dGateW = nullptr;
            void* dUpW = nullptr;
            void* dDownW = nullptr;
            void* dGate = nullptr;
            void* dUp = nullptr;
            void* dHidden = nullptr;
            void* dOutput = nullptr;
            cudaStream_t stream = nullptr;
            TensorDeviceBinding inputBinding;
            TensorDeviceBinding gateBinding;
            TensorDeviceBinding upBinding;
            TensorDeviceBinding downBinding;
            if (cudaStreamCreate(&stream) != cudaSuccess ||
                !BindTensorInput(input, inputBytes, stream, inputBinding) ||
                !BindTensorInput(gate, projBytes, stream, gateBinding) ||
                !BindTensorInput(up, projBytes, stream, upBinding) ||
                !BindTensorInput(down, downBytes, stream, downBinding) ||
                cudaMalloc(&dGate, intermediateBytes) != cudaSuccess ||
                cudaMalloc(&dUp, intermediateBytes) != cudaSuccess ||
                cudaMalloc(&dHidden, intermediateBytes) != cudaSuccess ||
                cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
                std::cout << "[TRTBuilder] CUDA TextMLP allocation failed." << std::endl;
                FreeOwnedBinding(inputBinding);
                FreeOwnedBinding(gateBinding);
                FreeOwnedBinding(upBinding);
                FreeOwnedBinding(downBinding);
                if (dGate) cudaFree(dGate);
                if (dUp) cudaFree(dUp);
                if (dHidden) cudaFree(dHidden);
                if (dOutput) cudaFree(dOutput);
                if (stream) cudaStreamDestroy(stream);
                return X::Value();
            }

            dInput = inputBinding.ptr;
            dGateW = gateBinding.ptr;
            dUpW = upBinding.ptr;
            dDownW = downBinding.ptr;

            auto linear = [&](const float* source, const void* weights, float* output,
                int rows, int inFeatures, int outFeatures) {
                return bf16Weights
                    ? runLinearTransposeBF16WeightFP32(source, static_cast<const bfloat16*>(weights),
                        output, rows, inFeatures, outFeatures, stream)
                    : runLinearTransposeFP32(source, static_cast<const float*>(weights),
                        output, rows, inFeatures, outFeatures, stream);
            };
            cudaError_t status = linear(
                static_cast<const float*>(dInput), dGateW, static_cast<float*>(dGate),
                tokens, hidden, intermediate);
            if (status == cudaSuccess) {
                status = linear(static_cast<const float*>(dInput), dUpW, static_cast<float*>(dUp),
                    tokens, hidden, intermediate);
            }
            if (status == cudaSuccess) {
                status = runSiluMulFP32(
                    static_cast<const float*>(dGate),
                    static_cast<const float*>(dUp),
                    static_cast<float*>(dHidden),
                    tokens * intermediate,
                    stream);
            }
            if (status == cudaSuccess) {
                status = linear(static_cast<const float*>(dHidden), dDownW, static_cast<float*>(dOutput),
                    tokens, intermediate, hidden);
            }
            if (status != cudaSuccess) {
                std::cout << "[TRTBuilder] CUDA TextMLP launch failed: "
                    << cudaGetErrorString(status) << std::endl;
                FreeOwnedBinding(inputBinding); FreeOwnedBinding(gateBinding); FreeOwnedBinding(upBinding); FreeOwnedBinding(downBinding);
                cudaFree(dGate); cudaFree(dUp); cudaFree(dHidden); cudaFree(dOutput);
                cudaStreamDestroy(stream);
                return X::Value();
            }

            X::Value output = MakeGPUBackedTensor2D(tokens, hidden, dOutput, outputBytes, stream);
            if (!output.IsValid()) {
                FreeOwnedBinding(inputBinding); FreeOwnedBinding(gateBinding); FreeOwnedBinding(upBinding); FreeOwnedBinding(downBinding);
                cudaFree(dGate); cudaFree(dUp); cudaFree(dHidden); cudaFree(dOutput);
                cudaStreamDestroy(stream);
                return X::Value();
            }

            FreeOwnedBinding(inputBinding); FreeOwnedBinding(gateBinding); FreeOwnedBinding(upBinding); FreeOwnedBinding(downBinding);
            cudaFree(dGate); cudaFree(dUp); cudaFree(dHidden);
            cudaStreamDestroy(stream);
            std::cout << "[TRTBuilder] CUDA TextMLP completed: [" << tokens
                << ", " << hidden << "]" << std::endl;
            return output;
        }

        auto context = GetCachedTRTExecutionContext(enginePath);
        if (!context) return X::Value();

        size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);
        size_t projBytes = static_cast<size_t>(intermediate) * static_cast<size_t>(hidden) * sizeof(float);
        size_t downBytes = static_cast<size_t>(hidden) * static_cast<size_t>(intermediate) * sizeof(float);
        size_t outputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);

        TensorDeviceBinding dInput;
        TensorDeviceBinding dGate;
        TensorDeviceBinding dUp;
        TensorDeviceBinding dDown;
        void* dOutput = nullptr;
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            return X::Value();
        }
        if (!BindTensorInput(input, inputBytes, stream, dInput) ||
            !BindTensorInput(gate, projBytes, stream, dGate) ||
            !BindTensorInput(up, projBytes, stream, dUp) ||
            !BindTensorInput(down, downBytes, stream, dDown) ||
            cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
            std::cout << "[TRTBuilder] TextMLP CUDA allocation failed." << std::endl;
            FreeOwnedBinding(dInput);
            FreeOwnedBinding(dGate);
            FreeOwnedBinding(dUp);
            FreeOwnedBinding(dDown);
            if (dOutput) cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }

        bool bound = context->setTensorAddress("x", dInput.ptr)
            && context->setTensorAddress("W_gate", dGate.ptr)
            && context->setTensorAddress("W_up", dUp.ptr)
            && context->setTensorAddress("W_down", dDown.ptr)
            && context->setTensorAddress("output", dOutput);
        if (!bound || !context->enqueueV3(stream)) {
            std::cout << "[TRTBuilder] TextMLP enqueue failed." << std::endl;
            FreeOwnedBinding(dInput);
            FreeOwnedBinding(dGate);
            FreeOwnedBinding(dUp);
            FreeOwnedBinding(dDown);
            cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }

        X::Value output = MakeGPUBackedTensor2D(tokens, hidden, dOutput, outputBytes, stream);
        if (!output.IsValid()) {
            std::cout << "[TRTBuilder] Failed to create TextMLP output tensor." << std::endl;
            FreeOwnedBinding(dInput);
            FreeOwnedBinding(dGate);
            FreeOwnedBinding(dUp);
            FreeOwnedBinding(dDown);
            cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }
        FreeOwnedBinding(dInput);
        FreeOwnedBinding(dGate);
        FreeOwnedBinding(dUp);
        FreeOwnedBinding(dDown);
        cudaStreamDestroy(stream);

        std::cout << "[TRTBuilder] RunTextMLPEngine completed: [" << tokens << ", " << hidden << "]" << std::endl;
        return output;
    }

    X::Value TRTBuilder::ExportTextQKVEngine(const std::string& enginePath, const std::vector<int>& inputShape, const std::vector<int>& qShape, const std::vector<int>& kShape, const std::vector<int>& vShape) {
        std::cout << "[TRTBuilder] ExportTextQKVEngine -> " << enginePath << std::endl;
        if (inputShape.size() != 2 || qShape.size() != 2 || kShape.size() != 2 || vShape.size() != 2) return X::Value();
        int tokens = inputShape[0];
        int hidden = inputShape[1];
        if (qShape[1] != hidden || kShape[1] != hidden || vShape[1] != hidden) {
            std::cout << "[TRTBuilder] TextQKV shape mismatch." << std::endl;
            return X::Value();
        }

        auto builder = createInferBuilder(gLogger);
        if (!builder) return X::Value();
        uint32_t flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(flags);
        if (!network) return X::Value();
        auto config = builder->createBuilderConfig();
        if (!config) return X::Value();
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 128ULL << 20);

        Dims xDims{};
        xDims.nbDims = 2;
        xDims.d[0] = tokens;
        xDims.d[1] = hidden;
        Dims qDims{};
        qDims.nbDims = 2;
        qDims.d[0] = qShape[0];
        qDims.d[1] = hidden;
        Dims kDims{};
        kDims.nbDims = 2;
        kDims.d[0] = kShape[0];
        kDims.d[1] = hidden;
        Dims vDims{};
        vDims.nbDims = 2;
        vDims.d[0] = vShape[0];
        vDims.d[1] = hidden;

        ITensor* x = network->addInput("x", DataType::kFLOAT, xDims);
        ITensor* wQ = network->addInput("W_q", DataType::kFLOAT, qDims);
        ITensor* wK = network->addInput("W_k", DataType::kFLOAT, kDims);
        ITensor* wV = network->addInput("W_v", DataType::kFLOAT, vDims);
        if (!x || !wQ || !wK || !wV) return X::Value();
        auto q = network->addMatrixMultiply(*x, MatrixOperation::kNONE, *wQ, MatrixOperation::kTRANSPOSE);
        auto k = network->addMatrixMultiply(*x, MatrixOperation::kNONE, *wK, MatrixOperation::kTRANSPOSE);
        auto v = network->addMatrixMultiply(*x, MatrixOperation::kNONE, *wV, MatrixOperation::kTRANSPOSE);
        if (!q || !k || !v || !q->getOutput(0) || !k->getOutput(0) || !v->getOutput(0)) return X::Value();
        ITensor* concatInputs[] = { q->getOutput(0), k->getOutput(0), v->getOutput(0) };
        auto concat = network->addConcatenation(concatInputs, 3);
        if (!concat || !concat->getOutput(0)) return X::Value();
        concat->setAxis(1);
        concat->getOutput(0)->setName("output");
        network->markOutput(*concat->getOutput(0));

        auto serialized = builder->buildSerializedNetwork(*network, *config);
        if (!serialized) return X::Value();
        std::filesystem::path outputPath(enginePath);
        std::filesystem::create_directories(outputPath.parent_path());
        std::ofstream outFile(outputPath, std::ios::binary);
        if (!outFile.is_open()) return X::Value();
        outFile.write(static_cast<const char*>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
        outFile.close();
        std::cout << "[TRTBuilder] Serialized TextQKV TensorRT engine bytes: " << serialized->size() << std::endl;
        return X::Value(enginePath);
    }

    X::Value TRTBuilder::RunTextQKVEngine(const std::string& enginePath, X::Value inputValue, X::Value qWeight, X::Value kWeight, X::Value vWeight) {
        std::cout << "[TRTBuilder] RunTextQKVEngine <- " << enginePath << std::endl;
        if (!inputValue.IsTensor() || !qWeight.IsTensor() || !kWeight.IsTensor() || !vWeight.IsTensor()) return X::Value();
        X::Tensor input(inputValue);
        X::Tensor q(qWeight);
        X::Tensor k(kWeight);
        X::Tensor v(vWeight);
        if (input->GetDataType() != X::TensorDataType::FLOAT32 || q->GetDataType() != X::TensorDataType::FLOAT32 ||
            k->GetDataType() != X::TensorDataType::FLOAT32 || v->GetDataType() != X::TensorDataType::FLOAT32) return X::Value();
        if (input->GetDimCount() != 2 || q->GetDimCount() != 2 || k->GetDimCount() != 2 || v->GetDimCount() != 2) return X::Value();
        int tokens = input->GetDimSize(0);
        int hidden = input->GetDimSize(1);
        int qOut = q->GetDimSize(0);
        int kOut = k->GetDimSize(0);
        int vOut = v->GetDimSize(0);
        if (q->GetDimSize(1) != hidden || k->GetDimSize(1) != hidden || v->GetDimSize(1) != hidden) return X::Value();

        auto context = GetCachedTRTExecutionContext(enginePath);
        if (!context) return X::Value();

        size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);
        size_t qBytes = static_cast<size_t>(qOut) * static_cast<size_t>(hidden) * sizeof(float);
        size_t kBytes = static_cast<size_t>(kOut) * static_cast<size_t>(hidden) * sizeof(float);
        size_t vBytes = static_cast<size_t>(vOut) * static_cast<size_t>(hidden) * sizeof(float);
        size_t outputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(qOut + kOut + vOut) * sizeof(float);
        TensorDeviceBinding dInput;
        TensorDeviceBinding dQ;
        TensorDeviceBinding dK;
        TensorDeviceBinding dV;
        void* dOutput = nullptr;
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            return X::Value();
        }
        if (!BindTensorInput(input, inputBytes, stream, dInput) ||
            !BindTensorInput(q, qBytes, stream, dQ) ||
            !BindTensorInput(k, kBytes, stream, dK) ||
            !BindTensorInput(v, vBytes, stream, dV) ||
            cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
            FreeOwnedBinding(dInput);
            FreeOwnedBinding(dQ);
            FreeOwnedBinding(dK);
            FreeOwnedBinding(dV);
            if (dOutput) cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }
        bool bound = context->setTensorAddress("x", dInput.ptr)
            && context->setTensorAddress("W_q", dQ.ptr)
            && context->setTensorAddress("W_k", dK.ptr)
            && context->setTensorAddress("W_v", dV.ptr)
            && context->setTensorAddress("output", dOutput);
        if (!bound || !context->enqueueV3(stream)) {
            FreeOwnedBinding(dInput);
            FreeOwnedBinding(dQ);
            FreeOwnedBinding(dK);
            FreeOwnedBinding(dV);
            cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        X::Value output = MakeGPUBackedTensor2D(tokens, qOut + kOut + vOut, dOutput, outputBytes, stream);
        if (!output.IsValid()) {
            FreeOwnedBinding(dInput);
            FreeOwnedBinding(dQ);
            FreeOwnedBinding(dK);
            FreeOwnedBinding(dV);
            cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }
        FreeOwnedBinding(dInput);
        FreeOwnedBinding(dQ);
        FreeOwnedBinding(dK);
        FreeOwnedBinding(dV);
        cudaStreamDestroy(stream);
        std::cout << "[TRTBuilder] RunTextQKVEngine completed: [" << tokens << ", " << (qOut + kOut + vOut) << "]" << std::endl;
        return output;
    }

    X::Value TRTBuilder::ExportTextQKVHeadNormEngine(const std::string& enginePath, const std::vector<int>& inputShape, const std::vector<int>& qShape, const std::vector<int>& kShape, const std::vector<int>& vShape, const std::vector<int>& qNormShape, const std::vector<int>& kNormShape, float eps) {
        std::cout << "[TRTBuilder] ExportTextQKVHeadNormEngine -> " << enginePath << std::endl;
        if (inputShape.size() != 2 || qShape.size() != 2 || kShape.size() != 2 || vShape.size() != 2 ||
            qNormShape.size() != 1 || kNormShape.size() != 1) {
            return X::Value();
        }
        int tokens = inputShape[0];
        int hidden = inputShape[1];
        int headDim = qNormShape[0];
        int qOut = qShape[0];
        int kOut = kShape[0];
        int vOut = vShape[0];
        if (headDim <= 0 || kNormShape[0] != headDim || qShape[1] != hidden || kShape[1] != hidden || vShape[1] != hidden ||
            qOut % headDim != 0 || kOut % headDim != 0 || vOut != kOut) {
            std::cout << "[TRTBuilder] TextQKVHeadNorm shape mismatch." << std::endl;
            return X::Value();
        }
        int qHeads = qOut / headDim;
        int kvHeads = kOut / headDim;

        auto builder = createInferBuilder(gLogger);
        if (!builder) return X::Value();
        uint32_t flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(flags);
        if (!network) return X::Value();
        auto config = builder->createBuilderConfig();
        if (!config) return X::Value();
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 128ULL << 20);

        Dims xDims{};
        xDims.nbDims = 2;
        xDims.d[0] = tokens;
        xDims.d[1] = hidden;
        Dims qDims{};
        qDims.nbDims = 2;
        qDims.d[0] = qOut;
        qDims.d[1] = hidden;
        Dims kDims{};
        kDims.nbDims = 2;
        kDims.d[0] = kOut;
        kDims.d[1] = hidden;
        Dims vDims{};
        vDims.nbDims = 2;
        vDims.d[0] = vOut;
        vDims.d[1] = hidden;
        Dims normDims{};
        normDims.nbDims = 3;
        normDims.d[0] = 1;
        normDims.d[1] = 1;
        normDims.d[2] = headDim;
        Dims scalarDims{};
        scalarDims.nbDims = 3;
        scalarDims.d[0] = 1;
        scalarDims.d[1] = 1;
        scalarDims.d[2] = 1;

        ITensor* x = network->addInput("x", DataType::kFLOAT, xDims);
        ITensor* wQ = network->addInput("W_q", DataType::kFLOAT, qDims);
        ITensor* wK = network->addInput("W_k", DataType::kFLOAT, kDims);
        ITensor* wV = network->addInput("W_v", DataType::kFLOAT, vDims);
        ITensor* qNormWeight = network->addInput("q_norm", DataType::kFLOAT, normDims);
        ITensor* kNormWeight = network->addInput("k_norm", DataType::kFLOAT, normDims);
        if (!x || !wQ || !wK || !wV || !qNormWeight || !kNormWeight) return X::Value();

        auto q = network->addMatrixMultiply(*x, MatrixOperation::kNONE, *wQ, MatrixOperation::kTRANSPOSE);
        auto k = network->addMatrixMultiply(*x, MatrixOperation::kNONE, *wK, MatrixOperation::kTRANSPOSE);
        auto v = network->addMatrixMultiply(*x, MatrixOperation::kNONE, *wV, MatrixOperation::kTRANSPOSE);
        if (!q || !k || !v || !q->getOutput(0) || !k->getOutput(0) || !v->getOutput(0)) return X::Value();

        auto headNorm = [&](ITensor* flat, ITensor* weight, int heads, int flatSize) -> ITensor* {
            Dims headShape{};
            headShape.nbDims = 3;
            headShape.d[0] = tokens;
            headShape.d[1] = heads;
            headShape.d[2] = headDim;
            auto reshape = network->addShuffle(*flat);
            if (!reshape) return nullptr;
            reshape->setReshapeDimensions(headShape);
            ITensor* headed = reshape->getOutput(0);
            if (!headed) return nullptr;
            auto square = network->addElementWise(*headed, *headed, ElementWiseOperation::kPROD);
            if (!square || !square->getOutput(0)) return nullptr;
            auto mean = network->addReduce(*square->getOutput(0), ReduceOperation::kAVG, 1U << 2, true);
            if (!mean || !mean->getOutput(0)) return nullptr;
            float epsValue = eps;
            Weights epsWeights{ DataType::kFLOAT, &epsValue, 1 };
            auto epsLayer = network->addConstant(scalarDims, epsWeights);
            if (!epsLayer || !epsLayer->getOutput(0)) return nullptr;
            auto variance = network->addElementWise(*mean->getOutput(0), *epsLayer->getOutput(0), ElementWiseOperation::kSUM);
            if (!variance || !variance->getOutput(0)) return nullptr;
            auto sqrt = network->addUnary(*variance->getOutput(0), UnaryOperation::kSQRT);
            if (!sqrt || !sqrt->getOutput(0)) return nullptr;
            auto inv = network->addUnary(*sqrt->getOutput(0), UnaryOperation::kRECIP);
            if (!inv || !inv->getOutput(0)) return nullptr;
            auto normalized = network->addElementWise(*headed, *inv->getOutput(0), ElementWiseOperation::kPROD);
            if (!normalized || !normalized->getOutput(0)) return nullptr;
            auto scaled = network->addElementWise(*normalized->getOutput(0), *weight, ElementWiseOperation::kPROD);
            if (!scaled || !scaled->getOutput(0)) return nullptr;
            Dims flatShape{};
            flatShape.nbDims = 2;
            flatShape.d[0] = tokens;
            flatShape.d[1] = flatSize;
            auto flatten = network->addShuffle(*scaled->getOutput(0));
            if (!flatten) return nullptr;
            flatten->setReshapeDimensions(flatShape);
            return flatten->getOutput(0);
        };

        ITensor* qNormed = headNorm(q->getOutput(0), qNormWeight, qHeads, qOut);
        ITensor* kNormed = headNorm(k->getOutput(0), kNormWeight, kvHeads, kOut);
        if (!qNormed || !kNormed) return X::Value();
        ITensor* concatInputs[] = { qNormed, kNormed, v->getOutput(0) };
        auto concat = network->addConcatenation(concatInputs, 3);
        if (!concat || !concat->getOutput(0)) return X::Value();
        concat->setAxis(1);
        concat->getOutput(0)->setName("output");
        network->markOutput(*concat->getOutput(0));

        auto serialized = builder->buildSerializedNetwork(*network, *config);
        if (!serialized) return X::Value();
        std::filesystem::path outputPath(enginePath);
        std::filesystem::create_directories(outputPath.parent_path());
        std::ofstream outFile(outputPath, std::ios::binary);
        if (!outFile.is_open()) return X::Value();
        outFile.write(static_cast<const char*>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
        outFile.close();
        std::cout << "[TRTBuilder] Serialized TextQKVHeadNorm TensorRT engine bytes: " << serialized->size() << std::endl;
        return X::Value(enginePath);
    }

    X::Value TRTBuilder::RunTextQKVHeadNormEngine(const std::string& enginePath, X::Value inputValue, X::Value qWeight, X::Value kWeight, X::Value vWeight, X::Value qNormWeight, X::Value kNormWeight) {
        std::cout << "[TRTBuilder] RunTextQKVHeadNormEngine <- " << enginePath << std::endl;
        if (!inputValue.IsTensor() || !qWeight.IsTensor() || !kWeight.IsTensor() || !vWeight.IsTensor() || !qNormWeight.IsTensor() || !kNormWeight.IsTensor()) return X::Value();
        X::Tensor input(inputValue);
        X::Tensor q(qWeight);
        X::Tensor k(kWeight);
        X::Tensor v(vWeight);
        X::Tensor qNorm(qNormWeight);
        X::Tensor kNorm(kNormWeight);
        bool bf16Weights = q->GetDataType() == X::TensorDataType::BFLOAT16 &&
            k->GetDataType() == X::TensorDataType::BFLOAT16 && v->GetDataType() == X::TensorDataType::BFLOAT16 &&
            qNorm->GetDataType() == X::TensorDataType::BFLOAT16 && kNorm->GetDataType() == X::TensorDataType::BFLOAT16;
        if (input->GetDataType() != X::TensorDataType::FLOAT32 ||
            (!bf16Weights && (q->GetDataType() != X::TensorDataType::FLOAT32 ||
                k->GetDataType() != X::TensorDataType::FLOAT32 || v->GetDataType() != X::TensorDataType::FLOAT32 ||
                qNorm->GetDataType() != X::TensorDataType::FLOAT32 || kNorm->GetDataType() != X::TensorDataType::FLOAT32))) return X::Value();
        if (input->GetDimCount() != 2 || q->GetDimCount() != 2 || k->GetDimCount() != 2 || v->GetDimCount() != 2 ||
            qNorm->GetDimCount() != 1 || kNorm->GetDimCount() != 1) return X::Value();
        int tokens = input->GetDimSize(0);
        int hidden = input->GetDimSize(1);
        int qOut = q->GetDimSize(0);
        int kOut = k->GetDimSize(0);
        int vOut = v->GetDimSize(0);
        int headDim = qNorm->GetDimSize(0);
        if (headDim <= 0 || kNorm->GetDimSize(0) != headDim || q->GetDimSize(1) != hidden || k->GetDimSize(1) != hidden ||
            v->GetDimSize(1) != hidden || qOut % headDim != 0 || kOut % headDim != 0 || vOut != kOut) return X::Value();

        auto context = bf16Weights ? nullptr : GetCachedTRTExecutionContext(enginePath);
        if (!bf16Weights && !context) return X::Value();

        size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);
        size_t qBytes = static_cast<size_t>(qOut) * static_cast<size_t>(hidden) * q->GetItemSize();
        size_t kBytes = static_cast<size_t>(kOut) * static_cast<size_t>(hidden) * k->GetItemSize();
        size_t vBytes = static_cast<size_t>(vOut) * static_cast<size_t>(hidden) * v->GetItemSize();
        size_t normBytes = static_cast<size_t>(headDim) * qNorm->GetItemSize();
        size_t outputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(qOut + kOut + vOut) * sizeof(float);
        TensorDeviceBinding dInput;
        TensorDeviceBinding dQ;
        TensorDeviceBinding dK;
        TensorDeviceBinding dV;
        TensorDeviceBinding dQNorm;
        TensorDeviceBinding dKNorm;
        void* dOutput = nullptr;
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            return X::Value();
        }
        if (!BindTensorInput(input, inputBytes, stream, dInput) ||
            !BindTensorInput(q, qBytes, stream, dQ) ||
            !BindTensorInput(k, kBytes, stream, dK) ||
            !BindTensorInput(v, vBytes, stream, dV) ||
            !BindTensorInput(qNorm, normBytes, stream, dQNorm) ||
            !BindTensorInput(kNorm, normBytes, stream, dKNorm) ||
            cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
            FreeOwnedBinding(dInput);
            FreeOwnedBinding(dQ);
            FreeOwnedBinding(dK);
            FreeOwnedBinding(dV);
            FreeOwnedBinding(dQNorm);
            FreeOwnedBinding(dKNorm);
            if (dOutput) cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }
        bool executed = bf16Weights
            ? runQKVHeadNormBF16WeightFP32(
                static_cast<const float*>(dInput.ptr),
                static_cast<const bfloat16*>(dQ.ptr), static_cast<const bfloat16*>(dK.ptr),
                static_cast<const bfloat16*>(dV.ptr), static_cast<const bfloat16*>(dQNorm.ptr),
                static_cast<const bfloat16*>(dKNorm.ptr), static_cast<float*>(dOutput),
                tokens, hidden, qOut, kOut, headDim, 1.0e-6f, stream) == cudaSuccess
            : (context->setTensorAddress("x", dInput.ptr)
                && context->setTensorAddress("W_q", dQ.ptr)
                && context->setTensorAddress("W_k", dK.ptr)
                && context->setTensorAddress("W_v", dV.ptr)
                && context->setTensorAddress("q_norm", dQNorm.ptr)
                && context->setTensorAddress("k_norm", dKNorm.ptr)
                && context->setTensorAddress("output", dOutput)
                && context->enqueueV3(stream));
        if (!executed) {
            FreeOwnedBinding(dInput);
            FreeOwnedBinding(dQ);
            FreeOwnedBinding(dK);
            FreeOwnedBinding(dV);
            FreeOwnedBinding(dQNorm);
            FreeOwnedBinding(dKNorm);
            cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        X::Value output = MakeGPUBackedTensor2D(tokens, qOut + kOut + vOut, dOutput, outputBytes, stream);
        if (!output.IsValid()) {
            FreeOwnedBinding(dInput);
            FreeOwnedBinding(dQ);
            FreeOwnedBinding(dK);
            FreeOwnedBinding(dV);
            FreeOwnedBinding(dQNorm);
            FreeOwnedBinding(dKNorm);
            cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }
        FreeOwnedBinding(dInput);
        FreeOwnedBinding(dQ);
        FreeOwnedBinding(dK);
        FreeOwnedBinding(dV);
        FreeOwnedBinding(dQNorm);
        FreeOwnedBinding(dKNorm);
        cudaStreamDestroy(stream);
        std::cout << "[TRTBuilder] RunTextQKVHeadNormEngine completed: [" << tokens << ", " << (qOut + kOut + vOut) << "]" << std::endl;
        return output;
    }

    X::Value TRTBuilder::ExportTextRoPEEngine(const std::string& enginePath, const std::vector<int>& qkvShape, const std::vector<int>& cosShape, const std::vector<int>& sinShape, int qHeads, int kvHeads, int headDim) {
        std::cout << "[TRTBuilder] ExportTextRoPEEngine -> " << enginePath << std::endl;
        if (qkvShape.size() != 2 || cosShape.size() != 2 || sinShape.size() != 2 || qHeads <= 0 || kvHeads <= 0 || headDim <= 0 || (headDim % 2) != 0) {
            std::cout << "[TRTBuilder] TextRoPE invalid shapes." << std::endl;
            return X::Value();
        }
        int tokens = qkvShape[0];
        int qOut = qHeads * headDim;
        int kOut = kvHeads * headDim;
        int vOut = kOut;
        int total = qOut + kOut + vOut;
        if (qkvShape[1] != total || cosShape[0] != tokens || sinShape[0] != tokens || cosShape[1] != headDim || sinShape[1] != headDim) {
            std::cout << "[TRTBuilder] TextRoPE shape mismatch." << std::endl;
            return X::Value();
        }

        auto builder = createInferBuilder(gLogger);
        if (!builder) return X::Value();
        uint32_t flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(flags);
        if (!network) return X::Value();
        auto config = builder->createBuilderConfig();
        if (!config) return X::Value();
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 128ULL << 20);

        Dims qkvDims{};
        qkvDims.nbDims = 2;
        qkvDims.d[0] = tokens;
        qkvDims.d[1] = total;
        Dims posDims{};
        posDims.nbDims = 2;
        posDims.d[0] = tokens;
        posDims.d[1] = headDim;
        ITensor* qkv = network->addInput("qkv", DataType::kFLOAT, qkvDims);
        ITensor* cos = network->addInput("cos", DataType::kFLOAT, posDims);
        ITensor* sin = network->addInput("sin", DataType::kFLOAT, posDims);
        if (!qkv || !cos || !sin) return X::Value();

        auto slice2d = [&](ITensor* src, int colStart, int width) -> ITensor* {
            Dims start{};
            start.nbDims = 2;
            start.d[0] = 0;
            start.d[1] = colStart;
            Dims size{};
            size.nbDims = 2;
            size.d[0] = tokens;
            size.d[1] = width;
            Dims stride{};
            stride.nbDims = 2;
            stride.d[0] = 1;
            stride.d[1] = 1;
            auto layer = network->addSlice(*src, start, size, stride);
            if (!layer) return nullptr;
            return layer->getOutput(0);
        };
        auto reshape3d = [&](ITensor* src, int heads) -> ITensor* {
            Dims shape{};
            shape.nbDims = 3;
            shape.d[0] = tokens;
            shape.d[1] = heads;
            shape.d[2] = headDim;
            auto layer = network->addShuffle(*src);
            if (!layer) return nullptr;
            layer->setReshapeDimensions(shape);
            return layer->getOutput(0);
        };
        auto reshapePos = [&](ITensor* src) -> ITensor* {
            Dims shape{};
            shape.nbDims = 3;
            shape.d[0] = tokens;
            shape.d[1] = 1;
            shape.d[2] = headDim;
            auto layer = network->addShuffle(*src);
            if (!layer) return nullptr;
            layer->setReshapeDimensions(shape);
            return layer->getOutput(0);
        };
        auto flatten2d = [&](ITensor* src, int width) -> ITensor* {
            Dims shape{};
            shape.nbDims = 2;
            shape.d[0] = tokens;
            shape.d[1] = width;
            auto layer = network->addShuffle(*src);
            if (!layer) return nullptr;
            layer->setReshapeDimensions(shape);
            return layer->getOutput(0);
        };
        auto rotateHalf = [&](ITensor* src, int heads) -> ITensor* {
            Dims start1{};
            start1.nbDims = 3;
            start1.d[0] = 0;
            start1.d[1] = 0;
            start1.d[2] = 0;
            Dims start2{};
            start2.nbDims = 3;
            start2.d[0] = 0;
            start2.d[1] = 0;
            start2.d[2] = headDim / 2;
            Dims size{};
            size.nbDims = 3;
            size.d[0] = tokens;
            size.d[1] = heads;
            size.d[2] = headDim / 2;
            Dims stride{};
            stride.nbDims = 3;
            stride.d[0] = 1;
            stride.d[1] = 1;
            stride.d[2] = 1;
            auto first = network->addSlice(*src, start1, size, stride);
            auto second = network->addSlice(*src, start2, size, stride);
            if (!first || !second || !first->getOutput(0) || !second->getOutput(0)) return nullptr;
            auto negSecond = network->addUnary(*second->getOutput(0), UnaryOperation::kNEG);
            if (!negSecond || !negSecond->getOutput(0)) return nullptr;
            ITensor* parts[] = { negSecond->getOutput(0), first->getOutput(0) };
            auto concat = network->addConcatenation(parts, 2);
            if (!concat || !concat->getOutput(0)) return nullptr;
            concat->setAxis(2);
            return concat->getOutput(0);
        };
        auto applyRope = [&](ITensor* src, ITensor* cos3d, ITensor* sin3d, int heads) -> ITensor* {
            auto xCos = network->addElementWise(*src, *cos3d, ElementWiseOperation::kPROD);
            ITensor* rotated = rotateHalf(src, heads);
            if (!xCos || !xCos->getOutput(0) || !rotated) return nullptr;
            auto rotSin = network->addElementWise(*rotated, *sin3d, ElementWiseOperation::kPROD);
            if (!rotSin || !rotSin->getOutput(0)) return nullptr;
            auto out = network->addElementWise(*xCos->getOutput(0), *rotSin->getOutput(0), ElementWiseOperation::kSUM);
            if (!out || !out->getOutput(0)) return nullptr;
            return out->getOutput(0);
        };

        ITensor* qFlat = slice2d(qkv, 0, qOut);
        ITensor* kFlat = slice2d(qkv, qOut, kOut);
        ITensor* vFlat = slice2d(qkv, qOut + kOut, vOut);
        ITensor* q3d = qFlat ? reshape3d(qFlat, qHeads) : nullptr;
        ITensor* k3d = kFlat ? reshape3d(kFlat, kvHeads) : nullptr;
        ITensor* cos3d = reshapePos(cos);
        ITensor* sin3d = reshapePos(sin);
        if (!q3d || !k3d || !vFlat || !cos3d || !sin3d) return X::Value();
        ITensor* qRope = applyRope(q3d, cos3d, sin3d, qHeads);
        ITensor* kRope = applyRope(k3d, cos3d, sin3d, kvHeads);
        if (!qRope || !kRope) return X::Value();
        ITensor* qOutFlat = flatten2d(qRope, qOut);
        ITensor* kOutFlat = flatten2d(kRope, kOut);
        if (!qOutFlat || !kOutFlat) return X::Value();
        ITensor* concatInputs[] = { qOutFlat, kOutFlat, vFlat };
        auto concat = network->addConcatenation(concatInputs, 3);
        if (!concat || !concat->getOutput(0)) return X::Value();
        concat->setAxis(1);
        concat->getOutput(0)->setName("output");
        network->markOutput(*concat->getOutput(0));

        auto serialized = builder->buildSerializedNetwork(*network, *config);
        if (!serialized) return X::Value();
        std::filesystem::path outputPath(enginePath);
        std::filesystem::create_directories(outputPath.parent_path());
        std::ofstream outFile(outputPath, std::ios::binary);
        if (!outFile.is_open()) return X::Value();
        outFile.write(static_cast<const char*>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
        outFile.close();
        std::cout << "[TRTBuilder] Serialized TextRoPE TensorRT engine bytes: " << serialized->size() << std::endl;
        return X::Value(enginePath);
    }

    X::Value TRTBuilder::RunTextRoPEEngine(const std::string& enginePath, X::Value qkvValue, X::Value cosValue, X::Value sinValue) {
        std::cout << "[TRTBuilder] RunTextRoPEEngine <- " << enginePath << std::endl;
        if (!qkvValue.IsTensor() || !cosValue.IsTensor() || !sinValue.IsTensor()) return X::Value();
        X::Tensor qkv(qkvValue);
        X::Tensor cos(cosValue);
        X::Tensor sin(sinValue);
        if (qkv->GetDataType() != X::TensorDataType::FLOAT32 || cos->GetDataType() != X::TensorDataType::FLOAT32 || sin->GetDataType() != X::TensorDataType::FLOAT32) return X::Value();
        if (qkv->GetDimCount() != 2 || cos->GetDimCount() != 2 || sin->GetDimCount() != 2) return X::Value();
        int tokens = qkv->GetDimSize(0);
        int total = qkv->GetDimSize(1);
        int headDim = cos->GetDimSize(1);
        if (cos->GetDimSize(0) != tokens || sin->GetDimSize(0) != tokens || sin->GetDimSize(1) != headDim) return X::Value();

        auto context = GetCachedTRTExecutionContext(enginePath);
        if (!context) return X::Value();

        size_t qkvBytes = static_cast<size_t>(tokens) * static_cast<size_t>(total) * sizeof(float);
        size_t posBytes = static_cast<size_t>(tokens) * static_cast<size_t>(headDim) * sizeof(float);
        TensorDeviceBinding dQKV;
        TensorDeviceBinding dCos;
        TensorDeviceBinding dSin;
        void* dOutput = nullptr;
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            return X::Value();
        }
        if (!BindTensorInput(qkv, qkvBytes, stream, dQKV) ||
            !BindTensorInput(cos, posBytes, stream, dCos) ||
            !BindTensorInput(sin, posBytes, stream, dSin) ||
            cudaMalloc(&dOutput, qkvBytes) != cudaSuccess) {
            FreeOwnedBinding(dQKV);
            FreeOwnedBinding(dCos);
            FreeOwnedBinding(dSin);
            if (dOutput) cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }
        bool bound = context->setTensorAddress("qkv", dQKV.ptr)
            && context->setTensorAddress("cos", dCos.ptr)
            && context->setTensorAddress("sin", dSin.ptr)
            && context->setTensorAddress("output", dOutput);
        if (!bound || !context->enqueueV3(stream)) {
            FreeOwnedBinding(dQKV);
            FreeOwnedBinding(dCos);
            FreeOwnedBinding(dSin);
            cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        X::Value output = MakeGPUBackedTensor2D(tokens, total, dOutput, qkvBytes, stream);
        if (!output.IsValid()) {
            FreeOwnedBinding(dQKV);
            FreeOwnedBinding(dCos);
            FreeOwnedBinding(dSin);
            cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }
        FreeOwnedBinding(dQKV);
        FreeOwnedBinding(dCos);
        FreeOwnedBinding(dSin);
        cudaStreamDestroy(stream);
        std::cout << "[TRTBuilder] RunTextRoPEEngine completed: [" << tokens << ", " << total << "]" << std::endl;
        return output;
    }

    X::Value TRTBuilder::ExportTextAttentionEngine(const std::string& enginePath, const std::vector<int>& qkvShape, int qHeads, int kvHeads, int headDim) {
        std::cout << "[TRTBuilder] ExportTextAttentionEngine -> " << enginePath << std::endl;
        if (qkvShape.size() != 2 || qHeads <= 0 || kvHeads <= 0 || headDim <= 0 || qHeads % kvHeads != 0) return X::Value();
        int tokens = qkvShape[0];
        int qOut = qHeads * headDim;
        int kOut = kvHeads * headDim;
        int vOut = kOut;
        int total = qOut + kOut + vOut;
        int repeat = qHeads / kvHeads;
        if (qkvShape[1] != total) return X::Value();

        auto builder = createInferBuilder(gLogger);
        if (!builder) return X::Value();
        uint32_t flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(flags);
        if (!network) return X::Value();
        auto config = builder->createBuilderConfig();
        if (!config) return X::Value();
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 256ULL << 20);

        Dims qkvDims{};
        qkvDims.nbDims = 2;
        qkvDims.d[0] = tokens;
        qkvDims.d[1] = total;
        ITensor* qkv = network->addInput("qkv", DataType::kFLOAT, qkvDims);
        if (!qkv) return X::Value();

        auto slice2d = [&](int colStart, int width) -> ITensor* {
            Dims start{};
            start.nbDims = 2;
            start.d[0] = 0;
            start.d[1] = colStart;
            Dims size{};
            size.nbDims = 2;
            size.d[0] = tokens;
            size.d[1] = width;
            Dims stride{};
            stride.nbDims = 2;
            stride.d[0] = 1;
            stride.d[1] = 1;
            auto layer = network->addSlice(*qkv, start, size, stride);
            if (!layer) return nullptr;
            return layer->getOutput(0);
        };
        auto reshapeTHD = [&](ITensor* src, int heads) -> ITensor* {
            Dims shape{};
            shape.nbDims = 3;
            shape.d[0] = tokens;
            shape.d[1] = heads;
            shape.d[2] = headDim;
            auto layer = network->addShuffle(*src);
            if (!layer) return nullptr;
            layer->setReshapeDimensions(shape);
            return layer->getOutput(0);
        };
        auto transposeTHDToHTD = [&](ITensor* src) -> ITensor* {
            auto layer = network->addShuffle(*src);
            if (!layer) return nullptr;
            Permutation perm{};
            perm.order[0] = 1;
            perm.order[1] = 0;
            perm.order[2] = 2;
            layer->setFirstTranspose(perm);
            return layer->getOutput(0);
        };
        auto transposeHTDToTHD = [&](ITensor* src) -> ITensor* {
            auto layer = network->addShuffle(*src);
            if (!layer) return nullptr;
            Permutation perm{};
            perm.order[0] = 1;
            perm.order[1] = 0;
            perm.order[2] = 2;
            layer->setFirstTranspose(perm);
            return layer->getOutput(0);
        };
        auto flattenTHD = [&](ITensor* src) -> ITensor* {
            Dims shape{};
            shape.nbDims = 2;
            shape.d[0] = tokens;
            shape.d[1] = qOut;
            auto layer = network->addShuffle(*src);
            if (!layer) return nullptr;
            layer->setReshapeDimensions(shape);
            return layer->getOutput(0);
        };
        auto repeatKV = [&](ITensor* src) -> ITensor* {
            std::vector<ITensor*> pieces;
            pieces.reserve(qHeads);
            for (int h = 0; h < kvHeads; ++h) {
                Dims start{};
                start.nbDims = 3;
                start.d[0] = h;
                start.d[1] = 0;
                start.d[2] = 0;
                Dims size{};
                size.nbDims = 3;
                size.d[0] = 1;
                size.d[1] = tokens;
                size.d[2] = headDim;
                Dims stride{};
                stride.nbDims = 3;
                stride.d[0] = 1;
                stride.d[1] = 1;
                stride.d[2] = 1;
                auto slice = network->addSlice(*src, start, size, stride);
                if (!slice || !slice->getOutput(0)) return nullptr;
                for (int r = 0; r < repeat; ++r) {
                    pieces.push_back(slice->getOutput(0));
                }
            }
            auto concat = network->addConcatenation(pieces.data(), static_cast<int32_t>(pieces.size()));
            if (!concat || !concat->getOutput(0)) return nullptr;
            concat->setAxis(0);
            return concat->getOutput(0);
        };

        ITensor* qFlat = slice2d(0, qOut);
        ITensor* kFlat = slice2d(qOut, kOut);
        ITensor* vFlat = slice2d(qOut + kOut, vOut);
        ITensor* qHTD = qFlat ? transposeTHDToHTD(reshapeTHD(qFlat, qHeads)) : nullptr;
        ITensor* kHTD = kFlat ? transposeTHDToHTD(reshapeTHD(kFlat, kvHeads)) : nullptr;
        ITensor* vHTD = vFlat ? transposeTHDToHTD(reshapeTHD(vFlat, kvHeads)) : nullptr;
        if (!qHTD || !kHTD || !vHTD) return X::Value();
        ITensor* kRepeated = repeatKV(kHTD);
        ITensor* vRepeated = repeatKV(vHTD);
        if (!kRepeated || !vRepeated) return X::Value();

        auto scores = network->addMatrixMultiply(*qHTD, MatrixOperation::kNONE, *kRepeated, MatrixOperation::kTRANSPOSE);
        if (!scores || !scores->getOutput(0)) return X::Value();
        float scaleValue = 1.0f / std::sqrt(static_cast<float>(headDim));
        Dims scalarDims{};
        scalarDims.nbDims = 3;
        scalarDims.d[0] = 1;
        scalarDims.d[1] = 1;
        scalarDims.d[2] = 1;
        Weights scaleWeights{ DataType::kFLOAT, &scaleValue, 1 };
        auto scaleConst = network->addConstant(scalarDims, scaleWeights);
        if (!scaleConst || !scaleConst->getOutput(0)) return X::Value();
        auto scaledScores = network->addElementWise(*scores->getOutput(0), *scaleConst->getOutput(0), ElementWiseOperation::kPROD);
        if (!scaledScores || !scaledScores->getOutput(0)) return X::Value();

        std::vector<float> mask(static_cast<size_t>(tokens) * static_cast<size_t>(tokens), 0.0f);
        for (int i = 0; i < tokens; ++i) {
            for (int j = i + 1; j < tokens; ++j) {
                mask[static_cast<size_t>(i) * static_cast<size_t>(tokens) + static_cast<size_t>(j)] = -10000.0f;
            }
        }
        Dims maskDims{};
        maskDims.nbDims = 3;
        maskDims.d[0] = 1;
        maskDims.d[1] = tokens;
        maskDims.d[2] = tokens;
        Weights maskWeights{ DataType::kFLOAT, mask.data(), static_cast<int64_t>(mask.size()) };
        auto maskConst = network->addConstant(maskDims, maskWeights);
        if (!maskConst || !maskConst->getOutput(0)) return X::Value();
        auto maskedScores = network->addElementWise(*scaledScores->getOutput(0), *maskConst->getOutput(0), ElementWiseOperation::kSUM);
        if (!maskedScores || !maskedScores->getOutput(0)) return X::Value();

        auto softmax = network->addSoftMax(*maskedScores->getOutput(0));
        if (!softmax || !softmax->getOutput(0)) return X::Value();
        softmax->setAxes(1U << 2);
        auto context = network->addMatrixMultiply(*softmax->getOutput(0), MatrixOperation::kNONE, *vRepeated, MatrixOperation::kNONE);
        if (!context || !context->getOutput(0)) return X::Value();
        ITensor* thd = transposeHTDToTHD(context->getOutput(0));
        ITensor* output = thd ? flattenTHD(thd) : nullptr;
        if (!output) return X::Value();
        output->setName("output");
        network->markOutput(*output);

        auto serialized = builder->buildSerializedNetwork(*network, *config);
        if (!serialized) return X::Value();
        std::filesystem::path outputPath(enginePath);
        std::filesystem::create_directories(outputPath.parent_path());
        std::ofstream outFile(outputPath, std::ios::binary);
        if (!outFile.is_open()) return X::Value();
        outFile.write(static_cast<const char*>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
        outFile.close();
        std::cout << "[TRTBuilder] Serialized TextAttention TensorRT engine bytes: " << serialized->size() << std::endl;
        return X::Value(enginePath);
    }

    X::Value TRTBuilder::RunTextAttentionEngine(const std::string& enginePath, X::Value qkvValue) {
        std::cout << "[TRTBuilder] RunTextAttentionEngine <- " << enginePath << std::endl;
        if (!qkvValue.IsTensor()) return X::Value();
        X::Tensor qkv(qkvValue);
        if (qkv->GetDataType() != X::TensorDataType::FLOAT32 || qkv->GetDimCount() != 2) return X::Value();
        int tokens = qkv->GetDimSize(0);
        int total = qkv->GetDimSize(1);

        auto context = GetCachedTRTExecutionContext(enginePath);
        if (!context) return X::Value();

        size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(total) * sizeof(float);
        size_t outputBytes = static_cast<size_t>(tokens) * 2048ULL * sizeof(float);
        TensorDeviceBinding dInput;
        void* dOutput = nullptr;
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            return X::Value();
        }
        if (!BindTensorInput(qkv, inputBytes, stream, dInput) ||
            cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
            FreeOwnedBinding(dInput);
            if (dOutput) cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }
        bool bound = context->setTensorAddress("qkv", dInput.ptr)
            && context->setTensorAddress("output", dOutput);
        if (!bound || !context->enqueueV3(stream)) {
            FreeOwnedBinding(dInput);
            cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        X::Value output = MakeGPUBackedTensor2D(tokens, 2048, dOutput, outputBytes, stream);
        if (!output.IsValid()) {
            FreeOwnedBinding(dInput);
            cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }
        FreeOwnedBinding(dInput);
        cudaStreamDestroy(stream);
        std::cout << "[TRTBuilder] RunTextAttentionEngine completed: [" << tokens << ", 2048]" << std::endl;
        return output;
    }

    X::Value TRTBuilder::ExportVisionAttentionEngine(const std::string& enginePath, const std::vector<int>& qkvShape, int heads, int headDim) {
        std::cout << "[TRTBuilder] ExportVisionAttentionEngine -> " << enginePath << std::endl;
        if (qkvShape.size() != 2 || heads <= 0 || headDim <= 0) return X::Value();
        int tokens = qkvShape[0];
        int hidden = heads * headDim;
        int total = hidden * 3;
        if (qkvShape[1] != total) return X::Value();
        if (tokens > 512) {
            std::cout << "[TRTBuilder] VisionAttention tokens=" << tokens
                << " uses CUDA exact attention path; skipping TensorRT score-matrix engine." << std::endl;
            return X::Value("cuda_exact_vision_attention");
        }

        auto builder = createInferBuilder(gLogger);
        if (!builder) return X::Value();
        uint32_t flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(flags);
        if (!network) return X::Value();
        auto config = builder->createBuilderConfig();
        if (!config) return X::Value();
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 256ULL << 20);

        Dims qkvDims{};
        qkvDims.nbDims = 2;
        qkvDims.d[0] = tokens;
        qkvDims.d[1] = total;
        ITensor* qkv = network->addInput("qkv", DataType::kFLOAT, qkvDims);
        if (!qkv) return X::Value();

        auto slice2d = [&](int colStart, int width) -> ITensor* {
            Dims start{};
            start.nbDims = 2;
            start.d[0] = 0;
            start.d[1] = colStart;
            Dims size{};
            size.nbDims = 2;
            size.d[0] = tokens;
            size.d[1] = width;
            Dims stride{};
            stride.nbDims = 2;
            stride.d[0] = 1;
            stride.d[1] = 1;
            auto layer = network->addSlice(*qkv, start, size, stride);
            if (!layer) return nullptr;
            return layer->getOutput(0);
        };
        auto reshapeTHD = [&](ITensor* src) -> ITensor* {
            Dims shape{};
            shape.nbDims = 3;
            shape.d[0] = tokens;
            shape.d[1] = heads;
            shape.d[2] = headDim;
            auto layer = network->addShuffle(*src);
            if (!layer) return nullptr;
            layer->setReshapeDimensions(shape);
            return layer->getOutput(0);
        };
        auto transposeTHDToHTD = [&](ITensor* src) -> ITensor* {
            auto layer = network->addShuffle(*src);
            if (!layer) return nullptr;
            Permutation perm{};
            perm.order[0] = 1;
            perm.order[1] = 0;
            perm.order[2] = 2;
            layer->setFirstTranspose(perm);
            return layer->getOutput(0);
        };
        auto transposeHTDToTHD = [&](ITensor* src) -> ITensor* {
            auto layer = network->addShuffle(*src);
            if (!layer) return nullptr;
            Permutation perm{};
            perm.order[0] = 1;
            perm.order[1] = 0;
            perm.order[2] = 2;
            layer->setFirstTranspose(perm);
            return layer->getOutput(0);
        };
        auto flattenTHD = [&](ITensor* src) -> ITensor* {
            Dims shape{};
            shape.nbDims = 2;
            shape.d[0] = tokens;
            shape.d[1] = hidden;
            auto layer = network->addShuffle(*src);
            if (!layer) return nullptr;
            layer->setReshapeDimensions(shape);
            return layer->getOutput(0);
        };

        ITensor* qFlat = slice2d(0, hidden);
        ITensor* kFlat = slice2d(hidden, hidden);
        ITensor* vFlat = slice2d(hidden * 2, hidden);
        ITensor* qHTD = qFlat ? transposeTHDToHTD(reshapeTHD(qFlat)) : nullptr;
        ITensor* kHTD = kFlat ? transposeTHDToHTD(reshapeTHD(kFlat)) : nullptr;
        ITensor* vHTD = vFlat ? transposeTHDToHTD(reshapeTHD(vFlat)) : nullptr;
        if (!qHTD || !kHTD || !vHTD) return X::Value();

        auto scores = network->addMatrixMultiply(*qHTD, MatrixOperation::kNONE, *kHTD, MatrixOperation::kTRANSPOSE);
        if (!scores || !scores->getOutput(0)) return X::Value();
        float scaleValue = 1.0f / std::sqrt(static_cast<float>(headDim));
        Dims scalarDims{};
        scalarDims.nbDims = 3;
        scalarDims.d[0] = 1;
        scalarDims.d[1] = 1;
        scalarDims.d[2] = 1;
        Weights scaleWeights{ DataType::kFLOAT, &scaleValue, 1 };
        auto scaleConst = network->addConstant(scalarDims, scaleWeights);
        if (!scaleConst || !scaleConst->getOutput(0)) return X::Value();
        auto scaledScores = network->addElementWise(*scores->getOutput(0), *scaleConst->getOutput(0), ElementWiseOperation::kPROD);
        if (!scaledScores || !scaledScores->getOutput(0)) return X::Value();
        auto softmax = network->addSoftMax(*scaledScores->getOutput(0));
        if (!softmax || !softmax->getOutput(0)) return X::Value();
        softmax->setAxes(1U << 2);
        auto context = network->addMatrixMultiply(*softmax->getOutput(0), MatrixOperation::kNONE, *vHTD, MatrixOperation::kNONE);
        if (!context || !context->getOutput(0)) return X::Value();
        ITensor* thd = transposeHTDToTHD(context->getOutput(0));
        ITensor* output = thd ? flattenTHD(thd) : nullptr;
        if (!output) return X::Value();
        output->setName("output");
        network->markOutput(*output);

        auto serialized = builder->buildSerializedNetwork(*network, *config);
        if (!serialized) return X::Value();
        std::filesystem::path outputPath(enginePath);
        std::filesystem::create_directories(outputPath.parent_path());
        std::ofstream outFile(outputPath, std::ios::binary);
        if (!outFile.is_open()) return X::Value();
        outFile.write(static_cast<const char*>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
        outFile.close();
        std::cout << "[TRTBuilder] Serialized VisionAttention TensorRT engine bytes: " << serialized->size() << std::endl;
        return X::Value(enginePath);
    }

    X::Value TRTBuilder::RunVisionAttentionEngine(const std::string& enginePath, X::Value qkvValue) {
        std::cout << "[TRTBuilder] RunVisionAttentionEngine <- " << enginePath << std::endl;
        if (!qkvValue.IsTensor()) return X::Value();
        X::Tensor qkv(qkvValue);
        if (qkv->GetDataType() != X::TensorDataType::FLOAT32 || qkv->GetDimCount() != 2 || qkv->GetDimSize(1) % 3 != 0) return X::Value();
        int tokens = qkv->GetDimSize(0);
        int total = qkv->GetDimSize(1);
        int hidden = total / 3;
        int heads = 16;
        int headDim = hidden / heads;
        if (hidden % heads != 0) {
            std::cout << "[TRTBuilder] VisionAttention hidden size is not divisible by heads." << std::endl;
            return X::Value();
        }

        if (enginePath == "cuda_exact_vision_attention" || tokens > 512) {
            std::cout << "[TRTBuilder] Running CUDA exact vision attention: tokens=" << tokens
                << ", heads=" << heads << ", headDim=" << headDim << std::endl;
            size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(total) * sizeof(float);
            size_t outputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);
            void* dInput = nullptr;
            void* dOutput = nullptr;
            cudaStream_t stream = nullptr;
            TensorDeviceBinding inputBinding;
            if (cudaStreamCreate(&stream) != cudaSuccess ||
                !BindTensorInput(qkv, inputBytes, stream, inputBinding) ||
                cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
                FreeOwnedBinding(inputBinding);
                if (dOutput) cudaFree(dOutput);
                if (stream) cudaStreamDestroy(stream);
                std::cout << "[TRTBuilder] CUDA exact vision attention allocation failed." << std::endl;
                return X::Value();
            }
            dInput = inputBinding.ptr;
            cudaError_t launchErr = runVisionAttentionFP32(
                static_cast<const float*>(dInput),
                static_cast<float*>(dOutput),
                tokens,
                heads,
                headDim,
                stream);
            if (launchErr != cudaSuccess) {
                std::cout << "[TRTBuilder] CUDA exact vision attention launch failed: "
                    << cudaGetErrorString(launchErr) << std::endl;
                FreeOwnedBinding(inputBinding); cudaFree(dOutput); cudaStreamDestroy(stream);
                return X::Value();
            }
            X::Value output = MakeGPUBackedTensor2D(tokens, hidden, dOutput, outputBytes, stream);
            if (!output.IsValid()) {
                FreeOwnedBinding(inputBinding); cudaFree(dOutput); cudaStreamDestroy(stream);
                return X::Value();
            }
            FreeOwnedBinding(inputBinding);
            cudaStreamDestroy(stream);
            std::cout << "[TRTBuilder] CUDA exact vision attention completed: [" << tokens
                << ", " << hidden << "]" << std::endl;
            return output;
        }

        auto context = GetCachedTRTExecutionContext(enginePath);
        if (!context) return X::Value();

        size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(total) * sizeof(float);
        size_t outputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);
        void* dInput = nullptr;
        void* dOutput = nullptr;
        cudaStream_t stream = nullptr;
        TensorDeviceBinding inputBinding;
        if (cudaStreamCreate(&stream) != cudaSuccess ||
            !BindTensorInput(qkv, inputBytes, stream, inputBinding) ||
            cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
            FreeOwnedBinding(inputBinding);
            if (dOutput) cudaFree(dOutput);
            if (stream) cudaStreamDestroy(stream);
            return X::Value();
        }
        dInput = inputBinding.ptr;
        bool bound = context->setTensorAddress("qkv", dInput)
            && context->setTensorAddress("output", dOutput);
        if (!bound || !context->enqueueV3(stream)) {
            FreeOwnedBinding(inputBinding); cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        X::Value output = MakeGPUBackedTensor2D(tokens, hidden, dOutput, outputBytes, stream);
        if (!output.IsValid()) {
            FreeOwnedBinding(inputBinding); cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        FreeOwnedBinding(inputBinding); cudaStreamDestroy(stream);
        std::cout << "[TRTBuilder] RunVisionAttentionEngine completed: [" << tokens << ", " << hidden << "]" << std::endl;
        return output;
    }

    X::Value TRTBuilder::ExportLinearTransposeEngine(const std::string& enginePath, const std::vector<int>& inputShape, const std::vector<int>& weightShape) {
        std::cout << "[TRTBuilder] ExportLinearTransposeEngine -> " << enginePath << std::endl;
        if (inputShape.size() != 2 || weightShape.size() != 2 || inputShape[1] != weightShape[1]) {
            std::cout << "[TRTBuilder] LinearTranspose shape mismatch." << std::endl;
            return X::Value();
        }
        int tokens = inputShape[0];
        int inFeatures = inputShape[1];
        int outFeatures = weightShape[0];

        auto builder = createInferBuilder(gLogger);
        if (!builder) return X::Value();
        uint32_t flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(flags);
        if (!network) return X::Value();
        auto config = builder->createBuilderConfig();
        if (!config) return X::Value();
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 64ULL << 20);

        Dims xDims{};
        xDims.nbDims = 2;
        xDims.d[0] = tokens;
        xDims.d[1] = inFeatures;
        Dims wDims{};
        wDims.nbDims = 2;
        wDims.d[0] = outFeatures;
        wDims.d[1] = inFeatures;
        ITensor* x = network->addInput("x", DataType::kFLOAT, xDims);
        ITensor* w = network->addInput("W", DataType::kFLOAT, wDims);
        if (!x || !w) return X::Value();
        auto out = network->addMatrixMultiply(*x, MatrixOperation::kNONE, *w, MatrixOperation::kTRANSPOSE);
        if (!out || !out->getOutput(0)) return X::Value();
        out->getOutput(0)->setName("output");
        network->markOutput(*out->getOutput(0));

        auto serialized = builder->buildSerializedNetwork(*network, *config);
        if (!serialized) return X::Value();
        std::filesystem::path outputPath(enginePath);
        std::filesystem::create_directories(outputPath.parent_path());
        std::ofstream outFile(outputPath, std::ios::binary);
        if (!outFile.is_open()) return X::Value();
        outFile.write(static_cast<const char*>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
        outFile.close();
        std::cout << "[TRTBuilder] Serialized LinearTranspose TensorRT engine bytes: " << serialized->size() << std::endl;
        return X::Value(enginePath);
    }

    X::Value TRTBuilder::RunLinearTransposeEngine(const std::string& enginePath, X::Value inputValue, X::Value weightValue) {
        std::cout << "[TRTBuilder] RunLinearTransposeEngine <- " << enginePath << std::endl;
        if (!inputValue.IsTensor() || !weightValue.IsTensor()) return X::Value();
        X::Tensor input(inputValue);
        X::Tensor weight(weightValue);
        bool bf16Weight = weight->GetDataType() == X::TensorDataType::BFLOAT16;
        if (input->GetDataType() != X::TensorDataType::FLOAT32 ||
            (!bf16Weight && weight->GetDataType() != X::TensorDataType::FLOAT32)) return X::Value();
        if (input->GetDimCount() != 2 || weight->GetDimCount() != 2) return X::Value();
        int tokens = input->GetDimSize(0);
        int inFeatures = input->GetDimSize(1);
        int outFeatures = weight->GetDimSize(0);
        if (weight->GetDimSize(1) != inFeatures) return X::Value();

        if (enginePath == "cuda_linear_transpose" || bf16Weight) {
            std::cout << "[TRTBuilder] Running CUDA linear transpose: ["
                << tokens << ", " << inFeatures << "] x [" << outFeatures << ", "
                << inFeatures << "]" << std::endl;
            size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(inFeatures) * sizeof(float);
            size_t weightBytes = static_cast<size_t>(outFeatures) * static_cast<size_t>(inFeatures) * weight->GetItemSize();
            size_t outputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(outFeatures) * sizeof(float);
            TensorDeviceBinding dInput;
            TensorDeviceBinding dWeight;
            void* dOutput = nullptr;
            cudaStream_t stream = nullptr;
            if (cudaStreamCreate(&stream) != cudaSuccess) {
                std::cout << "[TRTBuilder] CUDA linear transpose stream create failed." << std::endl;
                return X::Value();
            }
            if (!BindTensorInput(input, inputBytes, stream, dInput) ||
                !BindTensorInput(weight, weightBytes, stream, dWeight) ||
                cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
                size_t freeBytes = 0;
                size_t totalBytes = 0;
                cudaMemGetInfo(&freeBytes, &totalBytes);
                std::cout << "[TRTBuilder] CUDA linear transpose allocation/bind failed: "
                    << cudaGetErrorString(cudaGetLastError())
                    << ", free=" << (freeBytes / (1024 * 1024)) << " MiB"
                    << ", requested_output=" << (outputBytes / (1024 * 1024)) << " MiB"
                    << std::endl;
                FreeOwnedBinding(dInput);
                FreeOwnedBinding(dWeight);
                if (dOutput) cudaFree(dOutput);
                cudaStreamDestroy(stream);
                return X::Value();
            }
            cudaError_t status = bf16Weight
                ? runLinearTransposeBF16WeightFP32(
                    static_cast<const float*>(dInput.ptr),
                    static_cast<const bfloat16*>(dWeight.ptr),
                    static_cast<float*>(dOutput), tokens, inFeatures, outFeatures, stream)
                : runLinearTransposeFP32(
                    static_cast<const float*>(dInput.ptr),
                    static_cast<const float*>(dWeight.ptr),
                    static_cast<float*>(dOutput), tokens, inFeatures, outFeatures, stream);
            if (status != cudaSuccess) {
                std::cout << "[TRTBuilder] CUDA linear transpose launch failed: "
                    << cudaGetErrorString(status) << std::endl;
                FreeOwnedBinding(dInput);
                FreeOwnedBinding(dWeight);
                cudaFree(dOutput); cudaStreamDestroy(stream);
                return X::Value();
            }
            X::Value output = MakeGPUBackedTensor2D(tokens, outFeatures, dOutput, outputBytes, stream);
            if (!output.IsValid()) {
                std::cout << "[TRTBuilder] CUDA linear transpose output wrap failed." << std::endl;
                FreeOwnedBinding(dInput);
                FreeOwnedBinding(dWeight);
                cudaFree(dOutput);
                cudaStreamDestroy(stream);
                return X::Value();
            }
            FreeOwnedBinding(dInput);
            FreeOwnedBinding(dWeight);
            cudaStreamDestroy(stream);
            std::cout << "[TRTBuilder] CUDA linear transpose completed: [" << tokens
                << ", " << outFeatures << "]" << std::endl;
            return output;
        }

        auto context = GetCachedTRTExecutionContext(enginePath);
        if (!context) return X::Value();

        size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(inFeatures) * sizeof(float);
        size_t weightBytes = static_cast<size_t>(outFeatures) * static_cast<size_t>(inFeatures) * sizeof(float);
        size_t outputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(outFeatures) * sizeof(float);
        TensorDeviceBinding dInput;
        TensorDeviceBinding dWeight;
        void* dOutput = nullptr;
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            return X::Value();
        }
        if (!BindTensorInput(input, inputBytes, stream, dInput) ||
            !BindTensorInput(weight, weightBytes, stream, dWeight) ||
            cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
            FreeOwnedBinding(dInput);
            FreeOwnedBinding(dWeight);
            if (dOutput) cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }
        bool bound = context->setTensorAddress("x", dInput.ptr)
            && context->setTensorAddress("W", dWeight.ptr)
            && context->setTensorAddress("output", dOutput);
        if (!bound || !context->enqueueV3(stream)) {
            FreeOwnedBinding(dInput);
            FreeOwnedBinding(dWeight);
            cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        X::Value output = MakeGPUBackedTensor2D(tokens, outFeatures, dOutput, outputBytes, stream);
        if (!output.IsValid()) {
            FreeOwnedBinding(dInput);
            FreeOwnedBinding(dWeight);
            cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }
        FreeOwnedBinding(dInput);
        FreeOwnedBinding(dWeight);
        cudaStreamDestroy(stream);
        std::cout << "[TRTBuilder] RunLinearTransposeEngine completed: [" << tokens << ", " << outFeatures << "]" << std::endl;
        return output;
    }

    X::Value TRTBuilder::ExportLinearBiasTransposeEngine(const std::string& enginePath, const std::vector<int>& inputShape, const std::vector<int>& weightShape, const std::vector<int>& biasShape) {
        std::cout << "[TRTBuilder] ExportLinearBiasTransposeEngine -> " << enginePath << std::endl;
        if (inputShape.size() != 2 || weightShape.size() != 2 || biasShape.size() != 1 || inputShape[1] != weightShape[1] || biasShape[0] != weightShape[0]) {
            std::cout << "[TRTBuilder] LinearBiasTranspose shape mismatch." << std::endl;
            return X::Value();
        }
        int tokens = inputShape[0];
        int inFeatures = inputShape[1];
        int outFeatures = weightShape[0];

        auto builder = createInferBuilder(gLogger);
        if (!builder) return X::Value();
        uint32_t flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(flags);
        if (!network) return X::Value();
        auto config = builder->createBuilderConfig();
        if (!config) return X::Value();
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 64ULL << 20);

        Dims xDims{};
        xDims.nbDims = 2;
        xDims.d[0] = tokens;
        xDims.d[1] = inFeatures;
        Dims wDims{};
        wDims.nbDims = 2;
        wDims.d[0] = outFeatures;
        wDims.d[1] = inFeatures;
        Dims bDims{};
        bDims.nbDims = 1;
        bDims.d[0] = outFeatures;
        ITensor* x = network->addInput("x", DataType::kFLOAT, xDims);
        ITensor* w = network->addInput("W", DataType::kFLOAT, wDims);
        ITensor* b = network->addInput("B", DataType::kFLOAT, bDims);
        if (!x || !w || !b) return X::Value();
        auto linear = network->addMatrixMultiply(*x, MatrixOperation::kNONE, *w, MatrixOperation::kTRANSPOSE);
        if (!linear || !linear->getOutput(0)) return X::Value();
        auto biasShuffle = network->addShuffle(*b);
        if (!biasShuffle || !biasShuffle->getOutput(0)) return X::Value();
        Dims bias2d{};
        bias2d.nbDims = 2;
        bias2d.d[0] = 1;
        bias2d.d[1] = outFeatures;
        biasShuffle->setReshapeDimensions(bias2d);
        auto out = network->addElementWise(*linear->getOutput(0), *biasShuffle->getOutput(0), ElementWiseOperation::kSUM);
        if (!out || !out->getOutput(0)) return X::Value();
        out->getOutput(0)->setName("output");
        network->markOutput(*out->getOutput(0));

        auto serialized = builder->buildSerializedNetwork(*network, *config);
        if (!serialized) return X::Value();
        std::filesystem::path outputPath(enginePath);
        std::filesystem::create_directories(outputPath.parent_path());
        std::ofstream outFile(outputPath, std::ios::binary);
        if (!outFile.is_open()) return X::Value();
        outFile.write(static_cast<const char*>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
        outFile.close();
        std::cout << "[TRTBuilder] Serialized LinearBiasTranspose TensorRT engine bytes: " << serialized->size() << std::endl;
        return X::Value(enginePath);
    }

    X::Value TRTBuilder::RunLinearBiasTransposeEngine(const std::string& enginePath, X::Value inputValue, X::Value weightValue, X::Value biasValue) {
        std::cout << "[TRTBuilder] RunLinearBiasTransposeEngine <- " << enginePath << std::endl;
        if (!inputValue.IsTensor() || !weightValue.IsTensor() || !biasValue.IsTensor()) return X::Value();
        X::Tensor input(inputValue);
        X::Tensor weight(weightValue);
        X::Tensor bias(biasValue);
        bool bf16Weights = weight->GetDataType() == X::TensorDataType::BFLOAT16 &&
            bias->GetDataType() == X::TensorDataType::BFLOAT16;
        if (input->GetDataType() != X::TensorDataType::FLOAT32 ||
            (!bf16Weights && (weight->GetDataType() != X::TensorDataType::FLOAT32 ||
                bias->GetDataType() != X::TensorDataType::FLOAT32))) return X::Value();
        if (input->GetDimCount() != 2 || weight->GetDimCount() != 2 || bias->GetDimCount() != 1) return X::Value();
        int tokens = input->GetDimSize(0);
        int inFeatures = input->GetDimSize(1);
        int outFeatures = weight->GetDimSize(0);
        if (weight->GetDimSize(1) != inFeatures || bias->GetDimSize(0) != outFeatures) return X::Value();

        if (enginePath == "cuda_linear_bias_transpose" || bf16Weights) {
            std::cout << "[TRTBuilder] Running CUDA linear+bias transpose: ["
                << tokens << ", " << inFeatures << "] x [" << outFeatures << ", "
                << inFeatures << "]" << std::endl;
            size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(inFeatures) * sizeof(float);
            size_t weightBytes = static_cast<size_t>(outFeatures) * static_cast<size_t>(inFeatures) * weight->GetItemSize();
            size_t biasBytes = static_cast<size_t>(outFeatures) * bias->GetItemSize();
            size_t outputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(outFeatures) * sizeof(float);
            void* dInput = nullptr;
            void* dWeight = nullptr;
            void* dBias = nullptr;
            void* dOutput = nullptr;
            cudaStream_t stream = nullptr;
            TensorDeviceBinding inputBinding;
            TensorDeviceBinding weightBinding;
            TensorDeviceBinding biasBinding;
            if (cudaStreamCreate(&stream) != cudaSuccess ||
                !BindTensorInput(input, inputBytes, stream, inputBinding) ||
                !BindTensorInput(weight, weightBytes, stream, weightBinding) ||
                !BindTensorInput(bias, biasBytes, stream, biasBinding) ||
                cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
                FreeOwnedBinding(inputBinding);
                FreeOwnedBinding(weightBinding);
                FreeOwnedBinding(biasBinding);
                if (dOutput) cudaFree(dOutput);
                if (stream) cudaStreamDestroy(stream);
                std::cout << "[TRTBuilder] CUDA linear+bias allocation failed." << std::endl;
                return X::Value();
            }
            dInput = inputBinding.ptr;
            dWeight = weightBinding.ptr;
            dBias = biasBinding.ptr;
            cudaError_t launchErr = bf16Weights
                ? runLinearBiasTransposeBF16WeightFP32(
                    static_cast<const float*>(dInput),
                    static_cast<const bfloat16*>(dWeight),
                    static_cast<const bfloat16*>(dBias),
                    static_cast<float*>(dOutput), tokens, inFeatures, outFeatures, stream)
                : runLinearBiasTransposeFP32(
                    static_cast<const float*>(dInput),
                    static_cast<const float*>(dWeight),
                    static_cast<const float*>(dBias),
                    static_cast<float*>(dOutput), tokens, inFeatures, outFeatures, stream);
            if (launchErr != cudaSuccess) {
                std::cout << "[TRTBuilder] CUDA linear+bias launch failed: "
                    << cudaGetErrorString(launchErr) << std::endl;
                FreeOwnedBinding(inputBinding); FreeOwnedBinding(weightBinding); FreeOwnedBinding(biasBinding); cudaFree(dOutput); cudaStreamDestroy(stream);
                return X::Value();
            }
            X::Value output = MakeGPUBackedTensor2D(tokens, outFeatures, dOutput, outputBytes, stream);
            if (!output.IsValid()) {
                FreeOwnedBinding(inputBinding); FreeOwnedBinding(weightBinding); FreeOwnedBinding(biasBinding); cudaFree(dOutput); cudaStreamDestroy(stream);
                return X::Value();
            }
            FreeOwnedBinding(inputBinding);
            FreeOwnedBinding(weightBinding);
            FreeOwnedBinding(biasBinding);
            cudaStreamDestroy(stream);
            std::cout << "[TRTBuilder] CUDA linear+bias transpose completed: ["
                << tokens << ", " << outFeatures << "]" << std::endl;
            return output;
        }

        auto context = GetCachedTRTExecutionContext(enginePath);
        if (!context) return X::Value();

        size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(inFeatures) * sizeof(float);
        size_t weightBytes = static_cast<size_t>(outFeatures) * static_cast<size_t>(inFeatures) * sizeof(float);
        size_t biasBytes = static_cast<size_t>(outFeatures) * sizeof(float);
        size_t outputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(outFeatures) * sizeof(float);
        void* dInput = nullptr;
        void* dWeight = nullptr;
        void* dBias = nullptr;
        void* dOutput = nullptr;
        cudaStream_t stream = nullptr;
        TensorDeviceBinding inputBinding;
        TensorDeviceBinding weightBinding;
        TensorDeviceBinding biasBinding;
        if (cudaStreamCreate(&stream) != cudaSuccess ||
            !BindTensorInput(input, inputBytes, stream, inputBinding) ||
            !BindTensorInput(weight, weightBytes, stream, weightBinding) ||
            !BindTensorInput(bias, biasBytes, stream, biasBinding) ||
            cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
            FreeOwnedBinding(inputBinding);
            FreeOwnedBinding(weightBinding);
            FreeOwnedBinding(biasBinding);
            if (dOutput) cudaFree(dOutput);
            if (stream) cudaStreamDestroy(stream);
            return X::Value();
        }
        dInput = inputBinding.ptr;
        dWeight = weightBinding.ptr;
        dBias = biasBinding.ptr;
        bool bound = context->setTensorAddress("x", dInput)
            && context->setTensorAddress("W", dWeight)
            && context->setTensorAddress("B", dBias)
            && context->setTensorAddress("output", dOutput);
        if (!bound || !context->enqueueV3(stream)) {
            FreeOwnedBinding(inputBinding); FreeOwnedBinding(weightBinding); FreeOwnedBinding(biasBinding); cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        X::Value output = MakeGPUBackedTensor2D(tokens, outFeatures, dOutput, outputBytes, stream);
        if (!output.IsValid()) {
            FreeOwnedBinding(inputBinding); FreeOwnedBinding(weightBinding); FreeOwnedBinding(biasBinding); cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        FreeOwnedBinding(inputBinding); FreeOwnedBinding(weightBinding); FreeOwnedBinding(biasBinding); cudaStreamDestroy(stream);
        std::cout << "[TRTBuilder] RunLinearBiasTransposeEngine completed: [" << tokens << ", " << outFeatures << "]" << std::endl;
        return output;
    }

    X::Value TRTBuilder::ExportVisionMLPEngine(const std::string& enginePath, const std::vector<int>& inputShape, const std::vector<int>& fc1Shape, const std::vector<int>& fc2Shape) {
        std::cout << "[TRTBuilder] ExportVisionMLPEngine -> " << enginePath << std::endl;
        if (inputShape.size() != 2 || fc1Shape.size() != 2 || fc2Shape.size() != 2) {
            std::cout << "[TRTBuilder] VisionMLP export requires 2D input and weight shapes." << std::endl;
            return X::Value();
        }
        int tokens = inputShape[0];
        int hidden = inputShape[1];
        int intermediate = fc1Shape[0];
        if (fc1Shape[1] != hidden || fc2Shape[0] != hidden || fc2Shape[1] != intermediate) {
            std::cout << "[TRTBuilder] VisionMLP shape mismatch." << std::endl;
            return X::Value();
        }

        auto builder = createInferBuilder(gLogger);
        if (!builder) return X::Value();
        uint32_t flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(flags);
        if (!network) return X::Value();
        auto config = builder->createBuilderConfig();
        if (!config) return X::Value();
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 128ULL << 20);

        Dims xDims{};
        xDims.nbDims = 2;
        xDims.d[0] = tokens;
        xDims.d[1] = hidden;
        Dims fc1Dims{};
        fc1Dims.nbDims = 2;
        fc1Dims.d[0] = intermediate;
        fc1Dims.d[1] = hidden;
        Dims fc1BiasDims{};
        fc1BiasDims.nbDims = 2;
        fc1BiasDims.d[0] = 1;
        fc1BiasDims.d[1] = intermediate;
        Dims fc2Dims{};
        fc2Dims.nbDims = 2;
        fc2Dims.d[0] = hidden;
        fc2Dims.d[1] = intermediate;
        Dims fc2BiasDims{};
        fc2BiasDims.nbDims = 2;
        fc2BiasDims.d[0] = 1;
        fc2BiasDims.d[1] = hidden;

        ITensor* x = network->addInput("x", DataType::kFLOAT, xDims);
        ITensor* w1 = network->addInput("W_fc1", DataType::kFLOAT, fc1Dims);
        ITensor* b1 = network->addInput("b_fc1", DataType::kFLOAT, fc1BiasDims);
        ITensor* w2 = network->addInput("W_fc2", DataType::kFLOAT, fc2Dims);
        ITensor* b2 = network->addInput("b_fc2", DataType::kFLOAT, fc2BiasDims);
        if (!x || !w1 || !b1 || !w2 || !b2) {
            std::cout << "[TRTBuilder] VisionMLP addInput failed." << std::endl;
            return X::Value();
        }

        auto fc1 = network->addMatrixMultiply(*x, MatrixOperation::kNONE, *w1, MatrixOperation::kTRANSPOSE);
        if (!fc1 || !fc1->getOutput(0)) return X::Value();
        auto fc1Bias = network->addElementWise(*fc1->getOutput(0), *b1, ElementWiseOperation::kSUM);
        if (!fc1Bias || !fc1Bias->getOutput(0)) return X::Value();
        auto gelu = network->addActivation(*fc1Bias->getOutput(0), ActivationType::kGELU_TANH);
        if (!gelu || !gelu->getOutput(0)) return X::Value();
        auto fc2 = network->addMatrixMultiply(*gelu->getOutput(0), MatrixOperation::kNONE, *w2, MatrixOperation::kTRANSPOSE);
        if (!fc2 || !fc2->getOutput(0)) return X::Value();
        auto out = network->addElementWise(*fc2->getOutput(0), *b2, ElementWiseOperation::kSUM);
        if (!out || !out->getOutput(0)) return X::Value();
        out->getOutput(0)->setName("output");
        network->markOutput(*out->getOutput(0));

        auto serialized = builder->buildSerializedNetwork(*network, *config);
        if (!serialized) {
            std::cout << "[TRTBuilder] VisionMLP buildSerializedNetwork failed." << std::endl;
            return X::Value();
        }

        std::filesystem::path outputPath(enginePath);
        std::filesystem::create_directories(outputPath.parent_path());
        std::ofstream outFile(outputPath, std::ios::binary);
        if (!outFile.is_open()) return X::Value();
        outFile.write(static_cast<const char*>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
        outFile.close();

        std::cout << "[TRTBuilder] Serialized VisionMLP TensorRT engine bytes: " << serialized->size() << std::endl;
        return X::Value(enginePath);
    }

    X::Value TRTBuilder::RunVisionMLPEngine(const std::string& enginePath, X::Value inputValue, X::Value fc1Weight, X::Value fc1Bias, X::Value fc2Weight, X::Value fc2Bias) {
        std::cout << "[TRTBuilder] RunVisionMLPEngine <- " << enginePath << std::endl;
        if (!inputValue.IsTensor() || !fc1Weight.IsTensor() || !fc1Bias.IsTensor() || !fc2Weight.IsTensor() || !fc2Bias.IsTensor()) {
            std::cout << "[TRTBuilder] RunVisionMLPEngine requires tensor inputs." << std::endl;
            return X::Value();
        }

        X::Tensor input(inputValue);
        X::Tensor w1(fc1Weight);
        X::Tensor b1(fc1Bias);
        X::Tensor w2(fc2Weight);
        X::Tensor b2(fc2Bias);
        if (input->GetDataType() != X::TensorDataType::FLOAT32 || w1->GetDataType() != X::TensorDataType::FLOAT32 ||
            b1->GetDataType() != X::TensorDataType::FLOAT32 || w2->GetDataType() != X::TensorDataType::FLOAT32 ||
            b2->GetDataType() != X::TensorDataType::FLOAT32) {
            std::cout << "[TRTBuilder] RunVisionMLPEngine supports float32 only." << std::endl;
            return X::Value();
        }

        int tokens = input->GetDimSize(0);
        int hidden = input->GetDimSize(1);
        int intermediate = w1->GetDimSize(0);
        if (input->GetDimCount() != 2 || w1->GetDimCount() != 2 || w2->GetDimCount() != 2 ||
            w1->GetDimSize(1) != hidden || w2->GetDimSize(0) != hidden || w2->GetDimSize(1) != intermediate) {
            std::cout << "[TRTBuilder] RunVisionMLPEngine shape mismatch." << std::endl;
            return X::Value();
        }

        auto context = GetCachedTRTExecutionContext(enginePath);
        if (!context) return X::Value();

        size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);
        size_t fc1Bytes = static_cast<size_t>(intermediate) * static_cast<size_t>(hidden) * sizeof(float);
        size_t fc1BiasBytes = static_cast<size_t>(intermediate) * sizeof(float);
        size_t fc2Bytes = static_cast<size_t>(hidden) * static_cast<size_t>(intermediate) * sizeof(float);
        size_t fc2BiasBytes = static_cast<size_t>(hidden) * sizeof(float);
        size_t outputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);

        void* dInput = nullptr;
        void* dW1 = nullptr;
        void* dB1 = nullptr;
        void* dW2 = nullptr;
        void* dB2 = nullptr;
        void* dOutput = nullptr;
        cudaStream_t stream = nullptr;
        TensorDeviceBinding inputBinding;
        TensorDeviceBinding w1Binding;
        TensorDeviceBinding b1Binding;
        TensorDeviceBinding w2Binding;
        TensorDeviceBinding b2Binding;
        if (cudaStreamCreate(&stream) != cudaSuccess ||
            !BindTensorInput(input, inputBytes, stream, inputBinding) ||
            !BindTensorInput(w1, fc1Bytes, stream, w1Binding) ||
            !BindTensorInput(b1, fc1BiasBytes, stream, b1Binding) ||
            !BindTensorInput(w2, fc2Bytes, stream, w2Binding) ||
            !BindTensorInput(b2, fc2BiasBytes, stream, b2Binding) ||
            cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
            FreeOwnedBinding(inputBinding);
            FreeOwnedBinding(w1Binding);
            FreeOwnedBinding(b1Binding);
            FreeOwnedBinding(w2Binding);
            FreeOwnedBinding(b2Binding);
            if (dOutput) cudaFree(dOutput);
            if (stream) cudaStreamDestroy(stream);
            return X::Value();
        }

        dInput = inputBinding.ptr;
        dW1 = w1Binding.ptr;
        dB1 = b1Binding.ptr;
        dW2 = w2Binding.ptr;
        dB2 = b2Binding.ptr;

        bool bound = context->setTensorAddress("x", dInput)
            && context->setTensorAddress("W_fc1", dW1)
            && context->setTensorAddress("b_fc1", dB1)
            && context->setTensorAddress("W_fc2", dW2)
            && context->setTensorAddress("b_fc2", dB2)
            && context->setTensorAddress("output", dOutput);
        if (!bound || !context->enqueueV3(stream)) {
            FreeOwnedBinding(inputBinding); FreeOwnedBinding(w1Binding); FreeOwnedBinding(b1Binding); FreeOwnedBinding(w2Binding); FreeOwnedBinding(b2Binding); cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }

        X::Value output = MakeGPUBackedTensor2D(tokens, hidden, dOutput, outputBytes, stream);
        if (!output.IsValid()) {
            FreeOwnedBinding(inputBinding); FreeOwnedBinding(w1Binding); FreeOwnedBinding(b1Binding); FreeOwnedBinding(w2Binding); FreeOwnedBinding(b2Binding); cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }

        FreeOwnedBinding(inputBinding);
        FreeOwnedBinding(w1Binding);
        FreeOwnedBinding(b1Binding);
        FreeOwnedBinding(w2Binding);
        FreeOwnedBinding(b2Binding);
        cudaStreamDestroy(stream);

        std::cout << "[TRTBuilder] RunVisionMLPEngine completed: [" << tokens << ", " << hidden << "]" << std::endl;
        return output;
    }

    X::Value TRTBuilder::ExportRMSNormEngine(const std::string& enginePath, const std::vector<int>& inputShape, const std::vector<int>& weightShape, float eps) {
        std::cout << "[TRTBuilder] ExportRMSNormEngine -> " << enginePath << std::endl;
        if (inputShape.size() != 2 || weightShape.size() != 1 || inputShape[1] != weightShape[0]) {
            std::cout << "[TRTBuilder] RMSNorm shape mismatch." << std::endl;
            return X::Value();
        }
        int tokens = inputShape[0];
        int hidden = inputShape[1];

        auto builder = createInferBuilder(gLogger);
        if (!builder) return X::Value();
        uint32_t flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(flags);
        if (!network) return X::Value();
        auto config = builder->createBuilderConfig();
        if (!config) return X::Value();
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 128ULL << 20);

        Dims xDims{};
        xDims.nbDims = 2;
        xDims.d[0] = tokens;
        xDims.d[1] = hidden;
        Dims affineDims{};
        affineDims.nbDims = 2;
        affineDims.d[0] = 1;
        affineDims.d[1] = hidden;
        Dims scalarDims{};
        scalarDims.nbDims = 2;
        scalarDims.d[0] = 1;
        scalarDims.d[1] = 1;

        ITensor* x = network->addInput("x", DataType::kFLOAT, xDims);
        ITensor* weight = network->addInput("weight", DataType::kFLOAT, affineDims);
        if (!x || !weight) return X::Value();

        auto square = network->addElementWise(*x, *x, ElementWiseOperation::kPROD);
        if (!square || !square->getOutput(0)) return X::Value();
        auto mean = network->addReduce(*square->getOutput(0), ReduceOperation::kAVG, 1U << 1, true);
        if (!mean || !mean->getOutput(0)) return X::Value();
        float epsValue = eps;
        Weights epsWeights{ DataType::kFLOAT, &epsValue, 1 };
        auto epsLayer = network->addConstant(scalarDims, epsWeights);
        if (!epsLayer || !epsLayer->getOutput(0)) return X::Value();
        auto variance = network->addElementWise(*mean->getOutput(0), *epsLayer->getOutput(0), ElementWiseOperation::kSUM);
        if (!variance || !variance->getOutput(0)) return X::Value();
        auto sqrt = network->addUnary(*variance->getOutput(0), UnaryOperation::kSQRT);
        if (!sqrt || !sqrt->getOutput(0)) return X::Value();
        auto inv = network->addUnary(*sqrt->getOutput(0), UnaryOperation::kRECIP);
        if (!inv || !inv->getOutput(0)) return X::Value();
        auto normalized = network->addElementWise(*x, *inv->getOutput(0), ElementWiseOperation::kPROD);
        if (!normalized || !normalized->getOutput(0)) return X::Value();
        auto out = network->addElementWise(*normalized->getOutput(0), *weight, ElementWiseOperation::kPROD);
        if (!out || !out->getOutput(0)) return X::Value();
        out->getOutput(0)->setName("output");
        network->markOutput(*out->getOutput(0));

        auto serialized = builder->buildSerializedNetwork(*network, *config);
        if (!serialized) return X::Value();
        std::filesystem::path outputPath(enginePath);
        std::filesystem::create_directories(outputPath.parent_path());
        std::ofstream outFile(outputPath, std::ios::binary);
        if (!outFile.is_open()) return X::Value();
        outFile.write(static_cast<const char*>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
        outFile.close();
        std::cout << "[TRTBuilder] Serialized RMSNorm TensorRT engine bytes: " << serialized->size() << std::endl;
        return X::Value(enginePath);
    }

    X::Value TRTBuilder::RunRMSNormEngine(const std::string& enginePath, X::Value inputValue, X::Value weightValue) {
        std::cout << "[TRTBuilder] RunRMSNormEngine <- " << enginePath << std::endl;
        if (!inputValue.IsTensor() || !weightValue.IsTensor()) return X::Value();
        X::Tensor input(inputValue);
        X::Tensor weight(weightValue);
        if (input->GetDataType() != X::TensorDataType::FLOAT32 || weight->GetDataType() != X::TensorDataType::FLOAT32) return X::Value();
        if (input->GetDimCount() != 2 || weight->GetDimCount() != 1 || input->GetDimSize(1) != weight->GetDimSize(0)) return X::Value();
        int tokens = input->GetDimSize(0);
        int hidden = input->GetDimSize(1);

        size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);
        size_t weightBytes = static_cast<size_t>(hidden) * sizeof(float);
        TensorDeviceBinding dInput;
        TensorDeviceBinding dWeight;
        void* dOutput = nullptr;
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            return X::Value();
        }
        if (!BindTensorInput(input, inputBytes, stream, dInput) ||
            !BindTensorInput(weight, weightBytes, stream, dWeight) ||
            cudaMalloc(&dOutput, inputBytes) != cudaSuccess) {
            FreeOwnedBinding(dInput);
            FreeOwnedBinding(dWeight);
            if (dOutput) cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }
        cudaError_t status = runRMSNormFP32(
            static_cast<const float*>(dInput.ptr),
            static_cast<const float*>(dWeight.ptr),
            static_cast<float*>(dOutput),
            tokens, hidden, 1.0e-6f, stream);
        if (status != cudaSuccess) {
            std::cout << "[TRTBuilder] CUDA RMSNorm failed: "
                << cudaGetErrorString(status) << std::endl;
            FreeOwnedBinding(dInput);
            FreeOwnedBinding(dWeight);
            cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        X::Value output = MakeGPUBackedTensor2D(tokens, hidden, dOutput, inputBytes, stream);
        if (!output.IsValid()) {
            FreeOwnedBinding(dInput);
            FreeOwnedBinding(dWeight);
            cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }
        FreeOwnedBinding(dInput);
        FreeOwnedBinding(dWeight);
        cudaStreamDestroy(stream);
        std::cout << "[TRTBuilder] RunRMSNormEngine completed: [" << tokens << ", " << hidden << "]" << std::endl;
        return output;
    }

    X::Value TRTBuilder::ExportLayerNormEngine(const std::string& enginePath, const std::vector<int>& inputShape, const std::vector<int>& weightShape, float eps) {
        std::cout << "[TRTBuilder] ExportLayerNormEngine -> " << enginePath << std::endl;
        if (inputShape.size() != 2 || weightShape.size() != 1 || inputShape[1] != weightShape[0]) {
            std::cout << "[TRTBuilder] LayerNorm shape mismatch." << std::endl;
            return X::Value();
        }
        int tokens = inputShape[0];
        int hidden = inputShape[1];

        auto builder = createInferBuilder(gLogger);
        if (!builder) return X::Value();
        uint32_t flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(flags);
        if (!network) return X::Value();
        auto config = builder->createBuilderConfig();
        if (!config) return X::Value();
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 128ULL << 20);

        Dims xDims{};
        xDims.nbDims = 2;
        xDims.d[0] = tokens;
        xDims.d[1] = hidden;
        Dims affineDims{};
        affineDims.nbDims = 2;
        affineDims.d[0] = 1;
        affineDims.d[1] = hidden;
        Dims scalarDims{};
        scalarDims.nbDims = 2;
        scalarDims.d[0] = 1;
        scalarDims.d[1] = 1;

        ITensor* x = network->addInput("x", DataType::kFLOAT, xDims);
        ITensor* weight = network->addInput("weight", DataType::kFLOAT, affineDims);
        ITensor* bias = network->addInput("bias", DataType::kFLOAT, affineDims);
        if (!x || !weight || !bias) return X::Value();
        auto mean = network->addReduce(*x, ReduceOperation::kAVG, 1U << 1, true);
        if (!mean || !mean->getOutput(0)) return X::Value();
        auto centered = network->addElementWise(*x, *mean->getOutput(0), ElementWiseOperation::kSUB);
        if (!centered || !centered->getOutput(0)) return X::Value();
        auto square = network->addElementWise(*centered->getOutput(0), *centered->getOutput(0), ElementWiseOperation::kPROD);
        if (!square || !square->getOutput(0)) return X::Value();
        auto variance = network->addReduce(*square->getOutput(0), ReduceOperation::kAVG, 1U << 1, true);
        if (!variance || !variance->getOutput(0)) return X::Value();
        float epsValue = eps;
        Weights epsWeights{ DataType::kFLOAT, &epsValue, 1 };
        auto epsLayer = network->addConstant(scalarDims, epsWeights);
        if (!epsLayer || !epsLayer->getOutput(0)) return X::Value();
        auto varianceEps = network->addElementWise(*variance->getOutput(0), *epsLayer->getOutput(0), ElementWiseOperation::kSUM);
        if (!varianceEps || !varianceEps->getOutput(0)) return X::Value();
        auto sqrt = network->addUnary(*varianceEps->getOutput(0), UnaryOperation::kSQRT);
        if (!sqrt || !sqrt->getOutput(0)) return X::Value();
        auto inv = network->addUnary(*sqrt->getOutput(0), UnaryOperation::kRECIP);
        if (!inv || !inv->getOutput(0)) return X::Value();
        auto normalized = network->addElementWise(*centered->getOutput(0), *inv->getOutput(0), ElementWiseOperation::kPROD);
        if (!normalized || !normalized->getOutput(0)) return X::Value();
        auto scaled = network->addElementWise(*normalized->getOutput(0), *weight, ElementWiseOperation::kPROD);
        if (!scaled || !scaled->getOutput(0)) return X::Value();
        auto out = network->addElementWise(*scaled->getOutput(0), *bias, ElementWiseOperation::kSUM);
        if (!out || !out->getOutput(0)) return X::Value();
        out->getOutput(0)->setName("output");
        network->markOutput(*out->getOutput(0));

        auto serialized = builder->buildSerializedNetwork(*network, *config);
        if (!serialized) return X::Value();
        std::filesystem::path outputPath(enginePath);
        std::filesystem::create_directories(outputPath.parent_path());
        std::ofstream outFile(outputPath, std::ios::binary);
        if (!outFile.is_open()) return X::Value();
        outFile.write(static_cast<const char*>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
        outFile.close();
        std::cout << "[TRTBuilder] Serialized LayerNorm TensorRT engine bytes: " << serialized->size() << std::endl;
        return X::Value(enginePath);
    }

    X::Value TRTBuilder::RunLayerNormEngine(const std::string& enginePath, X::Value inputValue, X::Value weightValue, X::Value biasValue) {
        std::cout << "[TRTBuilder] RunLayerNormEngine <- " << enginePath << std::endl;
        if (!inputValue.IsTensor() || !weightValue.IsTensor() || !biasValue.IsTensor()) return X::Value();
        X::Tensor input(inputValue);
        X::Tensor weight(weightValue);
        X::Tensor bias(biasValue);
        if (input->GetDataType() != X::TensorDataType::FLOAT32 || weight->GetDataType() != X::TensorDataType::FLOAT32 || bias->GetDataType() != X::TensorDataType::FLOAT32) return X::Value();
        if (input->GetDimCount() != 2 || weight->GetDimCount() != 1 || bias->GetDimCount() != 1 || input->GetDimSize(1) != weight->GetDimSize(0) || weight->GetDimSize(0) != bias->GetDimSize(0)) return X::Value();
        int tokens = input->GetDimSize(0);
        int hidden = input->GetDimSize(1);

        auto context = GetCachedTRTExecutionContext(enginePath);
        if (!context) return X::Value();

        size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);
        size_t affineBytes = static_cast<size_t>(hidden) * sizeof(float);
        void* dInput = nullptr;
        void* dWeight = nullptr;
        void* dBias = nullptr;
        void* dOutput = nullptr;
        cudaStream_t stream = nullptr;
        TensorDeviceBinding inputBinding;
        TensorDeviceBinding weightBinding;
        TensorDeviceBinding biasBinding;
        if (cudaStreamCreate(&stream) != cudaSuccess ||
            !BindTensorInput(input, inputBytes, stream, inputBinding) ||
            !BindTensorInput(weight, affineBytes, stream, weightBinding) ||
            !BindTensorInput(bias, affineBytes, stream, biasBinding) ||
            cudaMalloc(&dOutput, inputBytes) != cudaSuccess) {
            FreeOwnedBinding(inputBinding);
            FreeOwnedBinding(weightBinding);
            FreeOwnedBinding(biasBinding);
            if (dOutput) cudaFree(dOutput);
            if (stream) cudaStreamDestroy(stream);
            return X::Value();
        }
        dInput = inputBinding.ptr;
        dWeight = weightBinding.ptr;
        dBias = biasBinding.ptr;
        bool bound = context->setTensorAddress("x", dInput)
            && context->setTensorAddress("weight", dWeight)
            && context->setTensorAddress("bias", dBias)
            && context->setTensorAddress("output", dOutput);
        if (!bound || !context->enqueueV3(stream)) {
            FreeOwnedBinding(inputBinding); FreeOwnedBinding(weightBinding); FreeOwnedBinding(biasBinding); cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        X::Value output = MakeGPUBackedTensor2D(tokens, hidden, dOutput, inputBytes, stream);
        if (!output.IsValid()) {
            FreeOwnedBinding(inputBinding); FreeOwnedBinding(weightBinding); FreeOwnedBinding(biasBinding); cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        FreeOwnedBinding(inputBinding); FreeOwnedBinding(weightBinding); FreeOwnedBinding(biasBinding); cudaStreamDestroy(stream);
        std::cout << "[TRTBuilder] RunLayerNormEngine completed: [" << tokens << ", " << hidden << "]" << std::endl;
        return output;
    }

    nvinfer1::ITensor* TRTBuilder::GetOrCreateTRTWeight(const std::string& weightName) {
        auto existing = weightTensorMap.find(weightName);
        if (existing != weightTensorMap.end()) {
            return existing->second;
        }
        if (!capturedWeightIndex || !capturedWeightFile.IsOpen()) {
            loweringError = "native safetensors mapping is unavailable for weight: " + weightName;
            return nullptr;
        }
        const SafeTensorMetadata* metadata = capturedWeightIndex->Find(weightName);
        if (!metadata) {
            loweringError = "weight is absent from safetensors index: " + weightName;
            return nullptr;
        }
        DataType dataType;
        if (metadata->dataType == "BF16") dataType = DataType::kBF16;
        else if (metadata->dataType == "F16") dataType = DataType::kHALF;
        else if (metadata->dataType == "F32") dataType = DataType::kFLOAT;
        else {
            loweringError = "unsupported TensorRT constant dtype for weight: " + weightName;
            return nullptr;
        }
        if (metadata->shape.empty() || metadata->shape.size() > Dims::MAX_DIMS) {
            loweringError = "unsupported TensorRT constant rank for weight: " + weightName;
            return nullptr;
        }
        Dims dimensions{};
        dimensions.nbDims = static_cast<int>(metadata->shape.size());
        std::int64_t elementCount = 1;
        for (int dimension = 0; dimension < dimensions.nbDims; ++dimension) {
            const long long size = metadata->shape[dimension];
            if (size <= 0 || size > std::numeric_limits<int>::max() ||
                elementCount > std::numeric_limits<std::int64_t>::max() / size) {
                loweringError = "invalid TensorRT constant shape for weight: " + weightName;
                return nullptr;
            }
            dimensions.d[dimension] = static_cast<int>(size);
            elementCount *= size;
        }
        const void* data = capturedWeightFile.DataAt(metadata->dataOffset, metadata->dataSize);
        if (!data) {
            loweringError = "safetensors range is unavailable for weight: " + weightName;
            return nullptr;
        }
        Weights weights{dataType, data, elementCount};
        auto* constant = network->addConstant(dimensions, weights);
        if (!constant || !constant->getOutput(0)) {
            loweringError = "TensorRT constant lowering failed for weight: " + weightName;
            return nullptr;
        }
        if (!network->setWeightsName(weights, weightName.c_str()) ||
            !network->markWeightsRefittable(weightName.c_str())) {
            loweringError = "TensorRT could not mark checkpoint weight refittable: " + weightName;
            return nullptr;
        }
        constant->setName(weightName.c_str());
        ITensor* output = constant->getOutput(0);
        output->setName(weightName.c_str());
        weightTensorMap[weightName] = output;
        return output;
    }

    nvinfer1::ITensor* TRTBuilder::GetOrCreateTRTTensor(X::Value value) {
        if (!value.IsObject()) {
            loweringError = "scalar operands are not implemented in generic TensorRT lowering";
            return nullptr;
        }
        const unsigned long long id = value.GetObj()->GetID();
        auto found = tensorMap.find(id);
        if (found != tensorMap.end()) {
            return found->second;
        }
        if (value.IsTensor()) {
            X::Tensor tensor(value);
            const std::string weightName = tensor->GetName().ToString();
            if (capturedWeightIndex && capturedWeightIndex->Find(weightName)) {
                ITensor* weight = GetOrCreateTRTWeight(weightName);
                if (weight) tensorMap[id] = weight;
                return weight;
            }
        }
        loweringError = "graph operand was not produced by an input or an earlier lowered operation";
        return nullptr;
    }

    nvinfer1::ITensor* TRTBuilder::BroadcastLastDimension(
        ITensor* tensor,
        int targetRank,
        const std::string& layerName) {
        if (!tensor) return nullptr;
        const Dims sourceDimensions = tensor->getDimensions();
        if (sourceDimensions.nbDims == targetRank) return tensor;
        if (sourceDimensions.nbDims != 1 || targetRank <= 1 || targetRank > Dims::MAX_DIMS) {
            loweringError = "cannot broadcast " + layerName + " to target rank";
            return nullptr;
        }
        Dims broadcastDimensions{};
        broadcastDimensions.nbDims = targetRank;
        for (int dimension = 0; dimension < targetRank - 1; ++dimension) {
            broadcastDimensions.d[dimension] = 1;
        }
        broadcastDimensions.d[targetRank - 1] = sourceDimensions.d[0];
        auto* reshape = network->addShuffle(*tensor);
        if (!reshape) {
            loweringError = "TensorRT broadcast reshape failed for " + layerName;
            return nullptr;
        }
        reshape->setReshapeDimensions(broadcastDimensions);
        return reshape->getOutput(0);
    }

    nvinfer1::ITensor* TRTBuilder::BroadcastMatrixWeight(ITensor* tensor, int targetRank) {
        if (!tensor) return nullptr;
        const Dims sourceDimensions = tensor->getDimensions();
        if (sourceDimensions.nbDims == targetRank) return tensor;
        if (sourceDimensions.nbDims != 2 || targetRank < 2 || targetRank > Dims::MAX_DIMS) {
            loweringError = "matrix weight cannot be broadcast to activation rank";
            return nullptr;
        }
        Dims broadcastDimensions{};
        broadcastDimensions.nbDims = targetRank;
        for (int dimension = 0; dimension < targetRank - 2; ++dimension) {
            broadcastDimensions.d[dimension] = 1;
        }
        broadcastDimensions.d[targetRank - 2] = sourceDimensions.d[0];
        broadcastDimensions.d[targetRank - 1] = sourceDimensions.d[1];
        auto* reshape = network->addShuffle(*tensor);
        if (reshape) reshape->setReshapeDimensions(broadcastDimensions);
        return reshape ? reshape->getOutput(0) : nullptr;
    }

    nvinfer1::ITensor* TRTBuilder::LowerVisionRope(
        ITensor* qkv,
        ITensor* positionIds,
        X::KWARGS& options) {
        const Dims qkvDims = qkv->getDimensions();
        const Dims positionDims = positionIds->getDimensions();
        auto* headsItem = options.find("num_heads");
        auto* headDimItem = options.find("head_dim");
        const int heads = headsItem ? static_cast<int>(headsItem->val.ToLongLong()) : 0;
        const int headDim = headDimItem ? static_cast<int>(headDimItem->val.ToLongLong()) : 0;
        if (qkvDims.nbDims != 2 || positionDims.nbDims != 2 || positionDims.d[1] != 2 ||
            heads <= 0 || headDim <= 0 || headDim % 4 != 0 ||
            qkvDims.d[1] != 3 * heads * headDim) {
            loweringError = "Qwen3-VL vision RoPE received incompatible static dimensions";
            return nullptr;
        }
        const int tokens = qkvDims.d[0];
        const int hidden = heads * headDim;
        auto slice = [&](ITensor* source, int axis, int startValue, int sizeValue) -> ITensor* {
            const Dims sourceDims = source->getDimensions();
            Dims start{};
            Dims size = sourceDims;
            Dims stride{};
            start.nbDims = sourceDims.nbDims;
            stride.nbDims = sourceDims.nbDims;
            for (int i = 0; i < sourceDims.nbDims; ++i) stride.d[i] = 1;
            start.d[axis] = startValue;
            size.d[axis] = sizeValue;
            auto* layer = network->addSlice(*source, start, size, stride);
            return layer ? layer->getOutput(0) : nullptr;
        };
        auto reshape = [&](ITensor* source, const Dims& dimensions) -> ITensor* {
            auto* layer = source ? network->addShuffle(*source) : nullptr;
            if (layer) layer->setReshapeDimensions(dimensions);
            return layer ? layer->getOutput(0) : nullptr;
        };

        Dims thd{};
        thd.nbDims = 3;
        thd.d[0] = tokens;
        thd.d[1] = heads;
        thd.d[2] = headDim;
        ITensor* q = reshape(slice(qkv, 1, 0, hidden), thd);
        ITensor* k = reshape(slice(qkv, 1, hidden, hidden), thd);
        ITensor* v = reshape(slice(qkv, 1, 2 * hidden, hidden), thd);
        auto* qFloat = q ? network->addCast(*q, DataType::kFLOAT) : nullptr;
        auto* kFloat = k ? network->addCast(*k, DataType::kFLOAT) : nullptr;
        q = qFloat ? qFloat->getOutput(0) : nullptr;
        k = kFloat ? kFloat->getOutput(0) : nullptr;

        auto* positionCast = network->addCast(*positionIds, DataType::kFLOAT);
        Dims positionExpanded{};
        positionExpanded.nbDims = 3;
        positionExpanded.d[0] = tokens;
        positionExpanded.d[1] = 2;
        positionExpanded.d[2] = 1;
        ITensor* positions = positionCast
            ? reshape(positionCast->getOutput(0), positionExpanded)
            : nullptr;
        const int frequencyCount = headDim / 4;
        vectorWeights.emplace_back(static_cast<size_t>(frequencyCount));
        for (int i = 0; i < frequencyCount; ++i) {
            vectorWeights.back()[i] = 1.0F / std::pow(
                10000.0F,
                static_cast<float>(2 * i) / static_cast<float>(headDim / 2));
        }
        Dims frequencyDims{};
        frequencyDims.nbDims = 3;
        frequencyDims.d[0] = 1;
        frequencyDims.d[1] = 1;
        frequencyDims.d[2] = frequencyCount;
        Weights frequencies{DataType::kFLOAT, vectorWeights.back().data(), frequencyCount};
        auto* frequency = network->addConstant(frequencyDims, frequencies);
        auto* phaseGrid = positions && frequency
            ? network->addElementWise(
                *positions,
                *frequency->getOutput(0),
                ElementWiseOperation::kPROD)
            : nullptr;
        Dims halfPhaseDims{};
        halfPhaseDims.nbDims = 2;
        halfPhaseDims.d[0] = tokens;
        halfPhaseDims.d[1] = headDim / 2;
        ITensor* halfPhase = phaseGrid
            ? reshape(phaseGrid->getOutput(0), halfPhaseDims)
            : nullptr;
        ITensor* phaseParts[] = {halfPhase, halfPhase};
        auto* phaseConcat = halfPhase ? network->addConcatenation(phaseParts, 2) : nullptr;
        if (phaseConcat) phaseConcat->setAxis(1);
        auto* cosine = phaseConcat
            ? network->addUnary(*phaseConcat->getOutput(0), UnaryOperation::kCOS)
            : nullptr;
        auto* sine = phaseConcat
            ? network->addUnary(*phaseConcat->getOutput(0), UnaryOperation::kSIN)
            : nullptr;
        Dims phaseBroadcastDims{};
        phaseBroadcastDims.nbDims = 3;
        phaseBroadcastDims.d[0] = tokens;
        phaseBroadcastDims.d[1] = 1;
        phaseBroadcastDims.d[2] = headDim;
        ITensor* cosBroadcast = cosine
            ? reshape(cosine->getOutput(0), phaseBroadcastDims)
            : nullptr;
        ITensor* sinBroadcast = sine
            ? reshape(sine->getOutput(0), phaseBroadcastDims)
            : nullptr;
        auto rotateHalf = [&](ITensor* source) -> ITensor* {
            ITensor* first = slice(source, 2, 0, headDim / 2);
            ITensor* second = slice(source, 2, headDim / 2, headDim / 2);
            auto* negativeSecond = second
                ? network->addUnary(*second, UnaryOperation::kNEG)
                : nullptr;
            ITensor* parts[] = {negativeSecond ? negativeSecond->getOutput(0) : nullptr, first};
            auto* concat = parts[0] && parts[1]
                ? network->addConcatenation(parts, 2)
                : nullptr;
            if (concat) concat->setAxis(2);
            return concat ? concat->getOutput(0) : nullptr;
        };
        auto applyRope = [&](ITensor* source) -> ITensor* {
            ITensor* rotated = rotateHalf(source);
            auto* base = source && cosBroadcast
                ? network->addElementWise(*source, *cosBroadcast, ElementWiseOperation::kPROD)
                : nullptr;
            auto* cross = rotated && sinBroadcast
                ? network->addElementWise(*rotated, *sinBroadcast, ElementWiseOperation::kPROD)
                : nullptr;
            auto* sum = base && cross
                ? network->addElementWise(
                    *base->getOutput(0),
                    *cross->getOutput(0),
                    ElementWiseOperation::kSUM)
                : nullptr;
            return sum ? sum->getOutput(0) : nullptr;
        };
        Dims flatHidden{};
        flatHidden.nbDims = 2;
        flatHidden.d[0] = tokens;
        flatHidden.d[1] = hidden;
        ITensor* qRotated = applyRope(q);
        ITensor* kRotated = applyRope(k);
        auto* qCast = qRotated ? network->addCast(*qRotated, qkv->getType()) : nullptr;
        auto* kCast = kRotated ? network->addCast(*kRotated, qkv->getType()) : nullptr;
        qRotated = reshape(qCast ? qCast->getOutput(0) : nullptr, flatHidden);
        kRotated = reshape(kCast ? kCast->getOutput(0) : nullptr, flatHidden);
        v = reshape(v, flatHidden);
        ITensor* packedParts[] = {qRotated, kRotated, v};
        auto* packed = qRotated && kRotated && v
            ? network->addConcatenation(packedParts, 3)
            : nullptr;
        if (packed) packed->setAxis(1);
        return packed ? packed->getOutput(0) : nullptr;
    }

    nvinfer1::ITensor* TRTBuilder::LowerVisionAttention(
        ITensor* qkv,
        ITensor* cuSeqlens,
        X::KWARGS& options) {
        const Dims qkvDims = qkv->getDimensions();
        const Dims sequenceDims = cuSeqlens->getDimensions();
        auto* headsItem = options.find("num_heads");
        auto* headDimItem = options.find("head_dim");
        const int heads = headsItem ? static_cast<int>(headsItem->val.ToLongLong()) : 0;
        const int headDim = headDimItem ? static_cast<int>(headDimItem->val.ToLongLong()) : 0;
        if (qkvDims.nbDims != 2 || sequenceDims.nbDims != 1 || sequenceDims.d[0] != 2 ||
            heads <= 0 || headDim <= 0 || qkvDims.d[1] != 3 * heads * headDim) {
            loweringError = "vision attention currently requires one packed vision sequence";
            return nullptr;
        }
        const int tokens = qkvDims.d[0];
        const int hidden = heads * headDim;
        auto sliceColumns = [&](int startColumn) -> ITensor* {
            Dims start{};
            start.nbDims = 2;
            start.d[1] = startColumn;
            Dims size{};
            size.nbDims = 2;
            size.d[0] = tokens;
            size.d[1] = hidden;
            Dims stride{};
            stride.nbDims = 2;
            stride.d[0] = 1;
            stride.d[1] = 1;
            auto* layer = network->addSlice(*qkv, start, size, stride);
            return layer ? layer->getOutput(0) : nullptr;
        };
        auto toHeadMajorFloat = [&](ITensor* source) -> ITensor* {
            Dims dimensions{};
            dimensions.nbDims = 3;
            dimensions.d[0] = tokens;
            dimensions.d[1] = heads;
            dimensions.d[2] = headDim;
            auto* shuffle = source ? network->addShuffle(*source) : nullptr;
            if (!shuffle) return nullptr;
            shuffle->setReshapeDimensions(dimensions);
            Permutation permutation{};
            permutation.order[0] = 1;
            permutation.order[1] = 0;
            permutation.order[2] = 2;
            shuffle->setSecondTranspose(permutation);
            auto* cast = network->addCast(*shuffle->getOutput(0), DataType::kFLOAT);
            return cast ? cast->getOutput(0) : nullptr;
        };
        ITensor* q = toHeadMajorFloat(sliceColumns(0));
        ITensor* k = toHeadMajorFloat(sliceColumns(hidden));
        ITensor* v = toHeadMajorFloat(sliceColumns(2 * hidden));
        auto* scores = q && k
            ? network->addMatrixMultiply(
                *q,
                MatrixOperation::kNONE,
                *k,
                MatrixOperation::kTRANSPOSE)
            : nullptr;
        scalarWeights.push_back(1.0F / std::sqrt(static_cast<float>(headDim)));
        Dims scaleDims{};
        scaleDims.nbDims = 3;
        scaleDims.d[0] = 1;
        scaleDims.d[1] = 1;
        scaleDims.d[2] = 1;
        Weights scaleWeights{DataType::kFLOAT, &scalarWeights.back(), 1};
        auto* scale = network->addConstant(scaleDims, scaleWeights);
        auto* scaledScores = scores && scale
            ? network->addElementWise(
                *scores->getOutput(0),
                *scale->getOutput(0),
                ElementWiseOperation::kPROD)
            : nullptr;
        auto* softmax = scaledScores
            ? network->addSoftMax(*scaledScores->getOutput(0))
            : nullptr;
        if (softmax) softmax->setAxes(1U << 2);
        auto* context = softmax && v
            ? network->addMatrixMultiply(
                *softmax->getOutput(0),
                MatrixOperation::kNONE,
                *v,
                MatrixOperation::kNONE)
            : nullptr;
        auto* contextCast = context
            ? network->addCast(*context->getOutput(0), qkv->getType())
            : nullptr;
        auto* output = contextCast
            ? network->addShuffle(*contextCast->getOutput(0))
            : nullptr;
        if (!output) return nullptr;
        Permutation permutation{};
        permutation.order[0] = 1;
        permutation.order[1] = 0;
        permutation.order[2] = 2;
        output->setFirstTranspose(permutation);
        Dims outputDims{};
        outputDims.nbDims = 2;
        outputDims.d[0] = tokens;
        outputDims.d[1] = hidden;
        output->setReshapeDimensions(outputDims);
        return output->getOutput(0);
    }

    nvinfer1::ITensor* TRTBuilder::LowerTextRope(
        ITensor* qkv,
        ITensor* positionIds,
        X::KWARGS& options) {
        const Dims qkvDims = qkv->getDimensions();
        const Dims positionDims = positionIds->getDimensions();
        auto* headsItem = options.find("num_heads");
        auto* kvHeadsItem = options.find("num_kv_heads");
        auto* headDimItem = options.find("head_dim");
        const int heads = headsItem ? static_cast<int>(headsItem->val.ToLongLong()) : 0;
        const int kvHeads = kvHeadsItem ? static_cast<int>(kvHeadsItem->val.ToLongLong()) : 0;
        const int headDim = headDimItem ? static_cast<int>(headDimItem->val.ToLongLong()) : 0;
        if (qkvDims.nbDims != 3 || positionDims.nbDims != 3 || positionDims.d[0] != 3 ||
            heads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim % 2 != 0 ||
            qkvDims.d[2] != (heads + 2 * kvHeads) * headDim) {
            loweringError = "Qwen3-VL text RoPE received incompatible static dimensions";
            return nullptr;
        }
        const int batch = qkvDims.d[0];
        const int tokens = qkvDims.d[1];
        const int qWidth = heads * headDim;
        const int kvWidth = kvHeads * headDim;
        auto reshape = [&](ITensor* source, const Dims& dimensions) -> ITensor* {
            auto* layer = source ? network->addShuffle(*source) : nullptr;
            if (layer) layer->setReshapeDimensions(dimensions);
            return layer ? layer->getOutput(0) : nullptr;
        };
        auto sliceLast = [&](ITensor* source, int startValue, int sizeValue) -> ITensor* {
            const Dims sourceDims = source->getDimensions();
            Dims start{};
            Dims size = sourceDims;
            Dims stride{};
            start.nbDims = sourceDims.nbDims;
            stride.nbDims = sourceDims.nbDims;
            for (int i = 0; i < sourceDims.nbDims; ++i) stride.d[i] = 1;
            start.d[sourceDims.nbDims - 1] = startValue;
            size.d[sourceDims.nbDims - 1] = sizeValue;
            auto* layer = network->addSlice(*source, start, size, stride);
            return layer ? layer->getOutput(0) : nullptr;
        };

        const int frequencyCount = headDim / 2;
        int sections[3] = {24, 20, 20};
        auto* sectionsItem = options.find("mrope_section");
        if (sectionsItem && sectionsItem->val.IsList()) {
            X::List sectionValues(sectionsItem->val);
            if (sectionValues->Size() == 3) {
                for (int i = 0; i < 3; ++i) {
                    sections[i] = static_cast<int>(sectionValues[i].ToLongLong());
                }
            }
        }
        integerVectorWeights.emplace_back(static_cast<size_t>(frequencyCount), 0);
        for (int i = 0; i < sections[1]; ++i) {
            const int index = 1 + 3 * i;
            if (index < frequencyCount) integerVectorWeights.back()[index] = 1;
        }
        for (int i = 0; i < sections[2]; ++i) {
            const int index = 2 + 3 * i;
            if (index < frequencyCount) integerVectorWeights.back()[index] = 2;
        }
        Dims selectorDims{};
        selectorDims.nbDims = 1;
        selectorDims.d[0] = frequencyCount;
        Weights selectorWeights{
            DataType::kINT32,
            integerVectorWeights.back().data(),
            frequencyCount};
        auto* selector = network->addConstant(selectorDims, selectorWeights);
        auto* selectedPositions = selector
            ? network->addGather(*positionIds, *selector->getOutput(0), 0)
            : nullptr;
        auto* positionTranspose = selectedPositions
            ? network->addShuffle(*selectedPositions->getOutput(0))
            : nullptr;
        if (positionTranspose) {
            Permutation permutation{};
            permutation.order[0] = 1;
            permutation.order[1] = 2;
            permutation.order[2] = 0;
            positionTranspose->setFirstTranspose(permutation);
        }
        auto* positionFloat = positionTranspose
            ? network->addCast(*positionTranspose->getOutput(0), DataType::kFLOAT)
            : nullptr;

        vectorWeights.emplace_back(static_cast<size_t>(frequencyCount));
        auto* thetaItem = options.find("rope_theta");
        const float theta = thetaItem ? static_cast<float>(thetaItem->val.ToDouble()) : 10000.0F;
        for (int i = 0; i < frequencyCount; ++i) {
            vectorWeights.back()[i] = 1.0F / std::pow(
                theta,
                static_cast<float>(2 * i) / static_cast<float>(headDim));
        }
        Dims frequencyDims{};
        frequencyDims.nbDims = 3;
        frequencyDims.d[0] = 1;
        frequencyDims.d[1] = 1;
        frequencyDims.d[2] = frequencyCount;
        Weights frequencyWeights{
            DataType::kFLOAT,
            vectorWeights.back().data(),
            frequencyCount};
        auto* frequencies = network->addConstant(frequencyDims, frequencyWeights);
        auto* phase = positionFloat && frequencies
            ? network->addElementWise(
                *positionFloat->getOutput(0),
                *frequencies->getOutput(0),
                ElementWiseOperation::kPROD)
            : nullptr;
        ITensor* phaseParts[] = {phase ? phase->getOutput(0) : nullptr, phase ? phase->getOutput(0) : nullptr};
        auto* fullPhase = phase ? network->addConcatenation(phaseParts, 2) : nullptr;
        if (fullPhase) fullPhase->setAxis(2);
        auto* cosine = fullPhase
            ? network->addUnary(*fullPhase->getOutput(0), UnaryOperation::kCOS)
            : nullptr;
        auto* sine = fullPhase
            ? network->addUnary(*fullPhase->getOutput(0), UnaryOperation::kSIN)
            : nullptr;
        Dims ropeBroadcastDims{};
        ropeBroadcastDims.nbDims = 4;
        ropeBroadcastDims.d[0] = batch;
        ropeBroadcastDims.d[1] = tokens;
        ropeBroadcastDims.d[2] = 1;
        ropeBroadcastDims.d[3] = headDim;
        ITensor* cosBroadcast = cosine
            ? reshape(cosine->getOutput(0), ropeBroadcastDims)
            : nullptr;
        ITensor* sinBroadcast = sine
            ? reshape(sine->getOutput(0), ropeBroadcastDims)
            : nullptr;

        Dims qHeadDims{};
        qHeadDims.nbDims = 4;
        qHeadDims.d[0] = batch;
        qHeadDims.d[1] = tokens;
        qHeadDims.d[2] = heads;
        qHeadDims.d[3] = headDim;
        Dims kvHeadDims = qHeadDims;
        kvHeadDims.d[2] = kvHeads;
        ITensor* q = reshape(sliceLast(qkv, 0, qWidth), qHeadDims);
        ITensor* k = reshape(sliceLast(qkv, qWidth, kvWidth), kvHeadDims);
        ITensor* v = sliceLast(qkv, qWidth + kvWidth, kvWidth);
        auto applyRope = [&](ITensor* source) -> ITensor* {
            auto* sourceFloatLayer = source ? network->addCast(*source, DataType::kFLOAT) : nullptr;
            ITensor* sourceFloat = sourceFloatLayer ? sourceFloatLayer->getOutput(0) : nullptr;
            ITensor* first = sliceLast(sourceFloat, 0, headDim / 2);
            ITensor* second = sliceLast(sourceFloat, headDim / 2, headDim / 2);
            auto* negativeSecond = second
                ? network->addUnary(*second, UnaryOperation::kNEG)
                : nullptr;
            ITensor* rotatedParts[] = {negativeSecond ? negativeSecond->getOutput(0) : nullptr, first};
            auto* rotated = rotatedParts[0] && rotatedParts[1]
                ? network->addConcatenation(rotatedParts, 2)
                : nullptr;
            if (rotated) rotated->setAxis(3);
            auto* base = sourceFloat && cosBroadcast
                ? network->addElementWise(*sourceFloat, *cosBroadcast, ElementWiseOperation::kPROD)
                : nullptr;
            auto* cross = rotated && sinBroadcast
                ? network->addElementWise(
                    *rotated->getOutput(0),
                    *sinBroadcast,
                    ElementWiseOperation::kPROD)
                : nullptr;
            auto* sum = base && cross
                ? network->addElementWise(
                    *base->getOutput(0),
                    *cross->getOutput(0),
                    ElementWiseOperation::kSUM)
                : nullptr;
            auto* cast = sum ? network->addCast(*sum->getOutput(0), qkv->getType()) : nullptr;
            return cast ? cast->getOutput(0) : nullptr;
        };
        q = applyRope(q);
        k = applyRope(k);
        Dims qFlatDims{};
        qFlatDims.nbDims = 3;
        qFlatDims.d[0] = batch;
        qFlatDims.d[1] = tokens;
        qFlatDims.d[2] = qWidth;
        Dims kvFlatDims = qFlatDims;
        kvFlatDims.d[2] = kvWidth;
        q = reshape(q, qFlatDims);
        k = reshape(k, kvFlatDims);
        ITensor* packedInputs[] = {q, k, v};
        auto* packed = q && k && v ? network->addConcatenation(packedInputs, 3) : nullptr;
        if (packed) packed->setAxis(2);
        return packed ? packed->getOutput(0) : nullptr;
    }

    nvinfer1::ITensor* TRTBuilder::LowerTextAttention(
        ITensor* qkv,
        ITensor* attentionMask,
        X::KWARGS& options) {
        const Dims qkvDims = qkv->getDimensions();
        const Dims maskDims = attentionMask->getDimensions();
        auto* headsItem = options.find("num_heads");
        auto* kvHeadsItem = options.find("num_key_value_heads");
        auto* headDimItem = options.find("head_dim");
        const int heads = headsItem ? static_cast<int>(headsItem->val.ToLongLong()) : 0;
        const int kvHeads = kvHeadsItem ? static_cast<int>(kvHeadsItem->val.ToLongLong()) : 0;
        const int headDim = headDimItem ? static_cast<int>(headDimItem->val.ToLongLong()) : 0;
        if (qkvDims.nbDims != 3 || maskDims.nbDims != 2 || heads <= 0 || kvHeads <= 0 ||
            heads % kvHeads != 0 || headDim <= 0 ||
            qkvDims.d[2] != (heads + 2 * kvHeads) * headDim) {
            loweringError = "Qwen3 text attention received incompatible static dimensions";
            return nullptr;
        }
        const int batch = qkvDims.d[0];
        const int tokens = qkvDims.d[1];
        const int qWidth = heads * headDim;
        const int kvWidth = kvHeads * headDim;
        auto sliceLast = [&](int startValue, int sizeValue) -> ITensor* {
            Dims start{};
            Dims size = qkvDims;
            Dims stride{};
            start.nbDims = qkvDims.nbDims;
            stride.nbDims = qkvDims.nbDims;
            for (int i = 0; i < qkvDims.nbDims; ++i) stride.d[i] = 1;
            start.d[2] = startValue;
            size.d[2] = sizeValue;
            auto* layer = network->addSlice(*qkv, start, size, stride);
            return layer ? layer->getOutput(0) : nullptr;
        };
        auto reshape = [&](ITensor* source, const Dims& dimensions) -> ITensor* {
            auto* layer = source ? network->addShuffle(*source) : nullptr;
            if (layer) layer->setReshapeDimensions(dimensions);
            return layer ? layer->getOutput(0) : nullptr;
        };
        Dims qHeadDims{};
        qHeadDims.nbDims = 4;
        qHeadDims.d[0] = batch;
        qHeadDims.d[1] = tokens;
        qHeadDims.d[2] = heads;
        qHeadDims.d[3] = headDim;
        Dims kvHeadDims = qHeadDims;
        kvHeadDims.d[2] = kvHeads;
        ITensor* q = reshape(sliceLast(0, qWidth), qHeadDims);
        ITensor* k = reshape(sliceLast(qWidth, kvWidth), kvHeadDims);
        ITensor* v = reshape(sliceLast(qWidth + kvWidth, kvWidth), kvHeadDims);

        integerVectorWeights.emplace_back(static_cast<size_t>(heads));
        const int repeats = heads / kvHeads;
        for (int head = 0; head < heads; ++head) {
            integerVectorWeights.back()[head] = head / repeats;
        }
        Dims repeatIndexDims{};
        repeatIndexDims.nbDims = 1;
        repeatIndexDims.d[0] = heads;
        Weights repeatIndexWeights{
            DataType::kINT32,
            integerVectorWeights.back().data(),
            heads};
        auto* repeatIndices = network->addConstant(repeatIndexDims, repeatIndexWeights);
        auto* repeatedK = repeatIndices ? network->addGather(*k, *repeatIndices->getOutput(0), 2) : nullptr;
        auto* repeatedV = repeatIndices ? network->addGather(*v, *repeatIndices->getOutput(0), 2) : nullptr;
        k = repeatedK ? repeatedK->getOutput(0) : nullptr;
        v = repeatedV ? repeatedV->getOutput(0) : nullptr;
        auto headMajorFloat = [&](ITensor* source) -> ITensor* {
            auto* shuffle = source ? network->addShuffle(*source) : nullptr;
            if (!shuffle) return nullptr;
            Permutation permutation{};
            permutation.order[0] = 0;
            permutation.order[1] = 2;
            permutation.order[2] = 1;
            permutation.order[3] = 3;
            shuffle->setFirstTranspose(permutation);
            auto* cast = network->addCast(*shuffle->getOutput(0), DataType::kFLOAT);
            return cast ? cast->getOutput(0) : nullptr;
        };
        q = headMajorFloat(q);
        k = headMajorFloat(k);
        v = headMajorFloat(v);
        auto* scores = q && k
            ? network->addMatrixMultiply(
                *q,
                MatrixOperation::kNONE,
                *k,
                MatrixOperation::kTRANSPOSE)
            : nullptr;
        scalarWeights.push_back(1.0F / std::sqrt(static_cast<float>(headDim)));
        Dims scalar4Dims{};
        scalar4Dims.nbDims = 4;
        scalar4Dims.d[0] = 1;
        scalar4Dims.d[1] = 1;
        scalar4Dims.d[2] = 1;
        scalar4Dims.d[3] = 1;
        Weights scaleWeights{DataType::kFLOAT, &scalarWeights.back(), 1};
        auto* scale = network->addConstant(scalar4Dims, scaleWeights);
        auto* scaledScores = scores && scale
            ? network->addElementWise(
                *scores->getOutput(0),
                *scale->getOutput(0),
                ElementWiseOperation::kPROD)
            : nullptr;

        integerWeights.push_back(0);
        Dims maskScalarDims{};
        maskScalarDims.nbDims = 2;
        maskScalarDims.d[0] = 1;
        maskScalarDims.d[1] = 1;
        Weights zeroWeights{DataType::kINT64, &integerWeights.back(), 1};
        auto* zero = network->addConstant(maskScalarDims, zeroWeights);
        auto* paddingMask = zero
            ? network->addElementWise(*attentionMask, *zero->getOutput(0), ElementWiseOperation::kEQUAL)
            : nullptr;
        auto* paddingFloat = paddingMask
            ? network->addCast(*paddingMask->getOutput(0), DataType::kFLOAT)
            : nullptr;
        Dims paddingDims{};
        paddingDims.nbDims = 4;
        paddingDims.d[0] = batch;
        paddingDims.d[1] = 1;
        paddingDims.d[2] = 1;
        paddingDims.d[3] = tokens;
        ITensor* padding = paddingFloat ? reshape(paddingFloat->getOutput(0), paddingDims) : nullptr;
        scalarWeights.push_back(-10000.0F);
        Weights negativeWeights{DataType::kFLOAT, &scalarWeights.back(), 1};
        auto* negative = network->addConstant(scalar4Dims, negativeWeights);
        auto* paddingBias = padding && negative
            ? network->addElementWise(*padding, *negative->getOutput(0), ElementWiseOperation::kPROD)
            : nullptr;

        vectorWeights.emplace_back(static_cast<size_t>(tokens) * static_cast<size_t>(tokens), 0.0F);
        for (int query = 0; query < tokens; ++query) {
            for (int key = query + 1; key < tokens; ++key) {
                vectorWeights.back()[static_cast<size_t>(query) * tokens + key] = -10000.0F;
            }
        }
        Dims causalDims{};
        causalDims.nbDims = 4;
        causalDims.d[0] = 1;
        causalDims.d[1] = 1;
        causalDims.d[2] = tokens;
        causalDims.d[3] = tokens;
        Weights causalWeights{
            DataType::kFLOAT,
            vectorWeights.back().data(),
            static_cast<int64_t>(tokens) * tokens};
        auto* causal = network->addConstant(causalDims, causalWeights);
        auto* withCausal = scaledScores && causal
            ? network->addElementWise(
                *scaledScores->getOutput(0),
                *causal->getOutput(0),
                ElementWiseOperation::kSUM)
            : nullptr;
        auto* maskedScores = withCausal && paddingBias
            ? network->addElementWise(
                *withCausal->getOutput(0),
                *paddingBias->getOutput(0),
                ElementWiseOperation::kSUM)
            : nullptr;
        auto* softmax = maskedScores ? network->addSoftMax(*maskedScores->getOutput(0)) : nullptr;
        if (softmax) softmax->setAxes(1U << 3);
        auto* context = softmax && v
            ? network->addMatrixMultiply(
                *softmax->getOutput(0),
                MatrixOperation::kNONE,
                *v,
                MatrixOperation::kNONE)
            : nullptr;
        auto* contextCast = context
            ? network->addCast(*context->getOutput(0), qkv->getType())
            : nullptr;
        auto* output = contextCast ? network->addShuffle(*contextCast->getOutput(0)) : nullptr;
        if (!output) return nullptr;
        Permutation permutation{};
        permutation.order[0] = 0;
        permutation.order[1] = 2;
        permutation.order[2] = 1;
        permutation.order[3] = 3;
        output->setFirstTranspose(permutation);
        Dims outputDims{};
        outputDims.nbDims = 3;
        outputDims.d[0] = batch;
        outputDims.d[1] = tokens;
        outputDims.d[2] = qWidth;
        output->setReshapeDimensions(outputDims);
        return output->getOutput(0);
    }

    bool TRTBuilder::BuildCapturedGraph(
        X::Value graph,
        X::Value forwardFunction,
        X::ARGS& graphArguments,
        X::ARGS& symbolicInputs,
        const SafeTensorsIndex* weightIndex,
        const std::string& enginePath,
        std::string& errorMessage) {
        tensorMap.clear();
        weightTensorMap.clear();
        scalarWeights.clear();
        integerWeights.clear();
        vectorWeights.clear();
        integerVectorWeights.clear();
        lastOutput = nullptr;
        pendingKVKeyPages = nullptr;
        pendingKVValuePages = nullptr;
        pendingKVPageTable = nullptr;
        pendingKVContextLength = nullptr;
        pendingKVSlotPosition = nullptr;
        for (auto* plugin : ownedPlugins) plugin->destroy();
        ownedPlugins.clear();
        loweringError.clear();
        loweringActive = true;
        branchParentActivity.clear();
        flowBranchTaken.clear();
        capturedWeightIndex = weightIndex;
        capturedWeightFile.Close();
        if (capturedWeightIndex && capturedWeightIndex->TensorCount() > 0) {
            if (!capturedWeightFile.Open(capturedWeightIndex->FilePath(), loweringError)) {
                errorMessage = loweringError;
                capturedWeightIndex = nullptr;
                return false;
            }
        }

        if (!EnsurePagedKVDecodePluginRegistered()) {
            errorMessage = "failed to register Garnet paged-KV TensorRT plugin";
            return false;
        }
        builder = createInferBuilder(gLogger);
        if (!builder) {
            errorMessage = "TensorRT createInferBuilder failed";
            return false;
        }
        const uint32_t flags =
            (1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)) |
            (1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED));
        network = builder->createNetworkV2(flags);
        config = builder->createBuilderConfig();
        if (!network || !config) {
            errorMessage = "TensorRT network/config creation failed";
            delete config;
            delete network;
            delete builder;
            config = nullptr;
            network = nullptr;
            builder = nullptr;
            return false;
        }
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 64ULL << 20);
        if (capturedWeightIndex && capturedWeightIndex->TensorCount() > 0) {
            config->setFlag(BuilderFlag::kREFIT_INDIVIDUAL);
            config->setFlag(BuilderFlag::kSTRIP_PLAN);
        }

        for (size_t index = 0; index < symbolicInputs.size(); ++index) {
            X::Value inputValue = symbolicInputs[index];
            if (!inputValue.IsTensor()) {
                loweringError = "compiled graph inputs must be tensors";
                break;
            }
            X::Tensor input(inputValue);
            DataType trtDataType;
            if (input->GetDataType() == X::TensorDataType::FLOAT32) {
                trtDataType = DataType::kFLOAT;
            }
            else if (input->GetDataType() == X::TensorDataType::BFLOAT16) {
                trtDataType = DataType::kBF16;
            }
            else if (input->GetDataType() == X::TensorDataType::LONGLONG) {
                trtDataType = DataType::kINT64;
            }
            else if (input->GetDataType() == X::TensorDataType::INT) {
                trtDataType = DataType::kINT32;
            }
            else {
                loweringError = "unsupported symbolic TensorRT input dtype";
                break;
            }
            Dims dimensions{};
            dimensions.nbDims = input->GetDimCount();
            if (dimensions.nbDims <= 0 || dimensions.nbDims > Dims::MAX_DIMS) {
                loweringError = "invalid symbolic input rank";
                break;
            }
            for (int dimension = 0; dimension < dimensions.nbDims; ++dimension) {
                dimensions.d[dimension] = input->GetDimSize(dimension);
            }
            const std::string name = "input_" + std::to_string(index);
            ITensor* trtInput = network->addInput(name.c_str(), trtDataType, dimensions);
            if (!trtInput) {
                loweringError = "TensorRT addInput failed for " + name;
                break;
            }
            tensorMap[inputValue.GetObj()->GetID()] = trtInput;
        }

        if (loweringError.empty()) {
            X::TensorGraph tensorGraph(graph);
            X::KWARGS runOptions;
            runOptions.Add("Func", forwardFunction);
            g_trtContext = this;
            const bool ran = tensorGraph->Run(graphArguments, runOptions);
            g_trtContext = nullptr;
            if (!ran && loweringError.empty()) {
                loweringError = "xlang TensorGraph replay failed";
            }
        }

        if (loweringError.empty() && !lastOutput) {
            loweringError = "captured graph produced no lowerable output";
        }
        if (loweringError.empty()) {
            lastOutput->setName("output_0");
            network->markOutput(*lastOutput);
            auto* serialized = builder->buildSerializedNetwork(*network, *config);
            if (!serialized) {
                loweringError = "TensorRT buildSerializedNetwork failed";
            }
            else {
                const std::filesystem::path outputPath(enginePath);
                std::filesystem::create_directories(outputPath.parent_path());
                const std::filesystem::path temporaryPath = outputPath.string() + ".tmp";
                std::ofstream output(temporaryPath, std::ios::binary | std::ios::trunc);
                output.write(
                    static_cast<const char*>(serialized->data()),
                    static_cast<std::streamsize>(serialized->size()));
                output.close();
                InvalidateCachedTRTExecution(enginePath);
                std::error_code fileError;
                std::filesystem::remove(outputPath, fileError);
                fileError.clear();
                std::filesystem::rename(temporaryPath, outputPath, fileError);
                if (fileError) {
                    loweringError = "failed to publish TensorRT engine atomically: " + fileError.message();
                }
                delete serialized;
            }
        }

        delete config;
        delete network;
        delete builder;
        config = nullptr;
        network = nullptr;
        builder = nullptr;
        for (auto* plugin : ownedPlugins) plugin->destroy();
        ownedPlugins.clear();
        tensorMap.clear();
        capturedWeightFile.Close();
        capturedWeightIndex = nullptr;
        lastOutput = nullptr;
        errorMessage = loweringError;
        return loweringError.empty();
    }

    X::Value TRTBuilder::RunCapturedEngine(
        const std::string& enginePath,
        X::Value inputsValue,
        const SafeTensorsIndex* weightIndex,
        std::string& errorMessage) {
        if (!EnsurePagedKVDecodePluginRegistered()) {
            errorMessage = "failed to register Garnet paged-KV TensorRT plugin";
            return X::Value();
        }
        if (!inputsValue.IsList()) {
            errorMessage = "compiled forward requires an inputs list";
            return X::Value();
        }

        ICudaEngine* cachedEngine = nullptr;
        IExecutionContext* cachedContext = nullptr;
        if (!GetCachedTRTExecutionObjects(enginePath, cachedEngine, cachedContext, weightIndex)) {
            errorMessage = "failed to load cached TensorRT engine";
            return X::Value();
        }

        X::List inputs(inputsValue);
        for (long long index = 0; index < inputs->Size(); ++index) {
            X::Value inputValue = inputs->Get(index);
            if (!inputValue.IsTensor()) {
                errorMessage = "compiled forward input_" + std::to_string(index) +
                    " must be an X::Tensor value";
                return X::Value();
            }
            X::Tensor input(inputValue);
            if (input->GetDataType() != X::TensorDataType::FLOAT32 &&
                input->GetDataType() != X::TensorDataType::BFLOAT16 &&
                input->GetDataType() != X::TensorDataType::INT &&
                input->GetDataType() != X::TensorDataType::LONGLONG) {
                errorMessage = "compiled forward accepts FLOAT32, BFLOAT16, INT32, or INT64 inputs";
                return X::Value();
            }
            if (TensorHelper::EnsureGPUMemory(input) != TensorOpStatus::Success) {
                errorMessage = "failed to make compiled input GPU-resident";
                return X::Value();
            }
            void* devicePointer = TensorHelper::GetGPUMemory(input);
            const std::string name = "input_" + std::to_string(index);
            if (!devicePointer || !cachedContext->setTensorAddress(name.c_str(), devicePointer)) {
                errorMessage = "failed to bind TensorRT input " + name;
                return X::Value();
            }
        }

        const Dims outputDimensions = cachedEngine->getTensorShape("output_0");
        if (outputDimensions.nbDims <= 0 || outputDimensions.nbDims > Dims::MAX_DIMS) {
            errorMessage = "TensorRT engine returned an invalid output shape";
            return X::Value();
        }
        size_t outputCount = 1;
        X::Port::vector<int> outputShape(outputDimensions.nbDims);
        for (int dimension = 0; dimension < outputDimensions.nbDims; ++dimension) {
            if (outputDimensions.d[dimension] <= 0) {
                errorMessage = "dynamic output shapes are not implemented in compiled fixture execution";
                return X::Value();
            }
            outputShape.push_back(outputDimensions.d[dimension]);
            outputCount *= static_cast<size_t>(outputDimensions.d[dimension]);
        }

        void* outputDevicePointer = nullptr;
        const DataType outputDataType = cachedEngine->getTensorDataType("output_0");
        X::TensorDataType xlangOutputDataType;
        size_t elementBytes = 0;
        if (outputDataType == DataType::kFLOAT) {
            xlangOutputDataType = X::TensorDataType::FLOAT32;
            elementBytes = sizeof(float);
        }
        else if (outputDataType == DataType::kBF16) {
            xlangOutputDataType = X::TensorDataType::BFLOAT16;
            elementBytes = sizeof(unsigned short);
        }
        else {
            errorMessage = "unsupported TensorRT output dtype";
            return X::Value();
        }
        const size_t outputBytes = outputCount * elementBytes;
        if (cudaMalloc(&outputDevicePointer, outputBytes) != cudaSuccess) {
            errorMessage = "failed to allocate TensorRT output on GPU";
            return X::Value();
        }
        if (!cachedContext->setTensorAddress("output_0", outputDevicePointer) ||
            !cachedContext->enqueueV3(cudaStreamPerThread)) {
            cudaFree(outputDevicePointer);
            errorMessage = "TensorRT enqueueV3 failed";
            return X::Value();
        }

        X::Tensor output(X::g_pXHost->CreateTensor());
        output->SetDataType(xlangOutputDataType);
        output->SetShape(outputShape);
        if (TensorHelper::AttachGPUMemory(output, outputDevicePointer) != TensorOpStatus::Success) {
            cudaFree(outputDevicePointer);
            errorMessage = "failed to attach TensorRT output to X::Tensor";
            return X::Value();
        }
        errorMessage.clear();
        return X::Value(output);
    }

    X::Value TRTBuilder::HandleBinaryOp(
        const std::string& opName,
        X::Value graph,
        X::ARGS& params,
        X::KWARGS& kwParams,
        X::Value input1,
        X::Value input2,
        X::Value output) {
        if (!network || loweringError.size() > 0) {
            return X::Value();
        }
        if (!loweringActive) {
            return X::Value(true);
        }
        const bool isLinear =
            opName == "linear" || opName == "qkv_linear" ||
            opName == "q_proj" || opName == "k_proj" ||
            opName == "v_proj" || opName == "o_proj" ||
            opName == "gate_proj" || opName == "up_proj" ||
            opName == "down_proj" || opName == "lm_head";
        const bool isElementwise =
            opName == "add" || opName == "minus" || opName == "mul";
        const bool isMatrix = opName == "matmul" || isLinear;
        const bool isVisionPositionInterpolate =
            opName == "qwen3_vl_pos_embed_interpolate";
        const bool isVisionRope = opName == "qwen3_vl_apply_vision_rope_packed";
        const bool isVisionAttention = opName == "vision_varlen_attention_packed";
        const bool isVisualEmbeddingMerge = opName == "qwen3_vl_merge_visual_embeddings";
        const bool isTextRope = opName == "qwen3_vl_apply_text_rope_packed";
        const bool isTextAttention = opName == "paged_attention_packed";
        const bool isDeepstackAdd = opName == "qwen3_vl_deepstack_add";
        const bool isPagedKVBinding =
            opName == "paged_kv_bind_key_pages" ||
            opName == "paged_kv_bind_value_pages" ||
            opName == "paged_kv_bind_page_table" ||
            opName == "paged_kv_bind_context_length" ||
            opName == "paged_kv_bind_slot_position";
        if (!isElementwise && !isMatrix && !isVisionPositionInterpolate &&
            !isVisionRope && !isVisionAttention && !isVisualEmbeddingMerge &&
            !isTextRope && !isTextAttention && !isDeepstackAdd && !isPagedKVBinding) {
            loweringError = "unsupported binary operation: " + opName;
            return X::Value();
        }
        if (opName == "mul" && (!input1.IsObject() || !input2.IsObject())) {
            // xlang represents `x * T.binary_op(name) * y` with an empty
            // structural mul followed by the named binary operation. The
            // named item owns the output identity and performs the lowering.
            return X::Value(true);
        }
        ITensor* left = GetOrCreateTRTTensor(input1);
        ITensor* right = GetOrCreateTRTTensor(input2);
        if (!left || !right) {
            return X::Value();
        }

        if (isPagedKVBinding) {
            if (opName == "paged_kv_bind_key_pages") pendingKVKeyPages = right;
            else if (opName == "paged_kv_bind_value_pages") pendingKVValuePages = right;
            else if (opName == "paged_kv_bind_page_table") pendingKVPageTable = right;
            else if (opName == "paged_kv_bind_context_length") pendingKVContextLength = right;
            else pendingKVSlotPosition = right;
            lastOutput = left;
        }

        else if (isDeepstackAdd) {
            auto* inputIdsItem = kwParams.find("input_ids");
            auto* imageTokenItem = kwParams.find("image_token_id");
            auto* videoTokenItem = kwParams.find("video_token_id");
            if (!inputIdsItem || !imageTokenItem || !videoTokenItem) {
                loweringError = "DeepStack add requires input_ids and visual token ids";
                return X::Value();
            }
            ITensor* inputIds = GetOrCreateTRTTensor(inputIdsItem->val);
            auto makeTokenConstant = [&](long long tokenId) -> ITensor* {
                integerWeights.push_back(tokenId);
                Dims dimensions{};
                dimensions.nbDims = 2;
                dimensions.d[0] = 1;
                dimensions.d[1] = 1;
                Weights weights{DataType::kINT64, &integerWeights.back(), 1};
                auto* constant = network->addConstant(dimensions, weights);
                return constant ? constant->getOutput(0) : nullptr;
            };
            ITensor* imageToken = makeTokenConstant(imageTokenItem->val.ToLongLong());
            ITensor* videoToken = makeTokenConstant(videoTokenItem->val.ToLongLong());
            auto* imageMask = inputIds && imageToken
                ? network->addElementWise(*inputIds, *imageToken, ElementWiseOperation::kEQUAL)
                : nullptr;
            auto* videoMask = inputIds && videoToken
                ? network->addElementWise(*inputIds, *videoToken, ElementWiseOperation::kEQUAL)
                : nullptr;
            auto* visualMask = imageMask && videoMask
                ? network->addElementWise(
                    *imageMask->getOutput(0),
                    *videoMask->getOutput(0),
                    ElementWiseOperation::kOR)
                : nullptr;
            auto* nonzero = visualMask
                ? network->addNonZero(*visualMask->getOutput(0), DataType::kINT32)
                : nullptr;
            auto* indices = nonzero ? network->addShuffle(*nonzero->getOutput(0)) : nullptr;
            if (indices) {
                Permutation transpose{};
                transpose.order[0] = 1;
                transpose.order[1] = 0;
                indices->setFirstTranspose(transpose);
            }
            auto* currentRows = indices
                ? network->addGatherV2(*left, *indices->getOutput(0), GatherMode::kND)
                : nullptr;
            auto* updatedRows = currentRows
                ? network->addElementWise(
                    *currentRows->getOutput(0),
                    *right,
                    ElementWiseOperation::kSUM)
                : nullptr;
            auto* scatter = indices && updatedRows
                ? network->addScatter(
                    *left,
                    *indices->getOutput(0),
                    *updatedRows->getOutput(0),
                    ScatterMode::kND)
                : nullptr;
            lastOutput = scatter ? scatter->getOutput(0) : nullptr;
        }
        else if (isTextAttention) {
            lastOutput = LowerTextAttention(left, right, kwParams);
        }
        else if (isTextRope) {
            lastOutput = LowerTextRope(left, right, kwParams);
        }
        else if (isVisualEmbeddingMerge) {
            auto* inputIdsItem = kwParams.find("input_ids");
            auto* imageTokenItem = kwParams.find("image_token_id");
            auto* videoTokenItem = kwParams.find("video_token_id");
            if (!inputIdsItem || !imageTokenItem || !videoTokenItem) {
                loweringError = "visual embedding merge requires input_ids and visual token ids";
                return X::Value();
            }
            ITensor* inputIds = GetOrCreateTRTTensor(inputIdsItem->val);
            if (!inputIds || inputIds->getDimensions().nbDims != 2 ||
                left->getDimensions().nbDims != 3 || right->getDimensions().nbDims != 2) {
                loweringError = "visual embedding merge received incompatible tensor ranks";
                return X::Value();
            }
            auto makeTokenConstant = [&](long long tokenId) -> ITensor* {
                integerWeights.push_back(tokenId);
                Dims dimensions{};
                dimensions.nbDims = 2;
                dimensions.d[0] = 1;
                dimensions.d[1] = 1;
                Weights weights{DataType::kINT64, &integerWeights.back(), 1};
                auto* constant = network->addConstant(dimensions, weights);
                return constant ? constant->getOutput(0) : nullptr;
            };
            ITensor* imageToken = makeTokenConstant(imageTokenItem->val.ToLongLong());
            ITensor* videoToken = makeTokenConstant(videoTokenItem->val.ToLongLong());
            auto* imageMask = imageToken
                ? network->addElementWise(*inputIds, *imageToken, ElementWiseOperation::kEQUAL)
                : nullptr;
            auto* videoMask = videoToken
                ? network->addElementWise(*inputIds, *videoToken, ElementWiseOperation::kEQUAL)
                : nullptr;
            auto* visualMask = imageMask && videoMask
                ? network->addElementWise(
                    *imageMask->getOutput(0),
                    *videoMask->getOutput(0),
                    ElementWiseOperation::kOR)
                : nullptr;
            auto* nonzero = visualMask
                ? network->addNonZero(*visualMask->getOutput(0), DataType::kINT32)
                : nullptr;
            auto* indices = nonzero ? network->addShuffle(*nonzero->getOutput(0)) : nullptr;
            if (indices) {
                Permutation transpose{};
                transpose.order[0] = 1;
                transpose.order[1] = 0;
                indices->setFirstTranspose(transpose);
            }
            auto* scatter = indices
                ? network->addScatter(
                    *left,
                    *indices->getOutput(0),
                    *right,
                    ScatterMode::kND)
                : nullptr;
            lastOutput = scatter ? scatter->getOutput(0) : nullptr;
        }
        else if (isVisionRope) {
            lastOutput = LowerVisionRope(left, right, kwParams);
        }
        else if (isVisionAttention) {
            lastOutput = LowerVisionAttention(left, right, kwParams);
        }
        else if (isVisionPositionInterpolate) {
            auto* weightNameItem = kwParams.find("weight_name");
            if (!weightNameItem) {
                loweringError = "vision position interpolation requires weight_name";
                return X::Value();
            }
            ITensor* positionTable = GetOrCreateTRTWeight(weightNameItem->val.ToString());
            auto* gather = positionTable
                ? network->addGather(*positionTable, *left, 0)
                : nullptr;
            const Dims weightDimensions = right->getDimensions();
            if (!gather || weightDimensions.nbDims != 2) {
                loweringError = "vision position interpolation requires rank-2 indices and weights";
                return X::Value();
            }
            Dims expandedWeightDimensions{};
            expandedWeightDimensions.nbDims = 3;
            expandedWeightDimensions.d[0] = weightDimensions.d[0];
            expandedWeightDimensions.d[1] = weightDimensions.d[1];
            expandedWeightDimensions.d[2] = 1;
            auto* expand = network->addShuffle(*right);
            if (expand) {
                expand->setReshapeDimensions(expandedWeightDimensions);
            }
            auto* weighted = expand
                ? network->addElementWise(
                    *gather->getOutput(0),
                    *expand->getOutput(0),
                    ElementWiseOperation::kPROD)
                : nullptr;
            auto* sum = weighted
                ? network->addReduce(
                    *weighted->getOutput(0),
                    ReduceOperation::kSUM,
                    1U << 1,
                    false)
                : nullptr;
            lastOutput = sum ? sum->getOutput(0) : nullptr;
        }
        else if (opName == "matmul" || isLinear) {
            auto* layer = network->addMatrixMultiply(
                *left,
                MatrixOperation::kNONE,
                *right,
                isLinear ? MatrixOperation::kTRANSPOSE : MatrixOperation::kNONE);
            if (!layer || !layer->getOutput(0)) {
                loweringError = "TensorRT matrix multiply lowering failed for " + opName;
                return X::Value();
            }
            lastOutput = layer->getOutput(0);
        }
        else if (opName == "add" || opName == "minus" || opName == "mul") {
            const ElementWiseOperation operation =
                opName == "add" ? ElementWiseOperation::kSUM :
                opName == "minus" ? ElementWiseOperation::kSUB :
                ElementWiseOperation::kPROD;
            auto* layer = network->addElementWise(*left, *right, operation);
            if (!layer || !layer->getOutput(0)) {
                loweringError = "TensorRT elementwise lowering failed for " + opName;
                return X::Value();
            }
            lastOutput = layer->getOutput(0);
        }
        if (!lastOutput) {
            if (loweringError.empty()) {
                loweringError = "TensorRT binary lowering failed for " + opName;
            }
            return X::Value();
        }
        if (!output.IsObject()) {
            loweringError = "binary operation has no graph output identity";
            return X::Value();
        }
        tensorMap[output.GetObj()->GetID()] = lastOutput;
        return X::Value(true);
    }

    X::Value TRTBuilder::HandleUnaryOp(
        const std::string& opName,
        X::Value graph,
        X::ARGS& params,
        X::KWARGS& kwParams,
        X::Value input,
        X::Value output) {
        if (!loweringActive) {
            return X::Value(true);
        }
        if (!network || loweringError.size() > 0) {
            return X::Value();
        }
        ITensor* source = GetOrCreateTRTTensor(input);
        if (!source) {
            loweringError = "unary operation " + opName + " input: " + loweringError;
            return X::Value();
        }

        const bool isLinear =
            opName == "linear" || opName == "qkv_linear" ||
            opName == "q_proj" || opName == "k_proj" ||
            opName == "v_proj" || opName == "o_proj" ||
            opName == "gate_proj" || opName == "up_proj" ||
            opName == "down_proj" || opName == "lm_head";

        if (opName == "paged_kv_select_layer") {
            auto* layerItem = kwParams.find("layer_idx");
            if (!layerItem) {
                loweringError = "paged_kv_select_layer requires layer_idx";
                return X::Value();
            }
            const int layerIndex = static_cast<int>(layerItem->val.ToLongLong());
            const Dims sourceDimensions = source->getDimensions();
            if (sourceDimensions.nbDims < 2 || layerIndex < 0 ||
                layerIndex >= sourceDimensions.d[0]) {
                loweringError = "paged_kv_select_layer index is outside the packed cache";
                return X::Value();
            }
            pendingKVLayerIndex = layerIndex;
            lastOutput = source;
        }

        else if (opName == "paged_kv_prefill_write_bf16") {
            auto getIntOption = [&](const char* name, int defaultValue) {
                auto* item = kwParams.find(name);
                return item ? static_cast<int>(item->val.ToLongLong()) : defaultValue;
            };
            const int pageSize = getIntOption("page_size", 16);
            const int qHeads = getIntOption("q_heads", 0);
            const int kvHeads = getIntOption("kv_heads", 0);
            const int headDim = getIntOption("head_dim", 0);
            if (!pendingKVKeyPages || !pendingKVValuePages || !pendingKVPageTable ||
                !pendingKVSlotPosition || pageSize <= 0 || qHeads <= 0 ||
                kvHeads <= 0 || headDim <= 0) {
                loweringError = "paged_kv_prefill_write_bf16 requires explicit cache bindings and geometry";
                return X::Value();
            }
            auto* plugin = new PagedKVPrefillWritePlugin(
                pageSize, qHeads, kvHeads, headDim, pendingKVLayerIndex);
            ownedPlugins.push_back(plugin);
            ITensor* pluginInputs[] = {
                source,
                pendingKVKeyPages,
                pendingKVValuePages,
                pendingKVPageTable,
                pendingKVSlotPosition,
            };
            auto* layer = network->addPluginV2(pluginInputs, 5, *plugin);
            lastOutput = layer ? layer->getOutput(0) : nullptr;
            pendingKVKeyPages = nullptr;
            pendingKVValuePages = nullptr;
            pendingKVPageTable = nullptr;
            pendingKVSlotPosition = nullptr;
            pendingKVLayerIndex = -1;
        }

        else if (opName == "paged_kv_decode_bf16") {
            auto getIntOption = [&](const char* name, int defaultValue) {
                auto* item = kwParams.find(name);
                return item ? static_cast<int>(item->val.ToLongLong()) : defaultValue;
            };
            const int pageSize = getIntOption("page_size", 16);
            const int qHeads = getIntOption("q_heads", 0);
            const int kvHeads = getIntOption("kv_heads", 0);
            const int headDim = getIntOption("head_dim", 0);
            if (!pendingKVKeyPages || !pendingKVValuePages || !pendingKVPageTable ||
                !pendingKVContextLength || !pendingKVSlotPosition || pageSize <= 0 ||
                qHeads <= 0 || kvHeads <= 0 || headDim <= 0) {
                loweringError = "paged_kv_decode_bf16 requires explicit cache bindings and geometry";
                return X::Value();
            }
            auto* plugin = new PagedKVDecodePlugin(
                pageSize, qHeads, kvHeads, headDim, pendingKVLayerIndex);
            ownedPlugins.push_back(plugin);
            ITensor* pluginInputs[] = {
                source,
                pendingKVKeyPages,
                pendingKVValuePages,
                pendingKVPageTable,
                pendingKVContextLength,
                pendingKVSlotPosition,
            };
            auto* layer = network->addPluginV2(pluginInputs, 6, *plugin);
            lastOutput = layer ? layer->getOutput(0) : nullptr;
            pendingKVKeyPages = nullptr;
            pendingKVValuePages = nullptr;
            pendingKVPageTable = nullptr;
            pendingKVContextLength = nullptr;
            pendingKVSlotPosition = nullptr;
            pendingKVLayerIndex = -1;
        }

        else if (opName == "embedding") {
            auto* weightNameItem = kwParams.find("weight_name");
            if (!weightNameItem) {
                loweringError = "embedding operation is missing weight_name";
                return X::Value();
            }
            const std::string weightName = weightNameItem->val.ToString();
            ITensor* weight = GetOrCreateTRTWeight(weightName);
            if (!weight) {
                loweringError = "embedding weight " + weightName + ": " + loweringError;
                return X::Value();
            }
            auto* layer = network->addGather(*weight, *source, 0);
            if (layer) {
                layer->setName("embedding");
            }
            lastOutput = layer ? layer->getOutput(0) : nullptr;
        }
        else if (isLinear) {
            auto* weightNameItem = kwParams.find("weight_name");
            if (!weightNameItem) {
                loweringError = opName + " operation is missing weight_name";
                return X::Value();
            }
            ITensor* weight = GetOrCreateTRTWeight(weightNameItem->val.ToString());
            weight = BroadcastMatrixWeight(weight, source->getDimensions().nbDims);
            auto* projection = weight
                ? network->addMatrixMultiply(
                    *source,
                    MatrixOperation::kNONE,
                    *weight,
                    MatrixOperation::kTRANSPOSE)
                : nullptr;
            if (!projection) {
                loweringError = opName + " matrix projection failed: " + loweringError;
                return X::Value();
            }
            lastOutput = projection->getOutput(0);
            auto* biasNameItem = kwParams.find("bias_name");
            if (biasNameItem && !biasNameItem->val.IsNone()) {
                const std::string biasName = biasNameItem->val.ToString();
                if (!biasName.empty()) {
                    ITensor* bias = GetOrCreateTRTWeight(biasName);
                    bias = BroadcastLastDimension(
                        bias,
                        lastOutput->getDimensions().nbDims,
                        opName + "_bias_broadcast");
                    auto* biased = bias
                        ? network->addElementWise(*lastOutput, *bias, ElementWiseOperation::kSUM)
                        : nullptr;
                    if (!biased) {
                        loweringError = opName + " bias addition failed: " + loweringError;
                        return X::Value();
                    }
                    lastOutput = biased->getOutput(0);
                }
            }
        }
        else if (opName == "layer_norm") {
            auto* weightNameItem = kwParams.find("weight_name");
            auto* biasNameItem = kwParams.find("bias_name");
            if (!weightNameItem || !biasNameItem) {
                loweringError = "layer_norm requires weight_name and bias_name";
                return X::Value();
            }
            ITensor* scale = GetOrCreateTRTWeight(weightNameItem->val.ToString());
            ITensor* bias = GetOrCreateTRTWeight(biasNameItem->val.ToString());
            const Dims sourceDimensions = source->getDimensions();
            scale = BroadcastLastDimension(scale, sourceDimensions.nbDims, "layer_norm_scale_broadcast");
            bias = BroadcastLastDimension(bias, sourceDimensions.nbDims, "layer_norm_bias_broadcast");
            if (!scale || !bias || sourceDimensions.nbDims <= 0) {
                loweringError = "layer_norm constants or input rank are invalid: " + loweringError;
                return X::Value();
            }
            auto* layer = network->addNormalization(
                *source,
                *scale,
                *bias,
                1U << (sourceDimensions.nbDims - 1));
            auto* epsilonItem = kwParams.find("eps");
            if (layer && epsilonItem) {
                layer->setEpsilon(static_cast<float>(epsilonItem->val.ToDouble()));
            }
            lastOutput = layer ? layer->getOutput(0) : nullptr;
        }
        else if (opName == "rms_norm") {
            auto* weightNameItem = kwParams.find("weight_name");
            if (!weightNameItem) {
                loweringError = "rms_norm requires weight_name";
                return X::Value();
            }
            ITensor* scale = GetOrCreateTRTWeight(weightNameItem->val.ToString());
            const Dims sourceDimensions = source->getDimensions();
            scale = BroadcastLastDimension(scale, sourceDimensions.nbDims, "rms_norm_scale_broadcast");
            if (!scale || sourceDimensions.nbDims <= 0) {
                loweringError = "rms_norm scale or input rank is invalid: " + loweringError;
                return X::Value();
            }
            auto* sourceFloatLayer = network->addCast(*source, DataType::kFLOAT);
            ITensor* sourceFloat = sourceFloatLayer ? sourceFloatLayer->getOutput(0) : nullptr;
            auto* scaleFloatLayer = scale ? network->addCast(*scale, DataType::kFLOAT) : nullptr;
            ITensor* scaleFloat = scaleFloatLayer ? scaleFloatLayer->getOutput(0) : nullptr;
            auto* square = sourceFloat
                ? network->addElementWise(*sourceFloat, *sourceFloat, ElementWiseOperation::kPROD)
                : nullptr;
            auto* mean = square
                ? network->addReduce(
                    *square->getOutput(0),
                    ReduceOperation::kAVG,
                    1U << (sourceDimensions.nbDims - 1),
                    true)
                : nullptr;
            auto* epsilonItem = kwParams.find("eps");
            scalarWeights.push_back(epsilonItem
                ? static_cast<float>(epsilonItem->val.ToDouble())
                : 1.0e-6F);
            Dims scalarDimensions{};
            scalarDimensions.nbDims = sourceDimensions.nbDims;
            for (int dimension = 0; dimension < scalarDimensions.nbDims; ++dimension) {
                scalarDimensions.d[dimension] = 1;
            }
            Weights epsilonWeights{DataType::kFLOAT, &scalarWeights.back(), 1};
            auto* epsilon = network->addConstant(scalarDimensions, epsilonWeights);
            auto* variance = mean && epsilon
                ? network->addElementWise(
                    *mean->getOutput(0),
                    *epsilon->getOutput(0),
                    ElementWiseOperation::kSUM)
                : nullptr;
            auto* root = variance
                ? network->addUnary(*variance->getOutput(0), UnaryOperation::kSQRT)
                : nullptr;
            auto* normalized = root
                ? network->addElementWise(
                    *sourceFloat,
                    *root->getOutput(0),
                    ElementWiseOperation::kDIV)
                : nullptr;
            auto* scaled = normalized
                ? network->addElementWise(
                    *normalized->getOutput(0),
                    *scaleFloat,
                    ElementWiseOperation::kPROD)
                : nullptr;
            auto* outputCast = scaled
                ? network->addCast(*scaled->getOutput(0), source->getType())
                : nullptr;
            lastOutput = outputCast ? outputCast->getOutput(0) : nullptr;
        }
        else if (opName == "qwen3_text_qkv_packed") {
            auto weight = [&](const char* name) -> ITensor* {
                auto* item = kwParams.find(name);
                return item ? GetOrCreateTRTWeight(item->val.ToString()) : nullptr;
            };
            ITensor* qWeight = weight("q_weight_name");
            ITensor* kWeight = weight("k_weight_name");
            ITensor* vWeight = weight("v_weight_name");
            ITensor* qNormWeight = weight("q_norm_weight_name");
            ITensor* kNormWeight = weight("k_norm_weight_name");
            auto* numHeadsItem = kwParams.find("num_heads");
            auto* numKvHeadsItem = kwParams.find("num_kv_heads");
            auto* headDimItem = kwParams.find("head_dim");
            const int numHeads = numHeadsItem ? static_cast<int>(numHeadsItem->val.ToLongLong()) : 0;
            const int numKvHeads = numKvHeadsItem ? static_cast<int>(numKvHeadsItem->val.ToLongLong()) : 0;
            const int headDim = headDimItem ? static_cast<int>(headDimItem->val.ToLongLong()) : 0;
            if (!qWeight || !kWeight || !vWeight || !qNormWeight || !kNormWeight ||
                numHeads <= 0 || numKvHeads <= 0 || headDim <= 0) {
                loweringError = "Qwen3 text packed QKV metadata or weights are incomplete";
                return X::Value();
            }
            auto project = [&](ITensor* projectionWeight) -> ITensor* {
                projectionWeight = BroadcastMatrixWeight(
                    projectionWeight,
                    source->getDimensions().nbDims);
                if (!projectionWeight) return nullptr;
                auto* layer = network->addMatrixMultiply(
                    *source,
                    MatrixOperation::kNONE,
                    *projectionWeight,
                    MatrixOperation::kTRANSPOSE);
                return layer ? layer->getOutput(0) : nullptr;
            };
            auto headRmsNorm = [&](ITensor* projected, ITensor* normWeight, int heads) -> ITensor* {
                if (!projected || !normWeight) return nullptr;
                const Dims projectedDimensions = projected->getDimensions();
                if (projectedDimensions.nbDims != 3) return nullptr;
                Dims headDimensions{};
                headDimensions.nbDims = 4;
                headDimensions.d[0] = projectedDimensions.d[0];
                headDimensions.d[1] = projectedDimensions.d[1];
                headDimensions.d[2] = heads;
                headDimensions.d[3] = headDim;
                auto* reshape = network->addShuffle(*projected);
                if (!reshape) return nullptr;
                reshape->setReshapeDimensions(headDimensions);
                ITensor* headed = reshape->getOutput(0);
                ITensor* broadcastScale = BroadcastLastDimension(
                    normWeight,
                    4,
                    "qwen3_text_head_norm_scale");
                auto* headedFloatLayer = network->addCast(*headed, DataType::kFLOAT);
                auto* scaleFloatLayer = broadcastScale
                    ? network->addCast(*broadcastScale, DataType::kFLOAT)
                    : nullptr;
                ITensor* headedFloat = headedFloatLayer ? headedFloatLayer->getOutput(0) : nullptr;
                auto* square = headedFloat
                    ? network->addElementWise(*headedFloat, *headedFloat, ElementWiseOperation::kPROD)
                    : nullptr;
                auto* mean = square
                    ? network->addReduce(
                        *square->getOutput(0),
                        ReduceOperation::kAVG,
                        1U << 3,
                        true)
                    : nullptr;
                auto* epsilonItem = kwParams.find("norm_eps");
                scalarWeights.push_back(epsilonItem
                    ? static_cast<float>(epsilonItem->val.ToDouble())
                    : 1.0e-6F);
                Dims scalarDimensions{};
                scalarDimensions.nbDims = 4;
                scalarDimensions.d[0] = 1;
                scalarDimensions.d[1] = 1;
                scalarDimensions.d[2] = 1;
                scalarDimensions.d[3] = 1;
                Weights epsilonWeights{DataType::kFLOAT, &scalarWeights.back(), 1};
                auto* epsilon = network->addConstant(scalarDimensions, epsilonWeights);
                auto* variance = mean && epsilon
                    ? network->addElementWise(
                        *mean->getOutput(0),
                        *epsilon->getOutput(0),
                        ElementWiseOperation::kSUM)
                    : nullptr;
                auto* root = variance
                    ? network->addUnary(*variance->getOutput(0), UnaryOperation::kSQRT)
                    : nullptr;
                auto* normalized = root
                    ? network->addElementWise(
                        *headedFloat,
                        *root->getOutput(0),
                        ElementWiseOperation::kDIV)
                    : nullptr;
                auto* scaled = normalized && scaleFloatLayer
                    ? network->addElementWise(
                        *normalized->getOutput(0),
                        *scaleFloatLayer->getOutput(0),
                        ElementWiseOperation::kPROD)
                    : nullptr;
                auto* cast = scaled
                    ? network->addCast(*scaled->getOutput(0), projected->getType())
                    : nullptr;
                auto* flatten = cast ? network->addShuffle(*cast->getOutput(0)) : nullptr;
                if (flatten) flatten->setReshapeDimensions(projectedDimensions);
                return flatten ? flatten->getOutput(0) : nullptr;
            };
            ITensor* q = headRmsNorm(project(qWeight), qNormWeight, numHeads);
            ITensor* k = headRmsNorm(project(kWeight), kNormWeight, numKvHeads);
            ITensor* v = project(vWeight);
            ITensor* packedInputs[] = {q, k, v};
            auto* packed = q && k && v ? network->addConcatenation(packedInputs, 3) : nullptr;
            if (packed) packed->setAxis(2);
            lastOutput = packed ? packed->getOutput(0) : nullptr;
        }
        else if (opName == "paged_kv_update_packed") {
            auto* enabledItem = kwParams.find("enabled");
            const bool enabled = enabledItem && enabledItem->val.ToLongLong() != 0;
            if (enabled) {
                loweringError = "paged KV update requires the decode engine profile and cache bindings";
                return X::Value();
            }
            lastOutput = source;
        }
        else if (opName == "merge_attention_heads") {
            // LowerTextAttention already returns [batch, tokens, hidden].
            lastOutput = source;
        }
        else if (opName == "qwen3_vl_patch_embed_conv3d") {
            auto* weightNameItem = kwParams.find("weight_name");
            auto* biasNameItem = kwParams.find("bias_name");
            if (!weightNameItem || !biasNameItem) {
                loweringError = "Qwen3-VL patch embedding requires weight_name and bias_name";
                return X::Value();
            }
            const std::string weightName = weightNameItem->val.ToString();
            const std::string biasName = biasNameItem->val.ToString();
            const SafeTensorMetadata* weightMetadata = capturedWeightIndex
                ? capturedWeightIndex->Find(weightName)
                : nullptr;
            if (!weightMetadata || weightMetadata->shape.size() != 5) {
                loweringError = "Qwen3-VL patch embedding weight must be rank 5: " + weightName;
                return X::Value();
            }
            ITensor* weight = GetOrCreateTRTWeight(weightName);
            ITensor* bias = GetOrCreateTRTWeight(biasName);
            if (!weight || !bias) {
                loweringError = "Qwen3-VL patch embedding constants: " + loweringError;
                return X::Value();
            }
            const long long outputWidth = weightMetadata->shape[0];
            long long inputWidth = 1;
            for (size_t dimension = 1; dimension < weightMetadata->shape.size(); ++dimension) {
                inputWidth *= weightMetadata->shape[dimension];
            }
            if (outputWidth > std::numeric_limits<int>::max() ||
                inputWidth > std::numeric_limits<int>::max()) {
                loweringError = "Qwen3-VL patch embedding dimensions exceed TensorRT limits";
                return X::Value();
            }
            Dims flattenedWeightShape{};
            flattenedWeightShape.nbDims = 2;
            flattenedWeightShape.d[0] = static_cast<int>(outputWidth);
            flattenedWeightShape.d[1] = static_cast<int>(inputWidth);
            auto* flatten = network->addShuffle(*weight);
            if (flatten) {
                flatten->setReshapeDimensions(flattenedWeightShape);
                flatten->setName("qwen3_vl_patch_embed_flatten_weight");
            }
            auto* projection = flatten
                ? network->addMatrixMultiply(
                    *source,
                    MatrixOperation::kNONE,
                    *flatten->getOutput(0),
                    MatrixOperation::kTRANSPOSE)
                : nullptr;
            bias = projection
                ? BroadcastLastDimension(
                    bias,
                    projection->getOutput(0)->getDimensions().nbDims,
                    "qwen3_vl_patch_embed_bias_broadcast")
                : nullptr;
            auto* biased = projection
                ? network->addElementWise(
                    *projection->getOutput(0),
                    *bias,
                    ElementWiseOperation::kSUM)
                : nullptr;
            lastOutput = biased ? biased->getOutput(0) : nullptr;
        }
        else if (opName == "qwen3_vl_patch_merger_shuffle") {
            const Dims sourceDimensions = source->getDimensions();
            auto* mergeSizeItem = kwParams.find("spatial_merge_size");
            const int mergeSize = mergeSizeItem
                ? static_cast<int>(mergeSizeItem->val.ToLongLong())
                : 0;
            const int mergeUnit = mergeSize * mergeSize;
            if (sourceDimensions.nbDims != 2 || mergeUnit <= 0 ||
                sourceDimensions.d[0] % mergeUnit != 0) {
                loweringError = "Qwen3-VL patch merger received incompatible static dimensions";
                return X::Value();
            }
            Dims mergedDimensions{};
            mergedDimensions.nbDims = 2;
            mergedDimensions.d[0] = sourceDimensions.d[0] / mergeUnit;
            mergedDimensions.d[1] = sourceDimensions.d[1] * mergeUnit;
            auto* layer = network->addShuffle(*source);
            if (layer) {
                layer->setReshapeDimensions(mergedDimensions);
            }
            lastOutput = layer ? layer->getOutput(0) : nullptr;
        }
        else if (opName == "silu") {
            auto* sigmoid = network->addActivation(*source, ActivationType::kSIGMOID);
            auto* product = sigmoid
                ? network->addElementWise(*source, *sigmoid->getOutput(0), ElementWiseOperation::kPROD)
                : nullptr;
            lastOutput = product ? product->getOutput(0) : nullptr;
        }
        else {
            ActivationType activation;
            if (opName == "relu") activation = ActivationType::kRELU;
            else if (opName == "sigmoid") activation = ActivationType::kSIGMOID;
            else if (opName == "tanh") activation = ActivationType::kTANH;
            else if (opName == "gelu_pytorch_tanh" || opName == "gelu_tanh") {
                activation = ActivationType::kGELU_TANH;
            }
            else {
                loweringError = "unsupported unary operation: " + opName;
                return X::Value();
            }
            auto* layer = network->addActivation(*source, activation);
            lastOutput = layer ? layer->getOutput(0) : nullptr;
        }
        if (!lastOutput || !output.IsObject()) {
            loweringError = "TensorRT unary lowering failed for " + opName;
            return X::Value();
        }
        tensorMap[output.GetObj()->GetID()] = lastOutput;
        return X::Value(true);
    }

    X::Value TRTBuilder::HandleBranchBegin(
        const std::string& condition,
        int branchType,
        unsigned long long flowId,
        int branchId) {
        const bool parentActive = loweringActive;
        branchParentActivity.push_back(parentActive);

        bool selectBranch = false;
        if (branchType == 2 || branchId == -1) {
            selectBranch = !flowBranchTaken[flowId];
        }
        else {
            bool conditionValue = false;
            if (condition == "True" || condition == "true" || condition == "1") {
                conditionValue = true;
            }
            else if (condition == "False" || condition == "false" || condition == "0") {
                conditionValue = false;
            }
            else {
                loweringError = "dynamic branch lowering is not implemented; condition=" + condition;
                loweringActive = false;
                return X::Value();
            }
            selectBranch = !flowBranchTaken[flowId] && conditionValue;
            if (conditionValue) {
                flowBranchTaken[flowId] = true;
            }
        }
        loweringActive = parentActive && selectBranch;
        return X::Value(true);
    }

    X::Value TRTBuilder::HandleBranchEnd() {
        if (branchParentActivity.empty()) {
            loweringError = "unbalanced branchEnd in captured graph";
            return X::Value();
        }
        loweringActive = branchParentActivity.back();
        branchParentActivity.pop_back();
        return X::Value(true);
    }
}
