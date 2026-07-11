#include "trt_builder.h"
#include "cuda_lib.h"
#include "garnet_tensor.h"
#include "garnet_tensor.h"
#include "tensor_helper.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <cstdlib>
#include <mutex>
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
        struct CachedTRTExecution {
            nvinfer1::IRuntime* runtime = nullptr;
            nvinfer1::ICudaEngine* engine = nullptr;
            nvinfer1::IExecutionContext* context = nullptr;
        };

        std::mutex g_trtExecutionCacheMutex;
        std::unordered_map<std::string, CachedTRTExecution> g_trtExecutionCache;

        nvinfer1::IExecutionContext* GetCachedTRTExecutionContext(const std::string& enginePath) {
            std::lock_guard<std::mutex> lock(g_trtExecutionCacheMutex);
            auto found = g_trtExecutionCache.find(enginePath);
            if (found != g_trtExecutionCache.end()) {
                return found->second.context;
            }

            std::ifstream in(enginePath, std::ios::binary | std::ios::ate);
            if (!in.is_open()) {
                std::cout << "[TRTBuilder] Failed to open engine for read: " << enginePath << std::endl;
                return nullptr;
            }
            std::streamsize size = in.tellg();
            in.seekg(0, std::ios::beg);
            std::vector<char> engineBytes(static_cast<size_t>(size));
            if (!in.read(engineBytes.data(), size)) {
                std::cout << "[TRTBuilder] Failed to read engine bytes: " << enginePath << std::endl;
                return nullptr;
            }

            CachedTRTExecution cached;
            cached.runtime = createInferRuntime(gLogger);
            if (!cached.runtime) {
                std::cout << "[TRTBuilder] createInferRuntime failed for cached engine: " << enginePath << std::endl;
                return nullptr;
            }
            cached.engine = cached.runtime->deserializeCudaEngine(engineBytes.data(), engineBytes.size());
            if (!cached.engine) {
                std::cout << "[TRTBuilder] deserializeCudaEngine failed for cached engine: " << enginePath << std::endl;
                return nullptr;
            }
            cached.context = cached.engine->createExecutionContext();
            if (!cached.context) {
                std::cout << "[TRTBuilder] createExecutionContext failed for cached engine: " << enginePath << std::endl;
                return nullptr;
            }

            auto inserted = g_trtExecutionCache.emplace(enginePath, cached);
            std::cout << "[TRTBuilder] Cached TensorRT execution context: " << enginePath << std::endl;
            return inserted.first->second.context;
        }

        bool ShouldSyncTRTOutputToCPU() {
            const char* value = std::getenv("GARNET_TRT_SYNC_CPU_OUTPUTS");
            if (!value) {
                return true;
            }
            return !(value[0] == '0' && value[1] == '\0');
        }

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

            // Tensor ownership is the device-memory contract. Promote a CPU
            // tensor once and retain its CUDA allocation for the tensor's
            // lifetime instead of uploading/freeing weights on every forward.
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
            cudaError_t syncErr = cudaStreamSynchronize(stream);
            if (syncErr != cudaSuccess) {
                std::cout << "[TRTBuilder] MakeGPUBackedTensor2D stream sync failed before tensor wrap: "
                    << cudaGetErrorString(syncErr) << std::endl;
                return X::Value();
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
            cudaError_t syncErr = cudaStreamSynchronize(stream);
            if (syncErr != cudaSuccess) {
                std::cout << "[TRTBuilder] RebindExistingTensor2D stream sync failed: "
                    << cudaGetErrorString(syncErr) << std::endl;
                return X::Value();
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

    X::Value TRTBuilder::BuildEngine(X::Value forwardFunc, X::Value inputShapes, X::Value weightsDict) {
        std::cout << "[TRTBuilder] Entered BuildEngine" << std::endl;
        // Initialize builder
        auto builder = createInferBuilder(gLogger);
        std::cout << "[TRTBuilder] createInferBuilder returned" << std::endl;
        if (!builder) return X::Value();

        std::cout << "[TRTBuilder] Calling createNetworkV2" << std::endl;
        uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(flag);
        if (!network) return X::Value();

        std::cout << "[TRTBuilder] Calling createBuilderConfig" << std::endl;
        auto config = builder->createBuilderConfig();
        if (!config) return X::Value();

        std::cout << "[TRTBuilder] Converting input shapes" << std::endl;
        
        X::ARGS params;
        if (inputShapes.IsList()) {
            X::List shapeList(inputShapes);
            long long list_size = shapeList->Size();
            std::cout << "[TRTBuilder] shapeList size: " << list_size << std::endl;
            params.resize(list_size);
            for (long long i = 0; i < list_size; ++i) {
                X::Value shape_val = shapeList->Get(i);
                std::cout << "[TRTBuilder] shape_val type: " << (int)shape_val.GetType() << ", is list: " << shape_val.IsList() << std::endl;
                if (shape_val.IsList()) {
                    X::List dims(shape_val);
                    long long dim_size = dims->Size();
                    X::Port::vector<int> shape_vec(dim_size);
                    std::cout << "[TRTBuilder] dim_size: " << dim_size << std::endl;
                    for (long long j = 0; j < dim_size; ++j) {
                        X::Value dim = dims->Get(j);
                        shape_vec.push_back((int)dim.ToLongLong());
                    }
                    std::cout << "[TRTBuilder] Created shape vec, dim0: " << shape_vec[0] << std::endl;
                    X::Tensor tensor(X::g_pXHost->CreateTensor());
                    if (tensor) {
                        tensor->SetShape(shape_vec);
                        std::cout << "[TRTBuilder] SetShape done" << std::endl;
                        params.push_back(tensor);
                    }
                }
            }
        }

        X::KWARGS kwParams;
        std::cout << "[TRTBuilder] Calling forwardFunc.ObjCall..." << std::endl;
        X::Value output = forwardFunc.ObjCall(params, kwParams);
        std::cout << "[TRTBuilder] ObjCall finished." << std::endl;
        
        X::ARGS params_t;
        if (output.IsList()) {
            X::List list(output);
            long long lsize = list->Size();
            params_t.resize(lsize);
            for (long long i = 0; i < lsize; ++i) {
                params_t.push_back(list->Get(i));
            }
        } else if (output.IsDict()) {
            X::Dict dict(output);
            // Too complex, skip for now, dict size not easily accessed via API
            params_t.resize(1);
            params_t.push_back(output);
        } else {
            params_t.resize(1);
            params_t.push_back(output);
        }

        std::cout << "[TRTBuilder] Creating TensorGraph" << std::endl;
        GarnetTensor* pHandler = new GarnetTensor();
        pHandler->m_trtContext = (long long)this;
        X::Value trtHandler(pHandler->APISET().GetProxy(pHandler));

        X::KWARGS kwParams_t;
        auto* pTensorGraph = X::g_pXHost->CreateTensorGraph();
        if (pTensorGraph) {
            std::cout << "[TRTBuilder] TensorGraph created. Calling Create..." << std::endl;
            pTensorGraph->Create(trtHandler.GetObj(), params_t, kwParams_t);
            std::cout << "[TRTBuilder] TensorGraph Create done." << std::endl;

            X::KWARGS kwArgs;
            kwArgs.Add("Func", forwardFunc);
            kwArgs.Add("TRT_Context", X::Value((long long)this)); // Keep this just in case

            std::cout << "[TRTBuilder] Running TensorGraph..." << std::endl;
            Garnet::g_trtContext = this;
            pTensorGraph->Run(params, kwArgs);
            Garnet::g_trtContext = nullptr;
            std::cout << "[TRTBuilder] TensorGraph finished." << std::endl;
        } else {
            std::cout << "[TRTBuilder] pTensorGraph is NULL!" << std::endl;
        }

        // Placeholder cleanup
        // Do not delete TRT objects yet to prevent crash if versions differ
        // delete config;
        // delete network;
        // delete builder;
        std::cout << "[TRTBuilder] BuildEngine completed successfully." << std::endl;

        return X::Value(true);
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
        if (input->GetDataType() != X::TensorDataType::FLOAT32 ||
            gate->GetDataType() != X::TensorDataType::FLOAT32 ||
            up->GetDataType() != X::TensorDataType::FLOAT32 ||
            down->GetDataType() != X::TensorDataType::FLOAT32) {
            std::cout << "[TRTBuilder] RunTextMLPEngine supports float32 only." << std::endl;
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

        if (enginePath == "cuda_text_mlp") {
            std::cout << "[TRTBuilder] Running CUDA TextMLP: tokens=" << tokens
                << ", hidden=" << hidden << ", intermediate=" << intermediate << std::endl;
            size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);
            size_t projBytes = static_cast<size_t>(intermediate) * static_cast<size_t>(hidden) * sizeof(float);
            size_t downBytes = static_cast<size_t>(hidden) * static_cast<size_t>(intermediate) * sizeof(float);
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

            cudaError_t status = runLinearTransposeFP32(
                static_cast<const float*>(dInput),
                static_cast<const float*>(dGateW),
                static_cast<float*>(dGate),
                tokens,
                hidden,
                intermediate,
                stream);
            if (status == cudaSuccess) {
                status = runLinearTransposeFP32(
                    static_cast<const float*>(dInput),
                    static_cast<const float*>(dUpW),
                    static_cast<float*>(dUp),
                    tokens,
                    hidden,
                    intermediate,
                    stream);
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
                status = runLinearTransposeFP32(
                    static_cast<const float*>(dHidden),
                    static_cast<const float*>(dDownW),
                    static_cast<float*>(dOutput),
                    tokens,
                    intermediate,
                    hidden,
                    stream);
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
        if (input->GetDataType() != X::TensorDataType::FLOAT32 || q->GetDataType() != X::TensorDataType::FLOAT32 ||
            k->GetDataType() != X::TensorDataType::FLOAT32 || v->GetDataType() != X::TensorDataType::FLOAT32 ||
            qNorm->GetDataType() != X::TensorDataType::FLOAT32 || kNorm->GetDataType() != X::TensorDataType::FLOAT32) return X::Value();
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

        auto context = GetCachedTRTExecutionContext(enginePath);
        if (!context) return X::Value();

        size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);
        size_t qBytes = static_cast<size_t>(qOut) * static_cast<size_t>(hidden) * sizeof(float);
        size_t kBytes = static_cast<size_t>(kOut) * static_cast<size_t>(hidden) * sizeof(float);
        size_t vBytes = static_cast<size_t>(vOut) * static_cast<size_t>(hidden) * sizeof(float);
        size_t normBytes = static_cast<size_t>(headDim) * sizeof(float);
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
        bool bound = context->setTensorAddress("x", dInput.ptr)
            && context->setTensorAddress("W_q", dQ.ptr)
            && context->setTensorAddress("W_k", dK.ptr)
            && context->setTensorAddress("W_v", dV.ptr)
            && context->setTensorAddress("q_norm", dQNorm.ptr)
            && context->setTensorAddress("k_norm", dKNorm.ptr)
            && context->setTensorAddress("output", dOutput);
        if (!bound || !context->enqueueV3(stream)) {
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
        if (input->GetDataType() != X::TensorDataType::FLOAT32 || weight->GetDataType() != X::TensorDataType::FLOAT32) return X::Value();
        if (input->GetDimCount() != 2 || weight->GetDimCount() != 2) return X::Value();
        int tokens = input->GetDimSize(0);
        int inFeatures = input->GetDimSize(1);
        int outFeatures = weight->GetDimSize(0);
        if (weight->GetDimSize(1) != inFeatures) return X::Value();

        if (enginePath == "cuda_linear_transpose") {
            std::cout << "[TRTBuilder] Running CUDA linear transpose: ["
                << tokens << ", " << inFeatures << "] x [" << outFeatures << ", "
                << inFeatures << "]" << std::endl;
            size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(inFeatures) * sizeof(float);
            size_t weightBytes = static_cast<size_t>(outFeatures) * static_cast<size_t>(inFeatures) * sizeof(float);
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
                std::cout << "[TRTBuilder] CUDA linear transpose allocation/bind failed: "
                    << cudaGetErrorString(cudaGetLastError()) << std::endl;
                FreeOwnedBinding(dInput);
                FreeOwnedBinding(dWeight);
                if (dOutput) cudaFree(dOutput);
                cudaStreamDestroy(stream);
                return X::Value();
            }
            cudaError_t status = runLinearTransposeFP32(
                static_cast<const float*>(dInput.ptr),
                static_cast<const float*>(dWeight.ptr),
                static_cast<float*>(dOutput),
                tokens,
                inFeatures,
                outFeatures,
                stream);
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
        if (input->GetDataType() != X::TensorDataType::FLOAT32 || weight->GetDataType() != X::TensorDataType::FLOAT32 || bias->GetDataType() != X::TensorDataType::FLOAT32) return X::Value();
        if (input->GetDimCount() != 2 || weight->GetDimCount() != 2 || bias->GetDimCount() != 1) return X::Value();
        int tokens = input->GetDimSize(0);
        int inFeatures = input->GetDimSize(1);
        int outFeatures = weight->GetDimSize(0);
        if (weight->GetDimSize(1) != inFeatures || bias->GetDimSize(0) != outFeatures) return X::Value();

        if (enginePath == "cuda_linear_bias_transpose") {
            std::cout << "[TRTBuilder] Running CUDA linear+bias transpose: ["
                << tokens << ", " << inFeatures << "] x [" << outFeatures << ", "
                << inFeatures << "]" << std::endl;
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
                std::cout << "[TRTBuilder] CUDA linear+bias allocation failed." << std::endl;
                return X::Value();
            }
            dInput = inputBinding.ptr;
            dWeight = weightBinding.ptr;
            dBias = biasBinding.ptr;
            cudaError_t launchErr = runLinearBiasTransposeFP32(
                static_cast<const float*>(dInput),
                static_cast<const float*>(dWeight),
                static_cast<const float*>(dBias),
                static_cast<float*>(dOutput),
                tokens,
                inFeatures,
                outFeatures,
                stream);
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

        auto context = GetCachedTRTExecutionContext(enginePath);
        if (!context) return X::Value();

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
        bool bound = context->setTensorAddress("x", dInput.ptr)
            && context->setTensorAddress("weight", dWeight.ptr)
            && context->setTensorAddress("output", dOutput);
        if (!bound || !context->enqueueV3(stream)) {
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

    X::Value TRTBuilder::HandleBinaryOp(const std::string& op_name, X::Value graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input1, X::Value input2) {
        // Here we map "matmul" -> addMatrixMultiply, etc.
        std::cout << "[TRTBuilder] Handling binary op: " << op_name << std::endl;
        return X::Value(); // Return ITensor* wrapped in X::Value eventually
    }

    X::Value TRTBuilder::HandleUnaryOp(const std::string& op_name, X::Value graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input) {
        std::cout << "[TRTBuilder] Handling unary op: " << op_name << std::endl;
        return X::Value(); // Return ITensor* wrapped in X::Value eventually
    }
}
