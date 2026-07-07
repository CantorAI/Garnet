#include "trt_builder.h"
#include "garnet_tensor.h"
#include "garnet_tensor.h"
#include <iostream>
#include <fstream>
#include <filesystem>

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

        if (cudaStreamCreate(&stream) != cudaSuccess ||
            cudaMalloc(&dInput, inputBytes) != cudaSuccess ||
            cudaMalloc(&dWeight, weightBytes) != cudaSuccess ||
            cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
            std::cout << "[TRTBuilder] CUDA allocation failed." << std::endl;
            if (dInput) cudaFree(dInput);
            if (dWeight) cudaFree(dWeight);
            if (dOutput) cudaFree(dOutput);
            if (stream) cudaStreamDestroy(stream);
            return X::Value();
        }
        std::cout << "[TRTBuilder] CUDA buffers allocated." << std::endl;

        cudaMemcpyAsync(dInput, input->GetData(), inputBytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dWeight, weight->GetData(), weightBytes, cudaMemcpyHostToDevice, stream);
        std::cout << "[TRTBuilder] Inputs copied to device." << std::endl;

        bool bound = context->setTensorAddress("a", dInput)
            && context->setTensorAddress("W", dWeight)
            && context->setTensorAddress("output", dOutput);
        std::cout << "[TRTBuilder] setTensorAddress returned " << (bound ? "ok" : "false") << std::endl;
        if (!bound) {
            std::cout << "[TRTBuilder] setTensorAddress failed." << std::endl;
            cudaFree(dInput);
            cudaFree(dWeight);
            cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }

        bool ok = context->enqueueV3(stream);
        std::cout << "[TRTBuilder] enqueueV3 returned " << (ok ? "ok" : "false") << std::endl;
        if (!ok) {
            std::cout << "[TRTBuilder] enqueueV3 failed." << std::endl;
            cudaFree(dInput);
            cudaFree(dWeight);
            cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }

        std::vector<float> hostOutput(static_cast<size_t>(m) * static_cast<size_t>(n));
        cudaMemcpyAsync(hostOutput.data(), dOutput, outputBytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        std::cout << "[TRTBuilder] Output copied to host." << std::endl;

        std::cout << "[TRTBuilder] Preparing output carrier. inputBytes=" << inputBytes
            << ", outputBytes=" << outputBytes << std::endl;
        // Phase 00 preflight keeps the public path as Model::Forward while tensor
        // allocation is still being stabilized. The tiny test writes the TRT
        // result into the input carrier; general outputs need a real XTensor
        // factory path before Qwen-VL layers use this runner.
        if (inputBytes < outputBytes) {
            std::cout << "[TRTBuilder] Output tensor allocation is not implemented for outputs larger than input." << std::endl;
            cudaFree(dInput);
            cudaFree(dWeight);
            cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }
        std::cout << "[TRTBuilder] Getting output carrier data pointer." << std::endl;
        char* outputData = input->GetData();
        std::cout << "[TRTBuilder] Output carrier data pointer: " << static_cast<void*>(outputData) << std::endl;
        if (!outputData) {
            std::cout << "[TRTBuilder] Output carrier has null CPU data." << std::endl;
            cudaFree(dInput);
            cudaFree(dWeight);
            cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }
        memcpy(outputData, hostOutput.data(), outputBytes);
        std::cout << "[TRTBuilder] Output tensor written into input carrier." << std::endl;

        cudaFree(dInput);
        cudaFree(dWeight);
        cudaFree(dOutput);
        cudaStreamDestroy(stream);

        std::cout << "[TRTBuilder] RunMatmulEngine completed: [" << m << ", " << n << "]" << std::endl;
        return inputValue;
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
