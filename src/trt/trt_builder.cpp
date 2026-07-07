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

        X::Tensor output;
        output->SetDataType(X::TensorDataType::FLOAT32);
        X::Port::vector<int> outputShape(2);
        outputShape.push_back(m);
        outputShape.push_back(n);
        output->SetShape(outputShape);

        X::Value initData;
        if (!output->Create(initData) || output->GetData() == nullptr) {
            std::cout << "[TRTBuilder] Failed to create output tensor storage." << std::endl;
            cudaFree(dInput);
            cudaFree(dWeight);
            cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }
        memcpy(output->GetData(), hostOutput.data(), outputBytes);
        std::cout << "[TRTBuilder] Output tensor created." << std::endl;

        cudaFree(dInput);
        cudaFree(dWeight);
        cudaFree(dOutput);
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

        std::ifstream in(enginePath, std::ios::binary | std::ios::ate);
        if (!in.is_open()) {
            std::cout << "[TRTBuilder] Failed to open TextMLP engine for read: " << enginePath << std::endl;
            return X::Value();
        }
        std::streamsize size = in.tellg();
        in.seekg(0, std::ios::beg);
        std::vector<char> engineBytes(static_cast<size_t>(size));
        if (!in.read(engineBytes.data(), size)) {
            std::cout << "[TRTBuilder] Failed to read TextMLP engine bytes." << std::endl;
            return X::Value();
        }

        auto runtime = createInferRuntime(gLogger);
        if (!runtime) return X::Value();
        auto engine = runtime->deserializeCudaEngine(engineBytes.data(), engineBytes.size());
        if (!engine) return X::Value();
        auto context = engine->createExecutionContext();
        if (!context) return X::Value();

        size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);
        size_t projBytes = static_cast<size_t>(intermediate) * static_cast<size_t>(hidden) * sizeof(float);
        size_t downBytes = static_cast<size_t>(hidden) * static_cast<size_t>(intermediate) * sizeof(float);
        size_t outputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);

        void* dInput = nullptr;
        void* dGate = nullptr;
        void* dUp = nullptr;
        void* dDown = nullptr;
        void* dOutput = nullptr;
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess ||
            cudaMalloc(&dInput, inputBytes) != cudaSuccess ||
            cudaMalloc(&dGate, projBytes) != cudaSuccess ||
            cudaMalloc(&dUp, projBytes) != cudaSuccess ||
            cudaMalloc(&dDown, downBytes) != cudaSuccess ||
            cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
            std::cout << "[TRTBuilder] TextMLP CUDA allocation failed." << std::endl;
            if (dInput) cudaFree(dInput);
            if (dGate) cudaFree(dGate);
            if (dUp) cudaFree(dUp);
            if (dDown) cudaFree(dDown);
            if (dOutput) cudaFree(dOutput);
            if (stream) cudaStreamDestroy(stream);
            return X::Value();
        }

        cudaMemcpyAsync(dInput, input->GetData(), inputBytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dGate, gate->GetData(), projBytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dUp, up->GetData(), projBytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dDown, down->GetData(), downBytes, cudaMemcpyHostToDevice, stream);

        bool bound = context->setTensorAddress("x", dInput)
            && context->setTensorAddress("W_gate", dGate)
            && context->setTensorAddress("W_up", dUp)
            && context->setTensorAddress("W_down", dDown)
            && context->setTensorAddress("output", dOutput);
        if (!bound || !context->enqueueV3(stream)) {
            std::cout << "[TRTBuilder] TextMLP enqueue failed." << std::endl;
            cudaFree(dInput);
            cudaFree(dGate);
            cudaFree(dUp);
            cudaFree(dDown);
            cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }

        std::vector<float> hostOutput(static_cast<size_t>(tokens) * static_cast<size_t>(hidden));
        cudaMemcpyAsync(hostOutput.data(), dOutput, outputBytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        X::Tensor output;
        output->SetDataType(X::TensorDataType::FLOAT32);
        X::Port::vector<int> outputShape(2);
        outputShape.push_back(tokens);
        outputShape.push_back(hidden);
        output->SetShape(outputShape);
        X::Value initData;
        if (!output->Create(initData) || output->GetData() == nullptr) {
            std::cout << "[TRTBuilder] Failed to create TextMLP output tensor." << std::endl;
            cudaFree(dInput);
            cudaFree(dGate);
            cudaFree(dUp);
            cudaFree(dDown);
            cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return X::Value();
        }
        memcpy(output->GetData(), hostOutput.data(), outputBytes);

        cudaFree(dInput);
        cudaFree(dGate);
        cudaFree(dUp);
        cudaFree(dDown);
        cudaFree(dOutput);
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

        std::ifstream in(enginePath, std::ios::binary | std::ios::ate);
        if (!in.is_open()) return X::Value();
        std::streamsize size = in.tellg();
        in.seekg(0, std::ios::beg);
        std::vector<char> engineBytes(static_cast<size_t>(size));
        if (!in.read(engineBytes.data(), size)) return X::Value();
        auto runtime = createInferRuntime(gLogger);
        if (!runtime) return X::Value();
        auto engine = runtime->deserializeCudaEngine(engineBytes.data(), engineBytes.size());
        if (!engine) return X::Value();
        auto context = engine->createExecutionContext();
        if (!context) return X::Value();

        size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);
        size_t qBytes = static_cast<size_t>(qOut) * static_cast<size_t>(hidden) * sizeof(float);
        size_t kBytes = static_cast<size_t>(kOut) * static_cast<size_t>(hidden) * sizeof(float);
        size_t vBytes = static_cast<size_t>(vOut) * static_cast<size_t>(hidden) * sizeof(float);
        size_t outputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(qOut + kOut + vOut) * sizeof(float);
        void* dInput = nullptr;
        void* dQ = nullptr;
        void* dK = nullptr;
        void* dV = nullptr;
        void* dOutput = nullptr;
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess ||
            cudaMalloc(&dInput, inputBytes) != cudaSuccess ||
            cudaMalloc(&dQ, qBytes) != cudaSuccess ||
            cudaMalloc(&dK, kBytes) != cudaSuccess ||
            cudaMalloc(&dV, vBytes) != cudaSuccess ||
            cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
            if (dInput) cudaFree(dInput);
            if (dQ) cudaFree(dQ);
            if (dK) cudaFree(dK);
            if (dV) cudaFree(dV);
            if (dOutput) cudaFree(dOutput);
            if (stream) cudaStreamDestroy(stream);
            return X::Value();
        }
        cudaMemcpyAsync(dInput, input->GetData(), inputBytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dQ, q->GetData(), qBytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dK, k->GetData(), kBytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dV, v->GetData(), vBytes, cudaMemcpyHostToDevice, stream);
        bool bound = context->setTensorAddress("x", dInput)
            && context->setTensorAddress("W_q", dQ)
            && context->setTensorAddress("W_k", dK)
            && context->setTensorAddress("W_v", dV)
            && context->setTensorAddress("output", dOutput);
        if (!bound || !context->enqueueV3(stream)) {
            cudaFree(dInput); cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        std::vector<float> hostOutput(static_cast<size_t>(tokens) * static_cast<size_t>(qOut + kOut + vOut));
        cudaMemcpyAsync(hostOutput.data(), dOutput, outputBytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        X::Tensor output;
        output->SetDataType(X::TensorDataType::FLOAT32);
        X::Port::vector<int> outputShape(2);
        outputShape.push_back(tokens);
        outputShape.push_back(qOut + kOut + vOut);
        output->SetShape(outputShape);
        X::Value initData;
        if (!output->Create(initData) || output->GetData() == nullptr) {
            cudaFree(dInput); cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        memcpy(output->GetData(), hostOutput.data(), outputBytes);
        cudaFree(dInput); cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dOutput); cudaStreamDestroy(stream);
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

        std::ifstream in(enginePath, std::ios::binary | std::ios::ate);
        if (!in.is_open()) return X::Value();
        std::streamsize size = in.tellg();
        in.seekg(0, std::ios::beg);
        std::vector<char> engineBytes(static_cast<size_t>(size));
        if (!in.read(engineBytes.data(), size)) return X::Value();
        auto runtime = createInferRuntime(gLogger);
        if (!runtime) return X::Value();
        auto engine = runtime->deserializeCudaEngine(engineBytes.data(), engineBytes.size());
        if (!engine) return X::Value();
        auto context = engine->createExecutionContext();
        if (!context) return X::Value();

        size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);
        size_t qBytes = static_cast<size_t>(qOut) * static_cast<size_t>(hidden) * sizeof(float);
        size_t kBytes = static_cast<size_t>(kOut) * static_cast<size_t>(hidden) * sizeof(float);
        size_t vBytes = static_cast<size_t>(vOut) * static_cast<size_t>(hidden) * sizeof(float);
        size_t normBytes = static_cast<size_t>(headDim) * sizeof(float);
        size_t outputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(qOut + kOut + vOut) * sizeof(float);
        void* dInput = nullptr;
        void* dQ = nullptr;
        void* dK = nullptr;
        void* dV = nullptr;
        void* dQNorm = nullptr;
        void* dKNorm = nullptr;
        void* dOutput = nullptr;
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess ||
            cudaMalloc(&dInput, inputBytes) != cudaSuccess ||
            cudaMalloc(&dQ, qBytes) != cudaSuccess ||
            cudaMalloc(&dK, kBytes) != cudaSuccess ||
            cudaMalloc(&dV, vBytes) != cudaSuccess ||
            cudaMalloc(&dQNorm, normBytes) != cudaSuccess ||
            cudaMalloc(&dKNorm, normBytes) != cudaSuccess ||
            cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
            if (dInput) cudaFree(dInput);
            if (dQ) cudaFree(dQ);
            if (dK) cudaFree(dK);
            if (dV) cudaFree(dV);
            if (dQNorm) cudaFree(dQNorm);
            if (dKNorm) cudaFree(dKNorm);
            if (dOutput) cudaFree(dOutput);
            if (stream) cudaStreamDestroy(stream);
            return X::Value();
        }
        cudaMemcpyAsync(dInput, input->GetData(), inputBytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dQ, q->GetData(), qBytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dK, k->GetData(), kBytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dV, v->GetData(), vBytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dQNorm, qNorm->GetData(), normBytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dKNorm, kNorm->GetData(), normBytes, cudaMemcpyHostToDevice, stream);
        bool bound = context->setTensorAddress("x", dInput)
            && context->setTensorAddress("W_q", dQ)
            && context->setTensorAddress("W_k", dK)
            && context->setTensorAddress("W_v", dV)
            && context->setTensorAddress("q_norm", dQNorm)
            && context->setTensorAddress("k_norm", dKNorm)
            && context->setTensorAddress("output", dOutput);
        if (!bound || !context->enqueueV3(stream)) {
            cudaFree(dInput); cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dQNorm); cudaFree(dKNorm); cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        std::vector<float> hostOutput(static_cast<size_t>(tokens) * static_cast<size_t>(qOut + kOut + vOut));
        cudaMemcpyAsync(hostOutput.data(), dOutput, outputBytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        X::Tensor output;
        output->SetDataType(X::TensorDataType::FLOAT32);
        X::Port::vector<int> outputShape(2);
        outputShape.push_back(tokens);
        outputShape.push_back(qOut + kOut + vOut);
        output->SetShape(outputShape);
        X::Value initData;
        if (!output->Create(initData) || output->GetData() == nullptr) {
            cudaFree(dInput); cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dQNorm); cudaFree(dKNorm); cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        memcpy(output->GetData(), hostOutput.data(), outputBytes);
        cudaFree(dInput); cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dQNorm); cudaFree(dKNorm); cudaFree(dOutput); cudaStreamDestroy(stream);
        std::cout << "[TRTBuilder] RunTextQKVHeadNormEngine completed: [" << tokens << ", " << (qOut + kOut + vOut) << "]" << std::endl;
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

        std::ifstream in(enginePath, std::ios::binary | std::ios::ate);
        if (!in.is_open()) return X::Value();
        std::streamsize size = in.tellg();
        in.seekg(0, std::ios::beg);
        std::vector<char> engineBytes(static_cast<size_t>(size));
        if (!in.read(engineBytes.data(), size)) return X::Value();
        auto runtime = createInferRuntime(gLogger);
        if (!runtime) return X::Value();
        auto engine = runtime->deserializeCudaEngine(engineBytes.data(), engineBytes.size());
        if (!engine) return X::Value();
        auto context = engine->createExecutionContext();
        if (!context) return X::Value();

        size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(inFeatures) * sizeof(float);
        size_t weightBytes = static_cast<size_t>(outFeatures) * static_cast<size_t>(inFeatures) * sizeof(float);
        size_t outputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(outFeatures) * sizeof(float);
        void* dInput = nullptr;
        void* dWeight = nullptr;
        void* dOutput = nullptr;
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess ||
            cudaMalloc(&dInput, inputBytes) != cudaSuccess ||
            cudaMalloc(&dWeight, weightBytes) != cudaSuccess ||
            cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
            if (dInput) cudaFree(dInput);
            if (dWeight) cudaFree(dWeight);
            if (dOutput) cudaFree(dOutput);
            if (stream) cudaStreamDestroy(stream);
            return X::Value();
        }
        cudaMemcpyAsync(dInput, input->GetData(), inputBytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dWeight, weight->GetData(), weightBytes, cudaMemcpyHostToDevice, stream);
        bool bound = context->setTensorAddress("x", dInput)
            && context->setTensorAddress("W", dWeight)
            && context->setTensorAddress("output", dOutput);
        if (!bound || !context->enqueueV3(stream)) {
            cudaFree(dInput); cudaFree(dWeight); cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        std::vector<float> hostOutput(static_cast<size_t>(tokens) * static_cast<size_t>(outFeatures));
        cudaMemcpyAsync(hostOutput.data(), dOutput, outputBytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        X::Tensor output;
        output->SetDataType(X::TensorDataType::FLOAT32);
        X::Port::vector<int> outputShape(2);
        outputShape.push_back(tokens);
        outputShape.push_back(outFeatures);
        output->SetShape(outputShape);
        X::Value initData;
        if (!output->Create(initData) || output->GetData() == nullptr) {
            cudaFree(dInput); cudaFree(dWeight); cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        memcpy(output->GetData(), hostOutput.data(), outputBytes);
        cudaFree(dInput); cudaFree(dWeight); cudaFree(dOutput); cudaStreamDestroy(stream);
        std::cout << "[TRTBuilder] RunLinearTransposeEngine completed: [" << tokens << ", " << outFeatures << "]" << std::endl;
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

        std::ifstream in(enginePath, std::ios::binary | std::ios::ate);
        if (!in.is_open()) return X::Value();
        std::streamsize size = in.tellg();
        in.seekg(0, std::ios::beg);
        std::vector<char> engineBytes(static_cast<size_t>(size));
        if (!in.read(engineBytes.data(), size)) return X::Value();
        auto runtime = createInferRuntime(gLogger);
        if (!runtime) return X::Value();
        auto engine = runtime->deserializeCudaEngine(engineBytes.data(), engineBytes.size());
        if (!engine) return X::Value();
        auto context = engine->createExecutionContext();
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
        if (cudaStreamCreate(&stream) != cudaSuccess ||
            cudaMalloc(&dInput, inputBytes) != cudaSuccess ||
            cudaMalloc(&dW1, fc1Bytes) != cudaSuccess ||
            cudaMalloc(&dB1, fc1BiasBytes) != cudaSuccess ||
            cudaMalloc(&dW2, fc2Bytes) != cudaSuccess ||
            cudaMalloc(&dB2, fc2BiasBytes) != cudaSuccess ||
            cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
            if (dInput) cudaFree(dInput);
            if (dW1) cudaFree(dW1);
            if (dB1) cudaFree(dB1);
            if (dW2) cudaFree(dW2);
            if (dB2) cudaFree(dB2);
            if (dOutput) cudaFree(dOutput);
            if (stream) cudaStreamDestroy(stream);
            return X::Value();
        }

        cudaMemcpyAsync(dInput, input->GetData(), inputBytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dW1, w1->GetData(), fc1Bytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dB1, b1->GetData(), fc1BiasBytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dW2, w2->GetData(), fc2Bytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dB2, b2->GetData(), fc2BiasBytes, cudaMemcpyHostToDevice, stream);

        bool bound = context->setTensorAddress("x", dInput)
            && context->setTensorAddress("W_fc1", dW1)
            && context->setTensorAddress("b_fc1", dB1)
            && context->setTensorAddress("W_fc2", dW2)
            && context->setTensorAddress("b_fc2", dB2)
            && context->setTensorAddress("output", dOutput);
        if (!bound || !context->enqueueV3(stream)) {
            cudaFree(dInput); cudaFree(dW1); cudaFree(dB1); cudaFree(dW2); cudaFree(dB2); cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }

        std::vector<float> hostOutput(static_cast<size_t>(tokens) * static_cast<size_t>(hidden));
        cudaMemcpyAsync(hostOutput.data(), dOutput, outputBytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        X::Tensor output;
        output->SetDataType(X::TensorDataType::FLOAT32);
        X::Port::vector<int> outputShape(2);
        outputShape.push_back(tokens);
        outputShape.push_back(hidden);
        output->SetShape(outputShape);
        X::Value initData;
        if (!output->Create(initData) || output->GetData() == nullptr) {
            cudaFree(dInput); cudaFree(dW1); cudaFree(dB1); cudaFree(dW2); cudaFree(dB2); cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        memcpy(output->GetData(), hostOutput.data(), outputBytes);

        cudaFree(dInput);
        cudaFree(dW1);
        cudaFree(dB1);
        cudaFree(dW2);
        cudaFree(dB2);
        cudaFree(dOutput);
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

        std::ifstream in(enginePath, std::ios::binary | std::ios::ate);
        if (!in.is_open()) return X::Value();
        std::streamsize size = in.tellg();
        in.seekg(0, std::ios::beg);
        std::vector<char> engineBytes(static_cast<size_t>(size));
        if (!in.read(engineBytes.data(), size)) return X::Value();
        auto runtime = createInferRuntime(gLogger);
        if (!runtime) return X::Value();
        auto engine = runtime->deserializeCudaEngine(engineBytes.data(), engineBytes.size());
        if (!engine) return X::Value();
        auto context = engine->createExecutionContext();
        if (!context) return X::Value();

        size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);
        size_t weightBytes = static_cast<size_t>(hidden) * sizeof(float);
        void* dInput = nullptr;
        void* dWeight = nullptr;
        void* dOutput = nullptr;
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess ||
            cudaMalloc(&dInput, inputBytes) != cudaSuccess ||
            cudaMalloc(&dWeight, weightBytes) != cudaSuccess ||
            cudaMalloc(&dOutput, inputBytes) != cudaSuccess) {
            if (dInput) cudaFree(dInput);
            if (dWeight) cudaFree(dWeight);
            if (dOutput) cudaFree(dOutput);
            if (stream) cudaStreamDestroy(stream);
            return X::Value();
        }
        cudaMemcpyAsync(dInput, input->GetData(), inputBytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dWeight, weight->GetData(), weightBytes, cudaMemcpyHostToDevice, stream);
        bool bound = context->setTensorAddress("x", dInput)
            && context->setTensorAddress("weight", dWeight)
            && context->setTensorAddress("output", dOutput);
        if (!bound || !context->enqueueV3(stream)) {
            cudaFree(dInput); cudaFree(dWeight); cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        std::vector<float> hostOutput(static_cast<size_t>(tokens) * static_cast<size_t>(hidden));
        cudaMemcpyAsync(hostOutput.data(), dOutput, inputBytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        X::Tensor output;
        output->SetDataType(X::TensorDataType::FLOAT32);
        X::Port::vector<int> outputShape(2);
        outputShape.push_back(tokens);
        outputShape.push_back(hidden);
        output->SetShape(outputShape);
        X::Value initData;
        if (!output->Create(initData) || output->GetData() == nullptr) {
            cudaFree(dInput); cudaFree(dWeight); cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        memcpy(output->GetData(), hostOutput.data(), inputBytes);
        cudaFree(dInput); cudaFree(dWeight); cudaFree(dOutput); cudaStreamDestroy(stream);
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

        std::ifstream in(enginePath, std::ios::binary | std::ios::ate);
        if (!in.is_open()) return X::Value();
        std::streamsize size = in.tellg();
        in.seekg(0, std::ios::beg);
        std::vector<char> engineBytes(static_cast<size_t>(size));
        if (!in.read(engineBytes.data(), size)) return X::Value();
        auto runtime = createInferRuntime(gLogger);
        if (!runtime) return X::Value();
        auto engine = runtime->deserializeCudaEngine(engineBytes.data(), engineBytes.size());
        if (!engine) return X::Value();
        auto context = engine->createExecutionContext();
        if (!context) return X::Value();

        size_t inputBytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);
        size_t affineBytes = static_cast<size_t>(hidden) * sizeof(float);
        void* dInput = nullptr;
        void* dWeight = nullptr;
        void* dBias = nullptr;
        void* dOutput = nullptr;
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess ||
            cudaMalloc(&dInput, inputBytes) != cudaSuccess ||
            cudaMalloc(&dWeight, affineBytes) != cudaSuccess ||
            cudaMalloc(&dBias, affineBytes) != cudaSuccess ||
            cudaMalloc(&dOutput, inputBytes) != cudaSuccess) {
            if (dInput) cudaFree(dInput);
            if (dWeight) cudaFree(dWeight);
            if (dBias) cudaFree(dBias);
            if (dOutput) cudaFree(dOutput);
            if (stream) cudaStreamDestroy(stream);
            return X::Value();
        }
        cudaMemcpyAsync(dInput, input->GetData(), inputBytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dWeight, weight->GetData(), affineBytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dBias, bias->GetData(), affineBytes, cudaMemcpyHostToDevice, stream);
        bool bound = context->setTensorAddress("x", dInput)
            && context->setTensorAddress("weight", dWeight)
            && context->setTensorAddress("bias", dBias)
            && context->setTensorAddress("output", dOutput);
        if (!bound || !context->enqueueV3(stream)) {
            cudaFree(dInput); cudaFree(dWeight); cudaFree(dBias); cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        std::vector<float> hostOutput(static_cast<size_t>(tokens) * static_cast<size_t>(hidden));
        cudaMemcpyAsync(hostOutput.data(), dOutput, inputBytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        X::Tensor output;
        output->SetDataType(X::TensorDataType::FLOAT32);
        X::Port::vector<int> outputShape(2);
        outputShape.push_back(tokens);
        outputShape.push_back(hidden);
        output->SetShape(outputShape);
        X::Value initData;
        if (!output->Create(initData) || output->GetData() == nullptr) {
            cudaFree(dInput); cudaFree(dWeight); cudaFree(dBias); cudaFree(dOutput); cudaStreamDestroy(stream);
            return X::Value();
        }
        memcpy(output->GetData(), hostOutput.data(), inputBytes);
        cudaFree(dInput); cudaFree(dWeight); cudaFree(dBias); cudaFree(dOutput); cudaStreamDestroy(stream);
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
