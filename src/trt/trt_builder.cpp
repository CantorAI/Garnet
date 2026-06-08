#include "trt_builder.h"
#include "garnet_tensor.h"
#include "garnet_tensor.h"
#include <iostream>

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
