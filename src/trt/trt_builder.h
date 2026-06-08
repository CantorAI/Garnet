#pragma once

#include "xlang.h"
#include "xpackage.h"
#include "trt_context.h"
#include <string>
#include <unordered_map>
#include <NvInfer.h>

namespace Garnet {
    class TRTEngine {
    public:
        // Holds nvinfer1::ICudaEngine, etc.
        TRTEngine() {}
        ~TRTEngine() {}

        BEGIN_PACKAGE(TRTEngine)
        END_PACKAGE
    };

    class TRTBuilder : public ITRTContext {
    public:
        TRTBuilder();
        ~TRTBuilder();

        X::Value BuildEngine(X::Value forwardFunc, X::Value inputShapes, X::Value weightsDict);

        X::Value HandleBinaryOp(const std::string& op_name, X::Value graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input1, X::Value input2) override;
        X::Value HandleUnaryOp(const std::string& op_name, X::Value graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input) override;

    private:
        nvinfer1::IBuilder* builder = nullptr;
        nvinfer1::INetworkDefinition* network = nullptr;
        nvinfer1::IBuilderConfig* config = nullptr;
        nvinfer1::ICudaEngine* engine = nullptr;

        std::unordered_map<unsigned long long, nvinfer1::ITensor*> tensorMap;

        nvinfer1::ITensor* GetOrCreateTRTTensor(X::Value garnetTensorVal);
    };

    extern thread_local ITRTContext* g_trtContext;

}
