#pragma once

#include "xlang.h"
#include "xpackage.h"
#include "trt_context.h"
#include <string>
#include <vector>
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
        X::Value ExportMatmulEngine(const std::string& enginePath, const std::vector<int>& inputShape, const std::vector<int>& weightShape);
        X::Value RunMatmulEngine(const std::string& enginePath, X::Value inputValue, X::Value weightValue);
        X::Value ExportTextMLPEngine(const std::string& enginePath, const std::vector<int>& inputShape, const std::vector<int>& gateShape, const std::vector<int>& upShape, const std::vector<int>& downShape);
        X::Value RunTextMLPEngine(const std::string& enginePath, X::Value inputValue, X::Value gateWeight, X::Value upWeight, X::Value downWeight);
        X::Value ExportTextQKVEngine(const std::string& enginePath, const std::vector<int>& inputShape, const std::vector<int>& qShape, const std::vector<int>& kShape, const std::vector<int>& vShape);
        X::Value RunTextQKVEngine(const std::string& enginePath, X::Value inputValue, X::Value qWeight, X::Value kWeight, X::Value vWeight);
        X::Value ExportTextQKVHeadNormEngine(const std::string& enginePath, const std::vector<int>& inputShape, const std::vector<int>& qShape, const std::vector<int>& kShape, const std::vector<int>& vShape, const std::vector<int>& qNormShape, const std::vector<int>& kNormShape, float eps);
        X::Value RunTextQKVHeadNormEngine(const std::string& enginePath, X::Value inputValue, X::Value qWeight, X::Value kWeight, X::Value vWeight, X::Value qNormWeight, X::Value kNormWeight);
        X::Value ExportTextRoPEEngine(const std::string& enginePath, const std::vector<int>& qkvShape, const std::vector<int>& cosShape, const std::vector<int>& sinShape, int qHeads, int kvHeads, int headDim);
        X::Value RunTextRoPEEngine(const std::string& enginePath, X::Value qkvValue, X::Value cosValue, X::Value sinValue);
        X::Value ExportTextAttentionEngine(const std::string& enginePath, const std::vector<int>& qkvShape, int qHeads, int kvHeads, int headDim);
        X::Value RunTextAttentionEngine(const std::string& enginePath, X::Value qkvValue);
        X::Value ExportVisionAttentionEngine(const std::string& enginePath, const std::vector<int>& qkvShape, int heads, int headDim);
        X::Value RunVisionAttentionEngine(const std::string& enginePath, X::Value qkvValue);
        X::Value ExportLinearTransposeEngine(const std::string& enginePath, const std::vector<int>& inputShape, const std::vector<int>& weightShape);
        X::Value RunLinearTransposeEngine(const std::string& enginePath, X::Value inputValue, X::Value weightValue);
        X::Value ExportLinearBiasTransposeEngine(const std::string& enginePath, const std::vector<int>& inputShape, const std::vector<int>& weightShape, const std::vector<int>& biasShape);
        X::Value RunLinearBiasTransposeEngine(const std::string& enginePath, X::Value inputValue, X::Value weightValue, X::Value biasValue);
        X::Value ExportVisionMLPEngine(const std::string& enginePath, const std::vector<int>& inputShape, const std::vector<int>& fc1Shape, const std::vector<int>& fc2Shape);
        X::Value RunVisionMLPEngine(const std::string& enginePath, X::Value inputValue, X::Value fc1Weight, X::Value fc1Bias, X::Value fc2Weight, X::Value fc2Bias);
        X::Value ExportRMSNormEngine(const std::string& enginePath, const std::vector<int>& inputShape, const std::vector<int>& weightShape, float eps);
        X::Value RunRMSNormEngine(const std::string& enginePath, X::Value inputValue, X::Value weightValue);
        X::Value ExportLayerNormEngine(const std::string& enginePath, const std::vector<int>& inputShape, const std::vector<int>& weightShape, float eps);
        X::Value RunLayerNormEngine(const std::string& enginePath, X::Value inputValue, X::Value weightValue, X::Value biasValue);

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
