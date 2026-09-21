// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "xlang3/xlang3.h"
#include "lowering_context.h"
#include "safetensors_index.h"
#include "weight_quantization.h"
#include "compiled_graph_capture.h"
#include <string>
#include <deque>
#include <memory>
#include <vector>
#include <unordered_map>
#include <NvInfer.h>

namespace Garnet {
    struct EnginePartitionBinding {
        std::string name;
        unsigned long long tensorId = 0;
        int requestInputIndex = -1;
        bool terminalOutput = false;
    };

    struct EnginePartitionSpec {
        int id = 0;
        std::string enginePath;
        std::vector<EnginePartitionBinding> inputs;
        std::vector<EnginePartitionBinding> outputs;
    };

    class TRTEngine {
    public:
        // Holds nvinfer1::ICudaEngine, etc.
        TRTEngine() {}
        ~TRTEngine() {}

        BEGIN_PACKAGE(TRTEngine)
        END_PACKAGE
    };

    class TRTBuilder : public ILoweringContext {
        X3PackageHost* host_;
        std::unordered_map<uint64_t, std::string> capturedTensorNames;
        std::vector<X::Value> constantTensors;
    public:
        explicit TRTBuilder(X3PackageHost* host);
        ~TRTBuilder();

        void SetCapturedWorkspaceBytes(unsigned long long bytes) {
            capturedWorkspaceBytes = bytes;
        }
        void SetCapturedOptimizationLevel(int level) {
            capturedOptimizationLevel = level;
        }
        void SetCapturedWeightProfile(const std::string& profile) {
            capturedWeightProfile = profile;
        }

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

        bool BuildCapturedGraph(
            X::Value graph,
            X::Value forwardFunction,
            X::ARGS& graphArguments,
            X::ARGS& symbolicInputs,
            const SafeTensorsIndex* weightIndex,
            const std::string& enginePath,
            std::string& errorMessage);
        bool AnalyzeCapturedGraph(
            X::Value graph,
            X::Value forwardFunction,
            X::ARGS& graphArguments,
            std::vector<CapturedTensorOperation>& operations,
            std::string& errorMessage);
        bool BuildCapturedPartitions(
            X::Value graph,
            X::Value forwardFunction,
            X::ARGS& graphArguments,
            X::ARGS& symbolicInputs,
            const SafeTensorsIndex* weightIndex,
            const std::string& baseEnginePath,
            const std::vector<CapturedTensorOperation>& operations,
            std::vector<EnginePartitionSpec>& partitions,
            std::string& errorMessage);
        bool PrepareCapturedEngine(
            const std::string& enginePath,
            const SafeTensorsIndex* weightIndex,
            std::string& errorMessage);
        bool PrepareCapturedPartitions(
            const std::vector<EnginePartitionSpec>& partitions,
            const SafeTensorsIndex* weightIndex,
            std::string& errorMessage);
        static void ReleaseCachedExecutions(const std::string& cacheRoot);
        static std::shared_ptr<void> RetainCachedExecution(
            const std::string& enginePath,
            const SafeTensorsIndex* weightIndex,
            std::string& errorMessage);
        X::Value RunCapturedEngine(
            const std::string& enginePath,
            X::Value inputs,
            const SafeTensorsIndex* weightIndex,
            X::Value reusableOutput,
            bool enableCudaGraph,
            std::string& errorMessage);
        X::Value RunCapturedPartitions(
            const std::vector<EnginePartitionSpec>& partitions,
            X::Value inputs,
            const SafeTensorsIndex* weightIndex,
            X::Value reusableOutput,
            std::string& errorMessage);

        X::Value HandleBinaryOp(const std::string& op_name, X::Value graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input1, X::Value input2, X::Value output) override;
        X::Value HandleUnaryOp(const std::string& op_name, X::Value graph, X::ARGS& params, X::KWARGS& kwParams, X::Value input, X::Value output) override;
        X::Value HandleBranchBegin(const std::string& condition, int branchType, unsigned long long flowId, int branchId) override;
        X::Value HandleBranchEnd() override;

    private:
        nvinfer1::IBuilder* builder = nullptr;
        nvinfer1::INetworkDefinition* network = nullptr;
        nvinfer1::IBuilderConfig* config = nullptr;
        nvinfer1::ICudaEngine* engine = nullptr;

        std::unordered_map<unsigned long long, nvinfer1::ITensor*> tensorMap;
        unsigned long long capturedWorkspaceBytes = 64ULL << 20;
        int capturedOptimizationLevel = 3;
        std::unordered_map<std::string, nvinfer1::ITensor*> weightTensorMap;
        std::deque<float> scalarWeights;
        std::deque<unsigned short> bfloat16ScalarWeights;
        std::deque<long long> integerWeights;
        std::deque<std::vector<float>> vectorWeights;
        std::deque<std::vector<long long>> integer64VectorWeights;
        std::deque<std::vector<int>> integerVectorWeights;
        std::deque<std::vector<unsigned char>> booleanVectorWeights;
        const SafeTensorsIndex* capturedWeightIndex = nullptr;
        std::unordered_map<
            std::string,
            std::unique_ptr<SafeTensorsMappedFile>> capturedWeightFiles;
        std::unordered_map<
            std::string,
            std::shared_ptr<QuantizedWeight>> capturedQuantizedWeights;
        std::string capturedWeightProfile = "bf16";
        nvinfer1::ITensor* lastOutput = nullptr;
        nvinfer1::ITensor* pendingKVKeyPages = nullptr;
        nvinfer1::ITensor* pendingKVValuePages = nullptr;
        nvinfer1::ITensor* pendingKVPageTable = nullptr;
        nvinfer1::ITensor* pendingKVContextLength = nullptr;
        nvinfer1::ITensor* pendingKVSlotPosition = nullptr;
        nvinfer1::ITensor* pendingKVActiveMask = nullptr;
        int pendingKVLayerIndex = -1;
        std::vector<nvinfer1::IPluginV2*> ownedPlugins;
        std::string loweringError;
        bool loweringActive = true;
        bool analysisActive = false;
        std::vector<CapturedTensorOperation> analyzedOperations;
        bool partitionBuildActive = false;
        int activePartition = 0;
        size_t replayOperationIndex = 0;
        std::unordered_map<unsigned long long, std::string> partitionInputNames;
        std::unordered_map<unsigned long long, std::string> partitionOutputNames;
        std::unordered_map<
            unsigned long long,
            std::pair<nvinfer1::DataType, nvinfer1::Dims>> partitionBoundaryMetadata;
        std::vector<bool> branchParentActivity;
        std::unordered_map<unsigned long long, bool> flowBranchTaken;

        nvinfer1::ITensor* GetOrCreateTRTTensor(X::Value garnetTensorVal);
        nvinfer1::ITensor* GetOrCreateTRTWeight(const std::string& weightName);
        nvinfer1::ITensor* GetOrCreateTRTWeightFP32(const std::string& weightName);
        nvinfer1::ITensor* BroadcastLastDimension(
            nvinfer1::ITensor* tensor,
            int targetRank,
            const std::string& layerName);
        nvinfer1::ITensor* BroadcastMatrixWeight(
            nvinfer1::ITensor* tensor,
            int targetRank);
        nvinfer1::ITensor* LowerVisionRope(
            nvinfer1::ITensor* qkv,
            nvinfer1::ITensor* positionIds,
            X::KWARGS& options);
        nvinfer1::ITensor* LowerVisionAttention(
            nvinfer1::ITensor* qkv,
            nvinfer1::ITensor* cuSeqlens,
            X::KWARGS& options);
        nvinfer1::ITensor* LowerTextRope(
            nvinfer1::ITensor* qkv,
            nvinfer1::ITensor* positionIds,
            X::KWARGS& options,
            bool multimodal);
        nvinfer1::ITensor* LowerTextAttention(
            nvinfer1::ITensor* qkv,
            nvinfer1::ITensor* attentionMask,
            X::KWARGS& options);
        nvinfer1::ITensor* LowerFusedTextAttention(
            nvinfer1::ITensor* qkv,
            nvinfer1::ITensor* attentionMask,
            X::KWARGS& options);
    };

}
