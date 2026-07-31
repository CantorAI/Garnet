#pragma once

#include "xlang.h"
#include "safetensors_index.h"
#include "trt_builder.h"

#include <cstdint>
#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace Garnet
{
    class CompiledModelRuntime
    {
    public:
        struct Diagnostics
        {
            long long rootXExecutions = 0;
            long long hardcodedQwenRunnerCalls = 0;
            long long pythonSubgraphCalls = 0;
            long long directInternalExportCalls = 0;
            long long cpuTensorIntermediates = 0;
            long long graphCacheHits = 0;
            long long graphCacheMisses = 0;
        };

    private:
        mutable std::mutex m_mutex;
        std::string m_rootXModel;
        std::string m_cacheDirectory;
        std::string m_weightsLocation;
        std::string m_entryFunction;
        std::string m_frontend;
        std::string m_backend = "tensorrt";
        std::string m_precision = "bf16";
        std::vector<std::vector<int>> m_inputShapes;
        FusionPartitionOptions m_partitionOptions;
        std::string m_state = "uninitialized";
        std::string m_errorCode;
        std::string m_errorMessage;
        bool m_ready = false;
        X::Value m_module;
        std::vector<X::Value> m_dependencyModules;
        X::Value m_rootFunction;
        X::Value m_graph;
        std::string m_graphSummary;
        std::string m_executionPlanJson;
        std::string m_enginePath;
        std::vector<EnginePartitionSpec> m_enginePartitions;
        SafeTensorsIndex m_weightIndex;
        std::string m_weightIndexError;
        std::unordered_map<std::string, X::Value> m_loadedWeights;
        long long m_loadedWeightBytes = 0;
        double m_enginePreparationMs = 0.0;
        double m_frontendPreparationMs = 0.0;
        bool m_enginesPrepared = false;
        bool m_frontendPrepared = false;
        bool m_cudaGraphEnabled = false;
        Diagnostics m_diagnostics;
        std::shared_ptr<CompiledModelRuntime> m_decodeRuntime;
        void* m_sampleTokenDevice = nullptr;
        void* m_sampleValueDevice = nullptr;
        int m_sampleCapacity = 0;
        X::Value m_reusableExecutionOutput;
        X::Value m_reusablePrefillKeyCache;
        X::Value m_reusablePrefillValueCache;
        std::uint64_t m_openVinoSessionId = 0;
    public:
        ~CompiledModelRuntime();

        bool Initialize(
            const std::string& rootXModel,
            const std::string& cacheDirectory,
            const std::string& weightsLocation,
            const std::string& entryFunction,
            const std::string& frontend,
            const std::vector<std::vector<int>>& inputShapes,
            const std::vector<std::string>& inputDataTypes,
            const FusionPartitionOptions& partitionOptions = {},
            const std::string& backend = "tensorrt",
            const std::string& precision = "");

        X::Value Status() const;
        X::Value Forward(X::Value request);
        X::Value DebugProbe(const std::string& probe, X::Value argument);
        void ReleaseDeviceMemory();
    };
}
