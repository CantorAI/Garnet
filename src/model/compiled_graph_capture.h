#pragma once

#include <string>
#include <vector>



namespace Garnet
{
    struct FusionAnnotation
    {
        std::string name;
        std::string functionName;
        std::string role = "generic";
        std::string boundary = "none";
        bool atomic = false;
        bool cudaGraph = false;
        std::string validationError;
    };

    struct CapturedFusionRegion
    {
        int id = -1;
        int parentId = -1;
        int depth = 0;
        int invocation = 0;
        int operationCount = 0;
        int inclusiveOperationCount = 0;
        std::vector<unsigned long long> inputTensorIds;
        std::vector<unsigned long long> outputTensorIds;
        FusionAnnotation annotation;
    };

    struct CapturedTensorOperation
    {
        int index = -1;
        int regionId = -1;
        int candidatePartition = 0;
        std::string partitionReason = "default";
        std::string name;
        std::vector<unsigned long long> inputTensorIds;
        unsigned long long outputTensorId = 0;
    };

    struct FusionPartitionOptions
    {
        bool enablePreferredBoundaries = true;
        int preferredMinOperations = 32;
        int maxAtomicRegionsPerPartition = 0;
        unsigned long long builderWorkspaceBytes = 64ULL << 20;
        int builderOptimizationLevel = 3;
    };

    bool IsCompiledGraphCaptureActive();
    class TensorGraphCapture;
    bool CaptureFusionGraph(const TensorGraphCapture& graph,
        const FusionPartitionOptions& options, std::string& errorMessage);
    const std::vector<CapturedFusionRegion>& GetCapturedFusionRegions();
    const std::vector<CapturedTensorOperation>& GetCapturedTensorOperations();
    const std::string& GetCompiledGraphCaptureError();
    bool AssignCapturedFusionOperations(
        std::vector<CapturedTensorOperation> operations,
        const FusionPartitionOptions& options,
        std::string& errorMessage);

    class ScopedCompiledGraphCapture
    {
        bool m_previous = false;

    public:
        ScopedCompiledGraphCapture();
        ~ScopedCompiledGraphCapture();

        ScopedCompiledGraphCapture(const ScopedCompiledGraphCapture&) = delete;
        ScopedCompiledGraphCapture& operator=(const ScopedCompiledGraphCapture&) = delete;
    };

}
