#include "compiled_graph_capture.h"

#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <algorithm>

namespace Garnet
{
    namespace
    {
        thread_local bool g_compiledGraphCaptureActive = false;
        thread_local int g_compiledFusionCaptureDepth = 0;
        thread_local std::vector<CapturedFusionRegion> g_capturedFusionRegions;
        thread_local std::vector<int> g_fusionRegionStack;
        thread_local std::unordered_map<std::string, int> g_fusionInvocationCounts;
        thread_local std::vector<CapturedTensorOperation> g_capturedTensorOperations;
        thread_local std::string g_compiledGraphCaptureError;

        void CollectTensorIds(
            const X::Value& value,
            std::vector<unsigned long long>& tensorIds,
            std::unordered_set<unsigned long long>& visited)
        {
            if (value.IsList()) {
                X::List list(value);
                for (long long index = 0; index < list->Size(); ++index) {
                    CollectTensorIds(list->Get(index), tensorIds, visited);
                }
                return;
            }
            if (value.IsDict()) {
                X::Dict dictionary(value);
                for (auto& entry : *dictionary) {
                    CollectTensorIds(entry.second(), tensorIds, visited);
                }
                return;
            }
            if (!value.IsObject() ||
                (value.GetObj()->GetType() != X::ObjType::TensorExpression &&
                 value.GetObj()->GetType() != X::ObjType::Tensor)) {
                return;
            }

            const unsigned long long tensorId = value.GetObj()->GetID();
            if (visited.insert(tensorId).second) tensorIds.push_back(tensorId);
        }
    }

    bool IsCompiledFusionCaptureRootActive()
    {
        return g_compiledFusionCaptureDepth > 0;
    }

    bool IsCompiledGraphCaptureActive()
    {
        return g_compiledGraphCaptureActive;
    }

    const std::vector<CapturedFusionRegion>& GetCapturedFusionRegions()
    {
        return g_capturedFusionRegions;
    }

    const std::vector<CapturedTensorOperation>& GetCapturedTensorOperations()
    {
        return g_capturedTensorOperations;
    }

    bool AssignCapturedFusionOperations(
        std::vector<CapturedTensorOperation> operations,
        const FusionPartitionOptions& options,
        std::string& errorMessage)
    {
        std::unordered_map<unsigned long long, int> producerByTensor;
        for (size_t index = 0; index < operations.size(); ++index) {
            operations[index].index = static_cast<int>(index);
            if (operations[index].outputTensorId != 0) {
                producerByTensor[operations[index].outputTensorId] =
                    static_cast<int>(index);
            }
        }

        std::vector<int> regionsByDepth(g_capturedFusionRegions.size());
        for (size_t index = 0; index < regionsByDepth.size(); ++index) {
            regionsByDepth[index] = static_cast<int>(index);
            g_capturedFusionRegions[index].operationCount = 0;
            g_capturedFusionRegions[index].inclusiveOperationCount = 0;
        }
        std::sort(
            regionsByDepth.begin(),
            regionsByDepth.end(),
            [](int left, int right) {
                return g_capturedFusionRegions[static_cast<size_t>(left)].depth >
                    g_capturedFusionRegions[static_cast<size_t>(right)].depth;
            });

        for (const int regionId : regionsByDepth) {
            const auto& region = g_capturedFusionRegions[static_cast<size_t>(regionId)];
            const std::unordered_set<unsigned long long> boundaries(
                region.inputTensorIds.begin(), region.inputTensorIds.end());
            std::unordered_set<int> visitedOperations;
            std::function<void(unsigned long long)> claimTensor =
                [&](unsigned long long tensorId) {
                    if (tensorId == 0 || boundaries.find(tensorId) != boundaries.end()) return;
                    const auto producer = producerByTensor.find(tensorId);
                    if (producer == producerByTensor.end()) return;
                    const int operationIndex = producer->second;
                    if (!visitedOperations.insert(operationIndex).second) return;
                    auto& operation = operations[static_cast<size_t>(operationIndex)];
                    if (operation.regionId >= 0) return;
                    operation.regionId = regionId;
                    for (const auto inputId : operation.inputTensorIds) {
                        claimTensor(inputId);
                    }
                };
            for (const auto outputId : region.outputTensorIds) claimTensor(outputId);
        }

        for (auto& operation : operations) {
            if (operation.regionId < 0 && !g_capturedFusionRegions.empty()) {
                operation.regionId = 0;
            }
            if (operation.regionId >= 0) {
                ++g_capturedFusionRegions[
                    static_cast<size_t>(operation.regionId)].operationCount;
            }
        }

        for (const auto& region : g_capturedFusionRegions) {
            int ancestor = region.id;
            while (ancestor >= 0) {
                g_capturedFusionRegions[static_cast<size_t>(ancestor)].inclusiveOperationCount +=
                    region.operationCount;
                ancestor = g_capturedFusionRegions[
                    static_cast<size_t>(ancestor)].parentId;
            }
        }

        std::unordered_set<int> selectedPreferredRegions;
        if (options.enablePreferredBoundaries) {
            for (const auto& region : g_capturedFusionRegions) {
                if (region.annotation.boundary == "preferred" &&
                    region.inclusiveOperationCount >= options.preferredMinOperations) {
                    selectedPreferredRegions.insert(region.id);
                }
            }
        }

        std::string previousKey;
        int currentPartition = -1;
        for (auto& operation : operations) {
            int regionId = operation.regionId;
            int requiredRegion = -1;
            int preferredRegion = -1;
            int atomicRegion = -1;
            while (regionId >= 0) {
                const auto& region = g_capturedFusionRegions[
                    static_cast<size_t>(regionId)];
                if (atomicRegion < 0 && region.annotation.atomic) atomicRegion = regionId;
                if (requiredRegion < 0 && region.parentId >= 0 &&
                    region.annotation.boundary == "required") {
                    requiredRegion = regionId;
                }
                if (preferredRegion < 0 &&
                    selectedPreferredRegions.find(regionId) !=
                        selectedPreferredRegions.end()) {
                    preferredRegion = regionId;
                }
                regionId = region.parentId;
            }

            std::string key = "default";
            std::string reason = "default";
            if (requiredRegion >= 0) {
                key = "required:" + std::to_string(requiredRegion);
                reason = key;
            }
            else if (preferredRegion >= 0) {
                key = "preferred:" + std::to_string(preferredRegion);
                reason = key;
            }
            if (atomicRegion >= 0 && options.maxAtomicRegionsPerPartition > 0) {
                const auto& atomic = g_capturedFusionRegions[
                    static_cast<size_t>(atomicRegion)];
                const int group = atomic.invocation /
                    options.maxAtomicRegionsPerPartition;
                key += ":atomic_group:" + atomic.annotation.functionName + ":" +
                    std::to_string(group);
                reason += ":atomic_group:" + std::to_string(group);
            }
            if (currentPartition < 0 || key != previousKey) {
                ++currentPartition;
                previousKey = key;
            }
            operation.candidatePartition = currentPartition;
            operation.partitionReason = reason;
        }

        g_capturedTensorOperations = std::move(operations);
        errorMessage.clear();
        return true;
    }

    const std::string& GetCompiledGraphCaptureError()
    {
        return g_compiledGraphCaptureError;
    }

    ScopedCompiledGraphCapture::ScopedCompiledGraphCapture()
        : m_previous(g_compiledGraphCaptureActive)
    {
        if (!m_previous) {
            g_capturedFusionRegions.clear();
            g_fusionRegionStack.clear();
            g_fusionInvocationCounts.clear();
            g_capturedTensorOperations.clear();
            g_compiledGraphCaptureError.clear();
        }
        g_compiledGraphCaptureActive = true;
    }

    ScopedCompiledGraphCapture::~ScopedCompiledGraphCapture()
    {
        g_compiledGraphCaptureActive = m_previous;
    }

    ScopedCompiledFusionCaptureRoot::ScopedCompiledFusionCaptureRoot()
    {
        ++g_compiledFusionCaptureDepth;
    }

    ScopedCompiledFusionCaptureRoot::~ScopedCompiledFusionCaptureRoot()
    {
        --g_compiledFusionCaptureDepth;
    }

    ScopedCompiledFusionRegion::ScopedCompiledFusionRegion(
        const FusionAnnotation& annotation)
    {
        if (!g_compiledGraphCaptureActive) return;
        if (!annotation.validationError.empty()) {
            g_compiledGraphCaptureError = annotation.validationError;
            return;
        }

        const int parentId = g_fusionRegionStack.empty()
            ? -1
            : g_fusionRegionStack.back();
        if (parentId >= 0 &&
            g_capturedFusionRegions[static_cast<size_t>(parentId)].annotation.atomic &&
            annotation.boundary == "required") {
            g_compiledGraphCaptureError =
                "fusion region '" + annotation.name +
                "' declares boundary='required' inside atomic region '" +
                g_capturedFusionRegions[static_cast<size_t>(parentId)].annotation.name + "'";
            return;
        }

        CapturedFusionRegion region;
        region.id = static_cast<int>(g_capturedFusionRegions.size());
        region.parentId = parentId;
        region.depth = static_cast<int>(g_fusionRegionStack.size());
        region.invocation = g_fusionInvocationCounts[annotation.name]++;
        region.annotation = annotation;
        g_capturedFusionRegions.push_back(region);
        g_fusionRegionStack.push_back(region.id);
        m_regionId = region.id;
        m_active = true;
    }

    ScopedCompiledFusionRegion::~ScopedCompiledFusionRegion()
    {
        if (m_active && !g_fusionRegionStack.empty()) {
            g_fusionRegionStack.pop_back();
        }
    }

    bool ScopedCompiledFusionRegion::IsValid() const
    {
        return !g_compiledGraphCaptureActive ||
            (m_active && g_compiledGraphCaptureError.empty());
    }

    void ScopedCompiledFusionRegion::CaptureInputs(X::ARGS& inputs)
    {
        if (!m_active || m_regionId < 0) return;
        std::unordered_set<unsigned long long> visited;
        auto& tensorIds = g_capturedFusionRegions[
            static_cast<size_t>(m_regionId)].inputTensorIds;
        for (size_t index = 0; index < inputs.size(); ++index) {
            CollectTensorIds(inputs[index], tensorIds, visited);
        }
    }

    void ScopedCompiledFusionRegion::CaptureResult(const X::Value& result)
    {
        if (!m_active || m_regionId < 0) return;
        std::unordered_set<unsigned long long> visited;
        CollectTensorIds(
            result,
            g_capturedFusionRegions[static_cast<size_t>(m_regionId)].outputTensorIds,
            visited);
    }
}
