#include "compiled_graph_capture.h"
#include "../tensor/graph_capture.h"

#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <algorithm>

namespace Garnet
{
    namespace
    {
        thread_local bool g_compiledGraphCaptureActive = false;
        thread_local std::vector<CapturedFusionRegion> g_capturedFusionRegions;
        thread_local std::vector<CapturedTensorOperation> g_capturedTensorOperations;
        thread_local std::string g_compiledGraphCaptureError;

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

    bool CaptureFusionGraph(const TensorGraphCapture& graph,
        const FusionPartitionOptions& options, std::string& errorMessage)
    {
        g_capturedFusionRegions.clear();
        g_capturedTensorOperations.clear();
        g_compiledGraphCaptureError.clear();
        try {
            std::unordered_map<uint64_t, int> regionIds;
            std::unordered_map<std::string, int> invocations;
            std::vector<CapturedTensorOperation> operations;
            std::vector<std::vector<int>> regionStacks;
            auto tensorId = [](const X::Value& value) -> uint64_t {
                if (!value.IsObject()) return 0;
                X3TensorInfo info{};
                info.size = sizeof(info);
                return x3_tensor_info(value.runtime(), value.raw(), &info) == X3_STATUS_OK ? info.id : 0;
            };
            for (const auto& node : graph.Operations()) {
                if (node.provider.empty()) continue;
                std::vector<int> stack;
                int parent = -1;
                for (uint64_t index = 0; index < node.regions.Size(); ++index) {
                    auto attributes = node.regions.Get(index);
                    auto lookup = [&](const char* name) {
                        for (uint64_t item = 0; item < attributes.Size(); ++item) {
                            X::Value key, value;
                            if (!attributes.DictEntry(item, key, value))
                                throw std::runtime_error("invalid fusion attributes");
                            if (key.ToString() == name) return value;
                        }
                        return X::Value();
                    };
                    const auto externalId = lookup("id").ToUInt64();
                    if (!externalId) throw std::runtime_error("fusion invocation has no identity");
                    auto found = regionIds.find(externalId);
                    int id = -1;
                    if (found == regionIds.end()) {
                        CapturedFusionRegion region;
                        region.id = static_cast<int>(g_capturedFusionRegions.size());
                        region.parentId = parent;
                        region.depth = static_cast<int>(index);
                        auto text = [&](const char* key, const char* fallback) {
                            auto value = lookup(key);
                            return value.IsString() ? value.ToString() : std::string(fallback);
                        };
                        region.annotation.name = text("name", "fusion");
                        region.annotation.functionName = text("function", region.annotation.name.c_str());
                        region.annotation.role = text("role", "generic");
                        region.annotation.boundary = text("boundary", "none");
                        region.annotation.atomic = lookup("atomic").ToLongLong() != 0;
                        region.annotation.cudaGraph = lookup("cuda_graph").ToLongLong() != 0;
                        if (region.annotation.boundary == "required") {
                            for (int ancestor : stack)
                                if (g_capturedFusionRegions[ancestor].annotation.atomic)
                                    throw std::runtime_error("required fusion boundary inside an atomic region");
                        }
                        region.invocation = invocations[region.annotation.functionName]++;
                        id = region.id;
                        regionIds.emplace(externalId, id);
                        g_capturedFusionRegions.push_back(std::move(region));
                    } else {
                        id = found->second;
                        if (g_capturedFusionRegions[id].parentId != parent)
                            throw std::runtime_error("fusion invocation has inconsistent ancestry");
                    }
                    stack.push_back(id);
                    parent = id;
                }
                CapturedTensorOperation operation;
                operation.name = node.name;
                operation.outputTensorId = node.id;
                operation.regionId = parent;
                for (const auto& input : node.inputs)
                    if (auto id = tensorId(input)) operation.inputTensorIds.push_back(id);
                operations.push_back(std::move(operation));
                regionStacks.push_back(std::move(stack));
            }
            std::unordered_map<uint64_t, size_t> producers;
            producers.reserve(operations.size());
            for (size_t index = 0; index < operations.size(); ++index)
                producers.emplace(operations[index].outputTensorId, index);
            std::unordered_set<uint64_t> graphOutputs;
            std::function<void(const X::Value&, unsigned)> collectOutputs =
                [&](const X::Value& value, unsigned depth) {
                    if (depth > 64) throw std::runtime_error("graph output nesting exceeds 64");
                    if (value.IsDict()) {
                        for (uint64_t index = 0; index < value.Size(); ++index) {
                            X::Value key, item;
                            if (!value.DictEntry(index, key, item)) throw std::runtime_error("invalid graph outputs");
                            collectOutputs(item, depth + 1);
                        }
                    } else if (value.IsList() || x3_value_object_kind(value.raw()) == X3_OBJECT_KIND_TUPLE) {
                        for (uint64_t index = 0; index < value.Size(); ++index)
                            collectOutputs(value.Get(index), depth + 1);
                    } else if (auto id = tensorId(value)) graphOutputs.insert(id);
                };
            collectOutputs(graph.Outputs(), 0);
            std::vector<std::unordered_set<uint64_t>> seenInputs(g_capturedFusionRegions.size());
            std::vector<std::unordered_set<uint64_t>> seenOutputs(g_capturedFusionRegions.size());
            auto addOutput = [&](int region, uint64_t id) {
                if (seenOutputs[region].insert(id).second)
                    g_capturedFusionRegions[region].outputTensorIds.push_back(id);
            };
            for (auto id : graphOutputs) {
                auto producer = producers.find(id);
                if (producer != producers.end())
                    for (int region : regionStacks[producer->second]) addOutput(region, id);
            }
            // Walk each dependency across its region boundary once. Scanning
            // every operation separately for every decoder layer is quadratic.
            const std::vector<int> external;
            for (size_t index = 0; index < operations.size(); ++index) {
                const auto& consumerStack = regionStacks[index];
                for (auto id : operations[index].inputTensorIds) {
                    auto producer = producers.find(id);
                    const auto& producerStack = producer == producers.end()
                        ? external : regionStacks[producer->second];
                    size_t common = 0;
                    while (common < consumerStack.size() && common < producerStack.size() &&
                        consumerStack[common] == producerStack[common]) ++common;
                    for (size_t depth = common; depth < consumerStack.size(); ++depth) {
                        const int region = consumerStack[depth];
                        if (seenInputs[region].insert(id).second)
                            g_capturedFusionRegions[region].inputTensorIds.push_back(id);
                    }
                    for (size_t depth = common; depth < producerStack.size(); ++depth)
                        addOutput(producerStack[depth], id);
                }
            }
            return AssignCapturedFusionOperations(std::move(operations), options, errorMessage);
        } catch (const std::exception& error) {
            g_capturedFusionRegions.clear();
            g_capturedTensorOperations.clear();
            errorMessage = error.what();
            g_compiledGraphCaptureError = errorMessage;
            return false;
        }
    }

    ScopedCompiledGraphCapture::ScopedCompiledGraphCapture()
        : m_previous(g_compiledGraphCaptureActive)
    {
        if (!m_previous) {
            g_capturedFusionRegions.clear();
            g_capturedTensorOperations.clear();
            g_compiledGraphCaptureError.clear();
        }
        g_compiledGraphCaptureActive = true;
    }

    ScopedCompiledGraphCapture::~ScopedCompiledGraphCapture()
    {
        g_compiledGraphCaptureActive = m_previous;
    }

}
