// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "compiled_model_runtime.h"
#include "compiled_graph_capture.h"
#include "graph_capture.h"
#include "md5.h"
#include "trt_builder.h"
#include "openvino_builder.h"
#include "garnet_tensor.h"
#include "tensor_helper.h"
#include "qwen_vl_compiled_frontend.h"
#include "qwen_text_compiled_frontend.h"
#include "qwen_asr_compiled_frontend.h"
#include "qwen_tts_compiled_frontend.h"
#include "cuda_lib.h"
#include "repetition_sampler.h"
#include "qwen_tokenizer.h"
#include "nlohmann/json.hpp"

#include <NvInferVersion.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <random>
#include <sstream>
#include <set>

namespace
{
    X::Value NativeValue(X3PackageHost*) { return {}; }
    X::Value NativeValue(X3PackageHost*, std::nullptr_t) { return X::Value(nullptr); }
    X::Value NativeValue(X3PackageHost* host, const std::string& value) { return X::Value::String(host, value); }
    X::Value NativeValue(X3PackageHost* host, const char* value) { return X::Value::String(host, value); }
    template<class T> X::Value NativeValue(X3PackageHost* host, T&& value) {
        if constexpr (std::is_convertible_v<T, std::string>) return X::Value::String(host, value);
        else return X::Value(std::forward<T>(value));
    }
    X::Value Lookup(const X::Value& dictionary, const char* name) {
        for (uint64_t i = 0; i < dictionary.Size(); ++i) {
            X::Value key, value;
            if (!dictionary.DictEntry(i, key, value)) throw std::runtime_error("invalid model dictionary");
            if (key.IsString() && key.ToString() == name) return value;
        }
        return {};
    }
    void ClearImportNamespace(X3PackageHost* host, std::string& name) {
        if (name.empty()) return;
        X::Module sys(host, "sys");
        auto registry = sys["modules"];
        std::vector<X::Value> keys;
        for (uint64_t i = 0; i < registry.Size(); ++i) {
            X::Value key, value;
            if (!registry.DictEntry(i, key, value)) break;
            if (!key.IsString()) continue;
            const auto text = key.ToString();
            if (text == name || text.compare(0, name.size() + 1, name + ".") == 0)
                keys.push_back(key);
        }
        for (const auto& key : keys) {
            if (x3_delete_item(host->runtime, registry.raw(), key.raw()) != X3_STATUS_OK)
                throw std::runtime_error(x3_runtime_last_error(host->runtime));
        }
        name.clear();
    }
    X::Tensor AdoptCudaTensor(X3PackageHost* host, X3TensorDType dtype,
        const std::vector<int64_t>& shape, void* allocation) {
        try {
            std::vector<int64_t> strides(shape.size());
            uint64_t bytes = Garnet::TensorHelper::ItemSize(dtype);
            for (size_t i = shape.size(); i-- > 0;) {
                if (shape[i] < 0 || bytes > INT64_MAX ||
                    (shape[i] && bytes > UINT64_MAX / shape[i]))
                    throw std::invalid_argument("invalid CUDA tensor shape");
                strides[i] = static_cast<int64_t>(bytes);
                bytes *= shape[i];
            }
            int device = 0;
            const auto status = cudaGetDevice(&device);
            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
            X3TensorInfo info{};
            info.size = sizeof(info); info.dtype = dtype;
            info.rank = static_cast<uint32_t>(shape.size());
            info.shape = shape.data(); info.strides = strides.data();
            info.data = allocation; info.byte_size = bytes;
            info.device_type = Garnet::TensorHelper::CudaDevice; info.device_id = device;
            return Garnet::TensorHelper::WrapGPU(host, info, allocation, device);
        }
        catch (...) {
            if (allocation) cudaFree(allocation);
            throw;
        }
    }

    constexpr const char* kGraphCacheMagic = "GARNET_RUNTIME_GRAPH_CACHE_V2";
    constexpr const char* kRuntimeSchema =
        "compiled_xmodel_runtime_v25_xlang3_python_graph";

    std::string ReadFile(const std::filesystem::path& path)
    {
        std::ifstream stream(path, std::ios::binary);
        return std::string(
            std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>());
    }

    X::Value JsonToXValue(X3PackageHost* host, const nlohmann::json& value)
    {
        if (value.is_null()) return NativeValue(host, nullptr);
        if (value.is_boolean()) return NativeValue(host, value.get<bool>());
        if (value.is_number_integer()) return NativeValue(host, value.get<long long>());
        if (value.is_number_unsigned()) return NativeValue(host, value.get<unsigned long long>());
        if (value.is_number_float()) return NativeValue(host, value.get<double>());
        if (value.is_string()) return NativeValue(host, value.get<std::string>());
        if (value.is_array()) {
            X::Value list = X::Value::List(host);
            for (const auto& item : value) {
                list.Append(JsonToXValue(host, item));
            }
            return NativeValue(host, list);
        }
        if (value.is_object()) {
            X::Value dictionary = X::Value::Dict(host);
            for (auto iterator = value.begin(); iterator != value.end(); ++iterator) {
                dictionary.SetItem(iterator.key(), JsonToXValue(host, iterator.value()));
            }
            return NativeValue(host, dictionary);
        }
        return NativeValue(host);
    }

    std::string BuildExecutionPlanJson(
        const std::vector<Garnet::CapturedFusionRegion>& regions,
        const std::vector<Garnet::CapturedTensorOperation>& operations,
        const std::vector<Garnet::EnginePartitionSpec>& enginePartitions,
        const Garnet::FusionPartitionOptions& partitionOptions)
    {
        nlohmann::json plan = {
            {"schema", "garnet_execution_plan_v8"},
            {"control_plane", "cpu"},
            {"tensor_plane", "gpu"},
            {"synchronization", "cuda_stream_ordering"},
            {"cross_partition_synchronization", "cuda_events_planned"},
            {"intermediate_host_copies", false},
            {"device_wide_synchronization", false},
            {"partition_state", enginePartitions.size() > 1
                ? "physical_engines_compiled"
                : "operation_dag_partitioned"},
            {"regions", nlohmann::json::array()},
            {"operations", nlohmann::json::array()},
            {"engine_partitions", nlohmann::json::array()},
            {"partition_candidates", nlohmann::json::array()}
        };
        plan["planner_options"] = {
            {"enable_preferred_boundaries", partitionOptions.enablePreferredBoundaries},
            {"preferred_min_operations", partitionOptions.preferredMinOperations},
            {"max_atomic_regions_per_partition", partitionOptions.maxAtomicRegionsPerPartition},
            {"builder_workspace_bytes", partitionOptions.builderWorkspaceBytes},
            {"builder_optimization_level", partitionOptions.builderOptimizationLevel}
        };

        int maxPartition = 0;
        for (const auto& region : regions) {
            int regionPartition = 0;
            for (const auto& operation : operations) {
                if (operation.regionId == region.id) {
                    regionPartition = operation.candidatePartition;
                    break;
                }
            }
            nlohmann::json item = {
                {"id", region.id},
                {"parent_id", region.parentId},
                {"depth", region.depth},
                {"invocation", region.invocation},
                {"operation_count", region.operationCount},
                {"inclusive_operation_count", region.inclusiveOperationCount},
                {"input_tensor_ids", region.inputTensorIds},
                {"output_tensor_ids", region.outputTensorIds},
                {"name", region.annotation.name},
                {"function", region.annotation.functionName},
                {"role", region.annotation.role},
                {"boundary", region.annotation.boundary},
                {"atomic", region.annotation.atomic},
                {"cuda_graph", region.annotation.cudaGraph},
                {"candidate_partition", regionPartition}
            };
            plan["regions"].push_back(item);
            if (region.annotation.boundary != "none") {
                plan["partition_candidates"].push_back({
                    {"region_id", region.id},
                    {"kind", region.annotation.boundary},
                    {"candidate_partition", regionPartition}
                });
            }
        }
        for (const auto& operation : operations) {
            maxPartition = std::max(maxPartition, operation.candidatePartition);
            plan["operations"].push_back({
                {"index", operation.index},
                {"name", operation.name},
                {"region_id", operation.regionId},
                {"candidate_partition", operation.candidatePartition},
                {"partition_reason", operation.partitionReason},
                {"input_tensor_ids", operation.inputTensorIds},
                {"output_tensor_id", operation.outputTensorId}
            });
        }
        plan["candidate_partition_count"] = operations.empty() ? 0 : maxPartition + 1;
        for (const auto& partition : enginePartitions) {
            nlohmann::json partitionJson = {
                {"id", partition.id},
                {"engine_path", partition.enginePath},
                {"inputs", nlohmann::json::array()},
                {"outputs", nlohmann::json::array()}
            };
            for (const auto& binding : partition.inputs) {
                partitionJson["inputs"].push_back({
                    {"name", binding.name},
                    {"tensor_id", binding.tensorId},
                    {"request_input_index", binding.requestInputIndex}
                });
            }
            for (const auto& binding : partition.outputs) {
                partitionJson["outputs"].push_back({
                    {"name", binding.name},
                    {"tensor_id", binding.tensorId},
                    {"terminal", binding.terminalOutput}
                });
            }
            plan["engine_partitions"].push_back(std::move(partitionJson));
        }
        plan["execution_mode"] = enginePartitions.size() > 1
            ? "partitioned_engines"
            : "monolithic_engine";
        return plan.dump();
    }

    bool ParseEnginePartitions(
        const std::string& executionPlanJson,
        std::vector<Garnet::EnginePartitionSpec>& partitions)
    {
        partitions.clear();
        try {
            const auto plan = nlohmann::json::parse(executionPlanJson);
            if (!plan.contains("engine_partitions") ||
                !plan["engine_partitions"].is_array()) return true;
            for (const auto& partitionJson : plan["engine_partitions"]) {
                Garnet::EnginePartitionSpec partition;
                partition.id = partitionJson.at("id").get<int>();
                partition.enginePath = partitionJson.at("engine_path").get<std::string>();
                for (const auto& inputJson : partitionJson.at("inputs")) {
                    Garnet::EnginePartitionBinding binding;
                    binding.name = inputJson.at("name").get<std::string>();
                    binding.tensorId = inputJson.at("tensor_id").get<unsigned long long>();
                    binding.requestInputIndex = inputJson.at("request_input_index").get<int>();
                    partition.inputs.push_back(std::move(binding));
                }
                for (const auto& outputJson : partitionJson.at("outputs")) {
                    Garnet::EnginePartitionBinding binding;
                    binding.name = outputJson.at("name").get<std::string>();
                    binding.tensorId = outputJson.at("tensor_id").get<unsigned long long>();
                    binding.terminalOutput = outputJson.at("terminal").get<bool>();
                    partition.outputs.push_back(std::move(binding));
                }
                partitions.push_back(std::move(partition));
            }
            return true;
        }
        catch (const std::exception&) {
            partitions.clear();
            return false;
        }
    }

    bool ExecutionPlanRequestsCudaGraph(const std::string& executionPlanJson)
    {
        try {
            const auto plan = nlohmann::json::parse(executionPlanJson);
            if (!plan.contains("regions") || !plan["regions"].is_array()) {
                return false;
            }
            for (const auto& region : plan["regions"]) {
                if (region.value("cuda_graph", false)) return true;
            }
        }
        catch (const std::exception&) {
        }
        return false;
    }

    X::Value MakeCudaTensorFromHost(X3PackageHost* host, X3TensorDType dataType,
        const std::vector<int>& dimensions, const void* source, size_t bytes)
    {
        const std::vector<int64_t> shape(dimensions.begin(), dimensions.end());
        uint64_t expected = Garnet::TensorHelper::ItemSize(dataType);
        for (auto dimension : shape) {
            if (dimension < 0 || (dimension && expected > UINT64_MAX / dimension))
                throw std::invalid_argument("invalid tensor shape");
            expected *= dimension;
        }
        if (expected != bytes) throw std::invalid_argument("tensor byte size mismatch");
        return Garnet::TensorHelper::CreateGPU(host, dataType, shape, source);
    }

    X::Value MakeCpuTensorFromHost(X3PackageHost* host, X3TensorDType dataType,
        const std::vector<int>& dimensions, const void* source, size_t bytes)
    {
        const std::vector<int64_t> shape(dimensions.begin(), dimensions.end());
        uint64_t expected = Garnet::TensorHelper::ItemSize(dataType);
        for (auto dimension : shape) {
            if (dimension < 0 || (dimension && expected > UINT64_MAX / dimension))
                throw std::invalid_argument("invalid tensor shape");
            expected *= dimension;
        }
        if (expected != bytes) throw std::invalid_argument("tensor byte size mismatch");
        return X::Tensor::Create(host, dataType, shape, source, source ? bytes : 0);
    }

    X::Value BuildSymbolicWeights(X3PackageHost* host, const Garnet::SafeTensorsIndex& index)
    {
        auto weights = X::Value::Dict(host);
        for (const auto& entry : index.Entries()) {
            const auto& name = entry.first;
            const auto& metadata = entry.second;
            X3TensorDType dtype;
            if (metadata.dataType == "BF16") dtype = X3_TENSOR_BFLOAT16;
            else if (metadata.dataType == "F16") dtype = X3_TENSOR_FLOAT16;
            else if (metadata.dataType == "F32") dtype = X3_TENSOR_FLOAT32;
            else continue;
            std::vector<int64_t> shape(metadata.shape.begin(), metadata.shape.end());
            if (std::any_of(shape.begin(), shape.end(), [](int64_t d) { return d < 0 || d > INT_MAX; }))
                throw std::invalid_argument("weight shape exceeds backend limits: " + name);
            weights.SetItem(name, X::Tensor::Input(host, name.c_str(), dtype, shape));
        }
        return weights;
    }

    void CollectXModelDependencies(
        const std::filesystem::path& sourcePath,
        std::set<std::filesystem::path>& dependencies)
    {
        const auto normalized = std::filesystem::absolute(sourcePath).lexically_normal();
        if (!dependencies.insert(normalized).second) {
            return;
        }

        for (auto parent = normalized.parent_path(); !parent.empty();) {
            const auto initializer = parent / "__init__.py";
            if (!std::filesystem::is_regular_file(initializer)) break;
            if (initializer != normalized) CollectXModelDependencies(initializer, dependencies);
            const auto next = parent.parent_path();
            if (next == parent) break;
            parent = next;
        }

        const std::string source = ReadFile(normalized);
        static const std::regex fromImport(
            R"(^\s*from\s+([A-Za-z0-9_.]+)\s+import\s+)",
            std::regex::ECMAScript);
        static const std::regex plainImport(
            R"(^\s*import\s+([A-Za-z0-9_.]+))",
            std::regex::ECMAScript);
        static const std::regex localImport(
            R"(^\s*from\s+\.\s+import\s+([A-Za-z0-9_]+))",
            std::regex::ECMAScript);
        static const std::regex explicitDependency(
            R"(^\s*#\s*garnet-dependency:\s*([^\s]+\.py)\s*$)",
            std::regex::ECMAScript);
        std::istringstream lines(source);
        std::string line;
        while (std::getline(lines, line)) {
            std::smatch match;
            if (std::regex_search(line, match, explicitDependency)) {
                const auto candidate = normalized.parent_path() / match[1].str();
                if (std::filesystem::is_regular_file(candidate)) {
                    CollectXModelDependencies(candidate, dependencies);
                }
                continue;
            }
            if (std::regex_search(line, match, localImport)) {
                const auto candidate = normalized.parent_path() /
                    (match[1].str() + ".py");
                if (std::filesystem::is_regular_file(candidate)) {
                    CollectXModelDependencies(candidate, dependencies);
                }
                continue;
            }
            if (!std::regex_search(line, match, fromImport) &&
                !std::regex_search(line, match, plainImport)) {
                continue;
            }
            std::string module = match[1].str();
            std::replace(module.begin(), module.end(), '.', '/');
            const auto candidate = normalized.parent_path() / (module + ".py");
            if (std::filesystem::is_regular_file(candidate)) {
                CollectXModelDependencies(candidate, dependencies);
            }
        }
    }

    std::string MakeGraphFingerprint(
        const std::filesystem::path& rootPath,
        const std::string& weightsLocation,
        const std::string& entryFunction,
        const std::string& backend,
        const std::string& precision,
        const std::vector<std::vector<int>>& inputShapes,
        const std::vector<std::string>& inputDataTypes,
        const Garnet::FusionPartitionOptions& partitionOptions)
    {
        std::set<std::filesystem::path> dependencies;
        CollectXModelDependencies(rootPath, dependencies);

        std::ostringstream material;
        material << kRuntimeSchema << '\n'
                 << "backend:" << backend << '\n'
                 << "precision:" << precision << '\n'
                 << "tensorrt:" << NV_TENSORRT_VERSION << '\n'
                 << entryFunction << '\n';
        if (backend == "openvino") {
            const char* device = std::getenv("GARNET_OPENVINO_DEVICE");
            material << "openvino_runtime_schema:"
                     << "per_layer_device_state_native_q4q8_lm_head_v12\n";
            material << "openvino_device:"
                     << (device && *device ? device : "CPU") << '\n';
            material << "openvino_precision:" << precision << '\n';
            const char* dynamicGroup = std::getenv(
                "GARNET_OPENVINO_CPU_DQ_GROUP_SIZE");
            material << "GARNET_OPENVINO_CPU_DQ_GROUP_SIZE:"
                     << (dynamicGroup && *dynamicGroup
                             ? dynamicGroup
                             : "default")
                     << '\n';
            const char* weightGroup = std::getenv(
                "GARNET_OPENVINO_CPU_WEIGHT_GROUP_SIZE");
            material << "GARNET_OPENVINO_CPU_WEIGHT_GROUP_SIZE:"
                     << (weightGroup && *weightGroup
                             ? weightGroup
                             : "128")
                     << '\n';
            const char* packedProjections = std::getenv(
                "GARNET_OPENVINO_CPU_PACKED_PROJECTIONS");
            material << "GARNET_OPENVINO_CPU_PACKED_PROJECTIONS:"
                     << (packedProjections && *packedProjections
                             ? packedProjections
                             : "0")
                     << '\n';
            const char* signedI4 = std::getenv(
                "GARNET_OPENVINO_CPU_SIGNED_I4");
            material << "GARNET_OPENVINO_CPU_SIGNED_I4:"
                     << (signedI4 && *signedI4 ? signedI4 : "0")
                     << '\n';
            const char* nativeQ4Q8 = std::getenv(
                "GARNET_OPENVINO_CPU_NATIVE_Q4Q8");
            material << "GARNET_OPENVINO_CPU_NATIVE_Q4Q8:"
                     << (nativeQ4Q8 && *nativeQ4Q8 ? nativeQ4Q8 : "0")
                     << '\n';
        }
        const char* fusedTextAttention = std::getenv("GARNET_FUSED_TEXT_ATTENTION");
        if (fusedTextAttention && std::string(fusedTextAttention) == "1") {
            material << "fused_text_attention:1\n";
        }
        int device = 0;
        cudaDeviceProp deviceProperties{};
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaGetDeviceProperties(&deviceProperties, device) == cudaSuccess) {
            material << "cuda_cc:" << deviceProperties.major << '.'
                     << deviceProperties.minor << '\n';
        }
        for (const auto& shape : inputShapes) {
            material << "shape";
            for (const int dimension : shape) {
                material << ':' << dimension;
            }
            material << '\n';
        }
        for (const auto& dataType : inputDataTypes) {
            material << "dtype:" << dataType << '\n';
        }
        material << "partition.enable_preferred:"
                 << partitionOptions.enablePreferredBoundaries << '\n'
                 << "partition.preferred_min_operations:"
                 << partitionOptions.preferredMinOperations << '\n'
                 << "partition.max_atomic_regions:"
                 << partitionOptions.maxAtomicRegionsPerPartition << '\n'
                 << "builder.workspace_bytes:"
                 << partitionOptions.builderWorkspaceBytes << '\n'
                 << "builder.optimization_level:"
                 << partitionOptions.builderOptimizationLevel << '\n';
        for (const auto& dependency : dependencies) {
            material << dependency.generic_string() << '\n';
            material << MD5(ReadFile(dependency)).hexdigest() << '\n';
        }

        if (weightsLocation.empty()) {
            material << "weights:none\n";
        }
        else {
            const std::filesystem::path weightsPath(weightsLocation);
            material << std::filesystem::absolute(weightsPath).lexically_normal().generic_string() << '\n';
            std::error_code error;
            std::vector<std::filesystem::path> files;
            if (std::filesystem::is_regular_file(weightsPath, error)) {
                files.push_back(weightsPath);
            }
            else if (std::filesystem::is_directory(weightsPath, error)) {
                for (std::filesystem::recursive_directory_iterator iterator(weightsPath, error), end;
                     !error && iterator != end;
                     iterator.increment(error)) {
                    if (iterator->is_regular_file(error)) {
                        files.push_back(iterator->path());
                    }
                }
            }
            std::sort(files.begin(), files.end());
            for (const auto& file : files) {
                error.clear();
                const auto canonical = std::filesystem::weakly_canonical(file, error);
                material << file.lexically_normal().generic_string() << '\n'
                         << canonical.generic_string() << '\n'
                         << std::filesystem::file_size(file, error) << '\n'
                         << std::filesystem::last_write_time(file, error).time_since_epoch().count() << '\n';
                if (file.extension() == ".json") {
                    material << MD5(ReadFile(file)).hexdigest() << '\n';
                }
            }
        }
        return MD5(material.str()).hexdigest();
    }

    std::string HexEncode(const std::string& value)
    {
        static constexpr char digits[] = "0123456789abcdef";
        std::string encoded;
        encoded.reserve(value.size() * 2);
        for (const unsigned char byte : value) {
            encoded.push_back(digits[byte >> 4]);
            encoded.push_back(digits[byte & 0x0f]);
        }
        return encoded;
    }

    bool HexDecode(const std::string& value, std::string& decoded)
    {
        if ((value.size() % 2) != 0) {
            return false;
        }
        auto nibble = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
        };
        decoded.clear();
        decoded.reserve(value.size() / 2);
        for (size_t index = 0; index < value.size(); index += 2) {
            const int high = nibble(value[index]);
            const int low = nibble(value[index + 1]);
            if (high < 0 || low < 0) {
                return false;
            }
            decoded.push_back(static_cast<char>((high << 4) | low));
        }
        return true;
    }

    bool LoadGraphCache(
        const std::filesystem::path& cachePath,
        const std::string& fingerprint,
        std::string& graphSummary,
        std::string& executionPlanJson)
    {
        std::ifstream stream(cachePath, std::ios::binary);
        std::string magic;
        std::string cachedFingerprint;
        std::string encodedSummary;
        std::string encodedExecutionPlan;
        if (!std::getline(stream, magic) ||
            !std::getline(stream, cachedFingerprint) ||
            !std::getline(stream, encodedSummary) ||
            !std::getline(stream, encodedExecutionPlan) ||
            magic != kGraphCacheMagic ||
            cachedFingerprint != fingerprint) {
            return false;
        }
        return HexDecode(encodedSummary, graphSummary) && !graphSummary.empty() &&
            HexDecode(encodedExecutionPlan, executionPlanJson) &&
            !executionPlanJson.empty();
    }

    bool StoreGraphCache(
        const std::filesystem::path& cachePath,
        const std::string& fingerprint,
        const std::string& graphSummary,
        const std::string& executionPlanJson)
    {
        const auto temporaryPath = cachePath.string() + ".tmp";
        {
            std::ofstream stream(temporaryPath, std::ios::binary | std::ios::trunc);
            if (!stream) {
                return false;
            }
            stream << kGraphCacheMagic << '\n'
                   << fingerprint << '\n'
                   << HexEncode(graphSummary) << '\n'
                   << HexEncode(executionPlanJson) << '\n';
            if (!stream.good()) {
                return false;
            }
        }
        std::error_code error;
        std::filesystem::remove(cachePath, error);
        error.clear();
        std::filesystem::rename(temporaryPath, cachePath, error);
        return !error;
    }
}

namespace Garnet
{
    namespace
    {
        std::uint64_t NextOpenVINOSessionId()
        {
            static std::atomic<std::uint64_t> nextSessionId{1};
            return nextSessionId.fetch_add(1, std::memory_order_relaxed);
        }
    }

    CompiledModelRuntime::~CompiledModelRuntime()
    {
        try { ReleaseDeviceMemory(); }
        catch (...) { }
    }

    void CompiledModelRuntime::ReleaseDeviceMemory()
    {
        std::lock_guard<std::mutex> guard(m_mutex);
        m_trtExecutions.clear();
        m_enginesPrepared = false;
        ClearImportNamespace(m_host, m_importNamespace);
        m_module = X::Value();
        m_rootFunction = X::Value();
        m_graph = X::Value();
        m_dependencyModules.clear();
        auto releaseTensor = [](X::Value& value) {
            value = X::Value();
        };
        releaseTensor(m_reusableExecutionOutput);
        releaseTensor(m_reusablePrefillKeyCache);
        releaseTensor(m_reusablePrefillValueCache);
        for (auto& weight : m_loadedWeights) {
            releaseTensor(weight.second);
        }
        m_loadedWeights.clear();
        m_loadedWeightBytes = 0;
        if (m_sampleTokenDevice) {
            cudaFree(m_sampleTokenDevice);
            m_sampleTokenDevice = nullptr;
        }
        if (m_sampleValueDevice) {
            cudaFree(m_sampleValueDevice);
            m_sampleValueDevice = nullptr;
        }
        m_sampleCapacity = 0;
        if (m_decodeRuntime) {
            m_decodeRuntime->ReleaseDeviceMemory();
            m_decodeRuntime.reset();
        }
        if (m_auxRuntime) {
            m_auxRuntime->ReleaseDeviceMemory();
            m_auxRuntime.reset();
        }
        if (m_codecRuntime) {
            m_codecRuntime->ReleaseDeviceMemory();
            m_codecRuntime.reset();
        }
        OpenVINOBuilder::ReleaseCachedSession(
            m_enginePath, m_openVinoSessionId);
        m_ready = false;
        m_state = "released";
    }

    bool CompiledModelRuntime::Initialize(
        const std::string& rootXModel,
        const std::string& cacheDirectory,
        const std::string& weightsLocation,
        const std::string& entryFunction,
        const std::string& frontend,
        const std::vector<std::vector<int>>& inputShapes,
        const std::vector<std::string>& inputDataTypes,
        const FusionPartitionOptions& partitionOptions,
        const std::string& backend,
        const std::string& precision)
    {
        auto* host = m_host;
        std::lock_guard<std::mutex> guard(m_mutex);
        m_trtExecutions.clear();
        // Failed initialization must not keep engines alive until DLL teardown.
        auto cleanupEngines = [](CompiledModelRuntime* runtime) noexcept {
            runtime->m_trtExecutions.clear();
            runtime->m_decodeRuntime.reset();
            runtime->m_auxRuntime.reset();
            runtime->m_codecRuntime.reset();
            runtime->m_enginesPrepared = false;
        };
        std::unique_ptr<CompiledModelRuntime, decltype(cleanupEngines)> engineGuard(this, cleanupEngines);
        ClearImportNamespace(host, m_importNamespace);
        m_rootXModel = std::filesystem::absolute(rootXModel).lexically_normal().string();
        if (std::filesystem::path(m_rootXModel).extension() == ".x") {
            auto pythonPath = std::filesystem::path(m_rootXModel).replace_extension(".py");
            if (std::filesystem::is_regular_file(pythonPath)) m_rootXModel = pythonPath.string();
        }
        m_cacheDirectory = std::filesystem::absolute(cacheDirectory).lexically_normal().string();
        m_weightsLocation = weightsLocation;
        m_entryFunction = entryFunction.empty() ? "Qwen3VLModel" : entryFunction;
        m_frontend = frontend;
        m_backend = backend.empty() ? "tensorrt" : backend;
        m_precision = precision.empty() ? "bf16" : precision;
        if (m_backend == "openvino" && m_openVinoSessionId == 0) {
            m_openVinoSessionId = NextOpenVINOSessionId();
        }
        m_inputShapes = inputShapes;
        m_partitionOptions = partitionOptions;
        m_ready = false;
        m_loadedWeights.clear();
        m_loadedWeightBytes = 0;
        m_enginePreparationMs = 0.0;
        m_frontendPreparationMs = 0.0;
        m_enginesPrepared = false;
        m_frontendPrepared = false;
        m_executionPlanJson.clear();
        m_enginePartitions.clear();
        m_errorCode.clear();
        m_errorMessage.clear();
        m_diagnostics = {};
        m_decodeRuntime.reset();
        m_auxRuntime.reset();
        m_codecRuntime.reset();
        if (m_backend != "tensorrt" && m_backend != "openvino") {
            m_state = "failed";
            m_errorCode = "backend_not_available";
            m_errorMessage =
                "backend '" + m_backend +
                "' is not available; expected tensorrt or openvino";
            return false;
        }
        if (m_backend == "openvino" && !OpenVINOBuilder::IsAvailable()) {
            m_state = "failed";
            m_errorCode = "backend_not_available";
            m_errorMessage =
                "OpenVINO was requested but this Garnet build does not include it";
            return false;
        }
        if (m_backend == "openvino" &&
            m_precision != "bf16" &&
            m_precision != "fp16" &&
            m_precision != "int4_fp16") {
            m_state = "failed";
            m_errorCode = "unsupported_precision";
            m_errorMessage =
                "OpenVINO precision must be 'bf16', 'fp16', or "
                "'int4_fp16'";
            return false;
        }
        if (m_backend == "tensorrt" &&
            m_precision != "bf16" &&
            m_precision != "int4_fp16") {
            m_state = "failed";
            m_errorCode = "unsupported_precision";
            m_errorMessage =
                "TensorRT precision must be 'bf16' or 'int4_fp16'";
            return false;
        }
        if (m_backend == "tensorrt" &&
            m_precision == "int4_fp16") {
            int device = 0;
            cudaDeviceProp properties{};
            if (cudaGetDevice(&device) == cudaSuccess &&
                cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
                properties.major < 9) {
                m_state = "failed";
                m_errorCode = "unsupported_precision";
                m_errorMessage =
                    "TensorRT INT4 weight-only execution in the full "
                    "TensorRT backend requires a Hopper-class GPU on this "
                    "runtime; use precision='bf16' on pre-Hopper GPUs";
                return false;
            }
        }

        std::error_code error;
        const std::filesystem::path rootPath(m_rootXModel);
        if (!std::filesystem::is_regular_file(rootPath, error) || rootPath.extension() != ".py") {
            m_state = "failed";
            m_errorCode = "invalid_root_xmodel";
            m_errorMessage = "compiled_xmodel mode requires an existing Python model file";
            return false;
        }

        std::filesystem::create_directories(m_cacheDirectory, error);
        if (error) {
            m_state = "failed";
            m_errorCode = "cache_directory_unavailable";
            m_errorMessage = error.message();
            return false;
        }

        m_weightIndexError.clear();
        if (!m_weightsLocation.empty()) {
            std::filesystem::path weightsPath(m_weightsLocation);
            if (!m_weightIndex.Open(weightsPath, m_weightIndexError)) {
                m_state = "failed";
                m_errorCode = "weights_index_invalid";
                m_errorMessage = m_weightIndexError;
                return false;
            }
        }
        std::ifstream sourceFile(rootPath, std::ios::binary);
        std::string source(
            (std::istreambuf_iterator<char>(sourceFile)),
            std::istreambuf_iterator<char>());
        if (!sourceFile.good() && source.empty()) {
            m_state = "failed";
            m_errorCode = "root_xmodel_read_failed";
            m_errorMessage = "failed to read model Python source";
            return false;
        }
        if (!inputDataTypes.empty() && inputDataTypes.size() != inputShapes.size()) {
            m_state = "failed";
            m_errorCode = "invalid_symbolic_input_dtype";
            m_errorMessage = "input_dtypes must contain one entry per input_shapes entry";
            return false;
        }

        const std::string graphFingerprint = MakeGraphFingerprint(
            rootPath, m_weightsLocation, m_entryFunction, m_backend, m_precision,
            inputShapes, inputDataTypes, m_partitionOptions);
        const std::filesystem::path graphCachePath =
            std::filesystem::path(m_cacheDirectory) / "runtime_graph.cache";
        const std::filesystem::path enginePath =
            std::filesystem::path(m_cacheDirectory) / "model.engine";
        m_enginePath = enginePath.string();
        auto createDecodeRuntime = [&]()
            -> std::pair<std::shared_ptr<CompiledModelRuntime>, std::string> {
            const bool qwenVL =
                m_frontend == "qwen3_vl" && m_inputShapes.size() == 15;
            const bool qwenText =
                m_frontend == "qwen3_text" && m_inputShapes.size() == 7;
            const bool qwenASR =
                m_frontend == "qwen3_asr" && m_inputShapes.size() == 9;
            const bool qwenTTS =
                m_frontend == "qwen3_tts" && m_inputShapes.size() == 7;
            if (!qwenVL && !qwenText && !qwenASR && !qwenTTS) {
                return {nullptr, {}};
            }
            const std::filesystem::path decodeModel =
                rootPath.parent_path() /
                (qwenVL ? "qwen_text_decode.py" :
                    (qwenTTS ? "talker_decode.py" : "decode.py"));
            const int keyIndex = qwenVL ? 11 : (qwenASR ? 5 : 3);
            const int valueIndex = qwenVL ? 12 : (qwenASR ? 6 : 4);
            const int tableIndex = qwenVL ? 13 : (qwenASR ? 7 : 5);
            std::vector<std::vector<int>> decodeShapes{
                {qwenTTS ? std::vector<int>{1, 16} : std::vector<int>{1, 1}},
                {qwenVL || qwenASR || qwenTTS ? 3 : 1, 1, 1},
                m_inputShapes[keyIndex], m_inputShapes[valueIndex],
                m_inputShapes[tableIndex], {1}, {1}};
            std::vector<std::string> decodeTypes{
                "int64", "int64", "bfloat16", "bfloat16", "int32", "int32", "int32"};
            FusionPartitionOptions decodePartitionOptions = m_partitionOptions;
            const char* decodeWorkspaceMb = std::getenv("GARNET_DECODE_WORKSPACE_MB");
            if (decodeWorkspaceMb && *decodeWorkspaceMb) {
                char* end = nullptr;
                const unsigned long long parsed = std::strtoull(decodeWorkspaceMb, &end, 10);
                if (end != decodeWorkspaceMb && *end == '\0' && parsed > 0) {
                    decodePartitionOptions.builderWorkspaceBytes = parsed << 20;
                }
            }
            const char* decodeOptimizationLevel = std::getenv("GARNET_DECODE_OPTIMIZATION_LEVEL");
            if (decodeOptimizationLevel && *decodeOptimizationLevel) {
                char* end = nullptr;
                const long parsed = std::strtol(decodeOptimizationLevel, &end, 10);
                if (end != decodeOptimizationLevel && *end == '\0' && parsed >= 0 && parsed <= 5) {
                    decodePartitionOptions.builderOptimizationLevel = static_cast<int>(parsed);
                }
            }
            auto decodeRuntime = std::make_shared<CompiledModelRuntime>(m_host);
            if (!decodeRuntime->Initialize(
                    decodeModel.string(),
                    (std::filesystem::path(m_cacheDirectory) / "decode").string(),
                    m_weightsLocation,
                    qwenVL ? "Qwen3TextDecode" : (qwenASR ? "Qwen3ASRDecode" :
                        (qwenTTS ? "Qwen3TTSTalkerDecode" : "Qwen3Decode")),
                    "",
                    decodeShapes,
                    decodeTypes,
                    decodePartitionOptions,
                    m_backend,
                    m_precision)) {
                X::Value decodeStatus(decodeRuntime->Status());
                return {nullptr, Lookup(decodeStatus, "error_message").ToString()};
            }
            return {std::move(decodeRuntime), {}};
        };
        auto prepareServingEngines = [&]() -> bool {
            const auto start = std::chrono::steady_clock::now();
            const bool needsDecodeRuntime =
                (m_frontend == "qwen3_vl" && m_inputShapes.size() == 15) ||
                (m_frontend == "qwen3_text" && m_inputShapes.size() == 7) ||
                (m_frontend == "qwen3_asr" && m_inputShapes.size() == 9);
            const bool needsTTSRuntimes =
                m_frontend == "qwen3_tts" && m_inputShapes.size() == 7;
            std::string preparationError;
            bool prepared = false;
            if (m_backend == "openvino") {
                if (m_enginePartitions.size() > 1) {
                    preparationError =
                        "partitioned OpenVINO execution is not implemented";
                }
                else {
                    OpenVINOBuilder builder;
                    prepared = builder.PrepareCapturedEngine(
                        m_enginePath, &m_weightIndex, preparationError);
                }
            }
            else {
                std::vector<std::shared_ptr<void>> owners;
                auto retain = [&](const std::string& path) {
                    auto owner = TRTBuilder::RetainCachedExecution(path, &m_weightIndex, preparationError);
                    if (!owner) return false;
                    owners.push_back(std::move(owner));
                    return true;
                };
                prepared = true;
                if (m_enginePartitions.size() > 1) {
                    for (const auto& partition : m_enginePartitions) {
                        if (!retain(partition.enginePath)) {
                            prepared = false;
                            break;
                        }
                    }
                }
                else prepared = retain(m_enginePath);
                if (prepared) m_trtExecutions = std::move(owners);
            }
            if (!prepared) {
                m_errorCode = "engine_preparation_failed";
                m_errorMessage = preparationError;
                return false;
            }
            m_enginesPrepared = true;
            std::pair<std::shared_ptr<CompiledModelRuntime>, std::string> decodeResult;
            if (needsDecodeRuntime || needsTTSRuntimes) decodeResult = createDecodeRuntime();
            if ((needsDecodeRuntime || needsTTSRuntimes) && !decodeResult.first) {
                m_errorCode = "decode_runtime_initialization_failed";
                m_errorMessage = decodeResult.second;
                return false;
            }
            m_decodeRuntime = std::move(decodeResult.first);
            if (needsTTSRuntimes) {
                int talkerHiddenSize = 0;
                try {
                    const auto ttsConfig = nlohmann::json::parse(ReadFile(
                        std::filesystem::path(m_weightsLocation) / "config.json"));
                    talkerHiddenSize = ttsConfig.at("talker_config")
                        .at("hidden_size").get<int>();
                }
                catch (const std::exception& exception) {
                    m_errorCode = "tts_config_invalid";
                    m_errorMessage = std::string(
                        "Qwen3-TTS talker hidden size is unavailable: ") +
                        exception.what();
                    return false;
                }
                if (talkerHiddenSize <= 0) {
                    m_errorCode = "tts_config_invalid";
                    m_errorMessage = "Qwen3-TTS talker hidden size is invalid";
                    return false;
                }
                auto auxiliary = std::make_shared<CompiledModelRuntime>(m_host);
                FusionPartitionOptions predictorOptions = m_partitionOptions;
                predictorOptions.builderOptimizationLevel = 0;
                if (!auxiliary->Initialize(
                        (rootPath.parent_path() / "code_predictor.py").string(),
                        (std::filesystem::path(m_cacheDirectory) / "code_predictor").string(),
                        m_weightsLocation, "Qwen3TTSCodePredictor", "",
                        {{1, 1, talkerHiddenSize}, {1, 1}},
                        {"float32", "int64"},
                        predictorOptions, m_backend, m_precision)) {
                    X::Value status(auxiliary->Status());
                    m_errorCode = "tts_code_predictor_initialization_failed";
                    m_errorMessage = Lookup(status, "error_message").ToString();
                    return false;
                }
                int codecFrames = 256;
                if (const char* value = std::getenv("GARNET_TTS_MAX_FRAMES")) {
                    codecFrames = std::max(1, std::atoi(value));
                }
                const std::filesystem::path codecWeights =
                    std::filesystem::path(m_weightsLocation) / "speech_tokenizer";
                auto codec = std::make_shared<CompiledModelRuntime>(m_host);
                if (!codec->Initialize(
                        (rootPath.parent_path() / "codec_decode.py").string(),
                        (std::filesystem::path(m_cacheDirectory) / "codec_decode").string(),
                        codecWeights.string(), "Qwen3TTSCodecDecode", "",
                        {{1, 16, codecFrames}}, {"int64"}, m_partitionOptions,
                        m_backend, m_precision)) {
                    X::Value status(codec->Status());
                    m_errorCode = "tts_codec_initialization_failed";
                    m_errorMessage = Lookup(status, "error_message").ToString();
                    return false;
                }
                m_auxRuntime = std::move(auxiliary);
                m_codecRuntime = std::move(codec);
            }
            m_enginePreparationMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            if (m_frontend == "qwen3_vl" || m_frontend == "qwen3_text" ||
                m_frontend == "qwen3_asr" || m_frontend == "qwen3_tts") {
                const auto frontendStart = std::chrono::steady_clock::now();
                std::string tokenizerError;
                if (!Tokenization::GetCachedQwenTokenizer(
                        m_weightsLocation, &tokenizerError)) {
                    m_errorCode = "frontend_preparation_failed";
                    m_errorMessage = tokenizerError;
                    return false;
                }
                m_frontendPreparationMs = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - frontendStart).count();
                m_frontendPrepared = true;
            }
            return true;
        };
        if (!inputShapes.empty() &&
            LoadGraphCache(
                graphCachePath,
                graphFingerprint,
                m_graphSummary,
                m_executionPlanJson)) {
            m_cudaGraphEnabled =
                ExecutionPlanRequestsCudaGraph(m_executionPlanJson);
            bool partitionCacheValid = ParseEnginePartitions(
                m_executionPlanJson, m_enginePartitions);
            for (const auto& partition : m_enginePartitions) {
                partitionCacheValid = partitionCacheValid &&
                    std::filesystem::is_regular_file(partition.enginePath);
            }
            if (partitionCacheValid && std::filesystem::is_regular_file(enginePath)) {
                ++m_diagnostics.graphCacheHits;
                m_ready = true;
                m_state = "engine_cache_loaded";
                if (!prepareServingEngines()) {
                    m_ready = false;
                    m_state = "failed";
                    return false;
                }
                engineGuard.release();
                return true;
            }
        }
        ++m_diagnostics.graphCacheMisses;

        try {
            // Keep each model's sibling modules in its own Python package.
            auto importRoot = rootPath.parent_path();
            std::string moduleName = rootPath.stem().string();
            while (std::filesystem::is_regular_file(importRoot / "__init__.py")) {
                moduleName = importRoot.filename().string() + "." + moduleName;
                importRoot = importRoot.parent_path();
            }
            static std::atomic<uint64_t> nextImportNamespace{1};
            m_importNamespace = "_garnet_model_" + graphFingerprint + "_" +
                std::to_string(nextImportNamespace.fetch_add(1, std::memory_order_relaxed));
            X::Module types(host, "types");
            X::Value package;
            if (!types["ModuleType"].Call({X::Value::String(host, m_importNamespace)}, package))
                throw std::runtime_error(x3_runtime_last_error(host->runtime));
            auto searchPaths = X::Value::List(host);
            if (!searchPaths.Append(X::Value::String(host, importRoot.string())) ||
                !package.SetAttr("__path__", searchPaths) ||
                !package.SetAttr("__package__", X::Value::String(host, m_importNamespace)))
                throw std::runtime_error(x3_runtime_last_error(host->runtime));
            X::Module sys(host, "sys");
            if (!sys["modules"].SetItem(m_importNamespace, package))
                throw std::runtime_error(x3_runtime_last_error(host->runtime));
            m_module = X::Module(host, (m_importNamespace + "." + moduleName).c_str());
            ++m_diagnostics.rootXExecutions;
            m_rootFunction = m_module[m_entryFunction.c_str()];
        }
        catch (const std::exception& exception) {
            m_state = "failed";
            m_errorCode = "root_model_import_failed";
            m_errorMessage = exception.what();
            return false;
        }
        if (!m_rootFunction.IsValid()) {
            m_state = "failed";
            m_errorCode = "root_forward_missing";
            m_errorMessage = "root Python module does not export " + m_entryFunction;
            return false;
        }

        if (!inputShapes.empty()) {
            X::ARGS symbolicInputs; symbolicInputs.reserve(inputShapes.size());
            for (size_t inputIndex = 0; inputIndex < inputShapes.size(); ++inputIndex) {
                std::vector<int64_t> shape;
                shape.reserve(inputShapes[inputIndex].size());
                for (const int size : inputShapes[inputIndex]) {
                    if (size <= 0) {
                        m_state = "failed";
                        m_errorCode = "invalid_symbolic_input_shape";
                        m_errorMessage = "symbolic input dimensions must be positive";
                        return false;
                    }
                    shape.push_back(size);
                }
                X3TensorDType dtype;
                const std::string dataType = inputDataTypes.empty()
                    ? "float32"
                    : inputDataTypes[inputIndex];
                if (dataType == "float32" || dataType == "fp32") {
                    dtype = X3_TENSOR_FLOAT32;
                }
                else if (dataType == "bfloat16" || dataType == "bf16") {
                    dtype = X3_TENSOR_BFLOAT16;
                }
                else if (dataType == "int64") {
                    dtype = X3_TENSOR_INT64;
                }
                else if (dataType == "int32") {
                    dtype = X3_TENSOR_INT32;
                }
                else {
                    m_state = "failed";
                    m_errorCode = "invalid_symbolic_input_dtype";
                    m_errorMessage = "unsupported symbolic input dtype: " + dataType;
                    return false;
                }
                auto tensor = X::Tensor::Input(host, ("input" + std::to_string(inputIndex)).c_str(), dtype, shape);
                symbolicInputs.push_back(NativeValue(host, tensor));
            }


            if (symbolicInputs.size() > 0) {
                X::ARGS rootArguments;
                X::Value modelSpec;
                X::Module builtins(host, "builtins");
                if (!builtins["getattr"].Call(
                        {m_module, X::Value::String(host, "GARNET_MODEL_SPEC"), X::Value(nullptr)}, modelSpec)) {
                    m_state = "failed";
                    m_errorCode = "invalid_model_spec";
                    m_errorMessage = x3_runtime_last_error(host->runtime);
                    return false;
                }
                if (modelSpec.IsDict()) {
                    X::Value spec(modelSpec);
                    X::Value argumentSpecsValue = Lookup(spec, "arguments");
                    if (!argumentSpecsValue.IsList()) {
                        m_state = "failed";
                        m_errorCode = "invalid_model_spec";
                        m_errorMessage = "GARNET_MODEL_SPEC.arguments must be a list";
                        return false;
                    }
                    X::Value argumentSpecs(argumentSpecsValue);
                    bool needsWeights = false;
                    bool needsConfig = false;
                    for (long long index = 0; index < argumentSpecs.Size(); ++index) {
                        X::Value itemValue = argumentSpecs.Get(index);
                        if (!itemValue.IsDict()) continue;
                        X::Value item(itemValue);
                        const std::string kind = Lookup(item, "kind").ToString();
                        needsWeights = needsWeights || kind == "weights";
                        needsConfig = needsConfig || kind == "config";
                    }

                    X::Value symbolicWeights;
                    if (needsWeights) {
                        symbolicWeights = BuildSymbolicWeights(host, m_weightIndex);
                        if (!symbolicWeights.IsDict() || m_weightIndex.TensorCount() == 0) {
                            m_state = "failed";
                            m_errorCode = "model_spec_weights_unavailable";
                            m_errorMessage = "model spec requires a native safetensors index";
                            return false;
                        }
                    }

                    X::Value configValue;
                    if (needsConfig) {
                        std::filesystem::path configPath(m_weightsLocation);
                        std::error_code configError;
                        if (std::filesystem::is_regular_file(configPath, configError)) {
                            configPath = configPath.parent_path();
                        }
                        configPath /= "config.json";
                        try {
                            configValue = JsonToXValue(host, nlohmann::json::parse(ReadFile(configPath)));
                        }
                        catch (const std::exception& exception) {
                            m_state = "failed";
                            m_errorCode = "model_config_invalid";
                            m_errorMessage = exception.what();
                            return false;
                        }
                    }

                    rootArguments.reserve(static_cast<int>(argumentSpecs.Size()));
                    size_t tensorIndex = 0;
                    for (long long argumentIndex = 0; argumentIndex < argumentSpecs.Size(); ++argumentIndex) {
                        X::Value argumentSpecValue = argumentSpecs.Get(argumentIndex);
                        if (!argumentSpecValue.IsDict()) {
                            m_state = "failed";
                            m_errorCode = "invalid_model_spec";
                            m_errorMessage = "each model argument spec must be a dictionary";
                            return false;
                        }
                        X::Value argumentSpec(argumentSpecValue);
                        const std::string kind = Lookup(argumentSpec, "kind").ToString();
                        if (kind == "tensor") {
                            if (tensorIndex >= symbolicInputs.size()) {
                                m_state = "failed";
                                m_errorCode = "model_input_count_mismatch";
                                m_errorMessage = "model spec requires more tensor inputs than input_shapes provides";
                                return false;
                            }
                            rootArguments.push_back(symbolicInputs[tensorIndex++]);
                        }
                        else if (kind == "weights") rootArguments.push_back(symbolicWeights);
                        else if (kind == "config") rootArguments.push_back(configValue);
                        else if (kind == "none") rootArguments.push_back(NativeValue(host, nullptr));
                        else if (kind == "bool") rootArguments.push_back(Lookup(argumentSpec, "value"));
                        else {
                            m_state = "failed";
                            m_errorCode = "invalid_model_spec";
                            m_errorMessage = "unsupported model argument kind: " + kind;
                            return false;
                        }
                    }

                    if (tensorIndex != symbolicInputs.size()) {
                        m_state = "failed";
                        m_errorCode = "model_input_count_mismatch";
                        m_errorMessage = "input_shapes provides more tensors than the model spec consumes";
                        return false;
                    }
                }
                else {
                    rootArguments = symbolicInputs;
                }
                {
                    ScopedCompiledGraphCapture capture;
                    X::KWARGS captureKwargs;
                    X::Value outputs;
                    if (!m_rootFunction.Call(rootArguments, captureKwargs, outputs)) {
                        m_state = "failed";
                        m_errorCode = "symbolic_graph_capture_failed";
                        m_errorMessage = x3_runtime_last_error(host->runtime);
                        return false;
                    }
                    m_graph = X::TensorGraph::IsGraph(outputs)
                        ? outputs : X::Value(X::TensorGraph(outputs));
                    if (!GetCompiledGraphCaptureError().empty()) {
                        m_state = "failed";
                        m_errorCode = "invalid_fusion_annotation";
                        m_errorMessage = GetCompiledGraphCaptureError();
                        return false;
                    }
                }
                if (!m_graph.IsValid()) {
                    m_state = "failed";
                    m_errorCode = "symbolic_graph_capture_failed";
                    m_errorMessage = "root fusion call did not return an xlang TensorGraph";
                    return false;
                }
                m_graphSummary = m_graph.ToString();
                TensorGraphCapture graphCapture(m_graph);
                if (!CaptureFusionGraph(graphCapture, m_partitionOptions, m_errorMessage)) {
                    m_state = "failed";
                    m_errorCode = "graph_partition_analysis_failed";
                    return false;
                }
                int partitionCount = 0;
                for (const auto& operation : GetCapturedTensorOperations()) {
                    partitionCount = std::max(
                        partitionCount, operation.candidatePartition + 1);
                }
                m_executionPlanJson = BuildExecutionPlanJson(
                    GetCapturedFusionRegions(),
                    GetCapturedTensorOperations(),
                    {},
                    m_partitionOptions);
                bool built = false;
                if (m_backend == "openvino") {
                    if (partitionCount > 1) {
                        m_errorMessage =
                            "partitioned OpenVINO graph lowering is not implemented";
                    }
                    else {
                        OpenVINOBuilder builder;
                        built = builder.BuildCapturedGraph(
                            m_graph, m_rootFunction, rootArguments,
                            symbolicInputs, &m_weightIndex, m_enginePath,
                            m_precision,
                            m_errorMessage);
                    }
                }
                else {
                    TRTBuilder builder(host);
                    builder.SetCapturedWeightProfile(m_precision);
                    builder.SetCapturedWorkspaceBytes(
                        m_partitionOptions.builderWorkspaceBytes);
                    builder.SetCapturedOptimizationLevel(
                        m_partitionOptions.builderOptimizationLevel);
                    built = partitionCount > 1
                        ? builder.BuildCapturedPartitions(
                            m_graph,
                            m_rootFunction,
                            rootArguments,
                            symbolicInputs,
                            &m_weightIndex,
                            m_enginePath,
                            GetCapturedTensorOperations(),
                            m_enginePartitions,
                            m_errorMessage)
                        : builder.BuildCapturedGraph(
                            m_graph,
                            m_rootFunction,
                            rootArguments,
                            symbolicInputs,
                            &m_weightIndex,
                            m_enginePath,
                            m_errorMessage);
                }
                if (!built) {
                    m_state = "failed";
                    m_errorCode = "generic_lowering_failed";
                    return false;
                }
                m_executionPlanJson = BuildExecutionPlanJson(
                    GetCapturedFusionRegions(),
                    GetCapturedTensorOperations(),
                    m_enginePartitions,
                    m_partitionOptions);
                m_cudaGraphEnabled =
                    ExecutionPlanRequestsCudaGraph(m_executionPlanJson);
                if (!StoreGraphCache(
                        graphCachePath,
                        graphFingerprint,
                        m_graphSummary,
                        m_executionPlanJson)) {
                    m_state = "failed";
                    m_errorCode = "graph_cache_write_failed";
                    m_errorMessage = "captured graph could not be stored atomically";
                    return false;
                }
                m_ready = true;
                m_state = "compiled_engine_ready";
                m_errorCode.clear();
                m_errorMessage.clear();
                if (!prepareServingEngines()) {
                    m_ready = false;
                    m_state = "failed";
                    return false;
                }
                engineGuard.release();
                return true;
            }
        }

        m_state = "root_function_loaded";
        m_errorCode = "symbolic_input_contract_not_implemented";
        m_errorMessage = "root fusion function loaded; symbolic model input construction is required";
        return true;
    }

    X::Value CompiledModelRuntime::Status() const
    {
        auto* host = m_host;
        std::lock_guard<std::mutex> guard(m_mutex);
        X::Value status = X::Value::Dict(host);
        status.SetItem("mode", NativeValue(host, "compiled_xmodel"));
        status.SetItem("state", NativeValue(host, m_state));
        status.SetItem("ready", NativeValue(host, m_ready));
        status.SetItem("root_xmodel", NativeValue(host, m_rootXModel));
        status.SetItem("cache_directory", NativeValue(host, m_cacheDirectory));
        status.SetItem("weights_location", NativeValue(host, m_weightsLocation));
        status.SetItem("entry_function", NativeValue(host, m_entryFunction));
        status.SetItem("frontend", NativeValue(host, m_frontend));
        status.SetItem("backend", NativeValue(host, m_backend));
        status.SetItem("precision", NativeValue(host, m_precision));
        status.SetItem("graph_summary", NativeValue(host, m_graphSummary));
        status.SetItem("scheduler", NativeValue(host, "cpu_control_gpu_execution"));
        status.SetItem("execution_plan_json", NativeValue(host, m_executionPlanJson));
        try {
            status.SetItem(
                "execution_plan",
                JsonToXValue(host, nlohmann::json::parse(m_executionPlanJson)));
        }
        catch (const std::exception&) {
            status.SetItem("execution_plan", NativeValue(host, nullptr));
        }
        status.SetItem("engine_path", NativeValue(host, m_enginePath));
        status.SetItem(
            "engine_partition_count",
            NativeValue(host, static_cast<long long>(
                m_enginePartitions.empty() ? 1 : m_enginePartitions.size())));
        X::Value partitionOptions = X::Value::Dict(host);
        partitionOptions.SetItem(
            "enable_preferred_boundaries",
            NativeValue(host, m_partitionOptions.enablePreferredBoundaries));
        partitionOptions.SetItem(
            "preferred_min_operations",
            NativeValue(host, m_partitionOptions.preferredMinOperations));
        partitionOptions.SetItem(
            "max_atomic_regions_per_partition",
            NativeValue(host, m_partitionOptions.maxAtomicRegionsPerPartition));
        partitionOptions.SetItem(
            "builder_workspace_bytes",
            NativeValue(host, m_partitionOptions.builderWorkspaceBytes));
        partitionOptions.SetItem(
            "builder_optimization_level",
            NativeValue(host, m_partitionOptions.builderOptimizationLevel));
        status.SetItem("partition_options", partitionOptions);
        status.SetItem("weight_tensor_count", NativeValue(host, static_cast<long long>(m_weightIndex.TensorCount())));
        status.SetItem("weight_tensor_bytes", NativeValue(host, static_cast<long long>(m_weightIndex.TensorBytes())));
        status.SetItem("weight_index_error", NativeValue(host, m_weightIndexError));
        status.SetItem("loaded_weight_count", NativeValue(host, static_cast<long long>(m_loadedWeights.size())));
        status.SetItem("loaded_weight_bytes", NativeValue(host, m_loadedWeightBytes));
        status.SetItem("engines_prepared", NativeValue(host, m_enginesPrepared));
        status.SetItem("engine_prepare_ms", NativeValue(host, m_enginePreparationMs));
        status.SetItem("frontend_prepared", NativeValue(host, m_frontendPrepared));
        status.SetItem("frontend_prepare_ms", NativeValue(host, m_frontendPreparationMs));
        status.SetItem("error_code", NativeValue(host, m_errorCode));
        status.SetItem("error_message", NativeValue(host, m_errorMessage));

        X::Value counters = X::Value::Dict(host);
        counters.SetItem("root_x_executions", NativeValue(host, m_diagnostics.rootXExecutions));
        counters.SetItem("hardcoded_qwen_runner_calls", NativeValue(host, m_diagnostics.hardcodedQwenRunnerCalls));
        counters.SetItem("python_subgraph_calls", NativeValue(host, m_diagnostics.pythonSubgraphCalls));
        counters.SetItem("direct_internal_export_calls", NativeValue(host, m_diagnostics.directInternalExportCalls));
        counters.SetItem("cpu_tensor_intermediates", NativeValue(host, m_diagnostics.cpuTensorIntermediates));
        counters.SetItem("graph_cache_hits", NativeValue(host, m_diagnostics.graphCacheHits));
        counters.SetItem("graph_cache_misses", NativeValue(host, m_diagnostics.graphCacheMisses));
        status.SetItem("forbidden_path_counters", counters);
        return status;
    }

    X::Value CompiledModelRuntime::Forward(X::Value request)
    {
        auto* host = m_host;
        std::lock_guard<std::mutex> guard(m_mutex);
        const auto requestStart = std::chrono::steady_clock::now();
        X::Value result = X::Value::Dict(host);
        if (!m_ready) {
            result.SetItem("status", NativeValue(host, "error"));
            result.SetItem("error_code", NativeValue(host, "compiled_graph_not_ready"));
            result.SetItem("error_message", NativeValue(host, m_errorMessage));
            return result;
        }

        if (!request.IsDict()) {
            result.SetItem("status", NativeValue(host, "error"));
            result.SetItem("error_code", NativeValue(host, "invalid_compiled_request"));
            result.SetItem("error_message", NativeValue(host, "compiled forward requires a request dictionary"));
            return result;
        }
        X::Value requestDict(request);
        const X::Value penaltyOption = Lookup(requestDict, "repetition_penalty");
        const float generationPenalty = penaltyOption.IsValid()
            ? static_cast<float>(penaltyOption.ToDouble()) : 1.0f;
        if (!std::isfinite(generationPenalty) || generationPenalty < 1.0f ||
            (generationPenalty != 1.0f && m_backend != "tensorrt")) {
            result.SetItem("status", NativeValue(host, "error"));
            result.SetItem("error_code", NativeValue(host, "invalid_repetition_penalty"));
            return result;
        }
        X::Value inputs = Lookup(requestDict, "inputs");
        QwenVLCompiledInputs frontendInputs;
        QwenTextCompiledInputs textFrontendInputs;
        QwenASRCompiledInputs asrFrontendInputs;
        QwenTTSCompiledInputs ttsFrontendInputs;
        bool frontendActive = false;
        bool frontendIsVL = false;
        bool frontendIsASR = false;
        bool frontendIsTTS = false;
        int frontendPromptTokenCount = 0;
        long long frontendPositionDelta = 0;
        if (!inputs.IsList() && m_frontend == "qwen3_vl") {
            X::Value imageSource = Lookup(requestDict, "image");
            if (!imageSource.IsValid()) imageSource = Lookup(requestDict, "image_path");
            const std::string prompt = Lookup(requestDict, "prompt").ToString();
            const int minPixels = Lookup(requestDict, "min_pixels").IsValid()
                ? static_cast<int>(Lookup(requestDict, "min_pixels").ToLongLong())
                : 65536;
            const int maxPixels = Lookup(requestDict, "max_pixels").IsValid()
                ? static_cast<int>(Lookup(requestDict, "max_pixels").ToLongLong())
                : 65536;
            frontendInputs = BuildQwenVLCompiledInputs(host,
                m_weightsLocation, imageSource, prompt, minPixels, maxPixels,
                m_inputShapes, m_reusablePrefillKeyCache,
                m_reusablePrefillValueCache);
            if (!frontendInputs.inputs.IsList()) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "compiled_frontend_failed"));
                result.SetItem("error_message", NativeValue(host, frontendInputs.error));
                return result;
            }
            {
                X::Value frontendList(frontendInputs.inputs);
                if (!(X::Tensor::IsTensor(m_reusablePrefillKeyCache))) {
                    m_reusablePrefillKeyCache = frontendList.Get(11);
                }
                if (!(X::Tensor::IsTensor(m_reusablePrefillValueCache))) {
                    m_reusablePrefillValueCache = frontendList.Get(12);
                }
            }
            inputs = frontendInputs.inputs;
            frontendActive = true;
            frontendIsVL = true;
            frontendPromptTokenCount = frontendInputs.promptTokenCount;
            frontendPositionDelta = frontendInputs.mropePositionDelta;
        }
        else if (!inputs.IsList() && m_frontend == "qwen3_text") {
            const std::string prompt = Lookup(requestDict, "prompt").ToString();
            const bool enableThinking =
                Lookup(requestDict, "enable_thinking").IsValid() &&
                Lookup(requestDict, "enable_thinking").ToLongLong() != 0;
            textFrontendInputs = BuildQwenTextCompiledInputs(host,
                m_weightsLocation, prompt, enableThinking, m_inputShapes,
                m_reusablePrefillKeyCache, m_reusablePrefillValueCache,
                m_backend == "openvino");
            if (!textFrontendInputs.inputs.IsList()) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "compiled_frontend_failed"));
                result.SetItem(
                    "error_message", NativeValue(host, textFrontendInputs.error));
                return result;
            }
            {
                X::Value frontendList(textFrontendInputs.inputs);
                if (!(X::Tensor::IsTensor(m_reusablePrefillKeyCache))) {
                    m_reusablePrefillKeyCache = frontendList.Get(3);
                }
                if (!(X::Tensor::IsTensor(m_reusablePrefillValueCache))) {
                    m_reusablePrefillValueCache = frontendList.Get(4);
                }
            }
            inputs = textFrontendInputs.inputs;
            frontendActive = true;
            frontendPromptTokenCount =
                textFrontendInputs.promptTokenCount;
        }
        else if (!inputs.IsList() && m_frontend == "qwen3_asr") {
            X::Value audioSource = Lookup(requestDict, "audio");
            if (!audioSource.IsValid()) audioSource = Lookup(requestDict, "audio_path");
            asrFrontendInputs = BuildQwenASRCompiledInputs(host,
                m_weightsLocation, audioSource,
                Lookup(requestDict, "context").ToString(),
                Lookup(requestDict, "language").ToString(), m_inputShapes,
                m_reusablePrefillKeyCache, m_reusablePrefillValueCache);
            if (!asrFrontendInputs.inputs.IsList()) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "compiled_frontend_failed"));
                result.SetItem("error_message", NativeValue(host, asrFrontendInputs.error));
                return result;
            }
            {
                X::Value frontendList(asrFrontendInputs.inputs);
                if (!(X::Tensor::IsTensor(m_reusablePrefillKeyCache))) {
                    m_reusablePrefillKeyCache = frontendList.Get(5);
                }
                if (!(X::Tensor::IsTensor(m_reusablePrefillValueCache))) {
                    m_reusablePrefillValueCache = frontendList.Get(6);
                }
            }
            inputs = asrFrontendInputs.inputs;
            frontendActive = true;
            frontendIsASR = true;
            frontendPromptTokenCount = asrFrontendInputs.promptTokenCount;
        }
        else if (!inputs.IsList() && m_frontend == "qwen3_tts") {
            ttsFrontendInputs = BuildQwenTTSCompiledInputs(host,
                m_weightsLocation, Lookup(requestDict, "text").ToString(),
                Lookup(requestDict, "speaker").ToString(),
                Lookup(requestDict, "language").ToString(),
                Lookup(requestDict, "instruct").ToString(), m_inputShapes,
                m_reusablePrefillKeyCache, m_reusablePrefillValueCache);
            if (!ttsFrontendInputs.inputs.IsList()) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "compiled_frontend_failed"));
                result.SetItem("error_message", NativeValue(host, ttsFrontendInputs.error));
                return result;
            }
            {
                X::Value frontendList(ttsFrontendInputs.inputs);
                if (!(X::Tensor::IsTensor(m_reusablePrefillKeyCache))) {
                    m_reusablePrefillKeyCache = frontendList.Get(3);
                    m_reusablePrefillValueCache = frontendList.Get(4);
                }
            }
            inputs = ttsFrontendInputs.inputs;
            frontendActive = true;
            frontendIsTTS = true;
            frontendPromptTokenCount = ttsFrontendInputs.promptTokenCount;
        }
        if (!inputs.IsList() || inputs.Size() != static_cast<long long>(m_inputShapes.size())) {
            result.SetItem("status", NativeValue(host, "error"));
            result.SetItem("error_code", NativeValue(host, "compiled_input_count_mismatch"));
            result.SetItem("error_message", NativeValue(host,
                "inputs must contain exactly " + std::to_string(m_inputShapes.size()) + " tensors"));
            return result;
        }
        std::string executionError;
        for (size_t index = 0; index < m_inputShapes.size(); ++index) {
            const auto input = inputs.Get(static_cast<long long>(index));
            if (!X::Tensor::IsTensor(input)) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "compiled_input_not_tensor"));
                return result;
            }
            const X::Tensor tensor(input);
            const auto& info = tensor.Info();
            const auto& shape = m_inputShapes[index];
            bool matches = static_cast<size_t>(info.rank) == shape.size();
            for (size_t dimension = 0; matches && dimension < shape.size(); ++dimension)
                matches = info.shape[dimension] == shape[dimension];
            if (!matches) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "compiled_input_shape_mismatch"));
                return result;
            }
        }
        const X::Value reusableOutput =
            Lookup(requestDict, "reuse_output").IsValid() &&
            Lookup(requestDict, "reuse_output").ToLongLong() != 0
                ? m_reusableExecutionOutput
                : NativeValue(host);
        X::Value output;
        if (m_backend == "openvino") {
            OpenVINOBuilder builder;
            const bool resetOpenVINOState =
                Lookup(requestDict, "reset_openvino_state").IsValid() &&
                Lookup(requestDict, "reset_openvino_state").ToLongLong() != 0;
            output = builder.RunCapturedEngine(
                m_enginePath, inputs, &m_weightIndex, reusableOutput,
                resetOpenVINOState,
                m_openVinoSessionId,
                executionError);
        }
        else {
            TRTBuilder builder(host);
            output = m_enginePartitions.size() > 1
                ? builder.RunCapturedPartitions(
                    m_enginePartitions,
                    inputs,
                    &m_weightIndex,
                    reusableOutput,
                    executionError)
                : builder.RunCapturedEngine(
                    m_enginePath,
                    inputs,
                    &m_weightIndex,
                    reusableOutput,
                    m_cudaGraphEnabled,
                    executionError);
        }
        if (!(X::Tensor::IsTensor(output))) {
            result.SetItem("status", NativeValue(host, "error"));
            result.SetItem("error_code", NativeValue(host, "compiled_engine_execution_failed"));
            result.SetItem("error_message", NativeValue(host, executionError));
            return result;
        }
        if (Lookup(requestDict, "reuse_output").IsValid() &&
            Lookup(requestDict, "reuse_output").ToLongLong() != 0 &&
            !(X::Tensor::IsTensor(m_reusableExecutionOutput))) {
            m_reusableExecutionOutput = output;
        }
        result.SetItem("status", NativeValue(host, "ok"));
        if (frontendIsTTS) {
            if (!m_decodeRuntime || !m_auxRuntime || !m_codecRuntime ||
                m_backend != "tensorrt") {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "compiled_tts_runtime_unavailable"));
                return result;
            }
            const int requestedFrames = Lookup(requestDict, "max_audio_frames").IsValid()
                ? std::max(1, static_cast<int>(
                    Lookup(requestDict, "max_audio_frames").ToLongLong()))
                : 256;
            int profileFrames = 256;
            if (const char* value = std::getenv("GARNET_TTS_MAX_FRAMES")) {
                profileFrames = std::max(1, std::atoi(value));
            }
            const int maxFrames = std::min(requestedFrames, profileFrames);
            const int vocabSize = 3072;
            const bool stochastic = !Lookup(requestDict, "do_sample").IsValid() ||
                Lookup(requestDict, "do_sample").ToLongLong() != 0;
            const int topK = Lookup(requestDict, "top_k").IsValid()
                ? std::clamp(static_cast<int>(Lookup(requestDict, "top_k").ToLongLong()), 1, vocabSize)
                : 50;
            const float temperature = Lookup(requestDict, "temperature").IsValid()
                ? std::max(0.01F, static_cast<float>(Lookup(requestDict, "temperature").ToDouble()))
                : 0.9F;
            const float repetitionPenalty = Lookup(requestDict, "repetition_penalty").IsValid()
                ? std::max(1.0F, static_cast<float>(Lookup(requestDict, "repetition_penalty").ToDouble()))
                : 1.05F;
            const uint64_t seed = Lookup(requestDict, "seed").IsValid()
                ? static_cast<uint64_t>(Lookup(requestDict, "seed").ToLongLong())
                : static_cast<uint64_t>(std::chrono::high_resolution_clock::now()
                    .time_since_epoch().count());
            std::mt19937_64 random(seed);
            std::vector<long long> firstCodeHistory;
            auto samplePacked = [&](X::Value packedValue, int row,
                                    long long& token, X::Value& hiddenValue,
                                    std::string& errorText) -> bool {
                if (!(X::Tensor::IsTensor(packedValue))) {
                    errorText = "talker returned no packed hidden/logit tensor";
                    return false;
                }
                X::Tensor packed(packedValue);
                if (packed.Info().rank != 3 ||
                    packed.Info().shape[2] <= vocabSize ||
                    packed.Info().shape[2] > std::numeric_limits<int>::max() ||
                    row < 0 || row >= packed.Info().shape[1]) {
                    errorText = "talker packed output has incompatible dimensions";
                    return false;
                }
                const int hiddenSize = static_cast<int>(packed.Info().shape[2]) - vocabSize;
                const size_t elementBytes = packed.Info().dtype ==
                    X3_TENSOR_FLOAT32 ? sizeof(float) : sizeof(unsigned short);
                const size_t rowElements = hiddenSize + vocabSize;
                const char* rowMemory = static_cast<const char*>(
                    TensorHelper::GetGPUMemory(packed)) +
                    static_cast<size_t>(row) * rowElements * elementBytes;
                auto hidden = TensorHelper::CreateGPU(host, packed.Info().dtype, {1, 1, hiddenSize});
                auto packedUse = TensorHelper::AcquireGPU({{packed, X3_TENSOR_READ}, {hidden, X3_TENSOR_WRITE}});
                void* hiddenMemory = TensorHelper::GetGPUMemory(hidden);
                if (cudaMemcpyAsync(hiddenMemory, rowMemory, hiddenSize * elementBytes,
                        cudaMemcpyDeviceToDevice, cudaStreamPerThread) != cudaSuccess) {
                    errorText = "talker hidden-state extraction failed";
                    return false;
                }
                hiddenValue = NativeValue(host, hidden);
                std::vector<unsigned char> raw(
                    static_cast<size_t>(vocabSize) * elementBytes);
                cudaError_t status = cudaMemcpyAsync(
                    raw.data(), rowMemory + hiddenSize * elementBytes,
                    raw.size(), cudaMemcpyDeviceToHost, cudaStreamPerThread);
                const cudaError_t copyCompletion = cudaStreamSynchronize(cudaStreamPerThread);
                if (status == cudaSuccess) status = copyCompletion;
                packedUse.Finish();
                if (status != cudaSuccess) {
                    errorText = cudaGetErrorString(status);
                    return false;
                }
                std::vector<float> logits(static_cast<size_t>(vocabSize));
                if (packed.Info().dtype == X3_TENSOR_FLOAT32) {
                    std::memcpy(logits.data(), raw.data(), raw.size());
                }
                else {
                    const auto* bits = reinterpret_cast<const uint16_t*>(raw.data());
                    for (int index = 0; index < vocabSize; ++index) {
                        const uint32_t expanded = static_cast<uint32_t>(bits[index]) << 16;
                        std::memcpy(&logits[index], &expanded, sizeof(float));
                    }
                }
                for (long long previous : firstCodeHistory) {
                    if (previous >= 0 && previous < vocabSize) {
                        float& value = logits[static_cast<size_t>(previous)];
                        value = value < 0.0F
                            ? value * repetitionPenalty : value / repetitionPenalty;
                    }
                }
                std::vector<int> order(static_cast<size_t>(vocabSize));
                for (int index = 0; index < vocabSize; ++index) order[index] = index;
                std::partial_sort(order.begin(), order.begin() + topK, order.end(),
                    [&](int left, int right) { return logits[left] > logits[right]; });
                if (!stochastic) token = order.front();
                else {
                    const float maximum = logits[order.front()] / temperature;
                    std::vector<double> probabilities(static_cast<size_t>(topK));
                    for (int index = 0; index < topK; ++index) {
                        probabilities[index] = std::exp(
                            logits[order[index]] / temperature - maximum);
                    }
                    std::discrete_distribution<int> distribution(
                        probabilities.begin(), probabilities.end());
                    token = order[distribution(random)];
                }
                return true;
            };

            X::Value activeInputs(inputs);
            X::Value packed = output;
            std::vector<int64_t> codes;
            codes.reserve(static_cast<size_t>(maxFrames) * 16);
            const auto decodeStart = std::chrono::steady_clock::now();
            for (int frame = 0; frame < maxFrames; ++frame) {
                long long firstCode = -1;
                X::Value hidden;
                std::string ttsError;
                const int packedRow = frame == 0
                    ? frontendPromptTokenCount - 1 : 0;
                if (!samplePacked(packed, packedRow, firstCode, hidden, ttsError)) {
                    result.SetItem("status", NativeValue(host, "error"));
                    result.SetItem("error_code", NativeValue(host, "compiled_tts_sampling_failed"));
                    result.SetItem("error_message", NativeValue(host, ttsError));
                    return result;
                }
                if (firstCode == ttsFrontendInputs.codecEosTokenId) break;
                firstCodeHistory.push_back(firstCode);
                X::Value firstCodeTensor = MakeCudaTensorFromHost(host,
                    X3_TENSOR_INT64, {1, 1}, &firstCode,
                    sizeof(firstCode));
                X::Value predictorInputs = X::Value::List(host);
                predictorInputs.Append(hidden);
                predictorInputs.Append(firstCodeTensor);
                X::Value predictorRequest = X::Value::Dict(host);
                predictorRequest.SetItem("inputs", NativeValue(host, predictorInputs));
                X::Value predictorValue = m_auxRuntime->Forward(predictorRequest);
                if (!predictorValue.IsDict()) {
                    result.SetItem("status", NativeValue(host, "error"));
                    result.SetItem("error_code", NativeValue(host, "compiled_tts_predictor_failed"));
                    return result;
                }
                X::Value predictorResult(predictorValue);
                if (Lookup(predictorResult, "status").ToString() != "ok") return predictorValue;
                X::Value frameCodes = Lookup(predictorResult, "output");
                if (!(X::Tensor::IsTensor(frameCodes))) {
                    result.SetItem("status", NativeValue(host, "error"));
                    result.SetItem("error_code", NativeValue(host, "compiled_tts_predictor_output_invalid"));
                    return result;
                }
                int64_t hostCodes[16]{};
                X::Tensor frameCodeTensor(frameCodes);
                auto frameUse = TensorHelper::AcquireGPU(frameCodeTensor);
                cudaError_t copyStatus = cudaMemcpyAsync(
                    hostCodes, TensorHelper::GetGPUMemory(frameCodeTensor),
                    sizeof(hostCodes), cudaMemcpyDeviceToHost, cudaStreamPerThread);
                const cudaError_t copyCompletion = cudaStreamSynchronize(cudaStreamPerThread);
                if (copyStatus == cudaSuccess) copyStatus = copyCompletion;
                frameUse.Finish();
                if (copyStatus != cudaSuccess) {
                    result.SetItem("status", NativeValue(host, "error"));
                    result.SetItem("error_code", NativeValue(host, "compiled_tts_code_download_failed"));
                    return result;
                }
                codes.insert(codes.end(), hostCodes, hostCodes + 16);
                if (frame + 1 >= maxFrames) break;

                const int slot = frontendPromptTokenCount + frame;
                const int context = slot + 1;
                const int64_t positions[3] = {slot, slot, slot};
                X::Value positionTensor = MakeCudaTensorFromHost(host,
                    X3_TENSOR_INT64, {3, 1, 1}, positions,
                    sizeof(positions));
                X::Value contextTensor = MakeCudaTensorFromHost(host,
                    X3_TENSOR_INT32, {1}, &context, sizeof(context));
                X::Value slotTensor = MakeCudaTensorFromHost(host,
                    X3_TENSOR_INT32, {1}, &slot, sizeof(slot));
                X::Value talkerInputs = X::Value::List(host);
                talkerInputs.Append(frameCodes);
                talkerInputs.Append(positionTensor);
                talkerInputs.Append(activeInputs.Get(3));
                talkerInputs.Append(activeInputs.Get(4));
                talkerInputs.Append(activeInputs.Get(5));
                talkerInputs.Append(contextTensor);
                talkerInputs.Append(slotTensor);
                X::Value talkerRequest = X::Value::Dict(host);
                talkerRequest.SetItem("inputs", NativeValue(host, talkerInputs));
                talkerRequest.SetItem("reuse_output", NativeValue(host, 1));
                X::Value talkerValue = m_decodeRuntime->Forward(talkerRequest);
                if (!talkerValue.IsDict()) {
                    result.SetItem("status", NativeValue(host, "error"));
                    result.SetItem("error_code", NativeValue(host, "compiled_tts_talker_decode_failed"));
                    return result;
                }
                X::Value talkerResult(talkerValue);
                if (Lookup(talkerResult, "status").ToString() != "ok") return talkerValue;
                packed = Lookup(talkerResult, "output");
            }
            if (codes.empty()) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "compiled_tts_generated_no_audio"));
                return result;
            }
            const int frameCount = static_cast<int>(codes.size() / 16);
            std::vector<int64_t> paddedCodes(
                static_cast<size_t>(16 * profileFrames), 0);
            for (int frame = 0; frame < frameCount; ++frame) {
                for (int group = 0; group < 16; ++group) {
                    paddedCodes[static_cast<size_t>(group * profileFrames + frame)] =
                        codes[static_cast<size_t>(frame * 16 + group)];
                }
            }
            X::Value codecInput = MakeCudaTensorFromHost(host,
                X3_TENSOR_INT64, {1, 16, profileFrames},
                paddedCodes.data(), paddedCodes.size() * sizeof(int64_t));
            X::Value codecInputs = X::Value::List(host);
            codecInputs.Append(codecInput);
            X::Value codecRequest = X::Value::Dict(host);
            codecRequest.SetItem("inputs", NativeValue(host, codecInputs));
            X::Value codecValue = m_codecRuntime->Forward(codecRequest);
            if (!codecValue.IsDict()) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "compiled_tts_codec_decode_failed"));
                return result;
            }
            X::Value codecResult(codecValue);
            if (Lookup(codecResult, "status").ToString() != "ok") return codecValue;
            X::Value fullAudioValue = Lookup(codecResult, "output");
            if (!(X::Tensor::IsTensor(fullAudioValue))) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "compiled_tts_waveform_invalid"));
                return result;
            }
            X::Tensor fullAudio(fullAudioValue);
            const int validSamples = frameCount * 1920;
            const size_t audioElementBytes = fullAudio.Info().dtype ==
                X3_TENSOR_FLOAT32 ? sizeof(float) : sizeof(unsigned short);
            auto audio = TensorHelper::CreateGPU(host, fullAudio.Info().dtype, {1, validSamples});
            auto audioUse = TensorHelper::AcquireGPU({{fullAudio, X3_TENSOR_READ}, {audio, X3_TENSOR_WRITE}});
            void* audioMemory = TensorHelper::GetGPUMemory(audio);
            cudaError_t audioStatus = cudaMemcpyAsync(
                audioMemory, TensorHelper::GetGPUMemory(fullAudio),
                static_cast<size_t>(validSamples) * audioElementBytes,
                cudaMemcpyDeviceToDevice, cudaStreamPerThread);
            audioUse.Finish();
            if (audioStatus != cudaSuccess) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "compiled_tts_waveform_trim_failed"));
                return result;
            }
            result.SetItem("audio", NativeValue(host, audio));
            result.SetItem("sample_rate", NativeValue(host, 24000));
            result.SetItem("audio_frame_count", NativeValue(host, frameCount));
            result.SetItem("audio_sample_count", NativeValue(host, validSamples));
            result.SetItem("audio_duration_seconds", NativeValue(host, frameCount / 12.5));
            result.SetItem("codec_codes", NativeValue(host, static_cast<long long>(codes.size())));
            result.SetItem("prompt_token_count", NativeValue(host, frontendPromptTokenCount));
            result.SetItem("decode_ms", NativeValue(host, std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - decodeStart).count()));
            result.SetItem("total_ms", NativeValue(host, std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - requestStart).count()));
            return result;
        }
        const int requestedNewTokens = Lookup(requestDict, "max_new_tokens").IsValid()
            ? std::max(0, static_cast<int>(Lookup(requestDict, "max_new_tokens").ToLongLong()))
            : 0;
        const std::string sampleMode = Lookup(requestDict, "sample").IsValid()
            ? Lookup(requestDict, "sample").ToString()
            : std::string();
        const bool sampleBatch = sampleMode == "greedy_batch";
        const bool sampleGreedy = requestedNewTokens > 0 ||
            sampleMode == "greedy" || sampleBatch;
        long long sampledTokenId = -1;
        int generationVocabSize = 0;
        if (sampleGreedy) {
            X::Tensor logits(output);
            if (logits.Info().rank != 3) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "compiled_sampling_shape_invalid"));
                return result;
            }
            const int tokenRows = static_cast<int>(logits.Info().shape[1]);
            const int vocabSize = static_cast<int>(logits.Info().shape[2]);
            generationVocabSize = vocabSize;
            const int batchSize = static_cast<int>(logits.Info().shape[0]);
            if (sampleBatch) {
                const int sampleRows = batchSize * tokenRows;
                if (sampleRows <= 0) {
                    result.SetItem("status", NativeValue(host, "error"));
                    result.SetItem(
                        "error_code",
                        NativeValue(host, "compiled_sampling_shape_invalid"));
                    return result;
                }
                cudaError_t sampleStatus = cudaSuccess;
                if (m_sampleCapacity < sampleRows) {
                    if (m_sampleTokenDevice) cudaFree(m_sampleTokenDevice);
                    if (m_sampleValueDevice) cudaFree(m_sampleValueDevice);
                    m_sampleTokenDevice = nullptr;
                    m_sampleValueDevice = nullptr;
                    sampleStatus = cudaMalloc(
                        &m_sampleTokenDevice,
                        static_cast<size_t>(sampleRows) * sizeof(long long));
                    if (sampleStatus == cudaSuccess) {
                        sampleStatus = cudaMalloc(
                            &m_sampleValueDevice,
                            static_cast<size_t>(sampleRows) * sizeof(float));
                    }
                    if (sampleStatus == cudaSuccess) {
                        m_sampleCapacity = sampleRows;
                    }
                }
                const void* logitsDevice = TensorHelper::GetGPUMemory(logits);
                auto logitsUse = TensorHelper::AcquireGPU(logits);
                if (sampleStatus == cudaSuccess &&
                    logits.Info().dtype == X3_TENSOR_FLOAT32) {
                    sampleStatus = runLogitsTop1BatchFP32(
                        static_cast<const float*>(logitsDevice),
                        static_cast<long long*>(m_sampleTokenDevice),
                        static_cast<float*>(m_sampleValueDevice),
                        sampleRows, vocabSize, cudaStreamPerThread);
                }
                else if (sampleStatus == cudaSuccess &&
                    logits.Info().dtype == X3_TENSOR_BFLOAT16) {
                    sampleStatus = runLogitsTop1BatchBF16(
                        static_cast<const bfloat16*>(logitsDevice),
                        static_cast<long long*>(m_sampleTokenDevice),
                        static_cast<float*>(m_sampleValueDevice),
                        sampleRows, vocabSize, cudaStreamPerThread);
                }
                else if (sampleStatus == cudaSuccess) {
                    sampleStatus = cudaErrorInvalidValue;
                }
                std::vector<long long> tokenIds(
                    static_cast<size_t>(sampleRows), -1);
                std::vector<float> tokenValues(
                    static_cast<size_t>(sampleRows), 0.0f);
                if (sampleStatus == cudaSuccess) {
                    sampleStatus = cudaMemcpyAsync(
                        tokenIds.data(), m_sampleTokenDevice,
                        tokenIds.size() * sizeof(long long),
                        cudaMemcpyDeviceToHost, cudaStreamPerThread);
                }
                if (sampleStatus == cudaSuccess) {
                    sampleStatus = cudaMemcpyAsync(
                        tokenValues.data(), m_sampleValueDevice,
                        tokenValues.size() * sizeof(float),
                        cudaMemcpyDeviceToHost, cudaStreamPerThread);
                }
                const cudaError_t sampleCompletion = cudaStreamSynchronize(cudaStreamPerThread);
                if (sampleStatus == cudaSuccess) sampleStatus = sampleCompletion;
                logitsUse.Finish();
                if (sampleStatus != cudaSuccess) {
                    result.SetItem("status", NativeValue(host, "error"));
                    result.SetItem(
                        "error_code",
                        NativeValue(host, "compiled_gpu_batch_sampling_failed"));
                    result.SetItem(
                        "error_message",
                        NativeValue(host, cudaGetErrorString(sampleStatus)));
                    return result;
                }
                X::Value tokenList = X::Value::List(host);
                X::Value valueList = X::Value::List(host);
                for (int row = 0; row < sampleRows; ++row) {
                    tokenList.Append(NativeValue(host, tokenIds[row]));
                    valueList.Append(NativeValue(host, tokenValues[row]));
                }
                result.SetItem("token_ids", NativeValue(host, tokenList));
                result.SetItem("token_values", NativeValue(host, valueList));
                sampledTokenId = tokenIds.front();
                const auto firstTokenReady = std::chrono::steady_clock::now();
                result.SetItem("time_to_first_token_ms", NativeValue(host,
                    std::chrono::duration<double, std::milli>(
                        firstTokenReady - requestStart).count()));
            }
            else {
            int selectedRow = tokenRows - 1;
            if (frontendPromptTokenCount > 0) {
                selectedRow =
                    std::min(frontendPromptTokenCount, tokenRows) - 1;
            }
            else if (Lookup(requestDict, "sample_row").IsValid()) {
                selectedRow = static_cast<int>(Lookup(requestDict, "sample_row").ToLongLong());
            }
            if (selectedRow < 0 || selectedRow >= tokenRows) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "compiled_sampling_row_invalid"));
                return result;
            }
            long long tokenId = -1;
            float tokenValue = 0.0f;
            if (m_backend == "openvino") {
                auto logitsUse = logits.Acquire();
                if (logits.Info().device_type != 0 ||
                    !logits.Info().data) {
                    result.SetItem("status", NativeValue(host, "error"));
                    result.SetItem(
                        "error_code",
                        NativeValue(host, "compiled_cpu_sampling_failed"));
                    return result;
                }
                if (vocabSize == 1 &&
                    logits.Info().dtype ==
                        X3_TENSOR_INT64) {
                    tokenId = reinterpret_cast<const int64_t*>(
                        logits.Info().data)[selectedRow];
                    tokenValue = 0.0F;
                }
                else {
                    tokenValue = -std::numeric_limits<float>::infinity();
                    for (int token = 0; token < vocabSize; ++token) {
                        float value = 0.0F;
                        const size_t offset =
                            static_cast<size_t>(selectedRow) * vocabSize +
                            token;
                        if (logits.Info().dtype ==
                            X3_TENSOR_FLOAT32) {
                            value = reinterpret_cast<const float*>(
                                logits.Info().data)[offset];
                        }
                        else if (logits.Info().dtype ==
                            X3_TENSOR_BFLOAT16) {
                            const uint16_t bits =
                                reinterpret_cast<const uint16_t*>(
                                    logits.Info().data)[offset];
                            const uint32_t expanded =
                                static_cast<uint32_t>(bits) << 16;
                            std::memcpy(&value, &expanded, sizeof(value));
                        }
                        else {
                            result.SetItem("status", NativeValue(host, "error"));
                            result.SetItem(
                                "error_code",
                                NativeValue(host,
                                    "compiled_cpu_sampling_dtype_invalid"));
                            return result;
                        }
                        if (value > tokenValue) {
                            tokenValue = value;
                            tokenId = token;
                        }
                    }
                }
            }
            else {
                cudaError_t sampleStatus = cudaSuccess;
                if (!m_sampleTokenDevice) {
                    sampleStatus = cudaMalloc(
                        &m_sampleTokenDevice, sizeof(long long));
                }
                if (sampleStatus == cudaSuccess && !m_sampleValueDevice) {
                    sampleStatus = cudaMalloc(
                        &m_sampleValueDevice, sizeof(float));
                }
                auto* deviceTokenId =
                    static_cast<long long*>(m_sampleTokenDevice);
                auto* deviceTokenValue =
                    static_cast<float*>(m_sampleValueDevice);
                const void* logitsDevice = TensorHelper::GetGPUMemory(logits);
                X::Value seenValue = Lookup(requestDict, "_sampling_seen");
                const bool penalize = seenValue.IsValid() && generationPenalty != 1.0f;
                X::Tensor seen;
                if (penalize && !X::Tensor::IsTensor(seenValue)) {
                    result.SetItem("status", NativeValue(host, "error"));
                    result.SetItem("error_code", NativeValue(host, "invalid_repetition_history"));
                    return result;
                }
                if (penalize) seen = X::Tensor(seenValue);
                if (penalize && (seen.Info().dtype != X3_TENSOR_UINT8 ||
                    seen.Info().rank != 1 || seen.Info().shape[0] != vocabSize)) {
                    result.SetItem("status", NativeValue(host, "error"));
                    result.SetItem("error_code", NativeValue(host, "invalid_repetition_history"));
                    return result;
                }
                auto logitsUse = penalize
                    ? TensorHelper::AcquireGPU(std::vector<std::pair<X::Tensor, X3TensorAccess>>{
                        {logits, X3_TENSOR_READ}, {seen, X3_TENSOR_READ}})
                    : TensorHelper::AcquireGPU(logits);
                const auto* seenDevice = penalize
                    ? static_cast<const unsigned char*>(TensorHelper::GetGPUMemory(seen)) : nullptr;
                if (sampleStatus == cudaSuccess &&
                    logits.Info().dtype == X3_TENSOR_FLOAT32) {
                    sampleStatus = penalize ? sampleRepetitionFP32(
                        static_cast<const float*>(logitsDevice) +
                            static_cast<size_t>(selectedRow) * vocabSize,
                        seenDevice, generationPenalty, deviceTokenId, deviceTokenValue,
                        vocabSize, cudaStreamPerThread) : runLogitsTop1FP32(
                        static_cast<const float*>(logitsDevice) +
                            static_cast<size_t>(selectedRow) * vocabSize,
                        deviceTokenId, deviceTokenValue, 1, vocabSize,
                        cudaStreamPerThread);
                }
                else if (sampleStatus == cudaSuccess &&
                    logits.Info().dtype == X3_TENSOR_BFLOAT16) {
                    sampleStatus = penalize ? sampleRepetitionBF16(
                        reinterpret_cast<const __nv_bfloat16*>(logitsDevice) +
                            static_cast<size_t>(selectedRow) * vocabSize,
                        seenDevice, generationPenalty, deviceTokenId, deviceTokenValue,
                        vocabSize, cudaStreamPerThread) : runLogitsTop1BF16(
                        static_cast<const bfloat16*>(logitsDevice) +
                            static_cast<size_t>(selectedRow) * vocabSize,
                        deviceTokenId, deviceTokenValue, 1, vocabSize,
                        cudaStreamPerThread);
                }
                else if (sampleStatus == cudaSuccess) {
                    sampleStatus = cudaErrorInvalidValue;
                }
                if (sampleStatus == cudaSuccess) {
                    sampleStatus = cudaMemcpyAsync(
                        &tokenId, deviceTokenId, sizeof(tokenId),
                        cudaMemcpyDeviceToHost, cudaStreamPerThread);
                }
                if (sampleStatus == cudaSuccess) {
                    sampleStatus = cudaMemcpyAsync(
                        &tokenValue, deviceTokenValue, sizeof(tokenValue),
                        cudaMemcpyDeviceToHost, cudaStreamPerThread);
                }
                const cudaError_t sampleCompletion = cudaStreamSynchronize(cudaStreamPerThread);
                if (sampleStatus == cudaSuccess) sampleStatus = sampleCompletion;
                logitsUse.Finish();
                if (sampleStatus != cudaSuccess) {
                    result.SetItem("status", NativeValue(host, "error"));
                    result.SetItem(
                        "error_code",
                        NativeValue(host, "compiled_gpu_sampling_failed"));
                    result.SetItem(
                        "error_message",
                        NativeValue(host, cudaGetErrorString(sampleStatus)));
                    return result;
                }
            }
            if (m_backend == "openvino" && tokenId < 0) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem(
                    "error_code",
                    NativeValue(host, "compiled_logits_non_finite"));
                result.SetItem(
                    "error_message",
                    NativeValue(host,
                        "OpenVINO produced no finite logits; this device or "
                        "driver does not safely execute the selected "
                        "precision profile"));
                return result;
            }
            result.SetItem("token_id", NativeValue(host, tokenId));
            result.SetItem("token_value", NativeValue(host, tokenValue));
            sampledTokenId = tokenId;
            const auto firstTokenReady = std::chrono::steady_clock::now();
            result.SetItem("time_to_first_token_ms", NativeValue(host,
                std::chrono::duration<double, std::milli>(firstTokenReady - requestStart).count()));
            }
        }

        if (requestedNewTokens > 0) {
            if (!frontendActive || !m_decodeRuntime || sampledTokenId < 0) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "compiled_generation_runtime_unavailable"));
                return result;
            }
            X::Value activeInputs(inputs);
            const int expectedInputCount = frontendIsVL ? 15 :
                (frontendIsASR ? 9 : 7);
            if (activeInputs.Size() != expectedInputCount) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "compiled_generation_cache_bindings_missing"));
                return result;
            }
            std::vector<int64_t> generatedTokens;
            X::Value repetitionSeenValue;
            if (generationPenalty != 1.0f) {
                std::vector<unsigned char> emptyHistory(static_cast<size_t>(generationVocabSize), 0);
                repetitionSeenValue = MakeCudaTensorFromHost(host, X3_TENSOR_UINT8,
                    {generationVocabSize}, emptyHistory.data(), emptyHistory.size());
                if (!X::Tensor::IsTensor(repetitionSeenValue)) {
                    result.SetItem("status", NativeValue(host, "error"));
                    result.SetItem("error_code", NativeValue(host, "repetition_history_allocation_failed"));
                    return result;
                }
            }
            generatedTokens.reserve(static_cast<size_t>(requestedNewTokens));
            generatedTokens.push_back(sampledTokenId);
            std::string tokenizerError;
            const auto tokenizer = Tokenization::GetCachedQwenTokenizer(m_weightsLocation, &tokenizerError);
            if (!tokenizer) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "compiled_generation_tokenizer_unavailable"));
                result.SetItem("error_message", NativeValue(host, tokenizerError));
                return result;
            }
            const int64_t endOfText = tokenizer->TokenId("<|endoftext|>");
            const int64_t imEnd = tokenizer->TokenId("<|im_end|>");
            const bool ignoreEos = Lookup(requestDict, "ignore_eos").IsValid() &&
                Lookup(requestDict, "ignore_eos").ToLongLong() != 0;
            const int cacheInputIndex = frontendIsVL ? 11 :
                (frontendIsASR ? 5 : 3);
            const int cacheCapacity =
                m_inputShapes.size() > static_cast<size_t>(cacheInputIndex) &&
                m_inputShapes[cacheInputIndex].size() >= 3
                ? m_inputShapes[cacheInputIndex][1] *
                    m_inputShapes[cacheInputIndex][2]
                : 0;
            if (cacheCapacity <= 0 ||
                frontendPromptTokenCount + requestedNewTokens - 1 >
                    cacheCapacity) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "compiled_generation_exceeds_kv_profile"));
                return result;
            }
            int64_t decodeTokenValue = sampledTokenId;
            int64_t decodeRopePositions[3] = {};
            const int positionComponents =
                frontendIsVL || frontendIsASR ? 3 : 1;
            int decodeContextLength = 0;
            int decodeSlotPosition = 0;
            auto makeDecodeTensor = m_backend == "openvino"
                ? MakeCpuTensorFromHost
                : MakeCudaTensorFromHost;
            X::Value decodeTokenTensor = makeDecodeTensor(host,
                X3_TENSOR_INT64, {1, 1}, &decodeTokenValue, sizeof(decodeTokenValue));
            X::Value decodePositionTensor = makeDecodeTensor(host,
                X3_TENSOR_INT64,
                {positionComponents, 1, 1},
                decodeRopePositions,
                static_cast<size_t>(positionComponents) * sizeof(int64_t));
            X::Value decodeContextTensor = makeDecodeTensor(host,
                X3_TENSOR_INT32, {1}, &decodeContextLength, sizeof(decodeContextLength));
            X::Value decodeSlotTensor = makeDecodeTensor(host,
                X3_TENSOR_INT32, {1}, &decodeSlotPosition, sizeof(decodeSlotPosition));
            if (!(X::Tensor::IsTensor(decodeTokenTensor)) || !(X::Tensor::IsTensor(decodePositionTensor)) ||
                !(X::Tensor::IsTensor(decodeContextTensor)) || !(X::Tensor::IsTensor(decodeSlotTensor))) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "compiled_decode_metadata_allocation_failed"));
                return result;
            }
            X::Tensor decodeTokenStorage(decodeTokenTensor);
            X::Tensor decodePositionStorage(decodePositionTensor);
            X::Tensor decodeContextStorage(decodeContextTensor);
            X::Tensor decodeSlotStorage(decodeSlotTensor);
            void* decodeTokenDevice = m_backend == "openvino"
                ? decodeTokenStorage.Info().data
                : TensorHelper::GetGPUMemory(decodeTokenStorage);
            void* decodePositionDevice = m_backend == "openvino"
                ? decodePositionStorage.Info().data
                : TensorHelper::GetGPUMemory(decodePositionStorage);
            void* decodeContextDevice = m_backend == "openvino"
                ? decodeContextStorage.Info().data
                : TensorHelper::GetGPUMemory(decodeContextStorage);
            void* decodeSlotDevice = m_backend == "openvino"
                ? decodeSlotStorage.Info().data
                : TensorHelper::GetGPUMemory(decodeSlotStorage);
            const auto decodeStart = std::chrono::steady_clock::now();
            for (int generatedIndex = 1; generatedIndex < requestedNewTokens; ++generatedIndex) {
                if (!ignoreEos && (sampledTokenId == endOfText || sampledTokenId == imEnd)) break;
                const int slotPosition =
                    frontendPromptTokenCount + generatedIndex - 1;
                const int contextLength = slotPosition + 1;
                const int64_t ropePosition = static_cast<int64_t>(slotPosition) +
                    frontendPositionDelta;
                const int64_t ropePositions[3] = {ropePosition, ropePosition, ropePosition};
                decodeTokenValue = sampledTokenId;
                decodeRopePositions[0] = ropePositions[0];
                decodeRopePositions[1] = ropePositions[1];
                decodeRopePositions[2] = ropePositions[2];
                decodeContextLength = contextLength;
                decodeSlotPosition = slotPosition;
                cudaError_t metadataStatus = cudaSuccess;
                if (m_backend == "openvino") {
                    auto metadataUse = X::Tensor::AcquireMany({
                        {decodeTokenStorage, X3_TENSOR_WRITE}, {decodePositionStorage, X3_TENSOR_WRITE},
                        {decodeContextStorage, X3_TENSOR_WRITE}, {decodeSlotStorage, X3_TENSOR_WRITE}});
                    std::memcpy(
                        decodeTokenDevice, &decodeTokenValue,
                        sizeof(decodeTokenValue));
                    std::memcpy(
                        decodePositionDevice, decodeRopePositions,
                        static_cast<size_t>(positionComponents) *
                            sizeof(int64_t));
                    std::memcpy(
                        decodeContextDevice, &decodeContextLength,
                        sizeof(decodeContextLength));
                    std::memcpy(
                        decodeSlotDevice, &decodeSlotPosition,
                        sizeof(decodeSlotPosition));
                }
                else {
                    auto metadataUse = TensorHelper::AcquireGPU({
                        {decodeTokenStorage, X3_TENSOR_WRITE}, {decodePositionStorage, X3_TENSOR_WRITE},
                        {decodeContextStorage, X3_TENSOR_WRITE}, {decodeSlotStorage, X3_TENSOR_WRITE}});
                    metadataStatus = cudaMemcpyAsync(
                        decodeTokenDevice, &decodeTokenValue,
                        sizeof(decodeTokenValue),
                        cudaMemcpyHostToDevice, cudaStreamPerThread);
                    if (metadataStatus == cudaSuccess) metadataStatus = cudaMemcpyAsync(
                        decodePositionDevice,
                        decodeRopePositions,
                        static_cast<size_t>(positionComponents) * sizeof(int64_t),
                        cudaMemcpyHostToDevice, cudaStreamPerThread);
                    if (metadataStatus == cudaSuccess) metadataStatus = cudaMemcpyAsync(
                        decodeContextDevice, &decodeContextLength, sizeof(decodeContextLength),
                        cudaMemcpyHostToDevice, cudaStreamPerThread);
                    if (metadataStatus == cudaSuccess) metadataStatus = cudaMemcpyAsync(
                        decodeSlotDevice, &decodeSlotPosition, sizeof(decodeSlotPosition),
                        cudaMemcpyHostToDevice, cudaStreamPerThread);
                    // The upload sources are mutable host locals, not retained tensor storage.
                    const cudaError_t uploadCompletion = cudaStreamSynchronize(cudaStreamPerThread);
                    if (metadataStatus == cudaSuccess) metadataStatus = uploadCompletion;
                    metadataUse.Finish();
                }
                if (metadataStatus != cudaSuccess) {
                    result.SetItem("status", NativeValue(host, "error"));
                    result.SetItem("error_code", NativeValue(host, "compiled_decode_metadata_upload_failed"));
                    result.SetItem("error_message", NativeValue(host, cudaGetErrorString(metadataStatus)));
                    return result;
                }
                X::Value decodeInputs = X::Value::List(host);
                const int keyInputIndex = frontendIsVL ? 11 :
                    (frontendIsASR ? 5 : 3);
                const int valueInputIndex = frontendIsVL ? 12 :
                    (frontendIsASR ? 6 : 4);
                const int tableInputIndex = frontendIsVL ? 13 :
                    (frontendIsASR ? 7 : 5);
                decodeInputs.Append(decodeTokenTensor);
                decodeInputs.Append(decodePositionTensor);
                decodeInputs.Append(activeInputs.Get(keyInputIndex));
                decodeInputs.Append(activeInputs.Get(valueInputIndex));
                decodeInputs.Append(activeInputs.Get(tableInputIndex));
                decodeInputs.Append(decodeContextTensor);
                decodeInputs.Append(decodeSlotTensor);
                X::Value decodeRequest = X::Value::Dict(host);
                if (generationPenalty != 1.0f) {
                    if (sampledTokenId < 0 || sampledTokenId >= generationVocabSize) {
                        result.SetItem("status", NativeValue(host, "error"));
                        result.SetItem("error_code", NativeValue(host, "invalid_generated_token"));
                        return result;
                    }
                    X::Tensor seen(repetitionSeenValue);
                    auto seenUse = TensorHelper::AcquireGPU(seen, X3_TENSOR_WRITE);
                    const auto markStatus = cudaMemsetAsync(
                        static_cast<unsigned char*>(TensorHelper::GetGPUMemory(seen)) + sampledTokenId,
                        1, 1, cudaStreamPerThread);
                    seenUse.Finish();
                    if (markStatus != cudaSuccess) {
                        result.SetItem("status", NativeValue(host, "error"));
                        result.SetItem("error_code", NativeValue(host, "repetition_history_update_failed"));
                        return result;
                    }
                    decodeRequest.SetItem("_sampling_seen", repetitionSeenValue);
                    decodeRequest.SetItem("repetition_penalty", NativeValue(host, generationPenalty));
                }
                decodeRequest.SetItem("inputs", NativeValue(host, decodeInputs));
                decodeRequest.SetItem("sample", NativeValue(host, "greedy"));
                decodeRequest.SetItem("reuse_output", NativeValue(host, 1));
                if (m_backend == "openvino" && generatedIndex == 1) {
                    decodeRequest.SetItem(
                        "reset_openvino_state", NativeValue(host, 1));
                }
                X::Value decodeValue = m_decodeRuntime->Forward(decodeRequest);
                if (!decodeValue.IsDict()) {
                    result.SetItem("status", NativeValue(host, "error"));
                    result.SetItem("error_code", NativeValue(host, "compiled_decode_failed"));
                    return result;
                }
                X::Value decodeResult(decodeValue);
                if (Lookup(decodeResult, "status").ToString() != "ok") return decodeValue;
                sampledTokenId = Lookup(decodeResult, "token_id").ToLongLong();
                generatedTokens.push_back(sampledTokenId);
            }
            const auto decodeEnd = std::chrono::steady_clock::now();
            const double decodeMs =
                std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count();
            X::Value tokenList = X::Value::List(host);
            for (const int64_t token : generatedTokens) tokenList.Append(NativeValue(host, token));
            result.SetItem("token_ids", NativeValue(host, tokenList));
            result.SetItem("text", NativeValue(host, tokenizer->Decode(generatedTokens, true)));
            result.SetItem("generated_token_count", NativeValue(host, static_cast<long long>(generatedTokens.size())));
            result.SetItem("decode_ms", NativeValue(host, decodeMs));
            result.SetItem("decode_tokens_per_second", NativeValue(host,
                generatedTokens.size() > 1 && decodeMs > 0.0
                    ? static_cast<double>(generatedTokens.size() - 1) * 1000.0 / decodeMs
                    : 0.0));
            result.SetItem("token_id", NativeValue(host, sampledTokenId));
        }
        const bool returnLogits = !sampleGreedy ||
            (Lookup(requestDict, "return_logits").IsValid() && Lookup(requestDict, "return_logits").ToLongLong() != 0);
        if (returnLogits) result.SetItem("output", output);
        if (frontendActive) {
            result.SetItem(
                "prompt_token_count", NativeValue(host, frontendPromptTokenCount));
        }
        if (frontendIsVL) {
            result.SetItem("visual_token_count", NativeValue(host, frontendInputs.visualTokenCount));
            result.SetItem("source_height", NativeValue(host, frontendInputs.sourceHeight));
            result.SetItem("source_width", NativeValue(host, frontendInputs.sourceWidth));
            result.SetItem("height", NativeValue(host, frontendInputs.resizedHeight));
            result.SetItem("width", NativeValue(host, frontendInputs.resizedWidth));
        }
        if (frontendIsASR) {
            result.SetItem(
                "audio_token_count", NativeValue(host, asrFrontendInputs.audioTokenCount));
            result.SetItem(
                "audio_sample_count", NativeValue(host, asrFrontendInputs.audioSampleCount));
            result.SetItem(
                "audio_duration_seconds",
                NativeValue(host, asrFrontendInputs.audioDurationSeconds));
        }
        result.SetItem("total_ms", NativeValue(host,
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - requestStart).count()));
        return result;
    }

    X::Value CompiledModelRuntime::DebugProbe(
        const std::string& probe,
        X::Value argument)
    {
        auto* host = m_host;
        std::lock_guard<std::mutex> guard(m_mutex);
        X::Value result = X::Value::Dict(host);
        result.SetItem("probe", NativeValue(host, probe));
        if (probe == "paged_kv_bf16") {
            if (!argument.IsDict()) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "paged_kv_probe_requires_dict"));
                return result;
            }
            X::Value options(argument);
            X::Value qkvValue = Lookup(options, "qkv");
            X::Value keyPagesValue = Lookup(options, "key_pages");
            X::Value valuePagesValue = Lookup(options, "value_pages");
            X::Value pageTableValue = Lookup(options, "page_table");
            if (pageTableValue.IsList()) {
                X::Value pageList(pageTableValue);
                auto pageTensor = X::Tensor::Create(host, X3_TENSOR_INT32, {static_cast<int64_t>(pageList.Size())});
                if (pageTensor.Info().data) {
                    auto pageUse = pageTensor.Acquire(X3_TENSOR_WRITE);
                    auto* pageData = reinterpret_cast<int*>(pageTensor.Info().data);
                    for (long long index = 0; index < pageList.Size(); ++index) {
                        pageData[index] = static_cast<int>(pageList.Get(index).ToLongLong());
                    }
                    pageUse.Finish();
                    if (TensorHelper::EnsureGPUMemory(pageTensor) == TensorOpStatus::Success) {
                        pageTableValue = NativeValue(host, pageTensor);
                    }
                }
            }
            if (!(X::Tensor::IsTensor(qkvValue)) || !(X::Tensor::IsTensor(keyPagesValue)) ||
                !(X::Tensor::IsTensor(valuePagesValue)) || !(X::Tensor::IsTensor(pageTableValue))) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "paged_kv_probe_tensor_missing"));
                return result;
            }
            X::Tensor qkv(qkvValue);
            X::Tensor keyPages(keyPagesValue);
            X::Tensor valuePages(valuePagesValue);
            X::Tensor pageTable(pageTableValue);
            const int tokenCount = static_cast<int>(Lookup(options, "token_count").ToLongLong());
            const int startPosition = static_cast<int>(Lookup(options, "start_position").ToLongLong());
            const int sequenceLength = static_cast<int>(Lookup(options, "sequence_length").ToLongLong());
            const int pageSize = static_cast<int>(Lookup(options, "page_size").ToLongLong());
            const int qHeads = static_cast<int>(Lookup(options, "q_heads").ToLongLong());
            const int kvHeads = static_cast<int>(Lookup(options, "kv_heads").ToLongLong());
            const int headDim = static_cast<int>(Lookup(options, "head_dim").ToLongLong());
            const int qWidth = qHeads * headDim;
            const int kvWidth = kvHeads * headDim;
            const int qkvWidth = qWidth + 2 * kvWidth;
            if (qkv.Info().dtype != X3_TENSOR_BFLOAT16 ||
                keyPages.Info().dtype != X3_TENSOR_BFLOAT16 ||
                valuePages.Info().dtype != X3_TENSOR_BFLOAT16 ||
                pageTable.Info().dtype != X3_TENSOR_INT32 ||
                qkv.Info().rank != 2 || qkv.Info().shape[0] < tokenCount ||
                qkv.Info().shape[1] != qkvWidth || tokenCount <= 0 ||
                sequenceLength <= 0 || startPosition < 0 || pageSize <= 0 ||
                qHeads <= 0 || kvHeads <= 0 || headDim <= 0 || qHeads % kvHeads != 0) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "paged_kv_probe_shape_or_dtype_invalid"));
                return result;
            }
            int pageTableElements = 1;
            for (int dimension = 0; dimension < pageTable.Info().rank; ++dimension) {
                const int64_t size = pageTable.Info().shape[dimension];
                if (size <= 0 || size > std::numeric_limits<int>::max() / pageTableElements) {
                    result.SetItem("status", NativeValue(host, "error"));
                    result.SetItem("error_code", NativeValue(host, "paged_kv_probe_capacity_out_of_range"));
                    return result;
                }
                pageTableElements *= static_cast<int>(size);
            }
            if (pageTableElements > std::numeric_limits<int>::max() / pageSize) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "paged_kv_probe_capacity_out_of_range"));
                return result;
            }
            const int maxSequenceLength = pageTableElements * pageSize;
            if (TensorHelper::EnsureGPUMemory(qkv) != TensorOpStatus::Success ||
                TensorHelper::EnsureGPUMemory(keyPages) != TensorOpStatus::Success ||
                TensorHelper::EnsureGPUMemory(valuePages) != TensorOpStatus::Success ||
                TensorHelper::EnsureGPUMemory(pageTable) != TensorOpStatus::Success) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "paged_kv_probe_gpu_residency_failed"));
                return result;
            }

            auto* qkvDevice = static_cast<bfloat16*>(TensorHelper::GetGPUMemory(qkv));
            auto* keyDevice = static_cast<bfloat16*>(TensorHelper::GetGPUMemory(keyPages));
            auto* valueDevice = static_cast<bfloat16*>(TensorHelper::GetGPUMemory(valuePages));
            auto* tableDevice = static_cast<int*>(TensorHelper::GetGPUMemory(pageTable));
            auto output = TensorHelper::CreateGPU(host, X3_TENSOR_BFLOAT16, {qHeads, headDim});
            auto probeUse = TensorHelper::AcquireGPU({
                {qkv, X3_TENSOR_READ}, {keyPages, X3_TENSOR_WRITE},
                {valuePages, X3_TENSOR_WRITE}, {pageTable, X3_TENSOR_READ},
                {output, X3_TENSOR_WRITE}});
            void* outputDevice = TensorHelper::GetGPUMemory(output);
            cudaError_t status = runTextPagedKVWriteBF16(
                qkvDevice, keyDevice, valueDevice, tableDevice,
                tokenCount, startPosition, pageSize, qHeads, kvHeads, headDim,
                cudaStreamPerThread);
            const bfloat16* lastQ = qkvDevice + static_cast<size_t>(tokenCount - 1) * qkvWidth;
            const std::string implementation = Lookup(options, "implementation").IsValid()
                ? Lookup(options, "implementation").ToString()
                : "reference";
            void* workspaceDevice = nullptr;
            int* metadataDevice = nullptr;
            if (status == cudaSuccess && implementation == "reference") {
                status = runTextPagedKVCachedAttentionBF16(
                    lastQ, keyDevice, valueDevice, tableDevice,
                    static_cast<bfloat16*>(outputDevice), sequenceLength, pageSize,
                    qHeads, kvHeads, headDim, cudaStreamPerThread);
            }
            else if (status == cudaSuccess &&
                     (implementation == "split" || implementation == "flash")) {
                const int splitCount = maxSequenceLength / 128 + (maxSequenceLength % 128 != 0);
                const size_t scoreFloats = static_cast<size_t>(qHeads) * maxSequenceLength;
                const size_t partialFloats = static_cast<size_t>(qHeads) * splitCount * headDim;
                status = cudaMalloc(&workspaceDevice,
                    (scoreFloats + partialFloats) * sizeof(float));
                if (status == cudaSuccess) status = cudaMalloc(&metadataDevice, 2 * sizeof(int));
                const int metadata[2]{sequenceLength, sequenceLength - 1};
                if (status == cudaSuccess) {
                    status = cudaMemcpyAsync(
                        metadataDevice, metadata, sizeof(metadata), cudaMemcpyHostToDevice,
                        cudaStreamPerThread);
                }
                if (status == cudaSuccess && implementation == "flash") {
                    status = runTextPagedKVDecodeFlashBF16DeviceMetadata(
                        lastQ, keyDevice, valueDevice, tableDevice,
                        metadataDevice, metadataDevice + 1,
                        static_cast<bfloat16*>(outputDevice),
                        static_cast<float*>(workspaceDevice),
                        static_cast<float*>(workspaceDevice) + scoreFloats,
                        1, maxSequenceLength, pageSize, qHeads, kvHeads, headDim,
                        cudaStreamPerThread);
                }
                else if (status == cudaSuccess) {
                    status = runTextPagedKVDecodeSplitKBF16DeviceMetadata(
                        lastQ, keyDevice, valueDevice, tableDevice,
                        metadataDevice, metadataDevice + 1,
                        static_cast<bfloat16*>(outputDevice),
                        static_cast<float*>(workspaceDevice),
                        static_cast<float*>(workspaceDevice) + scoreFloats,
                        maxSequenceLength, pageSize, qHeads, kvHeads, headDim, 1,
                        cudaStreamPerThread);
                }
                const cudaError_t completion = cudaStreamSynchronize(cudaStreamPerThread);
                if (status == cudaSuccess) status = completion;
            }
            else if (status == cudaSuccess) {
                status = cudaErrorInvalidValue;
            }
            if (metadataDevice) cudaFree(metadataDevice);
            if (workspaceDevice) cudaFree(workspaceDevice);
            probeUse.Finish();
            if (status != cudaSuccess) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "paged_kv_probe_kernel_failed"));
                result.SetItem("error_message", NativeValue(host, cudaGetErrorString(status)));
                return result;
            }
            result.SetItem("status", NativeValue(host, "ok"));
            result.SetItem("output", NativeValue(host, output));
            result.SetItem("key_pages", keyPages);
            result.SetItem("value_pages", valuePages);
            result.SetItem("sequence_length", NativeValue(host, sequenceLength));
            result.SetItem("implementation", NativeValue(host, implementation));
            return result;
        }
        if (probe != "weight") {
            result.SetItem("status", NativeValue(host, "error"));
            result.SetItem("error_code", NativeValue(host, "unknown_compiled_probe"));
            return result;
        }

        const std::string name = argument.ToString();
        auto cached = m_loadedWeights.find(name);
        if (cached != m_loadedWeights.end()) {
            result.SetItem("status", NativeValue(host, "ok"));
            result.SetItem("name", NativeValue(host, name));
            result.SetItem("cache_hit", NativeValue(host, true));
            result.SetItem("tensor", cached->second);
            return result;
        }

        const SafeTensorMetadata* metadata = m_weightIndex.Find(name);
        if (!metadata) {
            result.SetItem("status", NativeValue(host, "error"));
            result.SetItem("error_code", NativeValue(host, "weight_not_found"));
            result.SetItem("name", NativeValue(host, name));
            return result;
        }

        X3TensorDType dataType;
        if (metadata->dataType == "F32") dataType = X3_TENSOR_FLOAT32;
        else if (metadata->dataType == "BF16") dataType = X3_TENSOR_BFLOAT16;
        else if (metadata->dataType == "F16") dataType = X3_TENSOR_FLOAT16;
        else {
            result.SetItem("status", NativeValue(host, "error"));
            result.SetItem("error_code", NativeValue(host, "weight_dtype_unsupported"));
            result.SetItem("dtype", NativeValue(host, metadata->dataType));
            return result;
        }

        std::vector<int64_t> shape;
        shape.reserve(metadata->shape.size());
        for (const long long dimension : metadata->shape) {
            if (dimension < 0 || dimension > std::numeric_limits<int>::max()) {
                result.SetItem("status", NativeValue(host, "error"));
                result.SetItem("error_code", NativeValue(host, "weight_shape_unsupported"));
                return result;
            }
            shape.push_back(static_cast<int>(dimension));
        }

        void* pinnedMemory = nullptr;
        void* deviceMemory = nullptr;
        cudaError_t cudaError = cudaHostAlloc(&pinnedMemory, metadata->dataSize, cudaHostAllocDefault);
        if (cudaError == cudaSuccess) {
            std::ifstream stream(metadata->filePath, std::ios::binary);
            stream.seekg(static_cast<std::streamoff>(metadata->dataOffset));
            stream.read(static_cast<char*>(pinnedMemory), static_cast<std::streamsize>(metadata->dataSize));
            if (!stream) {
                cudaError = cudaErrorUnknown;
            }
        }
        if (cudaError == cudaSuccess) cudaError = cudaMalloc(&deviceMemory, metadata->dataSize);
        if (cudaError == cudaSuccess) {
            cudaError = cudaMemcpy(deviceMemory, pinnedMemory, metadata->dataSize, cudaMemcpyHostToDevice);
        }
        if (pinnedMemory) cudaFreeHost(pinnedMemory);
        if (cudaError != cudaSuccess) {
            if (deviceMemory) cudaFree(deviceMemory);
            result.SetItem("status", NativeValue(host, "error"));
            result.SetItem("error_code", NativeValue(host, "weight_gpu_load_failed"));
            result.SetItem("error_message", NativeValue(host, cudaGetErrorString(cudaError)));
            return result;
        }

        auto tensor = AdoptCudaTensor(host, dataType, shape, deviceMemory);

        X::Value tensorValue(tensor);
        m_loadedWeightBytes += static_cast<long long>(metadata->dataSize);
        m_loadedWeights.emplace(name, tensorValue);
        result.SetItem("status", NativeValue(host, "ok"));
        result.SetItem("name", NativeValue(host, name));
        result.SetItem("cache_hit", NativeValue(host, false));
        result.SetItem("bytes", NativeValue(host, static_cast<long long>(metadata->dataSize)));
        result.SetItem("tensor", tensorValue);
        return result;
    }
}
