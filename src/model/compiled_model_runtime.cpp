#include "compiled_model_runtime.h"
#include "compiled_graph_capture.h"
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
#include "qwen_tokenizer.h"
#include "nlohmann/json.hpp"

#include <NvInferVersion.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
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
    constexpr const char* kGraphCacheMagic = "GARNET_RUNTIME_GRAPH_CACHE_V2";
    constexpr const char* kRuntimeSchema =
        "compiled_xmodel_runtime_v24_qwen3_tts_predictor_fp32";

    std::string ReadFile(const std::filesystem::path& path)
    {
        std::ifstream stream(path, std::ios::binary);
        return std::string(
            std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>());
    }

    X::Value JsonToXValue(const nlohmann::json& value)
    {
        if (value.is_null()) return X::Value(X::ValueType::None);
        if (value.is_boolean()) return X::Value(value.get<bool>());
        if (value.is_number_integer()) return X::Value(value.get<long long>());
        if (value.is_number_unsigned()) return X::Value(value.get<unsigned long long>());
        if (value.is_number_float()) return X::Value(value.get<double>());
        if (value.is_string()) return X::Value(value.get<std::string>());
        if (value.is_array()) {
            X::V<X::XList> list;
            for (const auto& item : value) {
                list->AddItem(JsonToXValue(item));
            }
            return X::Value(list);
        }
        if (value.is_object()) {
            X::Dict dictionary;
            for (auto iterator = value.begin(); iterator != value.end(); ++iterator) {
                dictionary->Set(iterator.key(), JsonToXValue(iterator.value()));
            }
            return X::Value(dictionary);
        }
        return X::Value();
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

    X::Value MakeCudaTensorFromHost(
        X::TensorDataType dataType,
        const std::vector<int>& dimensions,
        const void* source,
        size_t bytes)
    {
        X::Tensor tensor(X::g_pXHost->CreateTensor());
        X::Port::vector<int> shape(static_cast<int>(dimensions.size()));
        for (const int dimension : dimensions) shape.push_back(dimension);
        tensor->SetDataType(dataType);
        tensor->SetShape(shape);
        void* deviceMemory = nullptr;
        if (cudaMalloc(&deviceMemory, bytes) != cudaSuccess) return X::Value();
        if (cudaMemcpy(deviceMemory, source, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            Garnet::TensorHelper::AttachGPUMemory(tensor, deviceMemory) !=
                Garnet::TensorOpStatus::Success) {
            cudaFree(deviceMemory);
            return X::Value();
        }
        return X::Value(tensor);
    }

    X::Value MakeCpuTensorFromHost(
        X::TensorDataType dataType,
        const std::vector<int>& dimensions,
        const void* source,
        size_t bytes)
    {
        X::Tensor tensor(X::g_pXHost->CreateTensor());
        X::Port::vector<int> shape(static_cast<int>(dimensions.size()));
        for (const int dimension : dimensions) shape.push_back(dimension);
        tensor->SetDataType(dataType);
        tensor->SetShape(shape);
        X::Value initial;
        tensor->Create(initial);
        if (!tensor->GetData() ||
            static_cast<size_t>(tensor->GetDataSize()) != bytes) {
            return X::Value();
        }
        std::memcpy(tensor->GetData(), source, bytes);
        return X::Value(tensor);
    }

    X::Value BuildSymbolicWeights(const Garnet::SafeTensorsIndex& index)
    {
        X::Dict weights;
        for (const auto& entry : index.Entries()) {
            const auto& name = entry.first;
            const auto& metadata = entry.second;
            X::TensorDataType dataType;
            if (metadata.dataType == "BF16") dataType = X::TensorDataType::BFLOAT16;
            else if (metadata.dataType == "F16") dataType = X::TensorDataType::FLOAT16;
            else if (metadata.dataType == "F32") dataType = X::TensorDataType::FLOAT32;
            else continue;

            X::Port::vector<int> shape(static_cast<int>(metadata.shape.size()));
            bool valid = true;
            for (const long long dimension : metadata.shape) {
                if (dimension < 0 || dimension > std::numeric_limits<int>::max()) {
                    valid = false;
                    break;
                }
                shape.push_back(static_cast<int>(dimension));
            }
            if (!valid) continue;
            X::Tensor tensor(X::g_pXHost->CreateTensor());
            tensor->SetDataType(dataType);
            tensor->SetShape(shape);
            X::Value tensorName(name);
            tensor->SetName(tensorName);
            weights->Set(name, X::Value(tensor));
        }
        return X::Value(weights);
    }

    void CollectXModelDependencies(
        const std::filesystem::path& sourcePath,
        std::set<std::filesystem::path>& dependencies)
    {
        const auto normalized = std::filesystem::absolute(sourcePath).lexically_normal();
        if (!dependencies.insert(normalized).second) {
            return;
        }

        const std::string source = ReadFile(normalized);
        static const std::regex fromImport(
            R"(^\s*from\s+([A-Za-z0-9_.]+)\s+import\s+)",
            std::regex::ECMAScript);
        static const std::regex plainImport(
            R"(^\s*import\s+([A-Za-z0-9_.]+))",
            std::regex::ECMAScript);
        static const std::regex localImport(
            "^\\s*from\\s+\"([^\"]*)\"\\s+import\\s+([A-Za-z0-9_]+)",
            std::regex::ECMAScript);
        static const std::regex explicitDependency(
            R"(^\s*#\s*garnet-dependency:\s*([^\s]+\.x)\s*$)",
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
                    match[1].str() / (match[2].str() + ".x");
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
            const auto candidate = normalized.parent_path() / (module + ".x");
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
        ReleaseDeviceMemory();
    }

    void CompiledModelRuntime::ReleaseDeviceMemory()
    {
        std::lock_guard<std::mutex> guard(m_mutex);
        auto releaseTensor = [](X::Value& value) {
            if (value.IsTensor()) {
                X::Tensor tensor(value);
                TensorHelper::ReleaseGPUMemory(tensor);
            }
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
        std::lock_guard<std::mutex> guard(m_mutex);
        m_rootXModel = std::filesystem::absolute(rootXModel).lexically_normal().string();
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
        if (!std::filesystem::is_regular_file(rootPath, error) || rootPath.extension() != ".x") {
            m_state = "failed";
            m_errorCode = "invalid_root_xmodel";
            m_errorMessage = "compiled_xmodel mode requires an existing root .x file";
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
            m_errorMessage = "failed to read root .x source";
            return false;
        }
        if (!inputDataTypes.empty() && inputDataTypes.size() != inputShapes.size()) {
            m_state = "failed";
            m_errorCode = "invalid_symbolic_input_dtype";
            m_errorMessage = "input_dtypes must contain one entry per input_shapes entry";
            return false;
        }

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
                (qwenVL ? "qwen_text_decode.x" :
                    (qwenTTS ? "talker_decode.x" : "decode.x"));
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
            auto decodeRuntime = std::make_shared<CompiledModelRuntime>();
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
                X::Dict decodeStatus(decodeRuntime->Status());
                return {nullptr, decodeStatus["error_message"].ToString()};
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
                TRTBuilder builder;
                prepared = m_enginePartitions.size() > 1
                    ? builder.PrepareCapturedPartitions(
                        m_enginePartitions, &m_weightIndex, preparationError)
                    : builder.PrepareCapturedEngine(
                        m_enginePath, &m_weightIndex, preparationError);
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
                auto auxiliary = std::make_shared<CompiledModelRuntime>();
                FusionPartitionOptions predictorOptions = m_partitionOptions;
                predictorOptions.builderOptimizationLevel = 0;
                if (!auxiliary->Initialize(
                        (rootPath.parent_path() / "code_predictor.x").string(),
                        (std::filesystem::path(m_cacheDirectory) / "code_predictor").string(),
                        m_weightsLocation, "Qwen3TTSCodePredictor", "",
                        {{1, 1, talkerHiddenSize}, {1, 1}},
                        {"float32", "int64"},
                        predictorOptions, m_backend, m_precision)) {
                    X::Dict status(auxiliary->Status());
                    m_errorCode = "tts_code_predictor_initialization_failed";
                    m_errorMessage = status["error_message"].ToString();
                    return false;
                }
                int codecFrames = 256;
                if (const char* value = std::getenv("GARNET_TTS_MAX_FRAMES")) {
                    codecFrames = std::max(1, std::atoi(value));
                }
                const std::filesystem::path codecWeights =
                    std::filesystem::path(m_weightsLocation) / "speech_tokenizer";
                auto codec = std::make_shared<CompiledModelRuntime>();
                if (!codec->Initialize(
                        (rootPath.parent_path() / "codec_decode.x").string(),
                        (std::filesystem::path(m_cacheDirectory) / "codec_decode").string(),
                        codecWeights.string(), "Qwen3TTSCodecDecode", "",
                        {{1, 16, codecFrames}}, {"int64"}, m_partitionOptions,
                        m_backend, m_precision)) {
                    X::Dict status(codec->Status());
                    m_errorCode = "tts_codec_initialization_failed";
                    m_errorMessage = status["error_message"].ToString();
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
        const std::string graphFingerprint = MakeGraphFingerprint(
            rootPath, m_weightsLocation, m_entryFunction, m_backend, m_precision,
            inputShapes,
            inputDataTypes, m_partitionOptions);
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
                return true;
            }
        }
        ++m_diagnostics.graphCacheMisses;

        m_dependencyModules.clear();
        std::set<std::filesystem::path> dependencies;
        CollectXModelDependencies(rootPath, dependencies);
        for (const auto& dependency : dependencies) {
            if (dependency == std::filesystem::absolute(rootPath).lexically_normal()) {
                continue;
            }
            const std::string dependencySource = ReadFile(dependency);
            X::Value dependencyModule;
            if (!X::g_pXHost->LoadModule(
                    dependency.string().c_str(),
                    dependencySource.c_str(),
                    static_cast<int>(dependencySource.size()),
                    dependencyModule)) {
                m_state = "failed";
                m_errorCode = "xmodel_dependency_load_failed";
                m_errorMessage = dependency.string();
                return false;
            }
            X::Value dependencyResult;
            if (!X::g_pXHost->RunModule(dependencyModule, dependencyResult, true)) {
                m_state = "failed";
                m_errorCode = "xmodel_dependency_execution_failed";
                m_errorMessage = dependency.string();
                return false;
            }
            m_dependencyModules.push_back(dependencyModule);
        }

        if (!X::g_pXHost->LoadModule(
                m_rootXModel.c_str(),
                source.c_str(),
                static_cast<int>(source.size()),
                m_module)) {
            m_state = "failed";
            m_errorCode = "root_xmodel_load_failed";
            m_errorMessage = "xlang failed to parse the root .x module";
            return false;
        }

        X::Value moduleResult;
        ++m_diagnostics.rootXExecutions;
        if (!X::g_pXHost->RunModule(m_module, moduleResult, true)) {
            m_state = "failed";
            m_errorCode = "root_xmodel_execution_failed";
            m_errorMessage = "xlang failed while executing root .x module declarations";
            return false;
        }

        m_rootFunction = X::g_pXHost->QueryMember(
            X::g_pXHost->GetCurrentRuntime(),
            m_module.GetObj(),
            m_entryFunction.c_str());
        if (!m_rootFunction.IsValid()) {
            m_state = "failed";
            m_errorCode = "root_forward_missing";
            m_errorMessage = "root .x module does not export " + m_entryFunction;
            return false;
        }

        if (!inputShapes.empty()) {
            X::ARGS symbolicInputs(static_cast<int>(inputShapes.size()));
            for (size_t inputIndex = 0; inputIndex < inputShapes.size(); ++inputIndex) {
                X::Port::vector<int> shape(static_cast<int>(inputShapes[inputIndex].size()));
                for (const int size : inputShapes[inputIndex]) {
                    if (size <= 0) {
                        m_state = "failed";
                        m_errorCode = "invalid_symbolic_input_shape";
                        m_errorMessage = "symbolic input dimensions must be positive";
                        return false;
                    }
                    shape.push_back(size);
                }
                X::Tensor tensor;
                const std::string dataType = inputDataTypes.empty()
                    ? "float32"
                    : inputDataTypes[inputIndex];
                if (dataType == "float32" || dataType == "fp32") {
                    tensor->SetDataType(X::TensorDataType::FLOAT32);
                }
                else if (dataType == "bfloat16" || dataType == "bf16") {
                    tensor->SetDataType(X::TensorDataType::BFLOAT16);
                }
                else if (dataType == "int64") {
                    tensor->SetDataType(X::TensorDataType::LONGLONG);
                }
                else if (dataType == "int32") {
                    tensor->SetDataType(X::TensorDataType::INT);
                }
                else {
                    m_state = "failed";
                    m_errorCode = "invalid_symbolic_input_dtype";
                    m_errorMessage = "unsupported symbolic input dtype: " + dataType;
                    return false;
                }
                tensor->SetShape(shape);
                symbolicInputs.push_back(X::Value(tensor));
            }
            symbolicInputs.Close();

            if (symbolicInputs.size() > 0) {
                X::ARGS rootArguments;
                X::Value modelSpec = X::g_pXHost->QueryMember(
                    X::g_pXHost->GetCurrentRuntime(),
                    m_module.GetObj(),
                    "GARNET_MODEL_SPEC");
                if (modelSpec.IsDict()) {
                    X::Dict spec(modelSpec);
                    X::Value argumentSpecsValue = spec["arguments"];
                    if (!argumentSpecsValue.IsList()) {
                        m_state = "failed";
                        m_errorCode = "invalid_model_spec";
                        m_errorMessage = "GARNET_MODEL_SPEC.arguments must be a list";
                        return false;
                    }
                    X::List argumentSpecs(argumentSpecsValue);
                    bool needsWeights = false;
                    bool needsConfig = false;
                    for (long long index = 0; index < argumentSpecs->Size(); ++index) {
                        X::Value itemValue = argumentSpecs->Get(index);
                        if (!itemValue.IsDict()) continue;
                        X::Dict item(itemValue);
                        const std::string kind = item["kind"].ToString();
                        needsWeights = needsWeights || kind == "weights";
                        needsConfig = needsConfig || kind == "config";
                    }

                    X::Value symbolicWeights;
                    if (needsWeights) {
                        symbolicWeights = BuildSymbolicWeights(m_weightIndex);
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
                            configValue = JsonToXValue(nlohmann::json::parse(ReadFile(configPath)));
                        }
                        catch (const std::exception& exception) {
                            m_state = "failed";
                            m_errorCode = "model_config_invalid";
                            m_errorMessage = exception.what();
                            return false;
                        }
                    }

                    rootArguments.resize(static_cast<int>(argumentSpecs->Size()));
                    size_t tensorIndex = 0;
                    for (long long argumentIndex = 0; argumentIndex < argumentSpecs->Size(); ++argumentIndex) {
                        X::Value argumentSpecValue = argumentSpecs->Get(argumentIndex);
                        if (!argumentSpecValue.IsDict()) {
                            m_state = "failed";
                            m_errorCode = "invalid_model_spec";
                            m_errorMessage = "each model argument spec must be a dictionary";
                            return false;
                        }
                        X::Dict argumentSpec(argumentSpecValue);
                        const std::string kind = argumentSpec["kind"].ToString();
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
                        else if (kind == "none") rootArguments.push_back(X::Value(X::ValueType::None));
                        else if (kind == "bool") rootArguments.push_back(argumentSpec["value"]);
                        else {
                            m_state = "failed";
                            m_errorCode = "invalid_model_spec";
                            m_errorMessage = "unsupported model argument kind: " + kind;
                            return false;
                        }
                    }
                    rootArguments.Close();
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
                    m_graph = m_rootFunction.ObjCall(rootArguments, captureKwargs);
                    if (!GetCompiledGraphCaptureError().empty()) {
                        m_state = "failed";
                        m_errorCode = "invalid_fusion_annotation";
                        m_errorMessage = GetCompiledGraphCaptureError();
                        return false;
                    }
                }
                if (!m_graph.IsObject() || m_graph.GetObj()->GetType() != X::ObjType::TensorGraph) {
                    m_state = "failed";
                    m_errorCode = "symbolic_graph_capture_failed";
                    m_errorMessage = "root fusion call did not return an xlang TensorGraph";
                    return false;
                }
                m_graphSummary = m_graph.ToString();
                std::vector<CapturedTensorOperation> analyzedOperations;
                bool analyzed = false;
                if (m_backend == "openvino") {
                    OpenVINOBuilder builder;
                    analyzed = builder.AnalyzeCapturedGraph(
                        m_graph, m_rootFunction, rootArguments,
                        analyzedOperations, m_errorMessage);
                }
                else {
                    TRTBuilder builder;
                    builder.SetCapturedWeightProfile(m_precision);
                    builder.SetCapturedWorkspaceBytes(
                        m_partitionOptions.builderWorkspaceBytes);
                    builder.SetCapturedOptimizationLevel(
                        m_partitionOptions.builderOptimizationLevel);
                    analyzed = builder.AnalyzeCapturedGraph(
                        m_graph, m_rootFunction, rootArguments,
                        analyzedOperations, m_errorMessage);
                }
                if (!analyzed ||
                    !AssignCapturedFusionOperations(
                        std::move(analyzedOperations),
                        m_partitionOptions,
                        m_errorMessage)) {
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
                    TRTBuilder builder;
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
        std::lock_guard<std::mutex> guard(m_mutex);
        X::Dict status;
        status->Set("mode", X::Value("compiled_xmodel"));
        status->Set("state", X::Value(m_state));
        status->Set("ready", X::Value(m_ready));
        status->Set("root_xmodel", X::Value(m_rootXModel));
        status->Set("cache_directory", X::Value(m_cacheDirectory));
        status->Set("weights_location", X::Value(m_weightsLocation));
        status->Set("entry_function", X::Value(m_entryFunction));
        status->Set("frontend", X::Value(m_frontend));
        status->Set("backend", X::Value(m_backend));
        status->Set("precision", X::Value(m_precision));
        status->Set("graph_summary", X::Value(m_graphSummary));
        status->Set("scheduler", X::Value("cpu_control_gpu_execution"));
        status->Set("execution_plan_json", X::Value(m_executionPlanJson));
        try {
            status->Set(
                "execution_plan",
                JsonToXValue(nlohmann::json::parse(m_executionPlanJson)));
        }
        catch (const std::exception&) {
            status->Set("execution_plan", X::Value(X::ValueType::None));
        }
        status->Set("engine_path", X::Value(m_enginePath));
        status->Set(
            "engine_partition_count",
            X::Value(static_cast<long long>(
                m_enginePartitions.empty() ? 1 : m_enginePartitions.size())));
        X::Dict partitionOptions;
        partitionOptions->Set(
            "enable_preferred_boundaries",
            X::Value(m_partitionOptions.enablePreferredBoundaries));
        partitionOptions->Set(
            "preferred_min_operations",
            X::Value(m_partitionOptions.preferredMinOperations));
        partitionOptions->Set(
            "max_atomic_regions_per_partition",
            X::Value(m_partitionOptions.maxAtomicRegionsPerPartition));
        partitionOptions->Set(
            "builder_workspace_bytes",
            X::Value(m_partitionOptions.builderWorkspaceBytes));
        partitionOptions->Set(
            "builder_optimization_level",
            X::Value(m_partitionOptions.builderOptimizationLevel));
        status->Set("partition_options", partitionOptions);
        status->Set("weight_tensor_count", X::Value(static_cast<long long>(m_weightIndex.TensorCount())));
        status->Set("weight_tensor_bytes", X::Value(static_cast<long long>(m_weightIndex.TensorBytes())));
        status->Set("weight_index_error", X::Value(m_weightIndexError));
        status->Set("loaded_weight_count", X::Value(static_cast<long long>(m_loadedWeights.size())));
        status->Set("loaded_weight_bytes", X::Value(m_loadedWeightBytes));
        status->Set("engines_prepared", X::Value(m_enginesPrepared));
        status->Set("engine_prepare_ms", X::Value(m_enginePreparationMs));
        status->Set("frontend_prepared", X::Value(m_frontendPrepared));
        status->Set("frontend_prepare_ms", X::Value(m_frontendPreparationMs));
        status->Set("error_code", X::Value(m_errorCode));
        status->Set("error_message", X::Value(m_errorMessage));

        X::Dict counters;
        counters->Set("root_x_executions", X::Value(m_diagnostics.rootXExecutions));
        counters->Set("hardcoded_qwen_runner_calls", X::Value(m_diagnostics.hardcodedQwenRunnerCalls));
        counters->Set("python_subgraph_calls", X::Value(m_diagnostics.pythonSubgraphCalls));
        counters->Set("direct_internal_export_calls", X::Value(m_diagnostics.directInternalExportCalls));
        counters->Set("cpu_tensor_intermediates", X::Value(m_diagnostics.cpuTensorIntermediates));
        counters->Set("graph_cache_hits", X::Value(m_diagnostics.graphCacheHits));
        counters->Set("graph_cache_misses", X::Value(m_diagnostics.graphCacheMisses));
        status->Set("forbidden_path_counters", counters);
        return status;
    }

    X::Value CompiledModelRuntime::Forward(X::Value request)
    {
        std::lock_guard<std::mutex> guard(m_mutex);
        const auto requestStart = std::chrono::steady_clock::now();
        X::Dict result;
        if (!m_ready) {
            result->Set("status", X::Value("error"));
            result->Set("error_code", X::Value("compiled_graph_not_ready"));
            result->Set("error_message", X::Value(m_errorMessage));
            return result;
        }

        if (!request.IsDict()) {
            result->Set("status", X::Value("error"));
            result->Set("error_code", X::Value("invalid_compiled_request"));
            result->Set("error_message", X::Value("compiled forward requires a request dictionary"));
            return result;
        }
        X::Dict requestDict(request);
        X::Value inputs = requestDict["inputs"];
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
            X::Value imageSource = requestDict["image"];
            if (!imageSource.IsValid()) imageSource = requestDict["image_path"];
            const std::string prompt = requestDict["prompt"].ToString();
            const int minPixels = requestDict["min_pixels"].IsValid()
                ? static_cast<int>(requestDict["min_pixels"].ToLongLong())
                : 65536;
            const int maxPixels = requestDict["max_pixels"].IsValid()
                ? static_cast<int>(requestDict["max_pixels"].ToLongLong())
                : 65536;
            frontendInputs = BuildQwenVLCompiledInputs(
                m_weightsLocation, imageSource, prompt, minPixels, maxPixels,
                m_inputShapes, m_reusablePrefillKeyCache,
                m_reusablePrefillValueCache);
            if (!frontendInputs.inputs.IsList()) {
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("compiled_frontend_failed"));
                result->Set("error_message", X::Value(frontendInputs.error));
                return result;
            }
            {
                X::List frontendList(frontendInputs.inputs);
                if (!m_reusablePrefillKeyCache.IsTensor()) {
                    m_reusablePrefillKeyCache = frontendList->Get(11);
                }
                if (!m_reusablePrefillValueCache.IsTensor()) {
                    m_reusablePrefillValueCache = frontendList->Get(12);
                }
            }
            inputs = frontendInputs.inputs;
            frontendActive = true;
            frontendIsVL = true;
            frontendPromptTokenCount = frontendInputs.promptTokenCount;
            frontendPositionDelta = frontendInputs.mropePositionDelta;
        }
        else if (!inputs.IsList() && m_frontend == "qwen3_text") {
            const std::string prompt = requestDict["prompt"].ToString();
            const bool enableThinking =
                requestDict["enable_thinking"].IsValid() &&
                requestDict["enable_thinking"].ToLongLong() != 0;
            textFrontendInputs = BuildQwenTextCompiledInputs(
                m_weightsLocation, prompt, enableThinking, m_inputShapes,
                m_reusablePrefillKeyCache, m_reusablePrefillValueCache,
                m_backend == "openvino");
            if (!textFrontendInputs.inputs.IsList()) {
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("compiled_frontend_failed"));
                result->Set(
                    "error_message", X::Value(textFrontendInputs.error));
                return result;
            }
            {
                X::List frontendList(textFrontendInputs.inputs);
                if (!m_reusablePrefillKeyCache.IsTensor()) {
                    m_reusablePrefillKeyCache = frontendList->Get(3);
                }
                if (!m_reusablePrefillValueCache.IsTensor()) {
                    m_reusablePrefillValueCache = frontendList->Get(4);
                }
            }
            inputs = textFrontendInputs.inputs;
            frontendActive = true;
            frontendPromptTokenCount =
                textFrontendInputs.promptTokenCount;
        }
        else if (!inputs.IsList() && m_frontend == "qwen3_asr") {
            X::Value audioSource = requestDict["audio"];
            if (!audioSource.IsValid()) audioSource = requestDict["audio_path"];
            asrFrontendInputs = BuildQwenASRCompiledInputs(
                m_weightsLocation, audioSource,
                requestDict["context"].ToString(),
                requestDict["language"].ToString(), m_inputShapes,
                m_reusablePrefillKeyCache, m_reusablePrefillValueCache);
            if (!asrFrontendInputs.inputs.IsList()) {
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("compiled_frontend_failed"));
                result->Set("error_message", X::Value(asrFrontendInputs.error));
                return result;
            }
            {
                X::List frontendList(asrFrontendInputs.inputs);
                if (!m_reusablePrefillKeyCache.IsTensor()) {
                    m_reusablePrefillKeyCache = frontendList->Get(5);
                }
                if (!m_reusablePrefillValueCache.IsTensor()) {
                    m_reusablePrefillValueCache = frontendList->Get(6);
                }
            }
            inputs = asrFrontendInputs.inputs;
            frontendActive = true;
            frontendIsASR = true;
            frontendPromptTokenCount = asrFrontendInputs.promptTokenCount;
        }
        else if (!inputs.IsList() && m_frontend == "qwen3_tts") {
            ttsFrontendInputs = BuildQwenTTSCompiledInputs(
                m_weightsLocation, requestDict["text"].ToString(),
                requestDict["speaker"].ToString(),
                requestDict["language"].ToString(),
                requestDict["instruct"].ToString(), m_inputShapes,
                m_reusablePrefillKeyCache, m_reusablePrefillValueCache);
            if (!ttsFrontendInputs.inputs.IsList()) {
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("compiled_frontend_failed"));
                result->Set("error_message", X::Value(ttsFrontendInputs.error));
                return result;
            }
            {
                X::List frontendList(ttsFrontendInputs.inputs);
                if (!m_reusablePrefillKeyCache.IsTensor()) {
                    m_reusablePrefillKeyCache = frontendList->Get(3);
                    m_reusablePrefillValueCache = frontendList->Get(4);
                }
            }
            inputs = ttsFrontendInputs.inputs;
            frontendActive = true;
            frontendIsTTS = true;
            frontendPromptTokenCount = ttsFrontendInputs.promptTokenCount;
        }
        std::string executionError;
        const X::Value reusableOutput =
            requestDict["reuse_output"].IsValid() &&
            requestDict["reuse_output"].ToLongLong() != 0
                ? m_reusableExecutionOutput
                : X::Value();
        X::Value output;
        if (m_backend == "openvino") {
            OpenVINOBuilder builder;
            const bool resetOpenVINOState =
                requestDict["reset_openvino_state"].IsValid() &&
                requestDict["reset_openvino_state"].ToLongLong() != 0;
            output = builder.RunCapturedEngine(
                m_enginePath, inputs, &m_weightIndex, reusableOutput,
                resetOpenVINOState,
                m_openVinoSessionId,
                executionError);
        }
        else {
            TRTBuilder builder;
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
        if (!output.IsTensor()) {
            result->Set("status", X::Value("error"));
            result->Set("error_code", X::Value("compiled_engine_execution_failed"));
            result->Set("error_message", X::Value(executionError));
            return result;
        }
        if (requestDict["reuse_output"].IsValid() &&
            requestDict["reuse_output"].ToLongLong() != 0 &&
            !m_reusableExecutionOutput.IsTensor()) {
            m_reusableExecutionOutput = output;
        }
        result->Set("status", X::Value("ok"));
        if (frontendIsTTS) {
            if (!m_decodeRuntime || !m_auxRuntime || !m_codecRuntime ||
                m_backend != "tensorrt") {
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("compiled_tts_runtime_unavailable"));
                return result;
            }
            const int requestedFrames = requestDict["max_audio_frames"].IsValid()
                ? std::max(1, static_cast<int>(
                    requestDict["max_audio_frames"].ToLongLong()))
                : 256;
            int profileFrames = 256;
            if (const char* value = std::getenv("GARNET_TTS_MAX_FRAMES")) {
                profileFrames = std::max(1, std::atoi(value));
            }
            const int maxFrames = std::min(requestedFrames, profileFrames);
            const int vocabSize = 3072;
            const bool stochastic = !requestDict["do_sample"].IsValid() ||
                requestDict["do_sample"].ToLongLong() != 0;
            const int topK = requestDict["top_k"].IsValid()
                ? std::clamp(static_cast<int>(requestDict["top_k"].ToLongLong()), 1, vocabSize)
                : 50;
            const float temperature = requestDict["temperature"].IsValid()
                ? std::max(0.01F, static_cast<float>(requestDict["temperature"].ToDouble()))
                : 0.9F;
            const float repetitionPenalty = requestDict["repetition_penalty"].IsValid()
                ? std::max(1.0F, static_cast<float>(requestDict["repetition_penalty"].ToDouble()))
                : 1.05F;
            const uint64_t seed = requestDict["seed"].IsValid()
                ? static_cast<uint64_t>(requestDict["seed"].ToLongLong())
                : static_cast<uint64_t>(std::chrono::high_resolution_clock::now()
                    .time_since_epoch().count());
            std::mt19937_64 random(seed);
            std::vector<long long> firstCodeHistory;
            auto samplePacked = [&](X::Value packedValue, int row,
                                    long long& token, X::Value& hiddenValue,
                                    std::string& errorText) -> bool {
                if (!packedValue.IsTensor()) {
                    errorText = "talker returned no packed hidden/logit tensor";
                    return false;
                }
                X::Tensor packed(packedValue);
                const int hiddenSize = packed->GetDimCount() == 3
                    ? packed->GetDimSize(2) - vocabSize : 0;
                if (packed->GetDimCount() != 3 ||
                    hiddenSize <= 0 ||
                    row < 0 || row >= packed->GetDimSize(1)) {
                    errorText = "talker packed output has incompatible dimensions";
                    return false;
                }
                const size_t elementBytes = packed->GetDataType() ==
                    X::TensorDataType::FLOAT32 ? sizeof(float) : sizeof(unsigned short);
                const size_t rowElements = hiddenSize + vocabSize;
                const char* rowMemory = static_cast<const char*>(
                    TensorHelper::GetGPUMemory(packed)) +
                    static_cast<size_t>(row) * rowElements * elementBytes;
                X::Tensor hidden(X::g_pXHost->CreateTensor());
                X::Port::vector<int> hiddenShape(3);
                hiddenShape.push_back(1); hiddenShape.push_back(1);
                hiddenShape.push_back(hiddenSize);
                hidden->SetDataType(packed->GetDataType());
                hidden->SetShape(hiddenShape);
                void* hiddenMemory = nullptr;
                if (cudaMalloc(&hiddenMemory, hiddenSize * elementBytes) != cudaSuccess ||
                    cudaMemcpyAsync(hiddenMemory, rowMemory, hiddenSize * elementBytes,
                        cudaMemcpyDeviceToDevice, cudaStreamPerThread) != cudaSuccess ||
                    TensorHelper::AttachGPUMemory(hidden, hiddenMemory) !=
                        TensorOpStatus::Success) {
                    if (hiddenMemory) cudaFree(hiddenMemory);
                    errorText = "talker hidden-state extraction failed";
                    return false;
                }
                hiddenValue = X::Value(hidden);
                std::vector<unsigned char> raw(
                    static_cast<size_t>(vocabSize) * elementBytes);
                cudaError_t status = cudaMemcpyAsync(
                    raw.data(), rowMemory + hiddenSize * elementBytes,
                    raw.size(), cudaMemcpyDeviceToHost, cudaStreamPerThread);
                if (status == cudaSuccess) status = cudaStreamSynchronize(cudaStreamPerThread);
                if (status != cudaSuccess) {
                    errorText = cudaGetErrorString(status);
                    return false;
                }
                std::vector<float> logits(static_cast<size_t>(vocabSize));
                if (packed->GetDataType() == X::TensorDataType::FLOAT32) {
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

            X::List activeInputs(inputs);
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
                    result->Set("status", X::Value("error"));
                    result->Set("error_code", X::Value("compiled_tts_sampling_failed"));
                    result->Set("error_message", X::Value(ttsError));
                    return result;
                }
                if (firstCode == ttsFrontendInputs.codecEosTokenId) break;
                firstCodeHistory.push_back(firstCode);
                X::Value firstCodeTensor = MakeCudaTensorFromHost(
                    X::TensorDataType::LONGLONG, {1, 1}, &firstCode,
                    sizeof(firstCode));
                X::V<X::XList> predictorInputs;
                predictorInputs->AddItem(hidden);
                predictorInputs->AddItem(firstCodeTensor);
                X::Dict predictorRequest;
                predictorRequest->Set("inputs", X::Value(predictorInputs));
                X::Value predictorValue = m_auxRuntime->Forward(predictorRequest);
                if (!predictorValue.IsDict()) {
                    result->Set("status", X::Value("error"));
                    result->Set("error_code", X::Value("compiled_tts_predictor_failed"));
                    return result;
                }
                X::Dict predictorResult(predictorValue);
                if (predictorResult["status"].ToString() != "ok") return predictorValue;
                X::Value frameCodes = predictorResult["output"];
                if (!frameCodes.IsTensor()) {
                    result->Set("status", X::Value("error"));
                    result->Set("error_code", X::Value("compiled_tts_predictor_output_invalid"));
                    return result;
                }
                int64_t hostCodes[16]{};
                X::Tensor frameCodeTensor(frameCodes);
                cudaError_t copyStatus = cudaMemcpyAsync(
                    hostCodes, TensorHelper::GetGPUMemory(frameCodeTensor),
                    sizeof(hostCodes), cudaMemcpyDeviceToHost, cudaStreamPerThread);
                if (copyStatus == cudaSuccess) {
                    copyStatus = cudaStreamSynchronize(cudaStreamPerThread);
                }
                if (copyStatus != cudaSuccess) {
                    result->Set("status", X::Value("error"));
                    result->Set("error_code", X::Value("compiled_tts_code_download_failed"));
                    return result;
                }
                codes.insert(codes.end(), hostCodes, hostCodes + 16);
                if (frame + 1 >= maxFrames) break;

                const int slot = frontendPromptTokenCount + frame;
                const int context = slot + 1;
                const int64_t positions[3] = {slot, slot, slot};
                X::Value positionTensor = MakeCudaTensorFromHost(
                    X::TensorDataType::LONGLONG, {3, 1, 1}, positions,
                    sizeof(positions));
                X::Value contextTensor = MakeCudaTensorFromHost(
                    X::TensorDataType::INT, {1}, &context, sizeof(context));
                X::Value slotTensor = MakeCudaTensorFromHost(
                    X::TensorDataType::INT, {1}, &slot, sizeof(slot));
                X::V<X::XList> talkerInputs;
                talkerInputs->AddItem(frameCodes);
                talkerInputs->AddItem(positionTensor);
                talkerInputs->AddItem(activeInputs->Get(3));
                talkerInputs->AddItem(activeInputs->Get(4));
                talkerInputs->AddItem(activeInputs->Get(5));
                talkerInputs->AddItem(contextTensor);
                talkerInputs->AddItem(slotTensor);
                X::Dict talkerRequest;
                talkerRequest->Set("inputs", X::Value(talkerInputs));
                talkerRequest->Set("reuse_output", X::Value(1));
                X::Value talkerValue = m_decodeRuntime->Forward(talkerRequest);
                if (!talkerValue.IsDict()) {
                    result->Set("status", X::Value("error"));
                    result->Set("error_code", X::Value("compiled_tts_talker_decode_failed"));
                    return result;
                }
                X::Dict talkerResult(talkerValue);
                if (talkerResult["status"].ToString() != "ok") return talkerValue;
                packed = talkerResult["output"];
            }
            if (codes.empty()) {
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("compiled_tts_generated_no_audio"));
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
            X::Value codecInput = MakeCudaTensorFromHost(
                X::TensorDataType::LONGLONG, {1, 16, profileFrames},
                paddedCodes.data(), paddedCodes.size() * sizeof(int64_t));
            X::V<X::XList> codecInputs;
            codecInputs->AddItem(codecInput);
            X::Dict codecRequest;
            codecRequest->Set("inputs", X::Value(codecInputs));
            X::Value codecValue = m_codecRuntime->Forward(codecRequest);
            if (!codecValue.IsDict()) {
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("compiled_tts_codec_decode_failed"));
                return result;
            }
            X::Dict codecResult(codecValue);
            if (codecResult["status"].ToString() != "ok") return codecValue;
            X::Value fullAudioValue = codecResult["output"];
            if (!fullAudioValue.IsTensor()) {
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("compiled_tts_waveform_invalid"));
                return result;
            }
            X::Tensor fullAudio(fullAudioValue);
            const int validSamples = frameCount * 1920;
            const size_t audioElementBytes = fullAudio->GetDataType() ==
                X::TensorDataType::FLOAT32 ? sizeof(float) : sizeof(unsigned short);
            X::Tensor audio(X::g_pXHost->CreateTensor());
            X::Port::vector<int> audioShape(2);
            audioShape.push_back(1); audioShape.push_back(validSamples);
            audio->SetDataType(fullAudio->GetDataType());
            audio->SetShape(audioShape);
            void* audioMemory = nullptr;
            cudaError_t audioStatus = cudaMalloc(
                &audioMemory, static_cast<size_t>(validSamples) * audioElementBytes);
            if (audioStatus == cudaSuccess) audioStatus = cudaMemcpyAsync(
                audioMemory, TensorHelper::GetGPUMemory(fullAudio),
                static_cast<size_t>(validSamples) * audioElementBytes,
                cudaMemcpyDeviceToDevice, cudaStreamPerThread);
            if (audioStatus != cudaSuccess ||
                TensorHelper::AttachGPUMemory(audio, audioMemory) !=
                    TensorOpStatus::Success) {
                if (audioMemory) cudaFree(audioMemory);
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("compiled_tts_waveform_trim_failed"));
                return result;
            }
            result->Set("audio", X::Value(audio));
            result->Set("sample_rate", X::Value(24000));
            result->Set("audio_frame_count", X::Value(frameCount));
            result->Set("audio_sample_count", X::Value(validSamples));
            result->Set("audio_duration_seconds", X::Value(frameCount / 12.5));
            result->Set("codec_codes", X::Value(static_cast<long long>(codes.size())));
            result->Set("prompt_token_count", X::Value(frontendPromptTokenCount));
            result->Set("decode_ms", X::Value(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - decodeStart).count()));
            result->Set("total_ms", X::Value(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - requestStart).count()));
            return result;
        }
        const int requestedNewTokens = requestDict["max_new_tokens"].IsValid()
            ? std::max(0, static_cast<int>(requestDict["max_new_tokens"].ToLongLong()))
            : 0;
        const std::string sampleMode = requestDict["sample"].IsValid()
            ? requestDict["sample"].ToString()
            : std::string();
        const bool sampleBatch = sampleMode == "greedy_batch";
        const bool sampleGreedy = requestedNewTokens > 0 ||
            sampleMode == "greedy" || sampleBatch;
        long long sampledTokenId = -1;
        if (sampleGreedy) {
            X::Tensor logits(output);
            if (logits->GetDimCount() != 3) {
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("compiled_sampling_shape_invalid"));
                return result;
            }
            const int tokenRows = static_cast<int>(logits->GetDimSize(1));
            const int vocabSize = static_cast<int>(logits->GetDimSize(2));
            const int batchSize = static_cast<int>(logits->GetDimSize(0));
            if (sampleBatch) {
                const int sampleRows = batchSize * tokenRows;
                if (sampleRows <= 0) {
                    result->Set("status", X::Value("error"));
                    result->Set(
                        "error_code",
                        X::Value("compiled_sampling_shape_invalid"));
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
                if (sampleStatus == cudaSuccess &&
                    logits->GetDataType() == X::TensorDataType::FLOAT32) {
                    sampleStatus = runLogitsTop1BatchFP32(
                        static_cast<const float*>(logitsDevice),
                        static_cast<long long*>(m_sampleTokenDevice),
                        static_cast<float*>(m_sampleValueDevice),
                        sampleRows, vocabSize, cudaStreamPerThread);
                }
                else if (sampleStatus == cudaSuccess &&
                    logits->GetDataType() == X::TensorDataType::BFLOAT16) {
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
                if (sampleStatus == cudaSuccess) {
                    sampleStatus = cudaStreamSynchronize(cudaStreamPerThread);
                }
                if (sampleStatus != cudaSuccess) {
                    result->Set("status", X::Value("error"));
                    result->Set(
                        "error_code",
                        X::Value("compiled_gpu_batch_sampling_failed"));
                    result->Set(
                        "error_message",
                        X::Value(cudaGetErrorString(sampleStatus)));
                    return result;
                }
                X::V<X::XList> tokenList;
                X::V<X::XList> valueList;
                for (int row = 0; row < sampleRows; ++row) {
                    tokenList->AddItem(X::Value(tokenIds[row]));
                    valueList->AddItem(X::Value(tokenValues[row]));
                }
                result->Set("token_ids", X::Value(tokenList));
                result->Set("token_values", X::Value(valueList));
                sampledTokenId = tokenIds.front();
                const auto firstTokenReady = std::chrono::steady_clock::now();
                result->Set("time_to_first_token_ms", X::Value(
                    std::chrono::duration<double, std::milli>(
                        firstTokenReady - requestStart).count()));
            }
            else {
            int selectedRow = tokenRows - 1;
            if (frontendPromptTokenCount > 0) {
                selectedRow =
                    std::min(frontendPromptTokenCount, tokenRows) - 1;
            }
            else if (requestDict["sample_row"].IsValid()) {
                selectedRow = static_cast<int>(requestDict["sample_row"].ToLongLong());
            }
            if (selectedRow < 0 || selectedRow >= tokenRows) {
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("compiled_sampling_row_invalid"));
                return result;
            }
            long long tokenId = -1;
            float tokenValue = 0.0f;
            if (m_backend == "openvino") {
                if (logits->GetDeviceType() != X::TensorDeviceType::CPU ||
                    !logits->GetData()) {
                    result->Set("status", X::Value("error"));
                    result->Set(
                        "error_code",
                        X::Value("compiled_cpu_sampling_failed"));
                    return result;
                }
                if (vocabSize == 1 &&
                    logits->GetDataType() ==
                        X::TensorDataType::LONGLONG) {
                    tokenId = reinterpret_cast<const int64_t*>(
                        logits->GetData())[selectedRow];
                    tokenValue = 0.0F;
                }
                else {
                    tokenValue = -std::numeric_limits<float>::infinity();
                    for (int token = 0; token < vocabSize; ++token) {
                        float value = 0.0F;
                        const size_t offset =
                            static_cast<size_t>(selectedRow) * vocabSize +
                            token;
                        if (logits->GetDataType() ==
                            X::TensorDataType::FLOAT32) {
                            value = reinterpret_cast<const float*>(
                                logits->GetData())[offset];
                        }
                        else if (logits->GetDataType() ==
                            X::TensorDataType::BFLOAT16) {
                            const uint16_t bits =
                                reinterpret_cast<const uint16_t*>(
                                    logits->GetData())[offset];
                            const uint32_t expanded =
                                static_cast<uint32_t>(bits) << 16;
                            std::memcpy(&value, &expanded, sizeof(value));
                        }
                        else {
                            result->Set("status", X::Value("error"));
                            result->Set(
                                "error_code",
                                X::Value(
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
                if (sampleStatus == cudaSuccess &&
                    logits->GetDataType() == X::TensorDataType::FLOAT32) {
                    sampleStatus = runLogitsTop1FP32(
                        static_cast<const float*>(logitsDevice) +
                            static_cast<size_t>(selectedRow) * vocabSize,
                        deviceTokenId, deviceTokenValue, 1, vocabSize,
                        cudaStreamPerThread);
                }
                else if (sampleStatus == cudaSuccess &&
                    logits->GetDataType() == X::TensorDataType::BFLOAT16) {
                    sampleStatus = runLogitsTop1BF16(
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
                if (sampleStatus == cudaSuccess) {
                    sampleStatus = cudaStreamSynchronize(cudaStreamPerThread);
                }
                if (sampleStatus != cudaSuccess) {
                    result->Set("status", X::Value("error"));
                    result->Set(
                        "error_code",
                        X::Value("compiled_gpu_sampling_failed"));
                    result->Set(
                        "error_message",
                        X::Value(cudaGetErrorString(sampleStatus)));
                    return result;
                }
            }
            if (m_backend == "openvino" && tokenId < 0) {
                result->Set("status", X::Value("error"));
                result->Set(
                    "error_code",
                    X::Value("compiled_logits_non_finite"));
                result->Set(
                    "error_message",
                    X::Value(
                        "OpenVINO produced no finite logits; this device or "
                        "driver does not safely execute the selected "
                        "precision profile"));
                return result;
            }
            result->Set("token_id", X::Value(tokenId));
            result->Set("token_value", X::Value(tokenValue));
            sampledTokenId = tokenId;
            const auto firstTokenReady = std::chrono::steady_clock::now();
            result->Set("time_to_first_token_ms", X::Value(
                std::chrono::duration<double, std::milli>(firstTokenReady - requestStart).count()));
            }
        }

        if (requestedNewTokens > 0) {
            if (!frontendActive || !m_decodeRuntime || sampledTokenId < 0) {
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("compiled_generation_runtime_unavailable"));
                return result;
            }
            X::List activeInputs(inputs);
            const int expectedInputCount = frontendIsVL ? 15 :
                (frontendIsASR ? 9 : 7);
            if (activeInputs->Size() != expectedInputCount) {
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("compiled_generation_cache_bindings_missing"));
                return result;
            }
            std::vector<int64_t> generatedTokens;
            generatedTokens.reserve(static_cast<size_t>(requestedNewTokens));
            generatedTokens.push_back(sampledTokenId);
            std::string tokenizerError;
            const auto tokenizer = Tokenization::GetCachedQwenTokenizer(m_weightsLocation, &tokenizerError);
            if (!tokenizer) {
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("compiled_generation_tokenizer_unavailable"));
                result->Set("error_message", X::Value(tokenizerError));
                return result;
            }
            const int64_t endOfText = tokenizer->TokenId("<|endoftext|>");
            const int64_t imEnd = tokenizer->TokenId("<|im_end|>");
            const bool ignoreEos = requestDict["ignore_eos"].IsValid() &&
                requestDict["ignore_eos"].ToLongLong() != 0;
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
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("compiled_generation_exceeds_kv_profile"));
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
            X::Value decodeTokenTensor = makeDecodeTensor(
                X::TensorDataType::LONGLONG, {1, 1}, &decodeTokenValue, sizeof(decodeTokenValue));
            X::Value decodePositionTensor = makeDecodeTensor(
                X::TensorDataType::LONGLONG,
                {positionComponents, 1, 1},
                decodeRopePositions,
                static_cast<size_t>(positionComponents) * sizeof(int64_t));
            X::Value decodeContextTensor = makeDecodeTensor(
                X::TensorDataType::INT, {1}, &decodeContextLength, sizeof(decodeContextLength));
            X::Value decodeSlotTensor = makeDecodeTensor(
                X::TensorDataType::INT, {1}, &decodeSlotPosition, sizeof(decodeSlotPosition));
            if (!decodeTokenTensor.IsTensor() || !decodePositionTensor.IsTensor() ||
                !decodeContextTensor.IsTensor() || !decodeSlotTensor.IsTensor()) {
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("compiled_decode_metadata_allocation_failed"));
                return result;
            }
            X::Tensor decodeTokenStorage(decodeTokenTensor);
            X::Tensor decodePositionStorage(decodePositionTensor);
            X::Tensor decodeContextStorage(decodeContextTensor);
            X::Tensor decodeSlotStorage(decodeSlotTensor);
            void* decodeTokenDevice = m_backend == "openvino"
                ? decodeTokenStorage->GetData()
                : TensorHelper::GetGPUMemory(decodeTokenStorage);
            void* decodePositionDevice = m_backend == "openvino"
                ? decodePositionStorage->GetData()
                : TensorHelper::GetGPUMemory(decodePositionStorage);
            void* decodeContextDevice = m_backend == "openvino"
                ? decodeContextStorage->GetData()
                : TensorHelper::GetGPUMemory(decodeContextStorage);
            void* decodeSlotDevice = m_backend == "openvino"
                ? decodeSlotStorage->GetData()
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
                }
                if (metadataStatus != cudaSuccess) {
                    result->Set("status", X::Value("error"));
                    result->Set("error_code", X::Value("compiled_decode_metadata_upload_failed"));
                    result->Set("error_message", X::Value(cudaGetErrorString(metadataStatus)));
                    return result;
                }
                X::V<X::XList> decodeInputs;
                const int keyInputIndex = frontendIsVL ? 11 :
                    (frontendIsASR ? 5 : 3);
                const int valueInputIndex = frontendIsVL ? 12 :
                    (frontendIsASR ? 6 : 4);
                const int tableInputIndex = frontendIsVL ? 13 :
                    (frontendIsASR ? 7 : 5);
                decodeInputs->AddItem(decodeTokenTensor);
                decodeInputs->AddItem(decodePositionTensor);
                decodeInputs->AddItem(activeInputs->Get(keyInputIndex));
                decodeInputs->AddItem(activeInputs->Get(valueInputIndex));
                decodeInputs->AddItem(activeInputs->Get(tableInputIndex));
                decodeInputs->AddItem(decodeContextTensor);
                decodeInputs->AddItem(decodeSlotTensor);
                X::Dict decodeRequest;
                decodeRequest->Set("inputs", X::Value(decodeInputs));
                decodeRequest->Set("sample", X::Value("greedy"));
                decodeRequest->Set("reuse_output", X::Value(1));
                if (m_backend == "openvino" && generatedIndex == 1) {
                    decodeRequest->Set(
                        "reset_openvino_state", X::Value(1));
                }
                X::Value decodeValue = m_decodeRuntime->Forward(decodeRequest);
                if (!decodeValue.IsDict()) {
                    result->Set("status", X::Value("error"));
                    result->Set("error_code", X::Value("compiled_decode_failed"));
                    return result;
                }
                X::Dict decodeResult(decodeValue);
                if (decodeResult["status"].ToString() != "ok") return decodeValue;
                sampledTokenId = decodeResult["token_id"].ToLongLong();
                generatedTokens.push_back(sampledTokenId);
            }
            const auto decodeEnd = std::chrono::steady_clock::now();
            const double decodeMs =
                std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count();
            X::V<X::XList> tokenList;
            for (const int64_t token : generatedTokens) tokenList->AddItem(X::Value(token));
            result->Set("token_ids", X::Value(tokenList));
            result->Set("text", X::Value(tokenizer->Decode(generatedTokens, true)));
            result->Set("generated_token_count", X::Value(static_cast<long long>(generatedTokens.size())));
            result->Set("decode_ms", X::Value(decodeMs));
            result->Set("decode_tokens_per_second", X::Value(
                generatedTokens.size() > 1 && decodeMs > 0.0
                    ? static_cast<double>(generatedTokens.size() - 1) * 1000.0 / decodeMs
                    : 0.0));
            result->Set("token_id", X::Value(sampledTokenId));
        }
        const bool returnLogits = !sampleGreedy ||
            (requestDict["return_logits"].IsValid() && requestDict["return_logits"].ToLongLong() != 0);
        if (returnLogits) result->Set("output", output);
        if (frontendActive) {
            result->Set(
                "prompt_token_count", X::Value(frontendPromptTokenCount));
        }
        if (frontendIsVL) {
            result->Set("visual_token_count", X::Value(frontendInputs.visualTokenCount));
            result->Set("source_height", X::Value(frontendInputs.sourceHeight));
            result->Set("source_width", X::Value(frontendInputs.sourceWidth));
            result->Set("height", X::Value(frontendInputs.resizedHeight));
            result->Set("width", X::Value(frontendInputs.resizedWidth));
        }
        if (frontendIsASR) {
            result->Set(
                "audio_token_count", X::Value(asrFrontendInputs.audioTokenCount));
            result->Set(
                "audio_sample_count", X::Value(asrFrontendInputs.audioSampleCount));
            result->Set(
                "audio_duration_seconds",
                X::Value(asrFrontendInputs.audioDurationSeconds));
        }
        result->Set("total_ms", X::Value(
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - requestStart).count()));
        if (frontendActive && inputs.IsList()) {
            X::List requestInputs(inputs);
            const int reusableKeyIndex = frontendIsVL ? 11 :
                (frontendIsASR ? 5 : 3);
            const int reusableValueIndex = frontendIsVL ? 12 :
                (frontendIsASR ? 6 : 4);
            for (long long index = 0; index < requestInputs->Size(); ++index) {
                if (index == reusableKeyIndex || index == reusableValueIndex) {
                    continue;
                }
                X::Value inputValue = requestInputs->Get(index);
                if (!inputValue.IsTensor()) continue;
                X::Tensor tensor(inputValue);
                TensorHelper::ReleaseGPUMemory(tensor);
            }
        }
        return result;
    }

    X::Value CompiledModelRuntime::DebugProbe(
        const std::string& probe,
        X::Value argument)
    {
        std::lock_guard<std::mutex> guard(m_mutex);
        X::Dict result;
        result->Set("probe", X::Value(probe));
        if (probe == "paged_kv_bf16") {
            if (!argument.IsDict()) {
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("paged_kv_probe_requires_dict"));
                return result;
            }
            X::Dict options(argument);
            X::Value qkvValue = options["qkv"];
            X::Value keyPagesValue = options["key_pages"];
            X::Value valuePagesValue = options["value_pages"];
            X::Value pageTableValue = options["page_table"];
            if (pageTableValue.IsList()) {
                X::List pageList(pageTableValue);
                X::Tensor pageTensor(X::g_pXHost->CreateTensor());
                X::Port::vector<int> pageShape(1);
                pageShape.push_back(static_cast<int>(pageList->Size()));
                pageTensor->SetDataType(X::TensorDataType::INT);
                pageTensor->SetShape(pageShape);
                X::Value init;
                if (pageTensor->Create(init) && pageTensor->GetData()) {
                    auto* pageData = reinterpret_cast<int*>(pageTensor->GetData());
                    for (long long index = 0; index < pageList->Size(); ++index) {
                        pageData[index] = static_cast<int>(pageList->Get(index).ToLongLong());
                    }
                    if (TensorHelper::EnsureGPUMemory(pageTensor) == TensorOpStatus::Success) {
                        pageTableValue = X::Value(pageTensor);
                    }
                }
            }
            if (!qkvValue.IsTensor() || !keyPagesValue.IsTensor() ||
                !valuePagesValue.IsTensor() || !pageTableValue.IsTensor()) {
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("paged_kv_probe_tensor_missing"));
                return result;
            }
            X::Tensor qkv(qkvValue);
            X::Tensor keyPages(keyPagesValue);
            X::Tensor valuePages(valuePagesValue);
            X::Tensor pageTable(pageTableValue);
            const int tokenCount = static_cast<int>(options["token_count"].ToLongLong());
            const int startPosition = static_cast<int>(options["start_position"].ToLongLong());
            const int sequenceLength = static_cast<int>(options["sequence_length"].ToLongLong());
            const int pageSize = static_cast<int>(options["page_size"].ToLongLong());
            const int qHeads = static_cast<int>(options["q_heads"].ToLongLong());
            const int kvHeads = static_cast<int>(options["kv_heads"].ToLongLong());
            const int headDim = static_cast<int>(options["head_dim"].ToLongLong());
            const int qWidth = qHeads * headDim;
            const int kvWidth = kvHeads * headDim;
            const int qkvWidth = qWidth + 2 * kvWidth;
            if (qkv->GetDataType() != X::TensorDataType::BFLOAT16 ||
                keyPages->GetDataType() != X::TensorDataType::BFLOAT16 ||
                valuePages->GetDataType() != X::TensorDataType::BFLOAT16 ||
                pageTable->GetDataType() != X::TensorDataType::INT ||
                qkv->GetDimCount() != 2 || qkv->GetDimSize(0) < tokenCount ||
                qkv->GetDimSize(1) != qkvWidth || tokenCount <= 0 ||
                sequenceLength <= 0 || startPosition < 0 || pageSize <= 0 ||
                qHeads <= 0 || kvHeads <= 0 || headDim <= 0 || qHeads % kvHeads != 0) {
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("paged_kv_probe_shape_or_dtype_invalid"));
                return result;
            }
            if (TensorHelper::EnsureGPUMemory(qkv) != TensorOpStatus::Success ||
                TensorHelper::EnsureGPUMemory(keyPages) != TensorOpStatus::Success ||
                TensorHelper::EnsureGPUMemory(valuePages) != TensorOpStatus::Success ||
                TensorHelper::EnsureGPUMemory(pageTable) != TensorOpStatus::Success) {
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("paged_kv_probe_gpu_residency_failed"));
                return result;
            }

            auto* qkvDevice = static_cast<bfloat16*>(TensorHelper::GetGPUMemory(qkv));
            auto* keyDevice = static_cast<bfloat16*>(TensorHelper::GetGPUMemory(keyPages));
            auto* valueDevice = static_cast<bfloat16*>(TensorHelper::GetGPUMemory(valuePages));
            auto* tableDevice = static_cast<int*>(TensorHelper::GetGPUMemory(pageTable));
            cudaError_t status = runTextPagedKVWriteBF16(
                qkvDevice, keyDevice, valueDevice, tableDevice,
                tokenCount, startPosition, pageSize, qHeads, kvHeads, headDim,
                cudaStreamPerThread);
            void* outputDevice = nullptr;
            if (status == cudaSuccess) {
                status = cudaMalloc(&outputDevice, static_cast<size_t>(qWidth) * sizeof(unsigned short));
            }
            const bfloat16* lastQ = qkvDevice + static_cast<size_t>(tokenCount - 1) * qkvWidth;
            const std::string implementation = options["implementation"].IsValid()
                ? options["implementation"].ToString()
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
                int pageTableElements = 1;
                for (int dimension = 0; dimension < pageTable->GetDimCount(); ++dimension) {
                    pageTableElements *= pageTable->GetDimSize(dimension);
                }
                const int maxSequenceLength = pageTableElements * pageSize;
                const int splitCount = (maxSequenceLength + 127) / 128;
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
                if (status == cudaSuccess) status = cudaStreamSynchronize(cudaStreamPerThread);
            }
            else if (status == cudaSuccess) {
                status = cudaErrorInvalidValue;
            }
            if (metadataDevice) cudaFree(metadataDevice);
            if (workspaceDevice) cudaFree(workspaceDevice);
            if (status != cudaSuccess) {
                if (outputDevice) cudaFree(outputDevice);
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("paged_kv_probe_kernel_failed"));
                result->Set("error_message", X::Value(cudaGetErrorString(status)));
                return result;
            }
            X::Tensor output(X::g_pXHost->CreateTensor());
            X::Port::vector<int> outputShape(2);
            outputShape.push_back(qHeads);
            outputShape.push_back(headDim);
            output->SetDataType(X::TensorDataType::BFLOAT16);
            output->SetShape(outputShape);
            if (TensorHelper::AttachGPUMemory(output, outputDevice) != TensorOpStatus::Success) {
                cudaFree(outputDevice);
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("paged_kv_probe_output_attach_failed"));
                return result;
            }
            result->Set("status", X::Value("ok"));
            result->Set("output", X::Value(output));
            result->Set("key_pages", keyPagesValue);
            result->Set("value_pages", valuePagesValue);
            result->Set("sequence_length", X::Value(sequenceLength));
            result->Set("implementation", X::Value(implementation));
            return result;
        }
        if (probe != "weight") {
            result->Set("status", X::Value("error"));
            result->Set("error_code", X::Value("unknown_compiled_probe"));
            return result;
        }

        const std::string name = argument.ToString();
        auto cached = m_loadedWeights.find(name);
        if (cached != m_loadedWeights.end()) {
            result->Set("status", X::Value("ok"));
            result->Set("name", X::Value(name));
            result->Set("cache_hit", X::Value(true));
            result->Set("tensor", cached->second);
            return result;
        }

        const SafeTensorMetadata* metadata = m_weightIndex.Find(name);
        if (!metadata) {
            result->Set("status", X::Value("error"));
            result->Set("error_code", X::Value("weight_not_found"));
            result->Set("name", X::Value(name));
            return result;
        }

        X::TensorDataType dataType;
        if (metadata->dataType == "F32") dataType = X::TensorDataType::FLOAT32;
        else if (metadata->dataType == "BF16") dataType = X::TensorDataType::BFLOAT16;
        else if (metadata->dataType == "F16") dataType = X::TensorDataType::FLOAT16;
        else {
            result->Set("status", X::Value("error"));
            result->Set("error_code", X::Value("weight_dtype_unsupported"));
            result->Set("dtype", X::Value(metadata->dataType));
            return result;
        }

        X::Port::vector<int> shape(static_cast<int>(metadata->shape.size()));
        for (const long long dimension : metadata->shape) {
            if (dimension > std::numeric_limits<int>::max()) {
                result->Set("status", X::Value("error"));
                result->Set("error_code", X::Value("weight_shape_unsupported"));
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
            result->Set("status", X::Value("error"));
            result->Set("error_code", X::Value("weight_gpu_load_failed"));
            result->Set("error_message", X::Value(cudaGetErrorString(cudaError)));
            return result;
        }

        X::Tensor tensor(X::g_pXHost->CreateTensor());
        tensor->SetDataType(dataType);
        tensor->SetShape(shape);
        if (TensorHelper::AttachGPUMemory(tensor, deviceMemory) != TensorOpStatus::Success) {
            cudaFree(deviceMemory);
            result->Set("status", X::Value("error"));
            result->Set("error_code", X::Value("weight_tensor_attach_failed"));
            return result;
        }

        X::Value tensorValue(tensor);
        m_loadedWeightBytes += static_cast<long long>(metadata->dataSize);
        m_loadedWeights.emplace(name, tensorValue);
        result->Set("status", X::Value("ok"));
        result->Set("name", X::Value(name));
        result->Set("cache_hit", X::Value(false));
        result->Set("bytes", X::Value(static_cast<long long>(metadata->dataSize)));
        result->Set("tensor", tensorValue);
        return result;
    }
}
