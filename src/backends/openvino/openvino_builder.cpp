#include "openvino_builder.h"

#include "garnet_tensor.h"
#include "q4q8_linear.h"
#include "qwen3_native_decode_engine.h"
#include "tensor_helper.h"
#include "weight_quantization.h"

#include <algorithm>
#include <climits>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <stdexcept>
#include <unordered_map>

#if defined(GARNET_WITH_OPENVINO)
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#endif

namespace Garnet
{
#if defined(GARNET_WITH_OPENVINO)
    namespace
    {
        struct OpenVINOExecutionSession
        {
            std::unique_ptr<ov::InferRequest> request;
            bool stateInitialized = false;
            bool profileDumped = false;
            std::mutex mutex;
        };

        struct CachedOpenVINOExecution
        {
            ov::Core core;
            ov::CompiledModel model;
            std::mutex sessionsMutex;
            std::unordered_map<
                std::uint64_t,
                std::shared_ptr<OpenVINOExecutionSession>> sessions;
            std::mutex nativeLMHeadMutex;
            std::shared_ptr<Q4Q8Linear> nativeLMHead;
        };

        std::mutex g_openVINOCacheMutex;
        std::unordered_map<
            std::string,
            std::shared_ptr<CachedOpenVINOExecution>> g_openVINOCache;
        std::mutex g_nativeDecodeCacheMutex;
        std::unordered_map<
            std::string,
            std::shared_ptr<Qwen3NativeDecodeEngine>> g_nativeDecodeCache;
        std::string OpenVINODevice();

        bool OpenVINOFullNativeCPUDecode()
        {
            const char* value = std::getenv(
                "GARNET_OPENVINO_CPU_NATIVE_DECODE");
            return value && std::string(value) == "1" &&
                OpenVINODevice().rfind("CPU", 0) == 0;
        }

        std::filesystem::path NativeDecodeMarker(
            const std::string& enginePath)
        {
            return std::filesystem::path(
                enginePath + ".garnet_qwen3_native_decode_v2");
        }

        std::filesystem::path NativeDecodePackedCache(
            const std::string& enginePath)
        {
            return std::filesystem::path(
                enginePath + ".garnet_q4q8_avx2_pack_v1");
        }

        bool OpenVINOPackedCPUProjections()
        {
            const char* value = std::getenv(
                "GARNET_OPENVINO_CPU_PACKED_PROJECTIONS");
            return value && std::string(value) == "1";
        }

        bool OpenVINOSignedCPUWeights()
        {
            const char* value = std::getenv(
                "GARNET_OPENVINO_CPU_SIGNED_I4");
            return value && std::string(value) == "1";
        }

        bool OpenVINONativeCPUQ4Q8()
        {
            const char* value = std::getenv(
                "GARNET_OPENVINO_CPU_NATIVE_Q4Q8");
            return value && std::string(value) == "1" &&
                OpenVINODevice().rfind("CPU", 0) == 0;
        }

        int OpenVINONativeCPUThreads()
        {
            const char* value = std::getenv(
                "GARNET_OPENVINO_CPU_THREADS");
            return value && *value ? std::max(1, std::stoi(value)) : 4;
        }

        int OpenVINOCPUWeightGroupSize()
        {
            const char* value = std::getenv(
                "GARNET_OPENVINO_CPU_WEIGHT_GROUP_SIZE");
            if (!value || !*value) return 128;
            const int parsed = std::stoi(value);
            if (parsed == -1 || parsed == 32 || parsed == 64 ||
                parsed == 128) {
                return parsed;
            }
            throw std::invalid_argument(
                "GARNET_OPENVINO_CPU_WEIGHT_GROUP_SIZE must be "
                "-1, 32, 64, or 128");
        }

        ov::AnyMap OpenVINOCPURuntimeProperties()
        {
            ov::AnyMap properties;
            if (OpenVINODevice().rfind("CPU", 0) != 0) return properties;
            if (const char* threads = std::getenv(
                    "GARNET_OPENVINO_CPU_THREADS");
                threads && *threads) {
                properties[ov::inference_num_threads.name()] =
                    static_cast<int32_t>(std::stoi(threads));
            }
            if (const char* hyperthreading = std::getenv(
                    "GARNET_OPENVINO_CPU_HYPERTHREADING");
                hyperthreading && *hyperthreading) {
                properties[ov::hint::enable_hyper_threading.name()] =
                    std::string(hyperthreading) != "0";
            }
            if (const char* profile = std::getenv(
                    "GARNET_OPENVINO_PROFILE");
                profile && std::string(profile) == "1") {
                properties[ov::enable_profiling.name()] = true;
            }
            properties[ov::hint::performance_mode.name()] =
                ov::hint::PerformanceMode::LATENCY;
            properties[ov::hint::execution_mode.name()] =
                ov::hint::ExecutionMode::PERFORMANCE;
            return properties;
        }

        ov::element::Type ToOpenVINOType(X::TensorDataType type)
        {
            switch (type) {
            case X::TensorDataType::FLOAT32: return ov::element::f32;
            case X::TensorDataType::BFLOAT16: return ov::element::bf16;
            case X::TensorDataType::INT: return ov::element::i32;
            case X::TensorDataType::LONGLONG: return ov::element::i64;
            default: return ov::element::dynamic;
            }
        }

        X::TensorDataType ToXLangType(const ov::element::Type& type)
        {
            if (type == ov::element::f32) return X::TensorDataType::FLOAT32;
            if (type == ov::element::bf16) return X::TensorDataType::BFLOAT16;
            if (type == ov::element::i32) return X::TensorDataType::INT;
            if (type == ov::element::i64) return X::TensorDataType::LONGLONG;
            return X::TensorDataType::UNKNOWN;
        }

        ov::Shape ToOpenVINOShape(X::Tensor tensor)
        {
            ov::Shape shape;
            shape.reserve(static_cast<size_t>(tensor->GetDimCount()));
            for (int index = 0; index < tensor->GetDimCount(); ++index) {
                shape.push_back(static_cast<size_t>(tensor->GetDimSize(index)));
            }
            return shape;
        }

        ov::Output<ov::Node> I64Scalar(int64_t value)
        {
            return ov::opset13::Constant::create(
                ov::element::i64, ov::Shape{}, {value});
        }

        ov::Output<ov::Node> I64Vector(
            std::initializer_list<int64_t> values)
        {
            return ov::opset13::Constant::create(
                ov::element::i64, ov::Shape{values.size()}, values);
        }

        ov::Output<ov::Node> Reshape(
            const ov::Output<ov::Node>& value,
            std::initializer_list<int64_t> shape)
        {
            return std::make_shared<ov::opset13::Reshape>(
                value, I64Vector(shape), false);
        }

        ov::Output<ov::Node> RMSNorm(
            const ov::Output<ov::Node>& value,
            const ov::Output<ov::Node>& scale,
            float epsilon)
        {
            const auto sourceType = value.get_element_type();
            auto sourceFloat = std::make_shared<ov::opset13::Convert>(
                value, ov::element::f32);
            auto square = std::make_shared<ov::opset13::Multiply>(
                sourceFloat, sourceFloat);
            auto mean = std::make_shared<ov::opset13::ReduceMean>(
                square, I64Vector({-1}), true);
            auto variance = std::make_shared<ov::opset13::Add>(
                mean, ov::opset13::Constant::create(
                    ov::element::f32, ov::Shape{}, {epsilon}));
            auto root = std::make_shared<ov::opset13::Sqrt>(variance);
            auto normalized = std::make_shared<ov::opset13::Divide>(
                sourceFloat, root);
            auto scaleFloat = std::make_shared<ov::opset13::Convert>(
                scale, ov::element::f32);
            auto scaled = std::make_shared<ov::opset13::Multiply>(
                normalized, scaleFloat);
            return std::make_shared<ov::opset13::Convert>(
                scaled, sourceType);
        }

        ov::Output<ov::Node> ExpandKVHeads(
            const ov::Output<ov::Node>& value,
            int heads,
            int kvHeads)
        {
            std::vector<int64_t> indices(static_cast<size_t>(heads));
            const int group = heads / kvHeads;
            for (int head = 0; head < heads; ++head) {
                indices[static_cast<size_t>(head)] = head / group;
            }
            auto indexNode = ov::opset13::Constant::create(
                ov::element::i64, ov::Shape{indices.size()}, indices);
            return std::make_shared<ov::opset13::Gather>(
                value, indexNode, I64Scalar(2));
        }

        ov::Output<ov::Node> ApplyRotary(
            const ov::Output<ov::Node>& value,
            const ov::Output<ov::Node>& positions,
            int headDim,
            float theta)
        {
            const auto sourceType = value.get_element_type();
            std::vector<float> inverse(static_cast<size_t>(headDim / 2));
            for (int index = 0; index < headDim / 2; ++index) {
                inverse[static_cast<size_t>(index)] =
                    std::pow(theta, -2.0F * static_cast<float>(index) /
                        static_cast<float>(headDim));
            }
            auto inverseNode = ov::opset13::Constant::create(
                ov::element::f32, ov::Shape{inverse.size()}, inverse);
            auto positionFloat = std::make_shared<ov::opset13::Convert>(
                positions, ov::element::f32);
            auto positionRows = Reshape(positionFloat, {-1, 1});
            auto angles = std::make_shared<ov::opset13::Multiply>(
                positionRows, inverseNode);
            auto frequencies = std::make_shared<ov::opset13::Concat>(
                ov::OutputVector{angles, angles}, 1);
            auto cosineFloat = Reshape(
                std::make_shared<ov::opset13::Cos>(frequencies),
                {1, -1, 1, headDim});
            auto sineFloat = Reshape(
                std::make_shared<ov::opset13::Sin>(frequencies),
                {1, -1, 1, headDim});
            auto valueFloat = std::make_shared<ov::opset13::Convert>(
                value, ov::element::f32);
            auto halves = std::make_shared<ov::opset13::VariadicSplit>(
                valueFloat, I64Scalar(-1),
                I64Vector({headDim / 2, headDim / 2}));
            auto rotated = std::make_shared<ov::opset13::Concat>(
                ov::OutputVector{
                    std::make_shared<ov::opset13::Negative>(halves->output(1)),
                    halves->output(0)},
                -1);
            auto resultFloat = std::make_shared<ov::opset13::Add>(
                std::make_shared<ov::opset13::Multiply>(
                    valueFloat, cosineFloat),
                std::make_shared<ov::opset13::Multiply>(
                    rotated, sineFloat));
            return std::make_shared<ov::opset13::Convert>(
                resultFloat, sourceType);
        }

        ov::Output<ov::Node> GroupedAttention(
            const ov::Output<ov::Node>& query,
            const ov::Output<ov::Node>& key,
            const ov::Output<ov::Node>& value,
            const ov::Output<ov::Node>& validKeys,
            int heads,
            int kvHeads,
            int headDim,
            bool causal)
        {
            auto expandedKey = ExpandKVHeads(key, heads, kvHeads);
            auto expandedValue = ExpandKVHeads(value, heads, kvHeads);
            auto q = std::make_shared<ov::opset13::Transpose>(
                query, I64Vector({0, 2, 1, 3}));
            auto k = std::make_shared<ov::opset13::Transpose>(
                expandedKey, I64Vector({0, 2, 3, 1}));
            auto v = std::make_shared<ov::opset13::Transpose>(
                expandedValue, I64Vector({0, 2, 1, 3}));
            auto scores = std::make_shared<ov::opset13::MatMul>(q, k);
            auto scoresFloat = std::make_shared<ov::opset13::Convert>(
                scores, ov::element::f32);
            auto scaled = std::make_shared<ov::opset13::Multiply>(
                scoresFloat,
                ov::opset13::Constant::create(
                    ov::element::f32, ov::Shape{}, {
                        1.0F / std::sqrt(static_cast<float>(headDim))}));
            auto keyMask = std::make_shared<ov::opset13::Convert>(
                Reshape(validKeys, {1, 1, 1, -1}), ov::element::boolean);
            ov::Output<ov::Node> allowed = keyMask;
            if (causal) {
                const int64_t queryLength =
                    query.get_partial_shape()[1].get_length();
                const int64_t keyLength =
                    key.get_partial_shape()[1].get_length();
                std::vector<uint8_t> causalValues(
                    static_cast<size_t>(queryLength * keyLength), 0);
                for (int64_t row = 0; row < queryLength; ++row) {
                    for (int64_t column = 0;
                         column < keyLength && column <= row;
                         ++column) {
                        causalValues[static_cast<size_t>(
                            row * keyLength + column)] = 1;
                    }
                }
                auto causalMask = std::make_shared<ov::opset13::Constant>(
                    ov::element::boolean,
                    ov::Shape{
                        1, 1, static_cast<size_t>(queryLength),
                        static_cast<size_t>(keyLength)},
                    causalValues.data());
                allowed = std::make_shared<ov::opset13::LogicalAnd>(
                    keyMask, causalMask);
            }
            auto masked = std::make_shared<ov::opset13::Select>(
                allowed, scaled,
                ov::opset13::Constant::create(
                    ov::element::f32, ov::Shape{}, {-1.0e9F}));
            auto probabilities = std::make_shared<ov::opset13::Softmax>(
                masked, -1);
            auto probabilitiesTyped = std::make_shared<ov::opset13::Convert>(
                probabilities, v->get_element_type());
            auto attended = std::make_shared<ov::opset13::MatMul>(
                probabilitiesTyped, v);
            auto tokenMajor = std::make_shared<ov::opset13::Transpose>(
                attended, I64Vector({0, 2, 1, 3}));
            const auto queryShape = query.get_partial_shape();
            return Reshape(
                tokenMajor,
                {
                    queryShape[0].get_length(),
                    queryShape[1].get_length(),
                    static_cast<int64_t>(heads) * headDim,
                });
        }

        std::string OpenVINODevice()
        {
            const char* configured = std::getenv("GARNET_OPENVINO_DEVICE");
            return configured && *configured ? configured : "CPU";
        }

        std::shared_ptr<CachedOpenVINOExecution> LoadCompiledModel(
            const std::string& enginePath,
            std::string& error)
        {
            std::lock_guard<std::mutex> cacheGuard(g_openVINOCacheMutex);
            const auto found = g_openVINOCache.find(enginePath);
            if (found != g_openVINOCache.end()) return found->second;
            try {
                std::ifstream stream(enginePath, std::ios::binary);
                if (!stream) {
                    error = "failed to open OpenVINO compiled model: " + enginePath;
                    return nullptr;
                }
                auto execution = std::make_shared<CachedOpenVINOExecution>();
                execution->model = execution->core.import_model(
                    stream,
                    OpenVINODevice(),
                    OpenVINOCPURuntimeProperties());
                g_openVINOCache[enginePath] = execution;
                return execution;
            }
            catch (const std::exception& exception) {
                error = std::string("OpenVINO import_model failed: ") + exception.what();
                return nullptr;
            }
        }

        std::shared_ptr<OpenVINOExecutionSession> GetExecutionSession(
            const std::shared_ptr<CachedOpenVINOExecution>& execution,
            std::uint64_t sessionId)
        {
            std::lock_guard<std::mutex> guard(execution->sessionsMutex);
            const auto found = execution->sessions.find(sessionId);
            if (found != execution->sessions.end()) return found->second;
            auto session = std::make_shared<OpenVINOExecutionSession>();
            session->request = std::make_unique<ov::InferRequest>(
                execution->model.create_infer_request());
            execution->sessions.emplace(sessionId, session);
            return session;
        }

        std::shared_ptr<Q4Q8Linear> GetNativeLMHead(
            const std::shared_ptr<CachedOpenVINOExecution>& execution,
            const SafeTensorsIndex* weightIndex,
            std::string& error)
        {
            if (!OpenVINONativeCPUQ4Q8() || !weightIndex) return nullptr;
            std::lock_guard<std::mutex> guard(execution->nativeLMHeadMutex);
            if (execution->nativeLMHead) return execution->nativeLMHead;
            const SafeTensorMetadata* metadata =
                weightIndex->Find("lm_head.weight");
            if (!metadata) {
                error = "native Q4Q8 lm_head.weight is missing";
                return nullptr;
            }
            auto mapped = std::make_shared<SafeTensorsMappedFile>();
            if (!mapped->Open(metadata->filePath, error)) return nullptr;
            const void* source = mapped->DataAt(
                metadata->dataOffset, metadata->dataSize);
            if (!source) {
                error = "native Q4Q8 failed to map lm_head.weight";
                return nullptr;
            }
            QuantizedWeight quantized;
            constexpr int groupSize = 128;
            if (!QuantizeSymmetricINT4(
                    "lm_head.weight", *metadata, source, groupSize,
                    quantized, error)) {
                return nullptr;
            }
            auto linear = std::make_shared<Q4Q8Linear>();
            if (!linear->Prepare(quantized, error)) return nullptr;
            execution->nativeLMHead = linear;
            return linear;
        }
    }

    struct OpenVINOBuilder::Implementation
    {
        struct CacheView
        {
            ov::Output<ov::Node> layer;
            int inputIndex = -1;
            int layerIndex = -1;
            bool key = false;
            std::shared_ptr<ov::op::util::Variable> variable;
            ov::element::Type stateType;
        };

        struct StatefulCacheSeed
        {
            unsigned long long tensorId = 0;
            int inputIndex = -1;
            bool key = false;
            ov::Shape layerShape;
            ov::element::Type type;
            ov::element::Type stateType;
        };

        struct TensorState
        {
            std::optional<CacheView> keyCache;
            std::optional<CacheView> valueCache;
            ov::Output<ov::Node> pageTable;
            ov::Output<ov::Node> contextLength;
            ov::Output<ov::Node> slotPosition;
            ov::Output<ov::Node> query;
            ov::Output<ov::Node> key;
            ov::Output<ov::Node> value;
        };

        struct CacheWrite
        {
            ov::Output<ov::Node> value;
            int inputIndex = -1;
            int layerIndex = -1;
            bool key = false;
        };

        std::unordered_map<unsigned long long, ov::Output<ov::Node>> tensors;
        std::unordered_map<unsigned long long, TensorState> states;
        std::unordered_map<unsigned long long, int> parameterIndices;
        std::unordered_map<int, ov::element::Type> inputTypes;
        std::unordered_map<unsigned long long, CacheView> cacheViews;
        std::unordered_map<std::string, ov::Output<ov::Node>> weights;
        std::unordered_map<
            std::string,
            std::shared_ptr<SafeTensorsMappedFile>> mappedWeightFiles;
        std::unordered_map<
            std::string,
            std::shared_ptr<QuantizedWeight>> quantizedWeights;
        std::unordered_map<
            std::string,
            std::shared_ptr<std::vector<uint8_t>>> openvinoPackedWeights;
        const SafeTensorsIndex* weightIndex = nullptr;
        ov::ParameterVector parameters;
        std::vector<CacheWrite> cacheWrites;
        bool statefulDecode = false;
        std::optional<StatefulCacheSeed> keyState;
        std::optional<StatefulCacheSeed> valueState;
        ov::SinkVector stateSinks;
        ov::op::util::VariableVector stateVariables;
        ov::Output<ov::Node> lastOutput;
        ov::element::Type computeType = ov::element::dynamic;
        bool int4WeightOnly = false;
        int quantizationBlockSize = 64;
        bool externalNativeLMHead = false;

        ov::Output<ov::Node> GetWeight(
            const std::string& name,
            std::string& error)
        {
            const auto existing = weights.find(name);
            if (existing != weights.end()) return existing->second;
            const SafeTensorMetadata* metadata =
                weightIndex ? weightIndex->Find(name) : nullptr;
            if (!metadata) {
                error = "OpenVINO checkpoint weight is missing: " + name;
                return {};
            }
            ov::element::Type type;
            if (metadata->dataType == "BF16") type = ov::element::bf16;
            else if (metadata->dataType == "F16") type = ov::element::f16;
            else if (metadata->dataType == "F32") type = ov::element::f32;
            else {
                error = "unsupported OpenVINO checkpoint dtype: " +
                    metadata->dataType;
                return {};
            }
            const std::string fileKey =
                metadata->filePath.lexically_normal().string();
            auto mapped = mappedWeightFiles.find(fileKey);
            if (mapped == mappedWeightFiles.end()) {
                auto file = std::make_shared<SafeTensorsMappedFile>();
                if (!file->Open(metadata->filePath, error)) return {};
                mapped = mappedWeightFiles.emplace(fileKey, std::move(file)).first;
            }
            const void* data = mapped->second->DataAt(
                metadata->dataOffset, metadata->dataSize);
            if (!data) {
                error = "failed to map OpenVINO checkpoint weight: " + name;
                return {};
            }
            const int effectiveBlockSize =
                quantizationBlockSize == -1
                    ? static_cast<int>(metadata->shape[1])
                    : quantizationBlockSize;
            if (int4WeightOnly &&
                IsQwenLinearWeightForINT4(
                    name, *metadata, effectiveBlockSize)) {
                auto quantized = std::make_shared<QuantizedWeight>();
                if (!QuantizeSymmetricINT4(
                        name, *metadata, data, effectiveBlockSize,
                        *quantized, error)) {
                    return {};
                }
                const auto scaleShape = quantized->ScaleShape();
                const int64_t rows = metadata->shape[0];
                const int64_t columns = metadata->shape[1];
                const int64_t groups = columns / quantized->blockSize;
                const bool signedWeights =
                    OpenVINODevice().rfind("CPU", 0) == 0 &&
                    OpenVINOSignedCPUWeights();
                // CPU receives Garnet's native symmetric signed I4 form.
                // Legacy Intel-GPU plugins require the equivalent U4 form:
                // u4 = i4 + 8, followed by a fixed zero-point subtraction.
                auto openvinoPacked =
                    std::make_shared<std::vector<uint8_t>>(
                        quantized->packedValues);
                if (!signedWeights) {
                    for (uint8_t& byte : *openvinoPacked) {
                        byte ^= 0x88U;
                    }
                }
                auto packed = std::make_shared<ov::opset13::Constant>(
                    signedWeights ? ov::element::i4 : ov::element::u4,
                    ov::Shape{
                        static_cast<size_t>(rows),
                        static_cast<size_t>(columns)},
                    openvinoPacked->data());
                auto blocked = Reshape(
                    packed, {rows, groups, quantized->blockSize});
                auto unpacked = std::make_shared<ov::opset13::Convert>(
                    blocked, ov::element::f16);
                ov::Output<ov::Node> centered = unpacked;
                if (!signedWeights) {
                    auto zeroPoint =
                        std::make_shared<ov::opset13::Constant>(
                            ov::element::f16,
                            ov::Shape{},
                            std::vector<float>{8.0F});
                    centered = std::make_shared<ov::opset13::Subtract>(
                        unpacked, zeroPoint);
                }
                auto scales = std::make_shared<ov::opset13::Constant>(
                    ov::element::f16,
                    ov::Shape{
                        static_cast<size_t>(scaleShape[0]),
                        static_cast<size_t>(scaleShape[1]),
                        1},
                    quantized->fp16Scales.data());
                auto dequantized = std::make_shared<ov::opset13::Multiply>(
                    centered, scales);
                ov::Output<ov::Node> output = Reshape(
                    dequantized, {rows, columns});
                openvinoPackedWeights[name] = std::move(openvinoPacked);
                quantizedWeights[name] = std::move(quantized);
                weights[name] = output;
                return output;
            }
            ov::Shape shape;
            shape.reserve(metadata->shape.size());
            for (const long long dimension : metadata->shape) {
                shape.push_back(static_cast<size_t>(dimension));
            }
            auto constant = std::make_shared<ov::opset13::Constant>(
                type, shape, data);
            constant->set_friendly_name(name);
            ov::Output<ov::Node> output = constant->output(0);
            if (computeType == ov::element::f16 &&
                type == ov::element::bf16) {
                output = std::make_shared<ov::opset13::Convert>(
                    output, computeType);
            }
            weights[name] = output;
            return output;
        }

        ov::Output<ov::Node> GetCombinedWeight(
            const std::vector<std::string>& names,
            std::string& error)
        {
            std::string cacheKey = "__combined";
            for (const std::string& name : names) {
                cacheKey += "\n" + name;
            }
            const auto existing = weights.find(cacheKey);
            if (existing != weights.end()) return existing->second;

            ov::OutputVector sourceWeights;
            sourceWeights.reserve(names.size());
            for (const std::string& name : names) {
                const auto weight = GetWeight(name, error);
                if (!weight.get_node_shared_ptr()) return {};
                sourceWeights.push_back(weight);
            }
            if (!int4WeightOnly) {
                auto combined = std::make_shared<ov::opset13::Concat>(
                    sourceWeights, 0);
                weights[cacheKey] = combined;
                return combined;
            }

            auto quantized = std::make_shared<QuantizedWeight>();
            auto packed = std::make_shared<std::vector<uint8_t>>();
            long long totalRows = 0;
            long long columns = 0;
            for (const std::string& name : names) {
                const auto item = quantizedWeights.find(name);
                const auto openvinoItem = openvinoPackedWeights.find(name);
                if (item == quantizedWeights.end() ||
                    openvinoItem == openvinoPackedWeights.end() ||
                    item->second->shape.size() != 2) {
                    error =
                        "OpenVINO combined INT4 weight is unavailable: " +
                        name;
                    return {};
                }
                const QuantizedWeight& source = *item->second;
                if (columns == 0) {
                    columns = source.shape[1];
                    quantized->blockSize = source.blockSize;
                    quantized->blockAxis = source.blockAxis;
                }
                if (source.shape[1] != columns ||
                    source.blockSize != quantized->blockSize) {
                    error =
                        "OpenVINO combined INT4 weights are incompatible";
                    return {};
                }
                totalRows += source.shape[0];
                packed->insert(
                    packed->end(),
                    openvinoItem->second->begin(),
                    openvinoItem->second->end());
                quantized->fp16Scales.insert(
                    quantized->fp16Scales.end(),
                    source.fp16Scales.begin(),
                    source.fp16Scales.end());
                quantized->maximumAbsoluteError = std::max(
                    quantized->maximumAbsoluteError,
                    source.maximumAbsoluteError);
            }
            quantized->shape = {totalRows, columns};
            const long long groups =
                columns / quantized->blockSize;
            const bool signedWeights =
                OpenVINODevice().rfind("CPU", 0) == 0 &&
                OpenVINOSignedCPUWeights();
            auto packedConstant =
                std::make_shared<ov::opset13::Constant>(
                    signedWeights ? ov::element::i4 : ov::element::u4,
                    ov::Shape{
                        static_cast<size_t>(totalRows),
                        static_cast<size_t>(columns)},
                    packed->data());
            auto blocked = Reshape(
                packedConstant,
                {totalRows, groups, quantized->blockSize});
            auto unpacked = std::make_shared<ov::opset13::Convert>(
                blocked, ov::element::f16);
            ov::Output<ov::Node> centered = unpacked;
            if (!signedWeights) {
                auto zeroPoint =
                    std::make_shared<ov::opset13::Constant>(
                        ov::element::f16,
                        ov::Shape{},
                        std::vector<float>{8.0F});
                centered = std::make_shared<ov::opset13::Subtract>(
                    unpacked, zeroPoint);
            }
            auto scales = std::make_shared<ov::opset13::Constant>(
                ov::element::f16,
                ov::Shape{
                    static_cast<size_t>(totalRows),
                    static_cast<size_t>(groups),
                    1},
                quantized->fp16Scales.data());
            auto dequantized = std::make_shared<ov::opset13::Multiply>(
                centered, scales);
            ov::Output<ov::Node> output = Reshape(
                dequantized, {totalRows, columns});
            openvinoPackedWeights[cacheKey] = std::move(packed);
            quantizedWeights[cacheKey] = std::move(quantized);
            weights[cacheKey] = output;
            return output;
        }
    };
#endif

    bool OpenVINOBuilder::IsAvailable()
    {
#if defined(GARNET_WITH_OPENVINO)
        return true;
#else
        return false;
#endif
    }

    bool OpenVINOBuilder::AnalyzeCapturedGraph(
        X::Value graph,
        X::Value forwardFunction,
        X::ARGS& graphArguments,
        std::vector<CapturedTensorOperation>& operations,
        std::string& errorMessage)
    {
        m_analyzedOperations.clear();
        m_analysisActive = true;
        X::TensorGraph tensorGraph(graph);
        X::KWARGS runOptions;
        runOptions.Add("Func", forwardFunction);
        ScopedLoweringContext loweringScope(*this);
        const bool ran = tensorGraph->Run(graphArguments, runOptions);
        m_analysisActive = false;
        if (!ran) {
            errorMessage = "xlang TensorGraph OpenVINO analysis replay failed";
            m_analyzedOperations.clear();
            return false;
        }
        operations = std::move(m_analyzedOperations);
        errorMessage.clear();
        return true;
    }

    bool OpenVINOBuilder::BuildCapturedGraph(
        X::Value graph,
        X::Value forwardFunction,
        X::ARGS& graphArguments,
        X::ARGS& symbolicInputs,
        const SafeTensorsIndex* weightIndex,
        const std::string& enginePath,
        const std::string& precision,
        std::string& errorMessage)
    {
#if !defined(GARNET_WITH_OPENVINO)
        errorMessage = "Garnet was built without the OpenVINO SDK";
        return false;
#else
        m_error.clear();
        m_loweringActive = true;
        Implementation implementation;
        implementation.weightIndex = weightIndex;
        implementation.computeType =
            precision == "fp16" || precision == "int4_fp16"
                ? ov::element::f16
                : ov::element::dynamic;
        implementation.int4WeightOnly = precision == "int4_fp16";
        implementation.quantizationBlockSize =
            implementation.int4WeightOnly &&
            OpenVINODevice().rfind("CPU", 0) == 0
                ? OpenVINOCPUWeightGroupSize()
                : 64;
        implementation.statefulDecode =
            symbolicInputs.size() == 7 &&
            symbolicInputs[2].IsTensor() &&
            symbolicInputs[3].IsTensor() &&
            X::Tensor(symbolicInputs[2])->GetDimCount() == 5 &&
            X::Tensor(symbolicInputs[3])->GetDimCount() == 5;
        m_implementation = &implementation;
        try {
            for (size_t index = 0; index < symbolicInputs.size(); ++index) {
                X::Value value = symbolicInputs[index];
                if (!value.IsTensor() || !value.IsObject()) {
                    m_error = "OpenVINO graph inputs must be tensors";
                    break;
                }
                X::Tensor tensor(value);
                const auto type = ToOpenVINOType(tensor->GetDataType());
                if (type == ov::element::dynamic) {
                    m_error = "unsupported symbolic OpenVINO input dtype";
                    break;
                }
                const auto tensorId = value.GetObj()->GetID();
                implementation.parameterIndices[tensorId] =
                    static_cast<int>(index);
                implementation.inputTypes[
                    static_cast<int>(index)] = type;
                if (implementation.statefulDecode &&
                    (index == 2 || index == 3)) {
                    const bool key = index == 2;
                    const auto fullShape = ToOpenVINOShape(tensor);
                    if (fullShape.size() != 5 || fullShape[0] == 0) {
                        m_error =
                            "OpenVINO decode cache shape is invalid";
                        break;
                    }
                    ov::Shape layerShape(
                        fullShape.begin() + 1, fullShape.end());
                    const std::string openVINODevice =
                        OpenVINODevice();
                    const bool gpuState =
                        openVINODevice.rfind("GPU", 0) == 0;
                    const auto internalType =
                        implementation.computeType == ov::element::f16 &&
                            type == ov::element::bf16
                            ? ov::element::f16
                            : type;
                    const auto stateType =
                        implementation.computeType == ov::element::f16 ||
                            (gpuState && type == ov::element::bf16)
                            ? ov::element::f16
                            : type;
                    Implementation::StatefulCacheSeed state{
                        tensorId,
                        static_cast<int>(index),
                        key,
                        std::move(layerShape),
                        internalType,
                        stateType};
                    implementation.tensors[tensorId] =
                        ov::opset13::Constant::create(
                            type, ov::Shape{}, {0});
                    if (key) implementation.keyState = std::move(state);
                    else implementation.valueState = std::move(state);
                    continue;
                }
                auto parameter = std::make_shared<ov::opset13::Parameter>(
                    type, ToOpenVINOShape(tensor));
                parameter->set_friendly_name("input_" + std::to_string(index));
                parameter->output(0).get_tensor().set_names(
                    {"input_" + std::to_string(index)});
                implementation.parameters.push_back(parameter);
                ov::Output<ov::Node> internalParameter = parameter;
                if (implementation.computeType == ov::element::f16 &&
                    type == ov::element::bf16) {
                    internalParameter =
                        std::make_shared<ov::opset13::Convert>(
                            parameter, ov::element::f16);
                }
                implementation.tensors[tensorId] = internalParameter;
            }
            if (m_error.empty()) {
                X::TensorGraph tensorGraph(graph);
                X::KWARGS runOptions;
                runOptions.Add("Func", forwardFunction);
                ScopedLoweringContext loweringScope(*this);
                if (!tensorGraph->Run(graphArguments, runOptions) && m_error.empty()) {
                    m_error = "xlang TensorGraph OpenVINO lowering replay failed";
                }
            }
            if (m_error.empty() && !implementation.lastOutput.get_node_shared_ptr()) {
                m_error = "captured graph produced no OpenVINO output";
            }
            if (m_error.empty()) {
                if (implementation.statefulDecode &&
                    !implementation.externalNativeLMHead) {
                    if (!implementation.keyState ||
                        !implementation.valueState ||
                        implementation.stateSinks.empty()) {
                        m_error =
                            "OpenVINO stateful decode cache variables are missing";
                    }
                    else {
                        auto top = std::make_shared<ov::opset13::TopK>(
                            implementation.lastOutput,
                            I64Scalar(1),
                            -1,
                            "max",
                            "none",
                            ov::element::i64);
                        implementation.lastOutput = top->output(1);
                    }
                }
            }
            if (m_error.empty()) {
                if (!implementation.statefulDecode &&
                    implementation.computeType == ov::element::f16 &&
                    implementation.lastOutput.get_element_type() ==
                        ov::element::f16) {
                    implementation.lastOutput =
                        std::make_shared<ov::opset13::Convert>(
                            implementation.lastOutput,
                            ov::element::bf16);
                }
                implementation.lastOutput.get_tensor().set_names({"output_0"});
                ov::OutputVector modelOutputs{implementation.lastOutput};
                for (const auto& write : implementation.cacheWrites) {
                    const std::string name =
                        std::string("cache_") +
                        (write.key ? "key" : "value") + "_input_" +
                        std::to_string(write.inputIndex) + "_layer_" +
                        std::to_string(write.layerIndex);
                    ov::Output<ov::Node> externalValue = write.value;
                    const auto inputType =
                        implementation.inputTypes.find(write.inputIndex);
                    if (inputType != implementation.inputTypes.end() &&
                        externalValue.get_element_type() !=
                            inputType->second) {
                        externalValue =
                            std::make_shared<ov::opset13::Convert>(
                                externalValue, inputType->second);
                    }
                    externalValue.get_tensor().set_names({name});
                    modelOutputs.push_back(externalValue);
                }
                auto model = implementation.statefulDecode
                    ? std::make_shared<ov::Model>(
                        modelOutputs,
                        implementation.stateSinks,
                        implementation.parameters,
                        implementation.stateVariables,
                        "garnet_tensor_graph")
                    : std::make_shared<ov::Model>(
                        modelOutputs,
                        implementation.parameters,
                        "garnet_tensor_graph");
                ov::Core core;
                ov::AnyMap compileProperties;
                if (implementation.computeType == ov::element::f16) {
                    compileProperties[
                        ov::hint::inference_precision.name()] =
                        ov::element::f16;
                    compileProperties[
                        ov::hint::performance_mode.name()] =
                        ov::hint::PerformanceMode::LATENCY;
                }
                const std::string device = OpenVINODevice();
                if (implementation.int4WeightOnly &&
                    device.rfind("CPU", 0) == 0) {
                    compileProperties[
                        ov::hint::performance_mode.name()] =
                        ov::hint::PerformanceMode::LATENCY;
                    compileProperties[
                        ov::hint::execution_mode.name()] =
                        ov::hint::ExecutionMode::PERFORMANCE;
                    if (const char* groupSize = std::getenv(
                            "GARNET_OPENVINO_CPU_DQ_GROUP_SIZE");
                        groupSize && *groupSize) {
                        compileProperties[
                            ov::hint::dynamic_quantization_group_size.name()] =
                            static_cast<uint64_t>(std::stoull(groupSize));
                    }
                    if (const char* threads = std::getenv(
                            "GARNET_OPENVINO_CPU_THREADS");
                        threads && *threads) {
                        compileProperties[
                            ov::inference_num_threads.name()] =
                            static_cast<int32_t>(std::stoi(threads));
                    }
                    if (const char* hyperthreading = std::getenv(
                            "GARNET_OPENVINO_CPU_HYPERTHREADING");
                        hyperthreading && *hyperthreading) {
                        compileProperties[
                            ov::hint::enable_hyper_threading.name()] =
                            std::string(hyperthreading) != "0";
                    }
                }
                ov::CompiledModel compiled = core.compile_model(
                    model, device, compileProperties);
                const std::filesystem::path outputPath(enginePath);
                std::filesystem::create_directories(outputPath.parent_path());
                const std::filesystem::path temporary =
                    outputPath.string() + ".tmp";
                {
                    std::ofstream stream(
                        temporary, std::ios::binary | std::ios::trunc);
                    compiled.export_model(stream);
                }
                std::error_code publishError;
                std::filesystem::remove(outputPath, publishError);
                publishError.clear();
                std::filesystem::rename(temporary, outputPath, publishError);
                if (publishError) {
                    m_error = "failed to publish OpenVINO compiled model: " +
                        publishError.message();
                }
                else {
                    if (implementation.statefulDecode) {
                        std::ofstream marker(
                            NativeDecodeMarker(enginePath),
                            std::ios::out | std::ios::trunc);
                        marker <<
                            "GARNET_XLANG_QWEN3_NATIVE_DECODE_V2\n";
                    }
                    std::lock_guard<std::mutex> cacheGuard(g_openVINOCacheMutex);
                    g_openVINOCache.erase(enginePath);
                }
            }
        }
        catch (const std::exception& exception) {
            m_error = std::string("OpenVINO graph compilation failed: ") +
                exception.what();
        }
        m_implementation = nullptr;
        m_loweringActive = false;
        errorMessage = m_error;
        return m_error.empty();
#endif
    }

    bool OpenVINOBuilder::PrepareCapturedEngine(
        const std::string& enginePath,
        const SafeTensorsIndex* weightIndex,
        std::string& errorMessage)
    {
#if defined(GARNET_WITH_OPENVINO)
        if (OpenVINOFullNativeCPUDecode() &&
            std::filesystem::is_regular_file(
                NativeDecodeMarker(enginePath))) {
            if (!weightIndex) {
                errorMessage =
                    "full native CPU decode requires a checkpoint index";
                return false;
            }
            std::lock_guard<std::mutex> guard(g_nativeDecodeCacheMutex);
            if (g_nativeDecodeCache.find(enginePath) ==
                g_nativeDecodeCache.end()) {
                auto engine =
                    std::make_shared<Qwen3NativeDecodeEngine>();
                if (!engine->Prepare(
                        *weightIndex,
                        NativeDecodePackedCache(enginePath),
                        errorMessage)) {
                    return false;
                }
                g_nativeDecodeCache.emplace(enginePath, std::move(engine));
            }
            errorMessage.clear();
            return true;
        }
        auto execution = LoadCompiledModel(enginePath, errorMessage);
        if (!execution) return false;
        if (OpenVINONativeCPUQ4Q8() && weightIndex &&
            !execution->model.outputs().empty()) {
            const SafeTensorMetadata* metadata =
                weightIndex->Find("lm_head.weight");
            const auto shape =
                execution->model.output(0).get_partial_shape();
            if (metadata && metadata->shape.size() == 2 &&
                shape.rank().is_static() &&
                shape.rank().get_length() == 3 &&
                shape[2].is_static() &&
                shape[2].get_length() == metadata->shape[1] &&
                !GetNativeLMHead(execution, weightIndex, errorMessage)) {
                return false;
            }
        }
        errorMessage.clear();
        return true;
#else
        errorMessage = "Garnet was built without the OpenVINO SDK";
        return false;
#endif
    }

    X::Value OpenVINOBuilder::RunCapturedEngine(
        const std::string& enginePath,
        X::Value inputsValue,
        const SafeTensorsIndex* weightIndex,
        X::Value,
        bool resetState,
        std::uint64_t sessionId,
        std::string& errorMessage)
    {
#if !defined(GARNET_WITH_OPENVINO)
        errorMessage = "Garnet was built without the OpenVINO SDK";
        return X::Value();
#else
        if (!inputsValue.IsList()) {
            errorMessage = "OpenVINO execution requires an inputs list";
            return X::Value();
        }
        if (OpenVINOFullNativeCPUDecode() &&
            std::filesystem::is_regular_file(
                NativeDecodeMarker(enginePath))) {
            std::shared_ptr<Qwen3NativeDecodeEngine> engine;
            {
                std::lock_guard<std::mutex> guard(
                    g_nativeDecodeCacheMutex);
                const auto found = g_nativeDecodeCache.find(enginePath);
                if (found != g_nativeDecodeCache.end()) {
                    engine = found->second;
                }
            }
            if (!engine) {
                errorMessage =
                    "full native CPU decode was not prepared";
                return X::Value();
            }
            X::List inputs(inputsValue);
            if (inputs->Size() != 7) {
                errorMessage =
                    "full native CPU decode requires seven graph inputs";
                return X::Value();
            }
            X::Tensor tensors[7];
            for (int index = 0; index < 7; ++index) {
                X::Value value = inputs->Get(index);
                if (!value.IsTensor()) {
                    errorMessage =
                        "full native CPU decode inputs must be tensors";
                    return X::Value();
                }
                tensors[index] = X::Tensor(value);
                if (tensors[index]->GetDeviceType() !=
                        X::TensorDeviceType::CPU ||
                    !tensors[index]->GetData()) {
                    errorMessage =
                        "full native CPU decode requires CPU tensors";
                    return X::Value();
                }
            }
            if (tensors[0]->GetDataType() !=
                    X::TensorDataType::LONGLONG ||
                tensors[1]->GetDataType() !=
                    X::TensorDataType::LONGLONG ||
                tensors[2]->GetDataType() !=
                    X::TensorDataType::BFLOAT16 ||
                tensors[3]->GetDataType() !=
                    X::TensorDataType::BFLOAT16 ||
                tensors[4]->GetDataType() != X::TensorDataType::INT ||
                tensors[5]->GetDataType() != X::TensorDataType::INT ||
                tensors[6]->GetDataType() != X::TensorDataType::INT ||
                tensors[2]->GetDimCount() != 5 ||
                tensors[3]->GetDimCount() != 5) {
                errorMessage =
                    "full native CPU decode input contract is invalid";
                return X::Value();
            }
            for (int dimension = 0; dimension < 5; ++dimension) {
                if (tensors[2]->GetDimSize(dimension) !=
                    tensors[3]->GetDimSize(dimension)) {
                    errorMessage =
                        "full native CPU key/value cache shapes differ";
                    return X::Value();
                }
            }
            int pageTableSize = 1;
            for (int dimension = 0;
                 dimension < tensors[4]->GetDimCount();
                 ++dimension) {
                const int size = tensors[4]->GetDimSize(dimension);
                if (size <= 0 || pageTableSize > INT_MAX / size) {
                    errorMessage =
                        "full native CPU page table shape is invalid";
                    return X::Value();
                }
                pageTableSize *= size;
            }
            Qwen3NativeDecodeEngine::DecodeInput input;
            input.tokenId = *reinterpret_cast<const long long*>(
                tensors[0]->GetData());
            input.position = *reinterpret_cast<const long long*>(
                tensors[1]->GetData());
            input.keyCache = reinterpret_cast<std::uint16_t*>(
                tensors[2]->GetData());
            input.valueCache = reinterpret_cast<std::uint16_t*>(
                tensors[3]->GetData());
            input.pageTable = reinterpret_cast<const int*>(
                tensors[4]->GetData());
            input.pageTableSize = pageTableSize;
            input.pages = tensors[2]->GetDimSize(1);
            input.pageSize = tensors[2]->GetDimSize(2);
            input.contextLength = *reinterpret_cast<const int*>(
                tensors[5]->GetData());
            input.slotPosition = *reinterpret_cast<const int*>(
                tensors[6]->GetData());
            long long token = -1;
            if (!engine->Decode(input, token, errorMessage)) {
                return X::Value();
            }
            X::Tensor output(X::g_pXHost->CreateTensor());
            X::Port::vector<int> shape(3);
            shape.push_back(1);
            shape.push_back(1);
            shape.push_back(1);
            output->SetDataType(X::TensorDataType::LONGLONG);
            output->SetShape(shape);
            X::Value initial;
            output->Create(initial);
            if (!output->GetData()) {
                errorMessage =
                    "failed to allocate full native CPU token output";
                return X::Value();
            }
            *reinterpret_cast<long long*>(output->GetData()) = token;
            errorMessage.clear();
            return X::Value(output);
        }
        auto execution = LoadCompiledModel(enginePath, errorMessage);
        if (!execution) return X::Value();
        try {
            auto session = GetExecutionSession(execution, sessionId);
            std::lock_guard<std::mutex> sessionGuard(session->mutex);
            X::List inputs(inputsValue);
            ov::InferRequest& request = *session->request;
            std::vector<X::Value> cpuCopies;
            cpuCopies.reserve(execution->model.inputs().size());
            const std::regex inputName(R"(input_([0-9]+))");
            for (size_t modelInputIndex = 0;
                 modelInputIndex < execution->model.inputs().size();
                 ++modelInputIndex) {
                const auto inputPort =
                    execution->model.input(modelInputIndex);
                std::string name;
                if (!inputPort.get_names().empty()) {
                    name = *inputPort.get_names().begin();
                }
                std::smatch match;
                if (!std::regex_match(name, match, inputName)) {
                    errorMessage =
                        "OpenVINO model input lacks original input identity";
                    return X::Value();
                }
                const int originalIndex = std::stoi(match[1].str());
                if (originalIndex < 0 || originalIndex >= inputs->Size()) {
                    errorMessage =
                        "OpenVINO execution input identity is out of range";
                    return X::Value();
                }
                X::Value inputValue = inputs->Get(originalIndex);
                if (!inputValue.IsTensor()) {
                    errorMessage = "OpenVINO inputs must be X::Tensor values";
                    return X::Value();
                }
                X::Tensor input(inputValue);
                if (input->GetDeviceType() != X::TensorDeviceType::CPU) {
                    inputValue = TensorHelper::CopyToCPUTensor(input);
                    if (!inputValue.IsTensor()) {
                        errorMessage = "failed to copy OpenVINO input to CPU";
                        return X::Value();
                    }
                    input = X::Tensor(inputValue);
                }
                cpuCopies.push_back(inputValue);
                const auto type = ToOpenVINOType(input->GetDataType());
                if (type == ov::element::dynamic || !input->GetData()) {
                    errorMessage = "unsupported or empty OpenVINO input tensor";
                    return X::Value();
                }
                request.set_input_tensor(
                    modelInputIndex,
                    ov::Tensor(type, ToOpenVINOShape(input), input->GetData()));
            }
            const auto variableStates = request.query_state();
            if (!variableStates.empty() &&
                (resetState || !session->stateInitialized)) {
                const std::regex stateName(
                    R"(cache_(key|value)_input_([0-9]+)_layer_([0-9]+))");
                for (auto variableState : request.query_state()) {
                    std::smatch match;
                    const std::string name = variableState.get_name();
                    if (!std::regex_match(name, match, stateName)) {
                        errorMessage =
                            "OpenVINO decode state identity is invalid: " +
                            name;
                        return X::Value();
                    }
                    const int originalIndex = std::stoi(match[2].str());
                    const int layerIndex = std::stoi(match[3].str());
                    if (originalIndex < 0 ||
                        originalIndex >= inputs->Size() ||
                        !inputs->Get(originalIndex).IsTensor()) {
                        errorMessage =
                            "OpenVINO decode state input is unavailable";
                        return X::Value();
                    }
                    X::Tensor cache(inputs->Get(originalIndex));
                    if (cache->GetDeviceType() !=
                            X::TensorDeviceType::CPU ||
                        !cache->GetData()) {
                        errorMessage =
                            "OpenVINO decode state requires a CPU cache seed";
                        return X::Value();
                    }
                    const auto type =
                        ToOpenVINOType(cache->GetDataType());
                    const ov::Tensor currentState =
                        variableState.get_state();
                    const size_t layerBytes =
                        currentState.get_byte_size();
                    const size_t offset =
                        static_cast<size_t>(layerIndex) * layerBytes;
                    if (offset + layerBytes >
                        static_cast<size_t>(cache->GetDataSize())) {
                        errorMessage =
                            "OpenVINO decode state seed exceeds cache input";
                        return X::Value();
                    }
                    if (currentState.get_element_type() == type) {
                        variableState.set_state(ov::Tensor(
                            type,
                            currentState.get_shape(),
                            cache->GetData() + offset));
                    }
                    else if (
                        type == ov::element::bf16 &&
                        currentState.get_element_type() ==
                            ov::element::f16) {
                        const auto* source =
                            reinterpret_cast<const ov::bfloat16*>(
                                cache->GetData() + offset);
                        ov::Tensor converted(
                            ov::element::f16,
                            currentState.get_shape());
                        auto* destination =
                            converted.data<ov::float16>();
                        for (size_t element = 0;
                             element < currentState.get_size();
                             ++element) {
                            destination[element] = ov::float16(
                                static_cast<float>(source[element]));
                        }
                        variableState.set_state(converted);
                    }
                    else {
                        errorMessage =
                            "OpenVINO decode state seed type conversion "
                            "is unsupported";
                        return X::Value();
                    }
                }
                session->stateInitialized = true;
            }
            request.infer();
            if (!session->profileDumped) {
                const char* profile = std::getenv("GARNET_OPENVINO_PROFILE");
                if (profile && std::string(profile) == "1") {
                    auto records = request.get_profiling_info();
                    std::sort(
                        records.begin(), records.end(),
                        [](const ov::ProfilingInfo& left,
                           const ov::ProfilingInfo& right) {
                            return left.real_time > right.real_time;
                        });
                    std::ofstream output(
                        enginePath + ".profile.txt",
                        std::ios::out | std::ios::trunc);
                    for (const auto& record : records) {
                        output << record.real_time.count() << '\t'
                               << record.cpu_time.count() << '\t'
                               << record.node_type << '\t'
                               << record.exec_type << '\t'
                               << record.node_name << '\n';
                    }
                    session->profileDumped = true;
                }
            }
            const std::regex cacheName(
                R"(cache_(key|value)_input_([0-9]+)_layer_([0-9]+))");
            for (size_t outputIndex = 1;
                 outputIndex < execution->model.outputs().size();
                 ++outputIndex) {
                const auto outputPort = execution->model.output(outputIndex);
                std::string outputName;
                if (!outputPort.get_names().empty()) {
                    outputName = *outputPort.get_names().begin();
                }
                std::smatch match;
                if (!std::regex_match(outputName, match, cacheName)) continue;
                const int inputIndex = std::stoi(match[2].str());
                const int layerIndex = std::stoi(match[3].str());
                if (inputIndex < 0 || inputIndex >= inputs->Size()) {
                    errorMessage = "OpenVINO cache output input index is invalid";
                    return X::Value();
                }
                X::Tensor cache(inputs->Get(inputIndex));
                if (cache->GetDeviceType() != X::TensorDeviceType::CPU ||
                    !cache->GetData() || cache->GetDimCount() < 1) {
                    errorMessage =
                        "OpenVINO Qwen cache inputs must be CPU tensors";
                    return X::Value();
                }
                const ov::Tensor cacheOutput =
                    request.get_output_tensor(outputIndex);
                const size_t layerBytes = cacheOutput.get_byte_size();
                const size_t offset =
                    static_cast<size_t>(layerIndex) * layerBytes;
                if (offset + layerBytes >
                    static_cast<size_t>(cache->GetDataSize())) {
                    errorMessage = "OpenVINO cache write exceeds input tensor";
                    return X::Value();
                }
                std::memcpy(
                    cache->GetData() + offset,
                    cacheOutput.data(),
                    layerBytes);
            }
            const ov::Tensor outputTensor = request.get_output_tensor(0);
            if (OpenVINONativeCPUQ4Q8() &&
                outputTensor.get_element_type() == ov::element::f32 &&
                outputTensor.get_shape().size() == 3) {
                auto nativeLMHead = GetNativeLMHead(
                    execution, weightIndex, errorMessage);
                if (!nativeLMHead) return X::Value();
                const auto& hiddenShape = outputTensor.get_shape();
                if (hiddenShape[0] == 1 && hiddenShape[1] == 1 &&
                    hiddenShape[2] ==
                        static_cast<size_t>(nativeLMHead->Columns())) {
                    std::vector<float> logits(
                        static_cast<size_t>(nativeLMHead->Rows()));
                    if (!nativeLMHead->Forward(
                            outputTensor.data<const float>(),
                            logits.data(),
                            OpenVINONativeCPUThreads(),
                            errorMessage)) {
                        return X::Value();
                    }
                    const auto maximum = std::max_element(
                        logits.begin(), logits.end());
                    const int64_t token = static_cast<int64_t>(
                        std::distance(logits.begin(), maximum));
                    X::Tensor output(X::g_pXHost->CreateTensor());
                    X::Port::vector<int> shape(3);
                    shape.push_back(1);
                    shape.push_back(1);
                    shape.push_back(1);
                    output->SetDataType(X::TensorDataType::LONGLONG);
                    output->SetShape(shape);
                    X::Value initial;
                    output->Create(initial);
                    if (!output->GetData()) {
                        errorMessage =
                            "failed to allocate native Q4Q8 token output";
                        return X::Value();
                    }
                    *reinterpret_cast<int64_t*>(output->GetData()) = token;
                    errorMessage.clear();
                    return X::Value(output);
                }
            }
            const X::TensorDataType outputType =
                ToXLangType(outputTensor.get_element_type());
            if (outputType == X::TensorDataType::UNKNOWN) {
                errorMessage = "unsupported OpenVINO output dtype";
                return X::Value();
            }
            X::Tensor output(X::g_pXHost->CreateTensor());
            X::Port::vector<int> shape(
                static_cast<int>(outputTensor.get_shape().size()));
            for (const size_t dimension : outputTensor.get_shape()) {
                shape.push_back(static_cast<int>(dimension));
            }
            output->SetDataType(outputType);
            output->SetShape(shape);
            X::Value initial;
            output->Create(initial);
            if (!output->GetData() ||
                static_cast<size_t>(output->GetDataSize()) !=
                    outputTensor.get_byte_size()) {
                errorMessage = "failed to allocate owned OpenVINO output tensor";
                return X::Value();
            }
            std::memcpy(
                output->GetData(),
                outputTensor.data(),
                outputTensor.get_byte_size());
            errorMessage.clear();
            return X::Value(output);
        }
        catch (const std::exception& exception) {
            errorMessage = std::string("OpenVINO inference failed: ") +
                exception.what();
            return X::Value();
        }
#endif
    }

    void OpenVINOBuilder::ReleaseCachedExecutions(const std::string& cacheRoot)
    {
#if defined(GARNET_WITH_OPENVINO)
        {
            std::lock_guard<std::mutex> guard(g_openVINOCacheMutex);
            for (auto iterator = g_openVINOCache.begin();
                 iterator != g_openVINOCache.end();) {
                if (iterator->first.rfind(cacheRoot, 0) == 0) {
                    iterator = g_openVINOCache.erase(iterator);
                }
                else {
                    ++iterator;
                }
            }
        }
        {
            std::lock_guard<std::mutex> guard(g_nativeDecodeCacheMutex);
            for (auto iterator = g_nativeDecodeCache.begin();
                 iterator != g_nativeDecodeCache.end();) {
                if (iterator->first.rfind(cacheRoot, 0) == 0) {
                    iterator = g_nativeDecodeCache.erase(iterator);
                }
                else {
                    ++iterator;
                }
            }
        }
#else
        (void)cacheRoot;
#endif
    }

    void OpenVINOBuilder::ReleaseCachedSession(
        const std::string& enginePath,
        std::uint64_t sessionId)
    {
#if defined(GARNET_WITH_OPENVINO)
        if (enginePath.empty() || sessionId == 0) return;
        std::lock_guard<std::mutex> cacheGuard(g_openVINOCacheMutex);
        const auto found = g_openVINOCache.find(enginePath);
        if (found == g_openVINOCache.end()) return;
        {
            std::lock_guard<std::mutex> sessionsGuard(
                found->second->sessionsMutex);
            found->second->sessions.erase(sessionId);
            if (!found->second->sessions.empty()) return;
        }
        g_openVINOCache.erase(found);
#else
        (void)enginePath;
        (void)sessionId;
#endif
    }

    X::Value OpenVINOBuilder::HandleBinaryOp(
        const std::string& opName,
        X::Value,
        X::ARGS&,
        X::KWARGS& keywordParams,
        X::Value input1,
        X::Value input2,
        X::Value output)
    {
        if (m_analysisActive) {
            CapturedTensorOperation operation;
            operation.index = static_cast<int>(m_analyzedOperations.size());
            operation.name = opName;
            if (input1.IsObject()) {
                operation.inputTensorIds.push_back(input1.GetObj()->GetID());
            }
            if (input2.IsObject()) {
                operation.inputTensorIds.push_back(input2.GetObj()->GetID());
            }
            if (output.IsObject()) operation.outputTensorId = output.GetObj()->GetID();
            m_analyzedOperations.push_back(std::move(operation));
            return X::Value(true);
        }
#if !defined(GARNET_WITH_OPENVINO)
        return X::Value();
#else
        if (!m_loweringActive || !m_implementation) return X::Value(true);
        const auto left = m_implementation->tensors.find(input1.GetObj()->GetID());
        const auto right = m_implementation->tensors.find(input2.GetObj()->GetID());
        if (left == m_implementation->tensors.end() ||
            right == m_implementation->tensors.end()) {
            m_error = "OpenVINO binary input is not available: " + opName;
            return X::Value();
        }
        const auto outputId = output.GetObj()->GetID();
        const auto leftStateFound =
            m_implementation->states.find(input1.GetObj()->GetID());
        Implementation::TensorState state =
            leftStateFound != m_implementation->states.end()
                ? leftStateFound->second
                : Implementation::TensorState{};
        auto keywordInt = [&](const char* name, int fallback) {
            auto* item = keywordParams.find(name);
            return item ? static_cast<int>(item->val.ToLongLong()) : fallback;
        };
        if (opName == "qwen3_apply_text_rope_packed") {
            const int heads = keywordInt("num_heads", 0);
            const int kvHeads = keywordInt("num_kv_heads", 0);
            const int headDim = keywordInt("head_dim", 0);
            auto* thetaItem = keywordParams.find("rope_theta");
            const float theta = thetaItem
                ? static_cast<float>(thetaItem->val.ToDouble())
                : 10000.0F;
            if (!state.query.get_node_shared_ptr() ||
                !state.key.get_node_shared_ptr() ||
                !state.value.get_node_shared_ptr() ||
                heads <= 0 || kvHeads <= 0 || headDim <= 0) {
                m_error = "OpenVINO text RoPE is missing packed QKV metadata";
                return X::Value();
            }
            state.query = ApplyRotary(
                state.query, right->second, headDim, theta);
            state.key = ApplyRotary(
                state.key, right->second, headDim, theta);
            auto packedHeads = std::make_shared<ov::opset13::Concat>(
                ov::OutputVector{state.query, state.key, state.value}, 2);
            const auto packedShape = packedHeads->get_output_partial_shape(0);
            auto packed = Reshape(
                packedHeads, {
                    packedShape[0].get_length(),
                    packedShape[1].get_length(),
                    static_cast<int64_t>(heads + 2 * kvHeads) * headDim});
            m_implementation->tensors[outputId] = packed;
            m_implementation->states[outputId] = state;
            m_implementation->lastOutput = packed;
            return X::Value(true);
        }
        if (
            opName == "paged_kv_bind_key_pages" ||
            opName == "paged_kv_bind_value_pages" ||
            opName == "paged_kv_bind_page_table" ||
            opName == "paged_kv_bind_context_length" ||
            opName == "paged_kv_bind_slot_position") {
            if (opName == "paged_kv_bind_key_pages" ||
                opName == "paged_kv_bind_value_pages") {
                const auto cache = m_implementation->cacheViews.find(
                    input2.GetObj()->GetID());
                if (cache == m_implementation->cacheViews.end()) {
                    m_error = "OpenVINO paged-KV binding is missing layer view";
                    return X::Value();
                }
                if (opName == "paged_kv_bind_key_pages") {
                    state.keyCache = cache->second;
                }
                else {
                    state.valueCache = cache->second;
                }
            }
            else if (opName == "paged_kv_bind_page_table") {
                state.pageTable = right->second;
            }
            else if (opName == "paged_kv_bind_context_length") {
                state.contextLength = right->second;
            }
            else {
                state.slotPosition = right->second;
            }
            m_implementation->tensors[outputId] = left->second;
            m_implementation->states[outputId] = state;
            m_implementation->lastOutput = left->second;
            return X::Value(true);
        }
        if (opName == "paged_attention_packed") {
            const int heads = keywordInt("num_heads", 0);
            const int kvHeads = keywordInt("num_key_value_heads", 0);
            const int headDim = keywordInt("head_dim", 0);
            if (!state.query.get_node_shared_ptr() ||
                !state.key.get_node_shared_ptr() ||
                !state.value.get_node_shared_ptr()) {
                m_error = "OpenVINO prefill attention is missing QKV state";
                return X::Value();
            }
            auto attended = GroupedAttention(
                state.query, state.key, state.value, right->second,
                heads, kvHeads, headDim, true);
            m_implementation->tensors[outputId] = attended;
            m_implementation->lastOutput = attended;
            return X::Value(true);
        }
        std::shared_ptr<ov::Node> node;
        if (opName == "add") {
            node = std::make_shared<ov::opset13::Add>(left->second, right->second);
        }
        else if (opName == "mul") {
            node = std::make_shared<ov::opset13::Multiply>(left->second, right->second);
        }
        else if (opName == "minus" || opName == "sub") {
            node = std::make_shared<ov::opset13::Subtract>(left->second, right->second);
        }
        else if (opName == "div") {
            node = std::make_shared<ov::opset13::Divide>(left->second, right->second);
        }
        else if (opName == "matmul") {
            node = std::make_shared<ov::opset13::MatMul>(
                left->second, right->second, false, false);
        }
        else if (opName == "linear") {
            node = std::make_shared<ov::opset13::MatMul>(
                left->second, right->second, false, true);
        }
        else {
            m_error = "unsupported OpenVINO binary operation: " + opName;
            return X::Value();
        }
        const auto lowered = node->output(0);
        m_implementation->tensors[outputId] = lowered;
        m_implementation->lastOutput = lowered;
        return X::Value(true);
#endif
    }

    X::Value OpenVINOBuilder::HandleUnaryOp(
        const std::string& opName,
        X::Value,
        X::ARGS&,
        X::KWARGS& keywordParams,
        X::Value input,
        X::Value output)
    {
        if (m_analysisActive) {
            CapturedTensorOperation operation;
            operation.index = static_cast<int>(m_analyzedOperations.size());
            operation.name = opName;
            if (input.IsObject()) {
                operation.inputTensorIds.push_back(input.GetObj()->GetID());
            }
            if (output.IsObject()) operation.outputTensorId = output.GetObj()->GetID();
            m_analyzedOperations.push_back(std::move(operation));
            return X::Value(true);
        }
#if !defined(GARNET_WITH_OPENVINO)
        return X::Value();
#else
        if (!m_loweringActive || !m_implementation) return X::Value(true);
        const auto source = m_implementation->tensors.find(input.GetObj()->GetID());
        if (source == m_implementation->tensors.end()) {
            m_error = "OpenVINO unary input is not available: " + opName;
            return X::Value();
        }
        const auto inputId = input.GetObj()->GetID();
        const auto outputId = output.GetObj()->GetID();
        const auto sourceStateFound = m_implementation->states.find(inputId);
        Implementation::TensorState state =
            sourceStateFound != m_implementation->states.end()
                ? sourceStateFound->second
                : Implementation::TensorState{};
        auto keywordInt = [&](const char* name, int fallback) {
            auto* item = keywordParams.find(name);
            return item ? static_cast<int>(item->val.ToLongLong()) : fallback;
        };
        if (opName == "qwen3_text_qkv_packed") {
            const int heads = keywordInt("num_heads", 0);
            const int kvHeads = keywordInt("num_kv_heads", 0);
            const int headDim = keywordInt("head_dim", 0);
            auto* qName = keywordParams.find("q_weight_name");
            auto* kName = keywordParams.find("k_weight_name");
            auto* vName = keywordParams.find("v_weight_name");
            auto* qNormName = keywordParams.find("q_norm_weight_name");
            auto* kNormName = keywordParams.find("k_norm_weight_name");
            if (!qName || !kName || !vName || !qNormName || !kNormName ||
                heads <= 0 || kvHeads <= 0 || headDim <= 0) {
                m_error = "OpenVINO packed text QKV parameters are invalid";
                return X::Value();
            }
            const bool packedProjections =
                OpenVINOPackedCPUProjections();
            const auto qWeight = packedProjections
                ? ov::Output<ov::Node>()
                : m_implementation->GetWeight(
                    qName->val.ToString(), m_error);
            const auto kWeight = packedProjections
                ? ov::Output<ov::Node>()
                : m_implementation->GetWeight(
                    kName->val.ToString(), m_error);
            const auto vWeight = packedProjections
                ? ov::Output<ov::Node>()
                : m_implementation->GetWeight(
                    vName->val.ToString(), m_error);
            const auto qkvWeight = packedProjections
                ? m_implementation->GetCombinedWeight(
                    {
                        qName->val.ToString(),
                        kName->val.ToString(),
                        vName->val.ToString(),
                    },
                    m_error)
                : ov::Output<ov::Node>();
            const auto qNorm = m_implementation->GetWeight(
                qNormName->val.ToString(), m_error);
            const auto kNorm = m_implementation->GetWeight(
                kNormName->val.ToString(), m_error);
            if (!m_error.empty()) return X::Value();
            auto* epsilonItem = keywordParams.find("norm_eps");
            const float epsilon = epsilonItem
                ? static_cast<float>(epsilonItem->val.ToDouble())
                : 1.0e-6F;
            const auto shape = source->second.get_partial_shape();
            const int64_t batch = shape[0].get_length();
            const int64_t sequence = shape[1].get_length();
            const int64_t qWidth =
                static_cast<int64_t>(heads) * headDim;
            const int64_t kvWidth =
                static_cast<int64_t>(kvHeads) * headDim;
            ov::Output<ov::Node> queryProjection;
            ov::Output<ov::Node> keyProjection;
            ov::Output<ov::Node> valueProjection;
            if (packedProjections) {
                auto qkvProjection =
                    std::make_shared<ov::opset13::MatMul>(
                        source->second, qkvWeight, false, true);
                auto qkvSplit =
                    std::make_shared<ov::opset13::VariadicSplit>(
                        qkvProjection,
                        I64Scalar(-1),
                        I64Vector({qWidth, kvWidth, kvWidth}));
                queryProjection = qkvSplit->output(0);
                keyProjection = qkvSplit->output(1);
                valueProjection = qkvSplit->output(2);
            }
            else {
                queryProjection =
                    std::make_shared<ov::opset13::MatMul>(
                        source->second, qWeight, false, true);
                keyProjection =
                    std::make_shared<ov::opset13::MatMul>(
                        source->second, kWeight, false, true);
                valueProjection =
                    std::make_shared<ov::opset13::MatMul>(
                        source->second, vWeight, false, true);
            }
            state.query = Reshape(
                queryProjection,
                {batch, sequence, heads, headDim});
            state.key = Reshape(
                keyProjection,
                {batch, sequence, kvHeads, headDim});
            state.value = Reshape(
                valueProjection,
                {batch, sequence, kvHeads, headDim});
            state.query = RMSNorm(state.query, qNorm, epsilon);
            state.key = RMSNorm(state.key, kNorm, epsilon);
            auto packedHeads = std::make_shared<ov::opset13::Concat>(
                ov::OutputVector{state.query, state.key, state.value}, 2);
            auto packed = Reshape(
                packedHeads,
                {batch, sequence,
                    static_cast<int64_t>(heads + 2 * kvHeads) * headDim});
            m_implementation->tensors[outputId] = packed;
            m_implementation->states[outputId] = state;
            m_implementation->lastOutput = packed;
            return X::Value(true);
        }
        if (opName == "paged_kv_select_layer") {
            const int layerIndex = keywordInt("layer_idx", -1);
            const auto parameter =
                m_implementation->parameterIndices.find(inputId);
            if (layerIndex < 0 ||
                parameter == m_implementation->parameterIndices.end()) {
                m_error = "OpenVINO paged-KV layer selection is invalid";
                return X::Value();
            }
            bool key = false;
            ov::Output<ov::Node> layer;
            std::shared_ptr<ov::op::util::Variable> variable;
            if (m_implementation->statefulDecode) {
                const Implementation::StatefulCacheSeed* seed = nullptr;
                if (m_implementation->keyState &&
                    inputId == m_implementation->keyState->tensorId) {
                    key = true;
                    seed = &*m_implementation->keyState;
                }
                else if (m_implementation->valueState &&
                    inputId == m_implementation->valueState->tensorId) {
                    seed = &*m_implementation->valueState;
                }
                else {
                    m_error =
                        "OpenVINO stateful cache tensor identity is invalid";
                    return X::Value();
                }
                const std::string variableName =
                    std::string("cache_") +
                    (key ? "key" : "value") + "_input_" +
                    std::to_string(seed->inputIndex) + "_layer_" +
                    std::to_string(layerIndex);
                ov::op::util::VariableInfo variableInfo{
                    ov::PartialShape(seed->layerShape),
                    seed->stateType,
                    variableName};
                variable =
                    std::make_shared<ov::op::util::Variable>(
                        std::move(variableInfo));
                auto read = std::make_shared<ov::op::v6::ReadValue>(
                    variable);
                layer = seed->stateType == seed->type
                    ? ov::Output<ov::Node>(read)
                    : ov::Output<ov::Node>(
                        std::make_shared<ov::opset13::Convert>(
                            read, seed->type));
                m_implementation->stateVariables.push_back(variable);
            }
            else {
                layer = std::make_shared<ov::opset13::Gather>(
                    source->second, I64Scalar(layerIndex), I64Scalar(0));
            }
            Implementation::CacheView view{
                layer,
                parameter->second,
                layerIndex,
                key,
                variable,
                m_implementation->statefulDecode
                    ? (key
                        ? m_implementation->keyState->stateType
                        : m_implementation->valueState->stateType)
                    : layer.get_element_type()};
            m_implementation->tensors[outputId] = layer;
            m_implementation->cacheViews[outputId] = view;
            m_implementation->lastOutput = layer;
            return X::Value(true);
        }
        if (
            opName == "paged_kv_prefill_write_bf16" ||
            opName == "paged_kv_decode_bf16") {
            const int heads = keywordInt("q_heads", 0);
            const int kvHeads = keywordInt("kv_heads", 0);
            const int headDim = keywordInt("head_dim", 0);
            if (!state.keyCache || !state.valueCache ||
                !state.query.get_node_shared_ptr() ||
                !state.key.get_node_shared_ptr() ||
                !state.value.get_node_shared_ptr()) {
                m_error = "OpenVINO paged-KV operation is missing state";
                return X::Value();
            }
            const auto layerShape =
                state.keyCache->layer.get_partial_shape();
            const int64_t pages = layerShape[0].get_length();
            const int64_t pageSize = layerShape[1].get_length();
            const int64_t capacity = pages * pageSize;
            auto keyFlat = Reshape(
                state.keyCache->layer, {capacity, kvHeads, headDim});
            auto valueFlat = Reshape(
                state.valueCache->layer, {capacity, kvHeads, headDim});
            ov::Output<ov::Node> indices;
            ov::Output<ov::Node> keyUpdates;
            ov::Output<ov::Node> valueUpdates;
            if (opName == "paged_kv_prefill_write_bf16") {
                const int64_t sequence =
                    state.key.get_partial_shape()[1].get_length();
                std::vector<int64_t> indexValues(
                    static_cast<size_t>(sequence));
                for (int64_t index = 0; index < sequence; ++index) {
                    indexValues[static_cast<size_t>(index)] = index;
                }
                indices = ov::opset13::Constant::create(
                    ov::element::i64, ov::Shape{indexValues.size()},
                    indexValues);
                keyUpdates = Reshape(
                    state.key, {sequence, kvHeads, headDim});
                valueUpdates = Reshape(
                    state.value, {sequence, kvHeads, headDim});
            }
            else {
                if (!state.slotPosition.get_node_shared_ptr() ||
                    !state.contextLength.get_node_shared_ptr()) {
                    m_error =
                        "OpenVINO decode requires slot and context tensors";
                    return X::Value();
                }
                indices = Reshape(state.slotPosition, {1});
                keyUpdates = Reshape(state.key, {1, kvHeads, headDim});
                valueUpdates = Reshape(state.value, {1, kvHeads, headDim});
            }
            auto updatedKeyFlat =
                std::make_shared<ov::opset13::ScatterUpdate>(
                    keyFlat, indices, keyUpdates, I64Scalar(0));
            auto updatedValueFlat =
                std::make_shared<ov::opset13::ScatterUpdate>(
                    valueFlat, indices, valueUpdates, I64Scalar(0));
            auto updatedKey = Reshape(
                updatedKeyFlat, {pages, pageSize, kvHeads, headDim});
            auto updatedValue = Reshape(
                updatedValueFlat, {pages, pageSize, kvHeads, headDim});
            if (m_implementation->statefulDecode) {
                if (!state.keyCache->variable ||
                    !state.valueCache->variable) {
                    m_error =
                        "OpenVINO per-layer cache variables are missing";
                    return X::Value();
                }
                ov::Output<ov::Node> keyStateValue = updatedKey;
                ov::Output<ov::Node> valueStateValue = updatedValue;
                if (updatedKey.get_element_type() !=
                    state.keyCache->stateType) {
                    keyStateValue =
                        std::make_shared<ov::opset13::Convert>(
                            updatedKey, state.keyCache->stateType);
                }
                if (updatedValue.get_element_type() !=
                    state.valueCache->stateType) {
                    valueStateValue =
                        std::make_shared<ov::opset13::Convert>(
                            updatedValue, state.valueCache->stateType);
                }
                m_implementation->stateSinks.push_back(
                    std::make_shared<ov::op::v6::Assign>(
                        keyStateValue, state.keyCache->variable));
                m_implementation->stateSinks.push_back(
                    std::make_shared<ov::op::v6::Assign>(
                        valueStateValue, state.valueCache->variable));
            }
            else {
                m_implementation->cacheWrites.push_back({
                    updatedKey, state.keyCache->inputIndex,
                    state.keyCache->layerIndex, true});
                m_implementation->cacheWrites.push_back({
                    updatedValue, state.valueCache->inputIndex,
                    state.valueCache->layerIndex, false});
            }
            if (opName == "paged_kv_prefill_write_bf16") {
                m_implementation->tensors[outputId] = source->second;
                m_implementation->states[outputId] = state;
                m_implementation->lastOutput = source->second;
                return X::Value(true);
            }
            auto fullKey = Reshape(
                updatedKeyFlat, {1, capacity, kvHeads, headDim});
            auto fullValue = Reshape(
                updatedValueFlat, {1, capacity, kvHeads, headDim});
            auto range = std::make_shared<ov::opset13::Range>(
                I64Scalar(0), I64Scalar(capacity), I64Scalar(1),
                ov::element::i64);
            auto context = std::make_shared<ov::opset13::Convert>(
                Reshape(state.contextLength, {}), ov::element::i64);
            auto valid = std::make_shared<ov::opset13::Less>(
                range, context);
            auto attended = GroupedAttention(
                state.query, fullKey, fullValue, valid,
                heads, kvHeads, headDim, false);
            m_implementation->tensors[outputId] = attended;
            m_implementation->lastOutput = attended;
            return X::Value(true);
        }
        if (opName == "merge_attention_heads") {
            // Attention lowerings return [batch, tokens, hidden] so backends
            // can fuse the head transpose/reshape with the attention kernel.
            m_implementation->tensors[outputId] = source->second;
            m_implementation->lastOutput = source->second;
            return X::Value(true);
        }
        std::shared_ptr<ov::Node> node;
        if (opName == "relu") {
            node = std::make_shared<ov::opset13::Relu>(source->second);
        }
        else if (opName == "sigmoid") {
            node = std::make_shared<ov::opset13::Sigmoid>(source->second);
        }
        else if (opName == "tanh") {
            node = std::make_shared<ov::opset13::Tanh>(source->second);
        }
        else if (opName == "silu") {
            node = std::make_shared<ov::opset13::Swish>(source->second);
        }
        else if (opName == "gelu" || opName == "gelu_pytorch_tanh") {
            node = std::make_shared<ov::opset13::Gelu>(
                source->second,
                opName == "gelu_pytorch_tanh"
                    ? ov::op::GeluApproximationMode::TANH
                    : ov::op::GeluApproximationMode::ERF);
        }
        else if (opName == "embedding") {
            auto* weightName = keywordParams.find("weight_name");
            if (!weightName) {
                m_error = "OpenVINO embedding requires weight_name";
                return X::Value();
            }
            const auto weight = m_implementation->GetWeight(
                weightName->val.ToString(), m_error);
            if (!weight.get_node_shared_ptr()) return X::Value();
            const auto axis = ov::opset13::Constant::create(
                ov::element::i64, ov::Shape{}, {0});
            node = std::make_shared<ov::opset13::Gather>(
                weight, source->second, axis);
        }
        else if (
            opName == "linear" || opName == "qkv_linear" ||
            opName == "q_proj" || opName == "k_proj" ||
            opName == "v_proj" || opName == "o_proj" ||
            opName == "gate_proj" || opName == "up_proj" ||
            opName == "down_proj" || opName == "lm_head") {
            auto* weightName = keywordParams.find("weight_name");
            if (!weightName) {
                m_error = "OpenVINO " + opName + " requires weight_name";
                return X::Value();
            }
            if (opName == "lm_head" &&
                m_implementation->statefulDecode &&
                m_implementation->int4WeightOnly &&
                OpenVINONativeCPUQ4Q8()) {
                node = std::make_shared<ov::opset13::Convert>(
                    source->second, ov::element::f32);
                m_implementation->externalNativeLMHead = true;
                const auto lowered = node->output(0);
                m_implementation->tensors[outputId] = lowered;
                m_implementation->lastOutput = lowered;
                return X::Value(true);
            }
            const auto weight = m_implementation->GetWeight(
                weightName->val.ToString(), m_error);
            if (!weight.get_node_shared_ptr()) return X::Value();
            auto projected = std::make_shared<ov::opset13::MatMul>(
                source->second, weight, false, true);
            auto* biasName = keywordParams.find("bias_name");
            if (biasName && !biasName->val.IsNone() &&
                !biasName->val.ToString().empty()) {
                const auto bias = m_implementation->GetWeight(
                    biasName->val.ToString(), m_error);
                if (!bias.get_node_shared_ptr()) return X::Value();
                node = std::make_shared<ov::opset13::Add>(projected, bias);
            }
            else {
                node = projected;
            }
        }
        else if (opName == "rms_norm") {
            auto* weightName = keywordParams.find("weight_name");
            if (!weightName) {
                m_error = "OpenVINO rms_norm requires weight_name";
                return X::Value();
            }
            const auto scale = m_implementation->GetWeight(
                weightName->val.ToString(), m_error);
            if (!scale.get_node_shared_ptr()) return X::Value();
            const auto sourceType = source->second.get_element_type();
            auto sourceFloat = std::make_shared<ov::opset13::Convert>(
                source->second, ov::element::f32);
            auto square = std::make_shared<ov::opset13::Multiply>(
                sourceFloat, sourceFloat);
            const int64_t lastAxis =
                static_cast<int64_t>(source->second.get_partial_shape().rank().get_length()) - 1;
            const auto axes = ov::opset13::Constant::create(
                ov::element::i64, ov::Shape{1}, {lastAxis});
            auto mean = std::make_shared<ov::opset13::ReduceMean>(
                square, axes, true);
            auto* epsilonItem = keywordParams.find("eps");
            const float epsilon = epsilonItem
                ? static_cast<float>(epsilonItem->val.ToDouble())
                : 1.0e-6F;
            const auto epsilonNode = ov::opset13::Constant::create(
                ov::element::f32, ov::Shape{}, {epsilon});
            auto variance = std::make_shared<ov::opset13::Add>(
                mean, epsilonNode);
            auto root = std::make_shared<ov::opset13::Sqrt>(variance);
            auto normalized = std::make_shared<ov::opset13::Divide>(
                sourceFloat, root);
            auto scaleFloat = std::make_shared<ov::opset13::Convert>(
                scale, ov::element::f32);
            auto scaled = std::make_shared<ov::opset13::Multiply>(
                normalized, scaleFloat);
            node = std::make_shared<ov::opset13::Convert>(
                scaled, sourceType);
        }
        else if (opName == "qwen3_mlp_gate_up_swiglu_packed") {
            auto* gateName = keywordParams.find("gate_weight_name");
            auto* upName = keywordParams.find("up_weight_name");
            if (!gateName || !upName) {
                m_error =
                    "OpenVINO packed SwiGLU requires gate/up weight names";
                return X::Value();
            }
            const bool packedProjections =
                OpenVINOPackedCPUProjections();
            const auto gateWeight = packedProjections
                ? ov::Output<ov::Node>()
                : m_implementation->GetWeight(
                    gateName->val.ToString(), m_error);
            const auto upWeight = packedProjections
                ? ov::Output<ov::Node>()
                : m_implementation->GetWeight(
                    upName->val.ToString(), m_error);
            const auto gateUpWeight = packedProjections
                ? m_implementation->GetCombinedWeight(
                    {
                        gateName->val.ToString(),
                        upName->val.ToString(),
                    },
                    m_error)
                : ov::Output<ov::Node>();
            if (!m_error.empty()) {
                return X::Value();
            }
            // Combine the packed U4 values and per-block scales before
            // dequantization. This preserves FullyConnectedCompressed while
            // amortizing one projection launch across gate and up.
            ov::Output<ov::Node> gate;
            ov::Output<ov::Node> up;
            if (packedProjections) {
                auto gateUp = std::make_shared<ov::opset13::MatMul>(
                    source->second, gateUpWeight, false, true);
                auto gateUpSplit =
                    std::make_shared<ov::opset13::Split>(
                        gateUp, I64Scalar(-1), 2);
                gate = gateUpSplit->output(0);
                up = gateUpSplit->output(1);
            }
            else {
                gate = std::make_shared<ov::opset13::MatMul>(
                    source->second, gateWeight, false, true);
                up = std::make_shared<ov::opset13::MatMul>(
                    source->second, upWeight, false, true);
            }
            auto sigmoid = std::make_shared<ov::opset13::Sigmoid>(
                gate);
            auto silu = std::make_shared<ov::opset13::Multiply>(
                gate, sigmoid);
            node = std::make_shared<ov::opset13::Multiply>(
                silu, up);
        }
        else {
            m_error = "unsupported OpenVINO unary operation: " + opName;
            return X::Value();
        }
        const auto lowered = node->output(0);
        m_implementation->tensors[outputId] = lowered;
        m_implementation->lastOutput = lowered;
        return X::Value(true);
#endif
    }

    X::Value OpenVINOBuilder::HandleBranchBegin(
        const std::string&,
        int,
        unsigned long long,
        int)
    {
        return X::Value(true);
    }

    X::Value OpenVINOBuilder::HandleBranchEnd()
    {
        return X::Value(true);
    }
}
