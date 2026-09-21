// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_native_decode_engine.h"

#include "nlohmann/json.hpp"
#include "weight_quantization.h"

#include <filesystem>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <memory>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Garnet
{
    namespace
    {
        float BF16ToFloat(std::uint16_t value)
        {
            std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
            float result = 0.0F;
            std::memcpy(&result, &bits, sizeof(result));
            return result;
        }

        std::uint16_t FloatToBF16(float value)
        {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            bits += 0x7FFFU + ((bits >> 16) & 1U);
            return static_cast<std::uint16_t>(bits >> 16);
        }

        float FP16ToFloat(std::uint16_t value)
        {
            const std::uint32_t sign =
                static_cast<std::uint32_t>(value & 0x8000U) << 16;
            std::uint32_t exponent = (value >> 10) & 0x1FU;
            std::uint32_t mantissa = value & 0x03FFU;
            std::uint32_t bits = 0;
            if (exponent == 0) {
                if (mantissa == 0) bits = sign;
                else {
                    int shift = 0;
                    while ((mantissa & 0x0400U) == 0) {
                        mantissa <<= 1;
                        ++shift;
                    }
                    mantissa &= 0x03FFU;
                    bits = sign |
                        (static_cast<std::uint32_t>(112 - shift) << 23) |
                        (mantissa << 13);
                }
            }
            else if (exponent == 0x1FU) {
                bits = sign | 0x7F800000U | (mantissa << 13);
            }
            else {
                bits = sign | ((exponent + 112U) << 23) |
                    (mantissa << 13);
            }
            float result = 0.0F;
            std::memcpy(&result, &bits, sizeof(result));
            return result;
        }

        float ReadElement(
            const SafeTensorMetadata& metadata,
            const void* data,
            std::uint64_t index)
        {
            if (metadata.dataType == "BF16") {
                return BF16ToFloat(
                    static_cast<const std::uint16_t*>(data)[index]);
            }
            if (metadata.dataType == "F16") {
                return FP16ToFloat(
                    static_cast<const std::uint16_t*>(data)[index]);
            }
            return static_cast<const float*>(data)[index];
        }

        void RMSNorm(
            const float* source,
            const std::vector<float>& weight,
            float epsilon,
            float* output)
        {
            __m256 sum = _mm256_setzero_ps();
            size_t index = 0;
            for (; index + 8 <= weight.size(); index += 8) {
                const __m256 values = _mm256_loadu_ps(source + index);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(values, values));
            }
            alignas(32) float lanes[8];
            _mm256_store_ps(lanes, sum);
            float squares = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                lanes[4] + lanes[5] + lanes[6] + lanes[7];
            for (; index < weight.size(); ++index) {
                squares += source[index] * source[index];
            }
            const float multiplier = 1.0F / std::sqrt(
                static_cast<float>(squares / weight.size()) + epsilon);
            const __m256 multiplierVector = _mm256_set1_ps(multiplier);
            index = 0;
            for (; index + 8 <= weight.size(); index += 8) {
                const __m256 values = _mm256_loadu_ps(source + index);
                const __m256 weights = _mm256_loadu_ps(weight.data() + index);
                _mm256_storeu_ps(
                    output + index,
                    _mm256_mul_ps(
                        _mm256_mul_ps(values, weights), multiplierVector));
            }
            for (; index < weight.size(); ++index) {
                output[index] = source[index] * weight[index] * multiplier;
            }
        }

        __m256 FastExp(__m256 value)
        {
            value = _mm256_min_ps(
                value, _mm256_set1_ps(88.3762626647949F));
            value = _mm256_max_ps(
                value, _mm256_set1_ps(-88.3762626647949F));
            __m256 exponent = _mm256_add_ps(
                _mm256_mul_ps(value, _mm256_set1_ps(1.44269504088896341F)),
                _mm256_set1_ps(0.5F));
            __m256i integerExponent = _mm256_cvttps_epi32(exponent);
            __m256 floored = _mm256_cvtepi32_ps(integerExponent);
            const __m256 correction = _mm256_and_ps(
                _mm256_cmp_ps(floored, exponent, _CMP_GT_OS),
                _mm256_set1_ps(1.0F));
            floored = _mm256_sub_ps(floored, correction);
            value = _mm256_sub_ps(
                value,
                _mm256_mul_ps(floored, _mm256_set1_ps(0.693359375F)));
            value = _mm256_sub_ps(
                value,
                _mm256_mul_ps(floored, _mm256_set1_ps(-2.12194440e-4F)));
            const __m256 square = _mm256_mul_ps(value, value);
            __m256 polynomial = _mm256_set1_ps(1.9875691500E-4F);
            polynomial = _mm256_add_ps(
                _mm256_mul_ps(polynomial, value),
                _mm256_set1_ps(1.3981999507E-3F));
            polynomial = _mm256_add_ps(
                _mm256_mul_ps(polynomial, value),
                _mm256_set1_ps(8.3334519073E-3F));
            polynomial = _mm256_add_ps(
                _mm256_mul_ps(polynomial, value),
                _mm256_set1_ps(4.1665795894E-2F));
            polynomial = _mm256_add_ps(
                _mm256_mul_ps(polynomial, value),
                _mm256_set1_ps(1.6666665459E-1F));
            polynomial = _mm256_add_ps(
                _mm256_mul_ps(polynomial, value),
                _mm256_set1_ps(5.0000001201E-1F));
            polynomial = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(polynomial, square), value),
                _mm256_set1_ps(1.0F));
            integerExponent = _mm256_cvttps_epi32(floored);
            integerExponent = _mm256_add_epi32(
                integerExponent, _mm256_set1_epi32(0x7F));
            integerExponent = _mm256_slli_epi32(integerExponent, 23);
            return _mm256_mul_ps(
                polynomial, _mm256_castsi256_ps(integerExponent));
        }

        __m256 LoadBF16(const std::uint16_t* source)
        {
            const __m128i packed = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(source));
            return _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_cvtepu16_epi32(packed), 16));
        }

        float HorizontalFloatSum(__m256 value)
        {
            const __m128 low = _mm256_castps256_ps128(value);
            const __m128 high = _mm256_extractf128_ps(value, 1);
            __m128 sum = _mm_add_ps(low, high);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            return _mm_cvtss_f32(sum);
        }

        QuantizedWeight Concatenate(
            const std::vector<QuantizedWeight>& inputs)
        {
            QuantizedWeight output;
            if (inputs.empty()) return output;
            output.shape = {0, inputs.front().shape[1]};
            output.blockSize = inputs.front().blockSize;
            output.blockAxis = 1;
            for (const auto& input : inputs) {
                output.shape[0] += input.shape[0];
                output.packedValues.insert(
                    output.packedValues.end(),
                    input.packedValues.begin(),
                    input.packedValues.end());
                output.fp16Scales.insert(
                    output.fp16Scales.end(),
                    input.fp16Scales.begin(),
                    input.fp16Scales.end());
                output.maximumAbsoluteError = std::max(
                    output.maximumAbsoluteError,
                    input.maximumAbsoluteError);
            }
            return output;
        }

        void Trace(const std::string& message)
        {
            const char* path = std::getenv("GARNET_CPU_INT4_TRACE");
            if (!path || !*path) return;
            std::ofstream stream(path, std::ios::out | std::ios::app);
            stream << message << '\n';
        }

        constexpr char kPackedCacheMagic[16] = {
            'G', 'A', 'R', 'N', 'E', 'T', 'Q', '4',
            'Q', '8', 'P', 'A', 'C', 'K', '1', '\0'};
        constexpr std::uint32_t kPackedCacheVersion = 1;

        template<typename T>
        bool WriteScalar(std::ofstream& stream, const T& value)
        {
            stream.write(
                reinterpret_cast<const char*>(&value), sizeof(value));
            return static_cast<bool>(stream);
        }
    }

    struct Qwen3NativeDecodeEngine::Implementation
    {
        struct Layer
        {
            std::vector<float> inputNorm;
            std::vector<float> postNorm;
            std::vector<float> qNorm;
            std::vector<float> kNorm;
            Q4Q8Linear qkv;
            Q4Q8Linear output;
            Q4Q8Linear gateUp;
            Q4Q8Linear down;
        };

        SafeTensorsIndex index;
        std::unordered_map<
            std::string,
            std::unique_ptr<SafeTensorsMappedFile>> files;
        std::unique_ptr<SafeTensorsMappedFile> packedFile;
        std::vector<Layer> layers;
        Q4Q8Linear lmHead;
        std::vector<float> finalNorm;
        const SafeTensorMetadata* embeddingMetadata = nullptr;
        const void* embeddingData = nullptr;
        std::uint64_t packedBytes = 0;
        int hidden = 0;
        int intermediate = 0;
        int layerCount = 0;
        int vocabulary = 0;
        int heads = 0;
        int kvHeads = 0;
        int headDim = 0;
        float normEpsilon = 1.0e-6F;
        float ropeTheta = 1000000.0F;
        int threads = 4;

        const void* Data(
            const SafeTensorMetadata& metadata,
            std::string& error)
        {
            const std::string key = metadata.filePath.string();
            auto found = files.find(key);
            if (found == files.end()) {
                auto file = std::make_unique<SafeTensorsMappedFile>();
                if (!file->Open(metadata.filePath, error)) return nullptr;
                found = files.emplace(key, std::move(file)).first;
            }
            const void* data = found->second->DataAt(
                metadata.dataOffset, metadata.dataSize);
            if (!data) error = "native CPU weight mapping is out of range";
            return data;
        }

        bool Quantize(
            const std::string& lookupName,
            const std::string& contractName,
            QuantizedWeight& output,
            std::string& error)
        {
            const SafeTensorMetadata* metadata = index.Find(lookupName);
            if (!metadata || metadata->shape.size() != 2) {
                error = "native CPU matrix is missing: " + lookupName;
                return false;
            }
            const void* data = Data(*metadata, error);
            return data && QuantizeSymmetricINT4(
                contractName, *metadata, data,
                static_cast<int>(metadata->shape[1]), output, error);
        }

        bool LoadVector(
            const std::string& name,
            int size,
            std::vector<float>& output,
            std::string& error)
        {
            const SafeTensorMetadata* metadata = index.Find(name);
            if (!metadata || metadata->shape !=
                    std::vector<long long>{size}) {
                error = "native CPU vector is missing: " + name;
                return false;
            }
            const void* data = Data(*metadata, error);
            if (!data) return false;
            output.resize(static_cast<size_t>(size));
            for (int item = 0; item < size; ++item) {
                output[static_cast<size_t>(item)] =
                    ReadElement(*metadata, data, item);
            }
            return true;
        }

        bool PrepareOne(
            const std::string& name,
            Q4Q8Linear& output,
            std::string& error)
        {
            QuantizedWeight weight;
            if (!Quantize(name, name, weight, error) ||
                !output.Prepare(weight, error)) return false;
            packedBytes += output.PackedBytes();
            return true;
        }

        bool PrepareMany(
            const std::vector<std::string>& names,
            Q4Q8Linear& output,
            std::string& error)
        {
            std::vector<QuantizedWeight> weights(names.size());
            for (size_t index = 0; index < names.size(); ++index) {
                if (!Quantize(
                        names[index], names[index],
                        weights[index], error)) return false;
            }
            QuantizedWeight combined = Concatenate(weights);
            if (!output.Prepare(combined, error)) return false;
            packedBytes += output.PackedBytes();
            return true;
        }

        struct MatrixBinding
        {
            Q4Q8Linear* operation = nullptr;
            int rows = 0;
            int columns = 0;
            int groupSize = 0;
        };

        std::vector<MatrixBinding> MatrixBindings()
        {
            std::vector<MatrixBinding> result;
            result.reserve(static_cast<size_t>(layerCount) * 4 + 1);
            const int qkvRows = hidden + 2 * kvHeads * headDim;
            for (auto& layer : layers) {
                result.push_back({&layer.qkv, qkvRows, hidden, hidden});
                result.push_back({&layer.output, hidden, hidden, hidden});
                result.push_back({
                    &layer.gateUp, 2 * intermediate, hidden, hidden});
                result.push_back({
                    &layer.down, hidden, intermediate, intermediate});
            }
            result.push_back({&lmHead, vocabulary, hidden, hidden});
            return result;
        }

        bool LoadPackedCache(
            const std::filesystem::path& path,
            std::string& error)
        {
            if (path.empty() || !std::filesystem::is_regular_file(path)) {
                error = "native Q4/Q8 packed cache is unavailable";
                return false;
            }
            std::error_code sizeError;
            const std::uint64_t diskSize =
                std::filesystem::file_size(path, sizeError);
            if (sizeError || diskSize < 84) {
                error = "native Q4/Q8 packed cache size is invalid";
                return false;
            }
            auto mapped = std::make_unique<SafeTensorsMappedFile>();
            if (!mapped->Open(path, error)) return false;
            std::uint64_t cursor = 0;
            const auto readBytes = [&](std::uint64_t bytes) -> const void* {
                if (bytes > diskSize || cursor > diskSize - bytes) {
                    return nullptr;
                }
                const void* data = mapped->DataAt(cursor, bytes);
                if (data) cursor += bytes;
                return data;
            };
            const auto readScalar = [&](auto& value) -> bool {
                using T = std::decay_t<decltype(value)>;
                const void* data = readBytes(sizeof(T));
                if (!data) return false;
                std::memcpy(&value, data, sizeof(T));
                return true;
            };
            const void* magic = readBytes(sizeof(kPackedCacheMagic));
            std::uint32_t version = 0;
            std::uint64_t declaredSize = 0;
            std::uint64_t tensorBytes = 0;
            std::uint64_t tensorCount = 0;
            std::int32_t cachedHidden = 0;
            std::int32_t cachedIntermediate = 0;
            std::int32_t cachedLayers = 0;
            std::int32_t cachedVocabulary = 0;
            std::int32_t cachedHeads = 0;
            std::int32_t cachedKVHeads = 0;
            std::int32_t cachedHeadDim = 0;
            float cachedEpsilon = 0.0F;
            float cachedTheta = 0.0F;
            std::uint32_t matrixCount = 0;
            if (!magic || std::memcmp(
                    magic, kPackedCacheMagic,
                    sizeof(kPackedCacheMagic)) != 0 ||
                !readScalar(version) ||
                !readScalar(declaredSize) ||
                !readScalar(tensorBytes) ||
                !readScalar(tensorCount) ||
                !readScalar(cachedHidden) ||
                !readScalar(cachedIntermediate) ||
                !readScalar(cachedLayers) ||
                !readScalar(cachedVocabulary) ||
                !readScalar(cachedHeads) ||
                !readScalar(cachedKVHeads) ||
                !readScalar(cachedHeadDim) ||
                !readScalar(cachedEpsilon) ||
                !readScalar(cachedTheta) ||
                !readScalar(matrixCount)) {
                error = "native Q4/Q8 packed cache header is truncated";
                return false;
            }
            const auto bindings = MatrixBindings();
            if (version != kPackedCacheVersion ||
                declaredSize != diskSize ||
                tensorBytes != index.TensorBytes() ||
                tensorCount != index.TensorCount() ||
                cachedHidden != hidden ||
                cachedIntermediate != intermediate ||
                cachedLayers != layerCount ||
                cachedVocabulary != vocabulary ||
                cachedHeads != heads || cachedKVHeads != kvHeads ||
                cachedHeadDim != headDim ||
                cachedEpsilon != normEpsilon ||
                cachedTheta != ropeTheta ||
                matrixCount != bindings.size()) {
                error = "native Q4/Q8 packed cache contract changed";
                return false;
            }
            packedBytes = 0;
            for (const auto& binding : bindings) {
                std::int32_t rows = 0;
                std::int32_t columns = 0;
                std::int32_t groupSize = 0;
                std::uint64_t weightBytes = 0;
                std::uint64_t scaleCount = 0;
                if (!readScalar(rows) || !readScalar(columns) ||
                    !readScalar(groupSize) || !readScalar(weightBytes) ||
                    !readScalar(scaleCount) ||
                    rows != binding.rows || columns != binding.columns ||
                    groupSize != binding.groupSize ||
                    weightBytes != static_cast<std::uint64_t>(rows) *
                        columns / 2 ||
                    scaleCount != static_cast<std::uint64_t>(rows) *
                        columns / groupSize) {
                    error = "native Q4/Q8 packed matrix contract is invalid";
                    return false;
                }
                const auto* weights = static_cast<const std::uint8_t*>(
                    readBytes(weightBytes));
                const auto* scales = static_cast<const float*>(
                    readBytes(scaleCount * sizeof(float)));
                if (!weights || !scales ||
                    !binding.operation->BindPackedView(
                        rows, columns, groupSize, weights,
                        static_cast<size_t>(weightBytes), scales,
                        static_cast<size_t>(scaleCount), error)) {
                    if (error.empty()) {
                        error = "native Q4/Q8 packed matrix is truncated";
                    }
                    return false;
                }
                packedBytes += binding.operation->PackedBytes();
            }
            if (cursor != diskSize) {
                error = "native Q4/Q8 packed cache has trailing data";
                return false;
            }
            const auto prefaultBegin = std::chrono::steady_clock::now();
            const auto* mappedBytes = static_cast<const volatile std::uint8_t*>(
                mapped->DataAt(0, diskSize));
            if (!mappedBytes) {
                error = "native Q4/Q8 packed cache mapping is unavailable";
                return false;
            }
            volatile std::uint8_t prefaultChecksum = 0;
            constexpr std::uint64_t kPageBytes = 4096;
            for (std::uint64_t offset = 0;
                 offset < diskSize;
                 offset += kPageBytes) {
                prefaultChecksum ^= mappedBytes[offset];
            }
            prefaultChecksum ^= mappedBytes[diskSize - 1];
            (void)prefaultChecksum;
            Trace("packed_cache_prefault_ms=" + std::to_string(
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - prefaultBegin)
                    .count()));
            packedFile = std::move(mapped);
            error.clear();
            return true;
        }

        bool SavePackedCache(
            const std::filesystem::path& path,
            std::string& error)
        {
            if (path.empty()) return true;
            const auto bindings = MatrixBindings();
            std::uint64_t fileSize = 84;
            for (const auto& binding : bindings) {
                if (!binding.operation ||
                    binding.operation->Rows() != binding.rows ||
                    binding.operation->Columns() != binding.columns ||
                    binding.operation->GroupSize() != binding.groupSize ||
                    !binding.operation->PackedWeightData() ||
                    !binding.operation->PackedScaleData()) {
                    error = "native Q4/Q8 operation cannot be persisted";
                    return false;
                }
                fileSize += 28 +
                    binding.operation->PackedWeightBytes() +
                    binding.operation->PackedScaleCount() * sizeof(float);
            }
            std::error_code directoryError;
            if (!path.parent_path().empty()) {
                std::filesystem::create_directories(
                    path.parent_path(), directoryError);
            }
            if (directoryError) {
                error = "failed to create native Q4/Q8 cache directory: " +
                    directoryError.message();
                return false;
            }
            const std::filesystem::path temporary = path.string() + ".tmp." +
                std::to_string(std::chrono::steady_clock::now()
                    .time_since_epoch().count());
            std::ofstream stream(
                temporary, std::ios::binary | std::ios::trunc);
            const std::uint64_t tensorBytes = index.TensorBytes();
            const std::uint64_t tensorCount = index.TensorCount();
            const std::int32_t values[] = {
                hidden, intermediate, layerCount, vocabulary,
                heads, kvHeads, headDim};
            const std::uint32_t matrixCount =
                static_cast<std::uint32_t>(bindings.size());
            stream.write(kPackedCacheMagic, sizeof(kPackedCacheMagic));
            bool valid = stream &&
                WriteScalar(stream, kPackedCacheVersion) &&
                WriteScalar(stream, fileSize) &&
                WriteScalar(stream, tensorBytes) &&
                WriteScalar(stream, tensorCount);
            for (const auto value : values) {
                valid = valid && WriteScalar(stream, value);
            }
            valid = valid && WriteScalar(stream, normEpsilon) &&
                WriteScalar(stream, ropeTheta) &&
                WriteScalar(stream, matrixCount);
            for (const auto& binding : bindings) {
                const std::int32_t rows = binding.rows;
                const std::int32_t columns = binding.columns;
                const std::int32_t groupSize = binding.groupSize;
                const std::uint64_t weightBytes =
                    binding.operation->PackedWeightBytes();
                const std::uint64_t scaleCount =
                    binding.operation->PackedScaleCount();
                valid = valid && WriteScalar(stream, rows) &&
                    WriteScalar(stream, columns) &&
                    WriteScalar(stream, groupSize) &&
                    WriteScalar(stream, weightBytes) &&
                    WriteScalar(stream, scaleCount);
                if (!valid) break;
                stream.write(
                    reinterpret_cast<const char*>(
                        binding.operation->PackedWeightData()),
                    static_cast<std::streamsize>(weightBytes));
                stream.write(
                    reinterpret_cast<const char*>(
                        binding.operation->PackedScaleData()),
                    static_cast<std::streamsize>(
                        scaleCount * sizeof(float)));
                valid = static_cast<bool>(stream);
            }
            stream.flush();
            valid = valid && static_cast<bool>(stream);
            stream.close();
            if (!valid) {
                std::error_code cleanupError;
                std::filesystem::remove(temporary, cleanupError);
                error = "failed to write native Q4/Q8 packed cache";
                return false;
            }
            std::error_code publishError;
#if defined(_WIN32)
            if (!MoveFileExW(
                    temporary.c_str(), path.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
                publishError = std::error_code(
                    static_cast<int>(GetLastError()),
                    std::system_category());
            }
#else
            std::filesystem::rename(temporary, path, publishError);
#endif
            if (publishError) {
                std::error_code cleanupError;
                std::filesystem::remove(temporary, cleanupError);
                error = "failed to publish native Q4/Q8 packed cache: " +
                    publishError.message();
                return false;
            }
            error.clear();
            return true;
        }
    };

    Qwen3NativeDecodeEngine::Qwen3NativeDecodeEngine()
        : m_implementation(std::make_unique<Implementation>())
    {
    }

    Qwen3NativeDecodeEngine::~Qwen3NativeDecodeEngine() = default;

    bool Qwen3NativeDecodeEngine::Prepare(
        const SafeTensorsIndex& index,
        std::string& errorMessage)
    {
        return Prepare(index, std::filesystem::path(), errorMessage);
    }

    bool Qwen3NativeDecodeEngine::Prepare(
        const SafeTensorsIndex& index,
        const std::filesystem::path& packedCachePath,
        std::string& errorMessage)
    {
        Trace("prepare_begin");
        auto next = std::make_unique<Implementation>();
        if (!next->index.Open(index.FilePath(), errorMessage)) return false;
        std::filesystem::path directory = index.FilePath().parent_path();
        std::ifstream stream(directory / "config.json");
        nlohmann::json config;
        try { stream >> config; }
        catch (const std::exception& exception) {
            errorMessage = std::string("native CPU config is invalid: ") +
                exception.what();
            return false;
        }
        next->hidden = config.value("hidden_size", 0);
        next->intermediate = config.value("intermediate_size", 0);
        next->layerCount = config.value("num_hidden_layers", 0);
        next->vocabulary = config.value("vocab_size", 0);
        next->heads = config.value("num_attention_heads", 0);
        next->kvHeads = config.value("num_key_value_heads", 0);
        next->headDim = config.value("head_dim", 0);
        next->normEpsilon = config.value("rms_norm_eps", 1.0e-6F);
        next->ropeTheta = config.value("rope_theta", 1000000.0F);
        if (const char* threads = std::getenv("GARNET_CPU_INT4_THREADS")) {
            next->threads = std::max(1, std::atoi(threads));
        }
        if (next->hidden <= 0 || next->intermediate <= 0 ||
            next->layerCount <= 0 || next->vocabulary <= 0 ||
            next->heads <= 0 || next->kvHeads <= 0 ||
            next->headDim <= 0 ||
            next->headDim % 8 != 0 ||
            next->hidden != next->heads * next->headDim ||
            next->heads % next->kvHeads != 0) {
            errorMessage = "native CPU Qwen3 config is incomplete";
            return false;
        }
        next->embeddingMetadata =
            next->index.Find("model.embed_tokens.weight");
        if (!next->embeddingMetadata ||
            next->embeddingMetadata->shape !=
                std::vector<long long>{next->vocabulary, next->hidden}) {
            errorMessage = "native CPU embedding is missing";
            return false;
        }
        next->embeddingData = next->Data(
            *next->embeddingMetadata, errorMessage);
        if (!next->embeddingData) return false;
        next->layers.resize(static_cast<size_t>(next->layerCount));
        for (int layerIndex = 0;
             layerIndex < next->layerCount;
             ++layerIndex) {
            auto& layer = next->layers[static_cast<size_t>(layerIndex)];
            const std::string prefix =
                "model.layers." + std::to_string(layerIndex);
            const std::string attention = prefix + ".self_attn";
            if (!next->LoadVector(
                    prefix + ".input_layernorm.weight",
                    next->hidden, layer.inputNorm, errorMessage) ||
                !next->LoadVector(
                    prefix + ".post_attention_layernorm.weight",
                    next->hidden, layer.postNorm, errorMessage) ||
                !next->LoadVector(
                    attention + ".q_norm.weight",
                    next->headDim, layer.qNorm, errorMessage) ||
                !next->LoadVector(
                    attention + ".k_norm.weight",
                    next->headDim, layer.kNorm, errorMessage)) {
                return false;
            }
        }
        if (!next->LoadVector(
                "model.norm.weight", next->hidden,
                next->finalNorm, errorMessage)) return false;
        if (!packedCachePath.empty()) {
            std::string cacheError;
            if (next->LoadPackedCache(packedCachePath, cacheError)) {
                Trace("packed_cache_hit");
                Trace("prepare_ready");
                m_implementation = std::move(next);
                errorMessage.clear();
                return true;
            }
            Trace("packed_cache_miss: " + cacheError);
        }
        for (int layerIndex = 0;
             layerIndex < next->layerCount;
             ++layerIndex) {
            Trace("layer_" + std::to_string(layerIndex) + "_begin");
            auto& layer = next->layers[static_cast<size_t>(layerIndex)];
            const std::string prefix =
                "model.layers." + std::to_string(layerIndex);
            const std::string attention = prefix + ".self_attn";
            const std::string mlp = prefix + ".mlp";
            if (
                !next->PrepareMany({
                    attention + ".q_proj.weight",
                    attention + ".k_proj.weight",
                    attention + ".v_proj.weight"},
                    layer.qkv, errorMessage) ||
                !next->PrepareOne(
                    attention + ".o_proj.weight",
                    layer.output, errorMessage) ||
                !next->PrepareMany({
                    mlp + ".gate_proj.weight",
                    mlp + ".up_proj.weight"},
                    layer.gateUp, errorMessage) ||
                !next->PrepareOne(
                    mlp + ".down_proj.weight",
                    layer.down, errorMessage)) {
                return false;
            }
            Trace("layer_" + std::to_string(layerIndex) + "_ready");
        }
        Trace("lm_head_begin");
        const std::string lookup = next->index.Find("lm_head.weight")
            ? "lm_head.weight"
            : "model.embed_tokens.weight";
        QuantizedWeight lmHead;
        if (!next->Quantize(
                lookup, "lm_head.weight", lmHead, errorMessage) ||
            !next->lmHead.Prepare(lmHead, errorMessage)) {
            return false;
        }
        next->packedBytes += next->lmHead.PackedBytes();
        if (!packedCachePath.empty()) {
            std::string cacheError;
            if (next->SavePackedCache(packedCachePath, cacheError)) {
                Trace("packed_cache_written");
            }
            else {
                Trace("packed_cache_write_failed: " + cacheError);
            }
        }
        Trace("prepare_ready");
        m_implementation = std::move(next);
        errorMessage.clear();
        return true;
    }

    std::uint64_t Qwen3NativeDecodeEngine::PackedBytes() const
    {
        return m_implementation ? m_implementation->packedBytes : 0;
    }

    bool Qwen3NativeDecodeEngine::Decode(
        const DecodeInput& input,
        long long& outputTokenId,
        std::string& errorMessage) const
    {
        using Clock = std::chrono::steady_clock;
        const bool profile = []() {
            const char* value = std::getenv("GARNET_CPU_INT4_PROFILE");
            return value && std::string(value) == "1";
        }();
        const auto decodeBegin = Clock::now();
        double qkvMilliseconds = 0.0;
        double outputMilliseconds = 0.0;
        double gateUpMilliseconds = 0.0;
        double downMilliseconds = 0.0;
        double lmHeadMilliseconds = 0.0;
        const auto elapsed = [](Clock::time_point begin) {
            return std::chrono::duration<double, std::milli>(
                Clock::now() - begin).count();
        };
        const auto& engine = *m_implementation;
        if (input.tokenId < 0 || input.tokenId >= engine.vocabulary ||
            !input.keyCache || !input.valueCache || !input.pageTable ||
            input.pages <= 0 || input.pageSize <= 0 ||
            input.contextLength <= 0 || input.slotPosition < 0) {
            errorMessage = "native Qwen3 decode input is invalid";
            return false;
        }
        const auto physical = [&](int position) -> int {
            const int logicalPage = position / input.pageSize;
            if (logicalPage < 0 || logicalPage >= input.pageTableSize) {
                return -1;
            }
            const int page = input.pageTable[logicalPage];
            return page >= 0 && page < input.pages
                ? page * input.pageSize + position % input.pageSize
                : -1;
        };
        if (physical(input.slotPosition) < 0 ||
            physical(input.contextLength - 1) < 0) {
            errorMessage = "native Qwen3 page table is invalid";
            return false;
        }
        thread_local std::vector<float> hidden, normalized, qkv;
        thread_local std::vector<float> attention, projection, gateUp, mlp;
        thread_local std::vector<float> logits, scores;
        thread_local std::vector<float> ropeCosines, ropeSines;
        thread_local std::vector<int> physicalPositions;
        hidden.resize(engine.hidden);
        normalized.resize(engine.hidden);
        qkv.resize(engine.hidden + 2 * engine.kvHeads * engine.headDim);
        attention.resize(engine.hidden);
        projection.resize(engine.hidden);
        gateUp.resize(2 * engine.intermediate);
        mlp.resize(engine.intermediate);
        logits.resize(engine.vocabulary);
        scores.resize(input.contextLength);
        ropeCosines.resize(engine.headDim / 2);
        ropeSines.resize(engine.headDim / 2);
        physicalPositions.resize(input.contextLength);
        for (int index = 0; index < engine.headDim / 2; ++index) {
            const float angle = static_cast<float>(input.position) *
                std::pow(engine.ropeTheta,
                    -2.0F * index / engine.headDim);
            ropeCosines[index] = std::cos(angle);
            ropeSines[index] = std::sin(angle);
        }
        for (int token = 0; token < input.contextLength; ++token) {
            physicalPositions[token] = physical(token);
        }
        const std::uint64_t embeddingOffset =
            static_cast<std::uint64_t>(input.tokenId) * engine.hidden;
        for (int item = 0; item < engine.hidden; ++item) {
            hidden[item] = ReadElement(
                *engine.embeddingMetadata, engine.embeddingData,
                embeddingOffset + item);
        }
        const size_t positionsPerLayer =
            static_cast<size_t>(input.pages) * input.pageSize *
            engine.kvHeads * engine.headDim;
        const float scoreScale = 1.0F / std::sqrt(
            static_cast<float>(engine.headDim));
        for (int layerIndex = 0;
             layerIndex < engine.layerCount;
             ++layerIndex) {
            const auto& layer = engine.layers[layerIndex];
            RMSNorm(hidden.data(), layer.inputNorm,
                engine.normEpsilon, normalized.data());
            const auto qkvBegin = Clock::now();
            if (!layer.qkv.Forward(normalized.data(), qkv.data(),
                    engine.threads, errorMessage)) return false;
            if (profile) qkvMilliseconds += elapsed(qkvBegin);
            float* query = qkv.data();
            float* key = query + engine.hidden;
            float* value = key + engine.kvHeads * engine.headDim;
            for (int head = 0; head < engine.heads; ++head) {
                RMSNorm(query + head * engine.headDim, layer.qNorm,
                    engine.normEpsilon, query + head * engine.headDim);
            }
            for (int head = 0; head < engine.kvHeads; ++head) {
                RMSNorm(key + head * engine.headDim, layer.kNorm,
                    engine.normEpsilon, key + head * engine.headDim);
            }
            for (int index = 0; index < engine.headDim / 2; ++index) {
                const float cosine = ropeCosines[index];
                const float sine = ropeSines[index];
                for (int head = 0; head < engine.heads; ++head) {
                    float* row = query + head * engine.headDim;
                    const float first = row[index];
                    const float second = row[index + engine.headDim / 2];
                    row[index] = first * cosine - second * sine;
                    row[index + engine.headDim / 2] =
                        second * cosine + first * sine;
                }
                for (int head = 0; head < engine.kvHeads; ++head) {
                    float* row = key + head * engine.headDim;
                    const float first = row[index];
                    const float second = row[index + engine.headDim / 2];
                    row[index] = first * cosine - second * sine;
                    row[index + engine.headDim / 2] =
                        second * cosine + first * sine;
                }
            }
            const size_t layerBase =
                static_cast<size_t>(layerIndex) * positionsPerLayer;
            const int write = physical(input.slotPosition);
            for (int head = 0; head < engine.kvHeads; ++head) {
                for (int dimension = 0;
                     dimension < engine.headDim; ++dimension) {
                    const size_t cache = layerBase +
                        (static_cast<size_t>(write) * engine.kvHeads + head) *
                        engine.headDim + dimension;
                    input.keyCache[cache] = FloatToBF16(
                        key[head * engine.headDim + dimension]);
                    input.valueCache[cache] = FloatToBF16(
                        value[head * engine.headDim + dimension]);
                }
            }
            std::fill(attention.begin(), attention.end(), 0.0F);
            const int queriesPerKV = engine.heads / engine.kvHeads;
            for (int head = 0; head < engine.heads; ++head) {
                const int kvHead = head / queriesPerKV;
                float maximum = -std::numeric_limits<float>::infinity();
                for (int token = 0; token < input.contextLength; ++token) {
                    const int position = physicalPositions[token];
                    const size_t cache = layerBase +
                        (static_cast<size_t>(position) * engine.kvHeads +
                            kvHead) * engine.headDim;
                    const float* queryHead =
                        query + head * engine.headDim;
                    __m256 dotVector = _mm256_setzero_ps();
                    for (int dimension = 0;
                         dimension < engine.headDim; dimension += 8) {
                        dotVector = _mm256_add_ps(
                            dotVector,
                            _mm256_mul_ps(
                                _mm256_loadu_ps(queryHead + dimension),
                                LoadBF16(
                                    input.keyCache + cache + dimension)));
                    }
                    const float dot = HorizontalFloatSum(dotVector);
                    scores[token] = dot * scoreScale;
                    maximum = std::max(maximum, scores[token]);
                }
                float denominator = 0.0F;
                for (int token = 0; token < input.contextLength; ++token) {
                    scores[token] = std::exp(scores[token] - maximum);
                    denominator += scores[token];
                }
                const float inverse = 1.0F / denominator;
                for (int token = 0; token < input.contextLength; ++token) {
                    const float probability = scores[token] * inverse;
                    const int position = physicalPositions[token];
                    const size_t cache = layerBase +
                        (static_cast<size_t>(position) * engine.kvHeads +
                            kvHead) * engine.headDim;
                    float* attentionHead =
                        attention.data() + head * engine.headDim;
                    const __m256 probabilityVector =
                        _mm256_set1_ps(probability);
                    for (int dimension = 0;
                         dimension < engine.headDim; dimension += 8) {
                        _mm256_storeu_ps(
                            attentionHead + dimension,
                            _mm256_add_ps(
                                _mm256_loadu_ps(
                                    attentionHead + dimension),
                                _mm256_mul_ps(
                                    probabilityVector,
                                    LoadBF16(input.valueCache + cache +
                                        dimension))));
                    }
                }
            }
            const auto outputBegin = Clock::now();
            if (!layer.output.Forward(attention.data(), projection.data(),
                    engine.threads, errorMessage)) return false;
            if (profile) outputMilliseconds += elapsed(outputBegin);
            for (int item = 0; item < engine.hidden; ++item) {
                hidden[item] += projection[item];
            }
            RMSNorm(hidden.data(), layer.postNorm,
                engine.normEpsilon, normalized.data());
            const auto gateUpBegin = Clock::now();
            if (!layer.gateUp.Forward(normalized.data(), gateUp.data(),
                    engine.threads, errorMessage)) return false;
            if (profile) gateUpMilliseconds += elapsed(gateUpBegin);
            int item = 0;
            const __m256 one = _mm256_set1_ps(1.0F);
            const __m256 sign = _mm256_set1_ps(-0.0F);
            for (; item + 8 <= engine.intermediate; item += 8) {
                const __m256 gate = _mm256_loadu_ps(gateUp.data() + item);
                const __m256 up = _mm256_loadu_ps(
                    gateUp.data() + engine.intermediate + item);
                const __m256 negativeGate = _mm256_xor_ps(gate, sign);
                const __m256 sigmoid = _mm256_div_ps(
                    one, _mm256_add_ps(one, FastExp(negativeGate)));
                _mm256_storeu_ps(
                    mlp.data() + item,
                    _mm256_mul_ps(_mm256_mul_ps(gate, sigmoid), up));
            }
            for (; item < engine.intermediate; ++item) {
                const float gate = gateUp[item];
                mlp[item] = gate / (1.0F + std::exp(-gate)) *
                    gateUp[engine.intermediate + item];
            }
            const auto downBegin = Clock::now();
            if (!layer.down.Forward(mlp.data(), projection.data(),
                    engine.threads, errorMessage)) return false;
            if (profile) downMilliseconds += elapsed(downBegin);
            for (int item = 0; item < engine.hidden; ++item) {
                hidden[item] += projection[item];
            }
        }
        RMSNorm(hidden.data(), engine.finalNorm,
            engine.normEpsilon, normalized.data());
        const auto lmHeadBegin = Clock::now();
        if (!engine.lmHead.Forward(normalized.data(), logits.data(),
                engine.threads, errorMessage)) return false;
        if (profile) lmHeadMilliseconds += elapsed(lmHeadBegin);
        outputTokenId = static_cast<long long>(std::distance(
            logits.begin(), std::max_element(logits.begin(), logits.end())));
        if (profile) {
            const double totalMilliseconds = elapsed(decodeBegin);
            const double linearMilliseconds = qkvMilliseconds +
                outputMilliseconds + gateUpMilliseconds +
                downMilliseconds + lmHeadMilliseconds;
            Trace(
                "decode_profile total_ms=" +
                std::to_string(totalMilliseconds) +
                " qkv_ms=" + std::to_string(qkvMilliseconds) +
                " output_ms=" + std::to_string(outputMilliseconds) +
                " gate_up_ms=" + std::to_string(gateUpMilliseconds) +
                " down_ms=" + std::to_string(downMilliseconds) +
                " lm_head_ms=" + std::to_string(lmHeadMilliseconds) +
                " other_ms=" +
                std::to_string(totalMilliseconds - linearMilliseconds));
        }
        errorMessage.clear();
        return true;
    }
}
