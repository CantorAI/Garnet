// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "qwen_asr_compiled_frontend.h"

#include "garnet_tensor.h"
#include "tensor_helper.h"
#include "qwen_tokenizer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace Garnet
{
    namespace
    {
        constexpr int kSampleRate = 16000;
        constexpr int kFftSize = 400;
        constexpr int kHopLength = 160;
        constexpr int kMelBins = 128;
        constexpr int kChunkFrames = 100;
        constexpr int kTokensPerChunk = 13;

        uint16_t ReadU16(const unsigned char* data)
        {
            return static_cast<uint16_t>(data[0]) |
                static_cast<uint16_t>(data[1] << 8);
        }

        uint32_t ReadU32(const unsigned char* data)
        {
            return static_cast<uint32_t>(data[0]) |
                (static_cast<uint32_t>(data[1]) << 8) |
                (static_cast<uint32_t>(data[2]) << 16) |
                (static_cast<uint32_t>(data[3]) << 24);
        }

        std::vector<float> DecodeWav(
            const unsigned char* bytes, size_t size, int& sourceRate)
        {
            if (!bytes || size < 44 || std::memcmp(bytes, "RIFF", 4) != 0 ||
                std::memcmp(bytes + 8, "WAVE", 4) != 0) {
                throw std::invalid_argument("Qwen3-ASR audio must be a RIFF/WAVE file");
            }
            uint16_t format = 0;
            uint16_t channels = 0;
            uint16_t bits = 0;
            const unsigned char* pcm = nullptr;
            size_t pcmBytes = 0;
            size_t offset = 12;
            while (offset + 8 <= size) {
                const unsigned char* chunk = bytes + offset;
                const uint32_t chunkSize = ReadU32(chunk + 4);
                const size_t dataOffset = offset + 8;
                if (dataOffset + chunkSize > size) {
                    throw std::invalid_argument("WAV chunk extends past the input");
                }
                if (std::memcmp(chunk, "fmt ", 4) == 0 && chunkSize >= 16) {
                    format = ReadU16(bytes + dataOffset);
                    channels = ReadU16(bytes + dataOffset + 2);
                    sourceRate = static_cast<int>(ReadU32(bytes + dataOffset + 4));
                    bits = ReadU16(bytes + dataOffset + 14);
                }
                else if (std::memcmp(chunk, "data", 4) == 0) {
                    pcm = bytes + dataOffset;
                    pcmBytes = chunkSize;
                }
                offset = dataOffset + chunkSize + (chunkSize & 1U);
            }
            if (!pcm || channels == 0 || sourceRate <= 0 ||
                !((format == 1 && bits == 16) || (format == 3 && bits == 32))) {
                throw std::invalid_argument(
                    "Qwen3-ASR supports PCM16 or IEEE-float32 WAV audio");
            }
            const size_t sampleBytes = bits / 8;
            const size_t frames = pcmBytes / (sampleBytes * channels);
            std::vector<float> mono(frames);
            for (size_t frame = 0; frame < frames; ++frame) {
                float sum = 0.0F;
                for (uint16_t channel = 0; channel < channels; ++channel) {
                    const size_t index = (frame * channels + channel) * sampleBytes;
                    if (format == 1) {
                        int16_t value = 0;
                        std::memcpy(&value, pcm + index, sizeof(value));
                        sum += static_cast<float>(value) / 32768.0F;
                    }
                    else {
                        float value = 0.0F;
                        std::memcpy(&value, pcm + index, sizeof(value));
                        sum += std::isfinite(value) ? value : 0.0F;
                    }
                }
                mono[frame] = sum / channels;
            }
            return mono;
        }

        std::vector<float> Resample(
            const std::vector<float>& source, int sourceRate)
        {
            if (sourceRate == kSampleRate) return source;
            if (source.empty()) return {};
            const size_t targetCount = static_cast<size_t>(std::ceil(
                source.size() * static_cast<double>(kSampleRate) / sourceRate));
            std::vector<float> target(targetCount);
            const double scale = static_cast<double>(sourceRate) / kSampleRate;
            for (size_t index = 0; index < targetCount; ++index) {
                const double position = index * scale;
                const size_t left = std::min(
                    static_cast<size_t>(position), source.size() - 1);
                const size_t right = std::min(left + 1, source.size() - 1);
                const float fraction = static_cast<float>(position - left);
                target[index] = source[left] +
                    (source[right] - source[left]) * fraction;
            }
            return target;
        }

        double HzToMel(double hz)
        {
            constexpr double minLogHz = 1000.0;
            constexpr double minLogMel = 15.0;
            constexpr double logStep = 0.06875177742094912;
            return hz >= minLogHz
                ? minLogMel + std::log(hz / minLogHz) / logStep
                : hz * 3.0 / 200.0;
        }

        double MelToHz(double mel)
        {
            constexpr double minLogHz = 1000.0;
            constexpr double minLogMel = 15.0;
            constexpr double logStep = 0.06875177742094912;
            return mel >= minLogMel
                ? minLogHz * std::exp(logStep * (mel - minLogMel))
                : mel * 200.0 / 3.0;
        }

        const std::vector<float>& MelFilterBank()
        {
            static const std::vector<float> filters = [] {
                constexpr int frequencies = kFftSize / 2 + 1;
                std::vector<double> points(kMelBins + 2);
                const double low = HzToMel(0.0);
                const double high = HzToMel(kSampleRate / 2.0);
                for (int index = 0; index < kMelBins + 2; ++index) {
                    points[index] = MelToHz(
                        low + (high - low) * index / (kMelBins + 1));
                }
                std::vector<float> result(kMelBins * frequencies, 0.0F);
                for (int mel = 0; mel < kMelBins; ++mel) {
                    const double left = points[mel];
                    const double center = points[mel + 1];
                    const double right = points[mel + 2];
                    const double normalization = 2.0 / (right - left);
                    for (int bin = 0; bin < frequencies; ++bin) {
                        const double hz =
                            static_cast<double>(bin) * kSampleRate / kFftSize;
                        const double lower = (hz - left) / (center - left);
                        const double upper = (right - hz) / (right - center);
                        result[mel * frequencies + bin] = static_cast<float>(
                            std::max(0.0, std::min(lower, upper)) * normalization);
                    }
                }
                return result;
            }();
            return filters;
        }

        int ReflectIndex(int index, int count)
        {
            if (count <= 1) return 0;
            while (index < 0 || index >= count) {
                index = index < 0 ? -index : 2 * count - 2 - index;
            }
            return index;
        }

        std::vector<float> LogMel(const std::vector<float>& samples, int frames)
        {
            constexpr int frequencies = kFftSize / 2 + 1;
            static const std::vector<float> window = [] {
                std::vector<float> values(kFftSize);
                for (int index = 0; index < kFftSize; ++index) {
                    values[index] = static_cast<float>(
                        0.5 - 0.5 * std::cos(
                            2.0 * 3.14159265358979323846 * index / kFftSize));
                }
                return values;
            }();
            static const std::vector<float> trig = [] {
                constexpr int tableFrequencies = kFftSize / 2 + 1;
                std::vector<float> values(
                    static_cast<size_t>(tableFrequencies) * kFftSize * 2);
                for (int bin = 0; bin < tableFrequencies; ++bin) {
                    for (int index = 0; index < kFftSize; ++index) {
                        const double phase =
                            2.0 * 3.14159265358979323846 * bin * index / kFftSize;
                        values[(bin * kFftSize + index) * 2] =
                            static_cast<float>(std::cos(phase));
                        values[(bin * kFftSize + index) * 2 + 1] =
                            static_cast<float>(std::sin(phase));
                    }
                }
                return values;
            }();
            const auto& filters = MelFilterBank();
            std::vector<float> output(
                static_cast<size_t>(kMelBins) * frames);
            float maximum = -std::numeric_limits<float>::infinity();
            std::vector<float> power(frequencies);
            for (int frame = 0; frame < frames; ++frame) {
                const int center = frame * kHopLength;
                for (int bin = 0; bin < frequencies; ++bin) {
                    double real = 0.0;
                    double imaginary = 0.0;
                    for (int index = 0; index < kFftSize; ++index) {
                        const int sampleIndex = ReflectIndex(
                            center + index - kFftSize / 2,
                            static_cast<int>(samples.size()));
                        const double value = samples[sampleIndex] * window[index];
                        real += value * trig[(bin * kFftSize + index) * 2];
                        imaginary -= value * trig[(bin * kFftSize + index) * 2 + 1];
                    }
                    power[bin] = static_cast<float>(real * real + imaginary * imaginary);
                }
                for (int mel = 0; mel < kMelBins; ++mel) {
                    double energy = 0.0;
                    for (int bin = 0; bin < frequencies; ++bin) {
                        energy += filters[mel * frequencies + bin] * power[bin];
                    }
                    const float value = std::log10(
                        std::max(static_cast<float>(energy), 1.0e-10F));
                    output[static_cast<size_t>(mel) * frames + frame] = value;
                    maximum = std::max(maximum, value);
                }
            }
            const float floor = maximum - 8.0F;
            for (float& value : output) value = (std::max(value, floor) + 4.0F) / 4.0F;
            return output;
        }

        X::Value MakeGpuTensor(X3PackageHost* host, X3TensorDType type,
            const std::vector<int>& dimensions, const void* data, size_t bytes)
        {
            std::vector<int64_t> shape(dimensions.begin(), dimensions.end());
            uint64_t expected = TensorHelper::ItemSize(type);
            for (auto dimension : shape) {
                if (dimension < 0 || (dimension && expected > UINT64_MAX / dimension))
                    throw std::invalid_argument("invalid frontend tensor shape");
                expected *= dimension;
            }
            if (expected != bytes) throw std::invalid_argument("frontend tensor byte size mismatch");
            return TensorHelper::CreateGPU(host, type, shape, data);
        }

        X::Value MakeZeroGpuTensor(X3PackageHost* host, X3TensorDType type,
            const std::vector<int>& dimensions, size_t elementBytes, X::Value reusable)
        {
            if (elementBytes != TensorHelper::ItemSize(type))
                throw std::invalid_argument("frontend tensor element size mismatch");
            const std::vector<int64_t> shape(dimensions.begin(), dimensions.end());
            if (X::Tensor::IsTensor(reusable)) {
                X::Tensor tensor(reusable);
                const auto info = tensor.Info();
                int device = -1;
                bool matches = cudaGetDevice(&device) == cudaSuccess &&
                    info.device_id == device && !info.readonly &&
                    reusable.runtime() == host->runtime &&
                    info.dtype == type && info.rank == shape.size();
                for (uint32_t i = 0; matches && i < info.rank; ++i)
                    matches = info.shape[i] == shape[i];
                uint64_t span = elementBytes;
                for (size_t i = shape.size(); matches && i-- > 0;) {
                    matches = info.strides[i] == static_cast<int64_t>(span) &&
                        shape[i] >= 0 && (!shape[i] || span <= UINT64_MAX / shape[i]);
                    if (matches) span *= shape[i];
                }
                matches = matches && span == info.byte_size;
                void* data = matches ? TensorHelper::GetGPUMemory(tensor) : nullptr;
                if (data) {
                    auto use = TensorHelper::AcquireGPU(tensor, X3_TENSOR_WRITE);
                    if (cudaMemsetAsync(data, 0, info.byte_size, cudaStreamPerThread) == cudaSuccess) {
                        use.Finish();
                        return reusable;
                    }
                }
            }
            return TensorHelper::CreateGPU(host, type, shape);
        }

        std::vector<unsigned char> ReadBytes(X::Value source)
        {
            if (source.IsBin()) {
                uint64_t size = 0;
                const auto* begin = static_cast<const unsigned char*>(source.BytesData(&size));
                if (!begin || !size) return {};
                size_t bytes = static_cast<size_t>(size);
                // Some language bridges expose a Binary capacity rather than
                // the exact payload length. RIFF carries its authoritative
                // byte length in the first chunk header, so avoid reading
                // uninitialized capacity past the WAV payload.
                if (bytes >= 12 && std::memcmp(begin, "RIFF", 4) == 0) {
                    const size_t declared = static_cast<size_t>(ReadU32(begin + 4)) + 8;
                    if (declared >= 12 && declared <= bytes) bytes = declared;
                }
                if (bytes > 512ULL * 1024ULL * 1024ULL) {
                    throw std::invalid_argument("Qwen3-ASR WAV binary is too large");
                }
                return {begin, begin + bytes};
            }
            const std::string path = source.ToString();
            std::ifstream stream(path, std::ios::binary);
            return stream
                ? std::vector<unsigned char>(
                    std::istreambuf_iterator<char>(stream), {})
                : std::vector<unsigned char>();
        }
    }

    QwenASRCompiledInputs BuildQwenASRCompiledInputs(X3PackageHost* host,
        const std::string& modelDirectory,
        X::Value audioSource,
        const std::string& context,
        const std::string& language,
        const std::vector<std::vector<int>>& shapes,
        X::Value reusableKeyCache,
        X::Value reusableValueCache)
    {
        QwenASRCompiledInputs result;
        std::string stage = "profile validation";
        try {
            if (shapes.size() != 9 || shapes[0].size() != 2 ||
                shapes[1].size() != 4 || shapes[2].size() != 1 ||
                shapes[3].size() != 3 || shapes[4].size() != 2 ||
                shapes[5].size() != 5 || shapes[6] != shapes[5] ||
                shapes[7].size() != 1 || shapes[8] != std::vector<int>{1}) {
                throw std::invalid_argument(
                    "Qwen3-ASR frontend requires the nine-input prefill profile");
            }
            if (shapes[1][1] != 1 || shapes[1][2] != kMelBins ||
                shapes[1][3] != kChunkFrames ||
                shapes[2][0] != shapes[1][0] + 1) {
                throw std::invalid_argument(
                    "Qwen3-ASR audio profile must be [chunks,1,128,100]");
            }
            stage = "WAV input";
            std::vector<unsigned char> wavBytes = ReadBytes(audioSource);
            if (wavBytes.empty()) throw std::invalid_argument("Qwen3-ASR audio is empty");
            int sourceRate = 0;
            stage = "WAV decode";
            std::vector<float> decoded = DecodeWav(
                wavBytes.data(), wavBytes.size(), sourceRate);
            stage = "audio resample";
            std::vector<float> samples = Resample(decoded, sourceRate);
            if (samples.empty()) throw std::invalid_argument("Qwen3-ASR WAV has no samples");
            result.audioSampleCount = static_cast<int>(samples.size());
            result.audioDurationSeconds =
                static_cast<double>(samples.size()) / kSampleRate;
            const int rawFrames = std::max(
                1, static_cast<int>((samples.size() + kHopLength - 1) / kHopLength));
            const int chunks = (rawFrames + kChunkFrames - 1) / kChunkFrames;
            if (chunks != shapes[1][0]) {
                throw std::invalid_argument(
                    "audio duration selects " + std::to_string(chunks) +
                    " one-second chunks but the compiled profile expects " +
                    std::to_string(shapes[1][0]));
            }
            const int paddedFrames = chunks * kChunkFrames;
            stage = "log-mel frontend";
            std::vector<float> mel = LogMel(samples, rawFrames);
            std::vector<float> chunked(
                static_cast<size_t>(chunks) * kMelBins * kChunkFrames, 0.0F);
            for (int chunk = 0; chunk < chunks; ++chunk) {
                for (int bin = 0; bin < kMelBins; ++bin) {
                    for (int frame = 0; frame < kChunkFrames; ++frame) {
                        const int sourceFrame = chunk * kChunkFrames + frame;
                        if (sourceFrame < rawFrames) {
                            chunked[(static_cast<size_t>(chunk) * kMelBins + bin) *
                                kChunkFrames + frame] =
                                mel[static_cast<size_t>(bin) * rawFrames + sourceFrame];
                        }
                    }
                }
            }
            std::vector<int> cu(static_cast<size_t>(chunks + 1));
            for (int chunk = 0; chunk < chunks; ++chunk) {
                const int chunkFrames = std::min(
                    kChunkFrames, rawFrames - chunk * kChunkFrames);
                const int encodedFrames = (chunkFrames + 7) / 8;
                cu[static_cast<size_t>(chunk + 1)] =
                    cu[static_cast<size_t>(chunk)] + encodedFrames;
            }
            result.audioTokenCount = cu.back();

            stage = "tokenizer load";
            std::string tokenizerError;
            auto tokenizer = Tokenization::GetCachedQwenTokenizer(
                modelDirectory, &tokenizerError);
            if (!tokenizer) throw std::runtime_error(tokenizerError);
            std::string prompt =
                "<|im_start|>system\n" + context + "<|im_end|>\n" +
                "<|im_start|>user\n<|audio_start|>";
            for (int index = 0; index < result.audioTokenCount; ++index) {
                prompt += "<|audio_pad|>";
            }
            prompt += "<|audio_end|><|im_end|>\n<|im_start|>assistant\n";
            if (!language.empty()) prompt += "language " + language + "<asr_text>";
            stage = "prompt tokenization";
            const auto promptIds = tokenizer->Encode(prompt, false);
            const int maxTokens = shapes[0][1];
            if (shapes[0][0] != 1 || promptIds.empty() ||
                promptIds.size() > static_cast<size_t>(maxTokens) ||
                shapes[3] != std::vector<int>({3, 1, maxTokens}) ||
                shapes[4] != std::vector<int>({1, maxTokens})) {
                throw std::invalid_argument(
                    "Qwen3-ASR prompt does not fit the compiled token profile");
            }
            result.promptTokenCount = static_cast<int>(promptIds.size());
            std::vector<int64_t> ids(maxTokens, 0);
            std::vector<int64_t> positions(static_cast<size_t>(3 * maxTokens), 0);
            std::vector<int64_t> mask(maxTokens, 0);
            std::copy(promptIds.begin(), promptIds.end(), ids.begin());
            for (int index = 0; index < result.promptTokenCount; ++index) {
                mask[index] = 1;
                for (int axis = 0; axis < 3; ++axis) {
                    positions[static_cast<size_t>(axis) * maxTokens + index] = index;
                }
            }
            std::vector<int> pageTable(static_cast<size_t>(shapes[7][0]));
            for (int index = 0; index < shapes[7][0]; ++index) pageTable[index] = index;
            const int start = 0;

            stage = "GPU input construction";
            X::Value inputs = X::Value::List(host);
            inputs.Append(MakeGpuTensor(host,
                X3_TENSOR_INT64, shapes[0], ids.data(),
                ids.size() * sizeof(int64_t)));
            inputs.Append(MakeGpuTensor(host,
                X3_TENSOR_FLOAT32, shapes[1], chunked.data(),
                chunked.size() * sizeof(float)));
            inputs.Append(MakeGpuTensor(host,
                X3_TENSOR_INT32, shapes[2], cu.data(),
                cu.size() * sizeof(int)));
            inputs.Append(MakeGpuTensor(host,
                X3_TENSOR_INT64, shapes[3], positions.data(),
                positions.size() * sizeof(int64_t)));
            inputs.Append(MakeGpuTensor(host,
                X3_TENSOR_INT64, shapes[4], mask.data(),
                mask.size() * sizeof(int64_t)));
            inputs.Append(MakeZeroGpuTensor(host,
                X3_TENSOR_BFLOAT16, shapes[5], sizeof(uint16_t),
                reusableKeyCache));
            inputs.Append(MakeZeroGpuTensor(host,
                X3_TENSOR_BFLOAT16, shapes[6], sizeof(uint16_t),
                reusableValueCache));
            inputs.Append(MakeGpuTensor(host,
                X3_TENSOR_INT32, shapes[7], pageTable.data(),
                pageTable.size() * sizeof(int)));
            inputs.Append(MakeGpuTensor(host,
                X3_TENSOR_INT32, shapes[8], &start, sizeof(start)));
            for (long long index = 0; index < inputs.Size(); ++index) {
                if (!X::Tensor::IsTensor(inputs.Get(index))) {
                    throw std::runtime_error(
                        "failed to construct Qwen3-ASR input_" +
                        std::to_string(index));
                }
            }
            result.inputs = X::Value(inputs);
            (void)paddedFrames;
        }
        catch (const std::exception& exception) {
            result.inputs = X::Value();
            result.error = stage + ": " + exception.what();
        }
        return result;
    }
}
