#include "qwen_tts_compiled_frontend.h"

#include "garnet_tensor.h"
#include "tensor_helper.h"
#include "qwen_tokenizer.h"
#include "nlohmann/json.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace Garnet
{
    namespace
    {
        X::Value MakeGpuTensor(X::TensorDataType type,
            const std::vector<int>& dimensions, const void* data, size_t bytes)
        {
            X::Tensor tensor(X::g_pXHost->CreateTensor());
            X::Port::vector<int> shape(static_cast<int>(dimensions.size()));
            for (int dimension : dimensions) shape.push_back(dimension);
            tensor->SetDataType(type);
            tensor->SetShape(shape);
            void* memory = nullptr;
            if (cudaMalloc(&memory, bytes) != cudaSuccess) return X::Value();
            if (bytes && data && cudaMemcpy(
                    memory, data, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                cudaFree(memory);
                return X::Value();
            }
            if (TensorHelper::AttachGPUMemory(tensor, memory) !=
                TensorOpStatus::Success) {
                cudaFree(memory);
                return X::Value();
            }
            return X::Value(tensor);
        }

        X::Value MakeZeroGpuTensor(X::TensorDataType type,
            const std::vector<int>& dimensions, size_t elementBytes,
            X::Value reusable)
        {
            size_t count = 1;
            for (int dimension : dimensions) count *= static_cast<size_t>(dimension);
            const size_t bytes = count * elementBytes;
            if (reusable.IsTensor()) {
                X::Tensor tensor(reusable);
                void* memory = TensorHelper::GetGPUMemory(tensor);
                if (memory && static_cast<size_t>(tensor->GetDataSize()) == bytes &&
                    cudaMemsetAsync(memory, 0, bytes, cudaStreamPerThread) == cudaSuccess) {
                    return reusable;
                }
            }
            X::Value value = MakeGpuTensor(type, dimensions, nullptr, bytes);
            if (value.IsTensor()) cudaMemset(
                TensorHelper::GetGPUMemory(X::Tensor(value)), 0, bytes);
            return value;
        }

        std::string Lower(std::string value)
        {
            std::transform(value.begin(), value.end(), value.begin(),
                [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
            return value;
        }
    }

    QwenTTSCompiledInputs BuildQwenTTSCompiledInputs(
        const std::string& modelDirectory, const std::string& text,
        const std::string& speaker, const std::string& language,
        const std::vector<std::vector<int>>& profileShapes,
        X::Value reusableKeyCache, X::Value reusableValueCache)
    {
        QwenTTSCompiledInputs result;
        try {
            if (profileShapes.size() != 7 || profileShapes[0].size() != 3 ||
                profileShapes[0][0] != 1 || profileShapes[0][2] != 2 ||
                profileShapes[1] != std::vector<int>({3, 1, profileShapes[0][1]}) ||
                profileShapes[2] != std::vector<int>({1, profileShapes[0][1]}) ||
                profileShapes[3].size() != 5 || profileShapes[4] != profileShapes[3]) {
                throw std::invalid_argument(
                    "Qwen3-TTS requires the seven-input aligned paged-prefill profile");
            }
            const int maxTokens = profileShapes[0][1];
            nlohmann::json config;
            std::ifstream configStream(
                std::filesystem::path(modelDirectory) / "config.json");
            if (!configStream || !(configStream >> config)) {
                throw std::runtime_error("Qwen3-TTS config.json is unavailable");
            }
            const auto& talker = config.at("talker_config");
            const int64_t ttsBos = config.value("tts_bos_token_id", 151672);
            const int64_t ttsEos = config.value("tts_eos_token_id", 151673);
            const int64_t ttsPad = config.value("tts_pad_token_id", 151671);
            const int64_t codecThink = talker.value("codec_think_id", 4202);
            const int64_t codecNoThink = talker.value("codec_nothink_id", 4203);
            const int64_t codecThinkBos = talker.value("codec_think_bos_id", 4204);
            const int64_t codecThinkEos = talker.value("codec_think_eos_id", 4205);
            const int64_t codecPad = talker.value("codec_pad_id", 4196);
            const int64_t codecBos = talker.value("codec_bos_id", 4197);
            result.codecEosTokenId = talker.value("codec_eos_token_id", 2150);

            std::string tokenizerError;
            auto tokenizer = Tokenization::GetCachedQwenTokenizer(
                modelDirectory, &tokenizerError);
            if (!tokenizer) throw std::runtime_error(tokenizerError);
            const auto textIds = tokenizer->Encode(
                "<|im_start|>assistant\n" + text +
                "<|im_end|>\n<|im_start|>assistant\n", false);
            if (textIds.size() < 8) {
                throw std::invalid_argument("Qwen3-TTS text prompt is malformed");
            }
            std::vector<int64_t> codecPrefix;
            const std::string normalizedLanguage = Lower(
                language.empty() ? "auto" : language);
            if (normalizedLanguage == "auto") {
                codecPrefix = {codecNoThink, codecThinkBos, codecThinkEos};
            }
            else {
                const auto& languages = talker.at("codec_language_id");
                if (!languages.contains(normalizedLanguage)) {
                    throw std::invalid_argument(
                        "unsupported Qwen3-TTS language: " + language);
                }
                codecPrefix = {codecThink, codecThinkBos,
                    languages.at(normalizedLanguage).get<int64_t>(), codecThinkEos};
            }
            const std::string normalizedSpeaker = Lower(speaker);
            const auto& speakers = talker.at("spk_id");
            if (normalizedSpeaker.empty() || !speakers.contains(normalizedSpeaker)) {
                throw std::invalid_argument(
                    "unsupported Qwen3-TTS speaker: " + speaker);
            }
            if ((normalizedLanguage == "auto" || normalizedLanguage == "chinese") &&
                talker.contains("spk_is_dialect")) {
                const auto& dialects = talker.at("spk_is_dialect");
                if (dialects.contains(normalizedSpeaker) &&
                    dialects.at(normalizedSpeaker).is_string()) {
                    const std::string dialect =
                        dialects.at(normalizedSpeaker).get<std::string>();
                    const auto& languages = talker.at("codec_language_id");
                    if (languages.contains(dialect)) {
                        codecPrefix = {codecThink, codecThinkBos,
                            languages.at(dialect).get<int64_t>(), codecThinkEos};
                    }
                }
            }
            codecPrefix.push_back(speakers.at(normalizedSpeaker).get<int64_t>());
            codecPrefix.push_back(codecPad);
            codecPrefix.push_back(codecBos);

            std::vector<std::pair<int64_t, int64_t>> aligned;
            for (size_t index = 0; index < 3; ++index) {
                aligned.emplace_back(textIds[index], -1);
            }
            for (size_t index = 0; index + 1 < codecPrefix.size(); ++index) {
                const int64_t textId = index + 2 == codecPrefix.size()
                    ? ttsBos : ttsPad;
                aligned.emplace_back(textId, codecPrefix[index]);
            }
            for (size_t index = 3; index + 5 < textIds.size(); ++index) {
                aligned.emplace_back(textIds[index], codecPad);
            }
            aligned.emplace_back(ttsEos, codecPad);
            aligned.emplace_back(ttsPad, codecBos);
            if (aligned.size() > static_cast<size_t>(maxTokens)) {
                throw std::invalid_argument(
                    "Qwen3-TTS prompt exceeds the compiled token profile");
            }
            result.promptTokenCount = static_cast<int>(aligned.size());
            std::vector<int64_t> ids(static_cast<size_t>(maxTokens) * 2);
            for (int index = 0; index < maxTokens; ++index) {
                ids[static_cast<size_t>(index) * 2] = ttsPad;
                ids[static_cast<size_t>(index) * 2 + 1] = codecPad;
            }
            for (size_t index = 0; index < aligned.size(); ++index) {
                ids[index * 2] = aligned[index].first;
                ids[index * 2 + 1] = aligned[index].second;
            }
            std::vector<int64_t> positions(static_cast<size_t>(3 * maxTokens));
            std::vector<int64_t> mask(static_cast<size_t>(maxTokens));
            for (int component = 0; component < 3; ++component) {
                for (int index = 0; index < result.promptTokenCount; ++index) {
                    positions[static_cast<size_t>(component * maxTokens + index)] = index;
                }
            }
            std::fill(mask.begin(), mask.begin() + result.promptTokenCount, 1);
            const int pages = profileShapes[5][0];
            std::vector<int> pageTable(static_cast<size_t>(pages));
            for (int index = 0; index < pages; ++index) pageTable[index] = index;
            const int start = 0;
            X::V<X::XList> inputs;
            inputs->AddItem(MakeGpuTensor(X::TensorDataType::LONGLONG,
                profileShapes[0], ids.data(), ids.size() * sizeof(int64_t)));
            inputs->AddItem(MakeGpuTensor(X::TensorDataType::LONGLONG,
                profileShapes[1], positions.data(), positions.size() * sizeof(int64_t)));
            inputs->AddItem(MakeGpuTensor(X::TensorDataType::LONGLONG,
                profileShapes[2], mask.data(), mask.size() * sizeof(int64_t)));
            inputs->AddItem(MakeZeroGpuTensor(X::TensorDataType::BFLOAT16,
                profileShapes[3], sizeof(unsigned short), reusableKeyCache));
            inputs->AddItem(MakeZeroGpuTensor(X::TensorDataType::BFLOAT16,
                profileShapes[4], sizeof(unsigned short), reusableValueCache));
            inputs->AddItem(MakeGpuTensor(X::TensorDataType::INT,
                profileShapes[5], pageTable.data(), pageTable.size() * sizeof(int)));
            inputs->AddItem(MakeGpuTensor(X::TensorDataType::INT,
                profileShapes[6], &start, sizeof(start)));
            for (long long index = 0; index < inputs->Size(); ++index) {
                if (!inputs->Get(index).IsTensor()) {
                    throw std::runtime_error("Qwen3-TTS GPU input allocation failed");
                }
            }
            result.inputs = X::Value(inputs);
        }
        catch (const std::exception& exception) {
            result.error = exception.what();
        }
        return result;
    }
}
