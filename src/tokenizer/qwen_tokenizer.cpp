#include "qwen_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include "nlohmann/json.hpp"

namespace Garnet::Tokenization
{
    namespace
    {
        void AppendUtf8(std::string& out, unsigned int cp)
        {
            if (cp <= 0x7F) {
                out.push_back(static_cast<char>(cp));
            }
            else if (cp <= 0x7FF) {
                out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
            else if (cp <= 0xFFFF) {
                out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
            else {
                out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
        }

        bool ReadTextFile(const std::string& path, std::string* out, std::string* error)
        {
            std::ifstream file(path, std::ios::binary);
            if (!file) {
                if (error) *error = "failed to open " + path;
                return false;
            }
            file.seekg(0, std::ios::end);
            std::streamoff size = file.tellg();
            file.seekg(0, std::ios::beg);
            if (size < 0) {
                if (error) *error = "failed to stat " + path;
                return false;
            }
            out->resize(static_cast<size_t>(size));
            if (size > 0 && !file.read(out->data(), size)) {
                if (error) *error = "failed to read " + path;
                return false;
            }
            return true;
        }

        std::string Utf8FromCodepoint(unsigned int cp)
        {
            std::string out;
            AppendUtf8(out, cp);
            return out;
        }

        bool IsAsciiLetter(unsigned char c)
        {
            return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
        }

        bool IsAsciiDigit(unsigned char c)
        {
            return c >= '0' && c <= '9';
        }

        bool IsSpaceButNotNewline(unsigned char c)
        {
            return c == ' ' || c == '\t' || c == '\f' || c == '\v';
        }

        bool StartsWithCaseInsensitive(const std::string& text, size_t pos, const char* needle)
        {
            size_t n = std::strlen(needle);
            if (pos + n > text.size()) return false;
            for (size_t i = 0; i < n; ++i) {
                unsigned char a = static_cast<unsigned char>(text[pos + i]);
                unsigned char b = static_cast<unsigned char>(needle[i]);
                if (std::tolower(a) != std::tolower(b)) return false;
            }
            return true;
        }

        std::string PairKey(const std::string& a, const std::string& b)
        {
            std::string key;
            key.reserve(a.size() + b.size() + 1);
            key.append(a);
            key.push_back('\x01');
            key.append(b);
            return key;
        }

        void SetError(char* errorMessage, int capacity, const char* message)
        {
            if (errorMessage && capacity > 0) {
                std::snprintf(errorMessage, static_cast<size_t>(capacity), "%s", message ? message : "");
            }
        }

        bool CopyIds(const std::vector<int64_t>& ids, long long* outputIds, int outputCapacity, int* outputCount)
        {
            if (outputCount) {
                *outputCount = static_cast<int>(ids.size());
            }
            if (!outputIds || outputCapacity < static_cast<int>(ids.size())) {
                return false;
            }
            for (size_t i = 0; i < ids.size(); ++i) {
                outputIds[i] = static_cast<long long>(ids[i]);
            }
            return true;
        }

        std::shared_ptr<const QwenTokenizer> GetCachedTokenizer(const char* modelDir, std::string* error)
        {
            static std::mutex cacheMutex;
            static std::unordered_map<std::string, std::shared_ptr<const QwenTokenizer>> cache;

            std::string key = modelDir ? modelDir : "";
            {
                std::lock_guard<std::mutex> lock(cacheMutex);
                auto it = cache.find(key);
                if (it != cache.end()) {
                    return it->second;
                }
            }

            auto tokenizer = std::make_shared<QwenTokenizer>();
            if (!tokenizer->LoadFromFolder(key, error)) {
                return nullptr;
            }

            std::lock_guard<std::mutex> lock(cacheMutex);
            cache[key] = tokenizer;
            return tokenizer;
        }
    }

    std::shared_ptr<const QwenTokenizer> GetCachedQwenTokenizer(
        const std::string& modelDir,
        std::string* error)
    {
        return GetCachedTokenizer(modelDir.c_str(), error);
    }

    void QwenTokenizer::InitByteLevelMaps()
    {
        m_byteToTokenChar.clear();
        m_tokenCharToByte.clear();
        std::vector<int> bytes;
        for (int i = static_cast<int>('!'); i <= static_cast<int>('~'); ++i) bytes.push_back(i);
        for (int i = 0xA1; i <= 0xAC; ++i) bytes.push_back(i);
        for (int i = 0xAE; i <= 0xFF; ++i) bytes.push_back(i);

        std::unordered_set<int> seen(bytes.begin(), bytes.end());
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            int cp = b;
            if (!seen.count(b)) {
                cp = 256 + n;
                ++n;
            }
            std::string tokenChar = Utf8FromCodepoint(static_cast<unsigned int>(cp));
            m_byteToTokenChar[static_cast<unsigned char>(b)] = tokenChar;
            m_tokenCharToByte[tokenChar] = static_cast<unsigned char>(b);
        }
    }

    bool QwenTokenizer::LoadFromFolder(const std::string& modelDir, std::string* error)
    {
        InitByteLevelMaps();
        m_tokenToId.clear();
        m_idToToken.clear();
        m_addedTokens.clear();
        m_specialTokens.clear();
        m_specialIds.clear();
        m_mergeRanks.clear();
        m_maxAddedTokenLength = 0;

        std::string jsonText;
        std::string path = modelDir + "/tokenizer.json";
        if (!ReadTextFile(path, &jsonText, error)) {
            path = modelDir + "\\tokenizer.json";
            if (!ReadTextFile(path, &jsonText, error)) {
                jsonText.clear();
            }
        }

        std::string fallbackMergesText;
        try {
            nlohmann::json root;
            if (!jsonText.empty()) {
                root = nlohmann::json::parse(jsonText);
            }
            else {
                std::string vocabText;
                std::string mergesText;
                std::string fallbackError;
                if (!ReadTextFile(
                        modelDir + "/vocab.json", &vocabText, &fallbackError)) {
                    fallbackError.clear();
                    if (!ReadTextFile(
                            modelDir + "\\vocab.json", &vocabText,
                            &fallbackError)) {
                        if (error) *error = fallbackError;
                        return false;
                    }
                }
                fallbackError.clear();
                if (!ReadTextFile(
                        modelDir + "/merges.txt", &mergesText, &fallbackError)) {
                    fallbackError.clear();
                    if (!ReadTextFile(
                            modelDir + "\\merges.txt", &mergesText,
                            &fallbackError)) {
                        if (error) *error = fallbackError;
                        return false;
                    }
                }
                root["model"]["vocab"] = nlohmann::json::parse(vocabText);
                fallbackMergesText = std::move(mergesText);
            }
            if (!root.contains("model") || !root["model"].is_object()) {
                if (error) *error = "tokenizer.json missing model object";
                return false;
            }
            const nlohmann::json& model = root["model"];
            if (!model.contains("vocab") || !model["vocab"].is_object()) {
                if (error) *error = "tokenizer.json missing model.vocab";
                return false;
            }
            for (auto it = model["vocab"].begin(); it != model["vocab"].end(); ++it) {
                int64_t id = it.value().get<int64_t>();
                m_tokenToId[it.key()] = id;
                m_idToToken[id] = it.key();
            }

            if (root.contains("added_tokens") && root["added_tokens"].is_array()) {
                for (const nlohmann::json& item : root["added_tokens"]) {
                    if (!item.is_object() || !item.contains("id") || !item.contains("content") || !item["content"].is_string()) {
                        continue;
                    }
                    int64_t id = item["id"].get<int64_t>();
                    std::string content = item["content"].get<std::string>();
                    m_tokenToId[content] = id;
                    m_idToToken[id] = content;
                    m_addedTokens.insert(content);
                    m_maxAddedTokenLength =
                        std::max(m_maxAddedTokenLength, content.size());
                    if (item.value("special", false)) {
                        m_specialTokens.insert(content);
                        m_specialIds.insert(id);
                    }
                }
            }

            if (!fallbackMergesText.empty()) {
                int rank = 0;
                std::istringstream lines(fallbackMergesText);
                std::string merge;
                while (std::getline(lines, merge)) {
                    if (!merge.empty() && merge.back() == '\r') merge.pop_back();
                    if (merge.empty() || merge[0] == '#') continue;
                    const size_t split = merge.find(' ');
                    if (split == std::string::npos) continue;
                    m_mergeRanks.emplace(
                        PairKey(merge.substr(0, split), merge.substr(split + 1)),
                        rank++);
                }
            }
            else if (model.contains("merges") && model["merges"].is_array()) {
                int rank = 0;
                for (const nlohmann::json& item : model["merges"]) {
                    std::string first;
                    std::string second;
                    if (item.is_string()) {
                        const std::string merge = item.get<std::string>();
                        const size_t split = merge.find(' ');
                        if (split == std::string::npos) continue;
                        first = merge.substr(0, split);
                        second = merge.substr(split + 1);
                    }
                    else if (item.is_array() && item.size() == 2 &&
                        item[0].is_string() && item[1].is_string()) {
                        // tokenizer.json v0.20+ stores BPE merge pairs as
                        // two-element arrays instead of space-delimited text.
                        first = item[0].get<std::string>();
                        second = item[1].get<std::string>();
                    }
                    else {
                        continue;
                    }
                    m_mergeRanks.emplace(PairKey(first, second), rank++);
                }
            }

            std::string tokenizerConfigText;
            std::string ignoredError;
            std::string tokenizerConfigPath =
                modelDir + "/tokenizer_config.json";
            if (!ReadTextFile(
                    tokenizerConfigPath, &tokenizerConfigText, &ignoredError)) {
                tokenizerConfigPath = modelDir + "\\tokenizer_config.json";
                ignoredError.clear();
                ReadTextFile(
                    tokenizerConfigPath, &tokenizerConfigText, &ignoredError);
            }
            if (!tokenizerConfigText.empty()) {
                const nlohmann::json tokenizerConfig =
                    nlohmann::json::parse(tokenizerConfigText);
                if (tokenizerConfig.contains("added_tokens_decoder") &&
                    tokenizerConfig["added_tokens_decoder"].is_object()) {
                    for (auto iterator =
                            tokenizerConfig["added_tokens_decoder"].begin();
                         iterator !=
                            tokenizerConfig["added_tokens_decoder"].end();
                         ++iterator) {
                        const auto& item = iterator.value();
                        if (!item.is_object() ||
                            !item.contains("content") ||
                            !item["content"].is_string()) {
                            continue;
                        }
                        const int64_t id = std::stoll(iterator.key());
                        const std::string content =
                            item["content"].get<std::string>();
                        m_tokenToId[content] = id;
                        m_idToToken[id] = content;
                        m_addedTokens.insert(content);
                        m_maxAddedTokenLength =
                            std::max(m_maxAddedTokenLength, content.size());
                        if (item.value("special", false)) {
                            m_specialTokens.insert(content);
                            m_specialIds.insert(id);
                        }
                    }
                }
            }
        }
        catch (const std::exception& exc) {
            if (error) *error = exc.what();
            return false;
        }

        return !m_tokenToId.empty();
    }

    int64_t QwenTokenizer::TokenId(const std::string& token) const
    {
        auto it = m_tokenToId.find(token);
        return it == m_tokenToId.end() ? -1 : it->second;
    }

    bool QwenTokenizer::IsSpecialId(int64_t id) const
    {
        return m_specialIds.find(id) != m_specialIds.end();
    }

    std::vector<std::string> QwenTokenizer::SplitSpecialAware(const std::string& text) const
    {
        std::vector<std::string> parts;
        size_t pos = 0;
        while (pos < text.size()) {
            std::string matched;
            size_t maxLen = std::min(m_maxAddedTokenLength, text.size() - pos);
            for (size_t len = maxLen; len > 0; --len) {
                std::string candidate = text.substr(pos, len);
                if (m_addedTokens.find(candidate) != m_addedTokens.end()) {
                    matched = candidate;
                    break;
                }
            }
            if (!matched.empty()) {
                parts.push_back(matched);
                pos += matched.size();
                continue;
            }
            size_t start = pos;
            ++pos;
            while (pos < text.size()) {
                bool atSpecial = false;
                maxLen = std::min(m_maxAddedTokenLength, text.size() - pos);
                for (size_t len = maxLen; len > 0; --len) {
                    if (m_addedTokens.find(text.substr(pos, len)) !=
                        m_addedTokens.end()) {
                        atSpecial = true;
                        break;
                    }
                }
                if (atSpecial) break;
                ++pos;
            }
            parts.push_back(text.substr(start, pos - start));
        }
        return parts;
    }

    std::vector<std::string> QwenTokenizer::PreTokenizeAscii(const std::string& text) const
    {
        std::vector<std::string> out;
        size_t i = 0;
        while (i < text.size()) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            const char* contractions[] = { "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };
            bool matchedContraction = false;
            for (const char* contraction : contractions) {
                if (StartsWithCaseInsensitive(text, i, contraction)) {
                    out.emplace_back(contraction, std::strlen(contraction));
                    i += std::strlen(contraction);
                    matchedContraction = true;
                    break;
                }
            }
            if (matchedContraction) continue;

            if (c == ' ' && i + 1 < text.size() && IsAsciiLetter(static_cast<unsigned char>(text[i + 1]))) {
                size_t start = i++;
                while (i < text.size() && IsAsciiLetter(static_cast<unsigned char>(text[i]))) ++i;
                out.push_back(text.substr(start, i - start));
            }
            else if (IsAsciiLetter(c)) {
                size_t start = i++;
                while (i < text.size() && IsAsciiLetter(static_cast<unsigned char>(text[i]))) ++i;
                out.push_back(text.substr(start, i - start));
            }
            else if (IsAsciiDigit(c)) {
                out.push_back(text.substr(i, 1));
                ++i;
            }
            else if (!std::isspace(c) && i + 1 < text.size() &&
                     IsAsciiLetter(static_cast<unsigned char>(text[i + 1]))) {
                // Qwen's pre-tokenizer permits one non-letter prefix before a
                // letter run (for example ",y" in JSON coordinates).
                size_t start = i++;
                while (i < text.size() && IsAsciiLetter(static_cast<unsigned char>(text[i]))) ++i;
                out.push_back(text.substr(start, i - start));
            }
            else if (c == ' ' && i + 1 < text.size() && !std::isspace(static_cast<unsigned char>(text[i + 1])) &&
                     !IsAsciiLetter(static_cast<unsigned char>(text[i + 1])) &&
                     !IsAsciiDigit(static_cast<unsigned char>(text[i + 1]))) {
                size_t start = i++;
                while (i < text.size()) {
                    unsigned char p = static_cast<unsigned char>(text[i]);
                    if (std::isspace(p) || IsAsciiLetter(p) || IsAsciiDigit(p)) break;
                    ++i;
                }
                while (i < text.size() && (text[i] == '\r' || text[i] == '\n')) ++i;
                out.push_back(text.substr(start, i - start));
            }
            else if (c == '\r' || c == '\n') {
                size_t start = i++;
                while (i < text.size() && (text[i] == '\r' || text[i] == '\n')) ++i;
                out.push_back(text.substr(start, i - start));
            }
            else if (std::isspace(c)) {
                size_t start = i++;
                while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])) &&
                       text[i] != '\r' && text[i] != '\n') {
                    ++i;
                }
                out.push_back(text.substr(start, i - start));
            }
            else {
                size_t start = i++;
                while (i < text.size()) {
                    unsigned char p = static_cast<unsigned char>(text[i]);
                    if (std::isspace(p) || IsAsciiLetter(p) || IsAsciiDigit(p)) break;
                    ++i;
                }
                while (i < text.size() && (text[i] == '\r' || text[i] == '\n')) ++i;
                out.push_back(text.substr(start, i - start));
            }
        }
        return out;
    }

    std::vector<std::string> QwenTokenizer::ByteLevelChars(const std::string& text) const
    {
        std::vector<std::string> chars;
        chars.reserve(text.size());
        for (unsigned char byte : text) {
            auto it = m_byteToTokenChar.find(byte);
            chars.push_back(it == m_byteToTokenChar.end() ? std::string(1, static_cast<char>(byte)) : it->second);
        }
        return chars;
    }

    std::vector<std::string> QwenTokenizer::ApplyBpe(const std::vector<std::string>& word) const
    {
        if (word.size() <= 1) {
            return word;
        }
        std::vector<std::string> parts = word;
        while (parts.size() > 1) {
            int bestRank = std::numeric_limits<int>::max();
            size_t bestIndex = 0;
            bool found = false;
            for (size_t i = 0; i + 1 < parts.size(); ++i) {
                auto it = m_mergeRanks.find(PairKey(parts[i], parts[i + 1]));
                if (it != m_mergeRanks.end() && it->second < bestRank) {
                    bestRank = it->second;
                    bestIndex = i;
                    found = true;
                }
            }
            if (!found) break;
            parts[bestIndex] += parts[bestIndex + 1];
            parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(bestIndex + 1));
        }
        return parts;
    }

    std::vector<int64_t> QwenTokenizer::EncodePlainSegment(const std::string& text) const
    {
        std::vector<int64_t> ids;
        for (const std::string& pretoken : PreTokenizeAscii(text)) {
            std::vector<std::string> bpeTokens = ApplyBpe(ByteLevelChars(pretoken));
            for (const std::string& token : bpeTokens) {
                auto it = m_tokenToId.find(token);
                if (it != m_tokenToId.end()) {
                    ids.push_back(it->second);
                }
            }
        }
        return ids;
    }

    std::vector<int64_t> QwenTokenizer::Encode(const std::string& text, bool) const
    {
        std::vector<int64_t> ids;
        for (const std::string& part : SplitSpecialAware(text)) {
            auto addedIt = m_addedTokens.find(part);
            if (addedIt != m_addedTokens.end()) {
                int64_t id = TokenId(part);
                if (id >= 0) ids.push_back(id);
                continue;
            }
            std::vector<int64_t> segment = EncodePlainSegment(part);
            ids.insert(ids.end(), segment.begin(), segment.end());
        }
        return ids;
    }

    std::string QwenTokenizer::Decode(const std::vector<int64_t>& tokenIds, bool skipSpecialTokens) const
    {
        std::string out;
        for (int64_t id : tokenIds) {
            if (skipSpecialTokens && IsSpecialId(id)) {
                continue;
            }
            auto it = m_idToToken.find(id);
            if (it == m_idToToken.end()) {
                continue;
            }
            if (m_specialIds.find(id) != m_specialIds.end()) {
                out += it->second;
                continue;
            }

            const std::string& token = it->second;
            for (size_t pos = 0; pos < token.size();) {
                size_t len = 1;
                unsigned char c = static_cast<unsigned char>(token[pos]);
                if ((c & 0xE0) == 0xC0) len = 2;
                else if ((c & 0xF0) == 0xE0) len = 3;
                else if ((c & 0xF8) == 0xF0) len = 4;
                std::string piece = token.substr(pos, len);
                auto byteIt = m_tokenCharToByte.find(piece);
                if (byteIt != m_tokenCharToByte.end()) {
                    out.push_back(static_cast<char>(byteIt->second));
                }
                pos += len;
            }
        }
        return out;
    }

    int64_t QwenVLPromptBuilder::VisualTokenCount(const int64_t imageGridTHW[3], int mergeSize)
    {
        if (!imageGridTHW || mergeSize <= 0) return 0;
        return imageGridTHW[0] * imageGridTHW[1] * imageGridTHW[2] / (mergeSize * mergeSize);
    }

    std::string QwenVLPromptBuilder::BuildSingleImagePromptText(
        const std::string& userPrompt,
        const int64_t imageGridTHW[3],
        int mergeSize)
    {
        int64_t visualTokens = VisualTokenCount(imageGridTHW, mergeSize);
        std::string text = "<|im_start|>user\n<|vision_start|>";
        for (int64_t i = 0; i < visualTokens; ++i) {
            text += "<|image_pad|>";
        }
        text += "<|vision_end|>";
        text += userPrompt;
        text += "<|im_end|>\n<|im_start|>assistant\n";
        return text;
    }

    std::vector<int64_t> QwenVLPromptBuilder::BuildSingleImagePromptIds(
        const QwenTokenizer& tokenizer,
        const std::string& userPrompt,
        const int64_t imageGridTHW[3],
        int mergeSize)
    {
        return tokenizer.Encode(BuildSingleImagePromptText(userPrompt, imageGridTHW, mergeSize), false);
    }

    QwenVLMRoPEMetadata QwenVLPromptBuilder::BuildSingleImageMRoPEMetadata(
        const std::vector<int64_t>& mmTokenTypeIds,
        const int64_t imageGridTHW[3],
        int mergeSize)
    {
        if (!imageGridTHW || mergeSize <= 0 || imageGridTHW[0] <= 0 ||
            imageGridTHW[1] <= 0 || imageGridTHW[2] <= 0 ||
            imageGridTHW[1] % mergeSize != 0 || imageGridTHW[2] % mergeSize != 0) {
            throw std::invalid_argument("invalid image grid for Qwen MRoPE metadata");
        }

        const int64_t tokenCount = static_cast<int64_t>(mmTokenTypeIds.size());
        QwenVLMRoPEMetadata result;
        result.positionIds.resize(static_cast<size_t>(3 * tokenCount));
        int64_t currentPosition = 0;
        int64_t tokenIndex = 0;
        bool consumedImage = false;

        auto setPosition = [&](int dimension, int64_t index, int64_t value) {
            result.positionIds[static_cast<size_t>(dimension * tokenCount + index)] = value;
        };

        while (tokenIndex < tokenCount) {
            const int64_t modality = mmTokenTypeIds[static_cast<size_t>(tokenIndex)];
            int64_t spanEnd = tokenIndex + 1;
            while (spanEnd < tokenCount &&
                mmTokenTypeIds[static_cast<size_t>(spanEnd)] == modality) {
                ++spanEnd;
            }
            const int64_t spanLength = spanEnd - tokenIndex;
            if (modality == 0) {
                for (int64_t offset = 0; offset < spanLength; ++offset) {
                    for (int dimension = 0; dimension < 3; ++dimension) {
                        setPosition(dimension, tokenIndex + offset, currentPosition + offset);
                    }
                }
                currentPosition += spanLength;
            }
            else if (modality == 1 && !consumedImage) {
                const int64_t llmGridT = imageGridTHW[0];
                const int64_t llmGridH = imageGridTHW[1] / mergeSize;
                const int64_t llmGridW = imageGridTHW[2] / mergeSize;
                if (spanLength != llmGridT * llmGridH * llmGridW) {
                    throw std::invalid_argument("Qwen visual token span does not match image grid");
                }
                int64_t offset = 0;
                for (int64_t t = 0; t < llmGridT; ++t) {
                    for (int64_t h = 0; h < llmGridH; ++h) {
                        for (int64_t w = 0; w < llmGridW; ++w, ++offset) {
                            setPosition(0, tokenIndex + offset, currentPosition + t);
                            setPosition(1, tokenIndex + offset, currentPosition + h);
                            setPosition(2, tokenIndex + offset, currentPosition + w);
                        }
                    }
                }
                currentPosition += std::max(imageGridTHW[1], imageGridTHW[2]) / mergeSize;
                consumedImage = true;
            }
            else {
                throw std::invalid_argument("single-image Qwen prompt contains an unsupported modality span");
            }
            tokenIndex = spanEnd;
        }

        int64_t maximumPosition = -1;
        for (const int64_t position : result.positionIds) {
            maximumPosition = std::max(maximumPosition, position);
        }
        result.positionDelta = maximumPosition + 1 - tokenCount;
        return result;
    }

extern "C" GARNET_TOKENIZER_EXPORT int GarnetQwenTokenizerEncode(
    const char* modelDir,
    const char* text,
    long long* outputIds,
    int outputCapacity,
    int* outputCount,
    char* errorMessage,
    int errorMessageCapacity)
{
    try {
        std::string error;
        auto tokenizer = GetCachedTokenizer(modelDir, &error);
        if (!tokenizer) {
            SetError(errorMessage, errorMessageCapacity, error.c_str());
            return 1;
        }
        std::vector<int64_t> ids = tokenizer->Encode(text ? text : "", false);
        if (!CopyIds(ids, outputIds, outputCapacity, outputCount)) {
            SetError(errorMessage, errorMessageCapacity, "output id buffer is too small");
            return 2;
        }
        SetError(errorMessage, errorMessageCapacity, "");
        return 0;
    }
    catch (const std::exception& exc) {
        SetError(errorMessage, errorMessageCapacity, exc.what());
        return 3;
    }
}

extern "C" GARNET_TOKENIZER_EXPORT int GarnetQwenTokenizerDecode(
    const char* modelDir,
    const long long* tokenIds,
    int tokenCount,
    int skipSpecialTokens,
    char* outputText,
    int outputTextCapacity,
    int* outputByteCount,
    char* errorMessage,
    int errorMessageCapacity)
{
    try {
        if (!tokenIds || tokenCount < 0) {
            SetError(errorMessage, errorMessageCapacity, "invalid token ids");
            return 1;
        }
        std::string error;
        auto tokenizer = GetCachedTokenizer(modelDir, &error);
        if (!tokenizer) {
            SetError(errorMessage, errorMessageCapacity, error.c_str());
            return 2;
        }
        std::vector<int64_t> ids;
        ids.reserve(static_cast<size_t>(tokenCount));
        for (int i = 0; i < tokenCount; ++i) ids.push_back(tokenIds[i]);
        std::string decoded = tokenizer->Decode(ids, skipSpecialTokens != 0);
        if (outputByteCount) {
            *outputByteCount = static_cast<int>(decoded.size());
        }
        if (!outputText || outputTextCapacity <= static_cast<int>(decoded.size())) {
            SetError(errorMessage, errorMessageCapacity, "output text buffer is too small");
            return 3;
        }
        std::memcpy(outputText, decoded.data(), decoded.size());
        outputText[decoded.size()] = '\0';
        SetError(errorMessage, errorMessageCapacity, "");
        return 0;
    }
    catch (const std::exception& exc) {
        SetError(errorMessage, errorMessageCapacity, exc.what());
        return 4;
    }
}

extern "C" GARNET_TOKENIZER_EXPORT int GarnetQwenVLBuildSingleImagePromptIds(
    const char* modelDir,
    const char* userPrompt,
    const long long* imageGridTHW,
    int mergeSize,
    long long* outputIds,
    int outputCapacity,
    int* outputCount,
    char* errorMessage,
    int errorMessageCapacity)
{
    try {
        if (!imageGridTHW) {
            SetError(errorMessage, errorMessageCapacity, "missing image_grid_thw");
            return 1;
        }
        std::string error;
        auto tokenizer = GetCachedTokenizer(modelDir, &error);
        if (!tokenizer) {
            SetError(errorMessage, errorMessageCapacity, error.c_str());
            return 2;
        }
        int64_t grid[3] = { imageGridTHW[0], imageGridTHW[1], imageGridTHW[2] };
        std::vector<int64_t> ids = Garnet::Tokenization::QwenVLPromptBuilder::BuildSingleImagePromptIds(
            *tokenizer,
            userPrompt ? userPrompt : "",
            grid,
            mergeSize);
        if (!CopyIds(ids, outputIds, outputCapacity, outputCount)) {
            SetError(errorMessage, errorMessageCapacity, "output id buffer is too small");
            return 3;
        }
        SetError(errorMessage, errorMessageCapacity, "");
        return 0;
    }
    catch (const std::exception& exc) {
        SetError(errorMessage, errorMessageCapacity, exc.what());
        return 4;
    }
}

extern "C" GARNET_TOKENIZER_EXPORT long long GarnetQwenTokenizerTokenId(
    const char* modelDir,
    const char* token)
{
    auto tokenizer = GetCachedTokenizer(modelDir, nullptr);
    if (!tokenizer) {
        return -1;
    }
    return static_cast<long long>(tokenizer->TokenId(token ? token : ""));
}

}
