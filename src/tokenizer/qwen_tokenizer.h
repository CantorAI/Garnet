#pragma once

#include "tokenizer.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#define GARNET_TOKENIZER_EXPORT __declspec(dllexport)
#else
#define GARNET_TOKENIZER_EXPORT
#endif

namespace Garnet::Tokenization
{
    struct QwenVLMRoPEMetadata
    {
        std::vector<int64_t> positionIds;
        int64_t positionDelta = 0;
    };

    class QwenTokenizer : public Tokenizer
    {
    public:
        bool LoadFromFolder(const std::string& modelDir, std::string* error = nullptr) override;
        std::vector<int64_t> Encode(const std::string& text, bool addSpecialTokens = false) const override;
        std::string Decode(const std::vector<int64_t>& tokenIds, bool skipSpecialTokens = true) const override;
        int64_t TokenId(const std::string& token) const override;

        bool IsSpecialId(int64_t id) const;

    private:
        std::unordered_map<std::string, int64_t> m_tokenToId;
        std::unordered_map<int64_t, std::string> m_idToToken;
        std::unordered_set<std::string> m_addedTokens;
        std::unordered_set<std::string> m_specialTokens;
        std::unordered_set<int64_t> m_specialIds;
        std::unordered_map<std::string, int> m_mergeRanks;
        std::unordered_map<unsigned char, std::string> m_byteToTokenChar;
        std::unordered_map<std::string, unsigned char> m_tokenCharToByte;
        size_t m_maxAddedTokenLength = 0;

        void InitByteLevelMaps();
        std::vector<std::string> SplitSpecialAware(const std::string& text) const;
        std::vector<std::string> PreTokenizeAscii(const std::string& text) const;
        std::vector<std::string> ByteLevelChars(const std::string& text) const;
        std::vector<std::string> ApplyBpe(const std::vector<std::string>& word) const;
        std::vector<int64_t> EncodePlainSegment(const std::string& text) const;
    };

    class QwenVLPromptBuilder
    {
    public:
        static int64_t VisualTokenCount(const int64_t imageGridTHW[3], int mergeSize = 2);

        static std::string BuildSingleImagePromptText(
            const std::string& userPrompt,
            const int64_t imageGridTHW[3],
            int mergeSize = 2);

        static std::vector<int64_t> BuildSingleImagePromptIds(
            const QwenTokenizer& tokenizer,
            const std::string& userPrompt,
            const int64_t imageGridTHW[3],
            int mergeSize = 2);

        static QwenVLMRoPEMetadata BuildSingleImageMRoPEMetadata(
            const std::vector<int64_t>& mmTokenTypeIds,
            const int64_t imageGridTHW[3],
            int mergeSize = 2);

        // C entry points use long long; int64_t is long on LP64 platforms.
        template<typename Integer>
        static std::vector<int64_t> BuildSingleImagePromptIds(
            const QwenTokenizer& tokenizer, const std::string& userPrompt,
            const Integer (&grid)[3], int mergeSize = 2) {
            const int64_t normalized[3] = {grid[0], grid[1], grid[2]};
            return BuildSingleImagePromptIds(tokenizer, userPrompt, normalized, mergeSize);
        }

        template<typename Integer>
        static QwenVLMRoPEMetadata BuildSingleImageMRoPEMetadata(
            const std::vector<int64_t>& mmTokenTypeIds,
            const Integer (&grid)[3], int mergeSize = 2) {
            const int64_t normalized[3] = {grid[0], grid[1], grid[2]};
            return BuildSingleImageMRoPEMetadata(mmTokenTypeIds, normalized, mergeSize);
        }
    };

    std::shared_ptr<const QwenTokenizer> GetCachedQwenTokenizer(
        const std::string& modelDir,
        std::string* error = nullptr);
}

extern "C" GARNET_TOKENIZER_EXPORT int GarnetQwenTokenizerEncode(
    const char* modelDir,
    const char* text,
    long long* outputIds,
    int outputCapacity,
    int* outputCount,
    char* errorMessage,
    int errorMessageCapacity);

extern "C" GARNET_TOKENIZER_EXPORT int GarnetQwenTokenizerDecode(
    const char* modelDir,
    const long long* tokenIds,
    int tokenCount,
    int skipSpecialTokens,
    char* outputText,
    int outputTextCapacity,
    int* outputByteCount,
    char* errorMessage,
    int errorMessageCapacity);

extern "C" GARNET_TOKENIZER_EXPORT int GarnetQwenVLBuildSingleImagePromptIds(
    const char* modelDir,
    const char* userPrompt,
    const long long* imageGridTHW,
    int mergeSize,
    long long* outputIds,
    int outputCapacity,
    int* outputCount,
    char* errorMessage,
    int errorMessageCapacity);

extern "C" GARNET_TOKENIZER_EXPORT long long GarnetQwenTokenizerTokenId(
    const char* modelDir,
    const char* token);
