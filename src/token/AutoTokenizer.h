// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <memory>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include "xlang3/xlang3.h"


namespace tokenizer {

    /**
     * Represents a token added to the vocabulary with special properties
     */
    class AddedToken {
    private:
        std::string content_;
        bool lstrip_;
        bool rstrip_;
        bool single_word_;
        bool normalized_;
        bool special_;

    public:
		AddedToken() : content_(""), lstrip_(false), rstrip_(false),
            single_word_(false), normalized_(true), special_(false) {}
        AddedToken(const std::string& content, bool lstrip = false, bool rstrip = false,
            bool single_word = false, bool normalized = true, bool special = false)
            : content_(content), lstrip_(lstrip), rstrip_(rstrip), single_word_(single_word),
            normalized_(normalized), special_(special) {
        }

        // Getters
        const std::string& content() const { return content_; }
        bool lstrip() const { return lstrip_; }
        bool rstrip() const { return rstrip_; }
        bool single_word() const { return single_word_; }
        bool normalized() const { return normalized_; }
        bool special() const { return special_; }
    };

    /**
     * Trie data structure for token detection and splitting
     */
    class Trie {
    private:
        struct TrieNode {
            std::unordered_map<char, std::shared_ptr<TrieNode>> children;
            bool is_end_of_word;

            TrieNode() : is_end_of_word(false) {}
        };

        std::shared_ptr<TrieNode> root_;
        std::unordered_set<std::string> tokens_;

    public:
        Trie() : root_(std::make_shared<TrieNode>()) {}

        void add(const std::string& word);
        void update(const std::vector<std::string>& words);
        std::vector<std::string> split(const std::string& text);
        const std::unordered_set<std::string>& tokens() const { return tokens_; }
    };

    enum class TruncationStrategy {
        LONGEST_FIRST,
        ONLY_FIRST,
        ONLY_SECOND,
        DO_NOT_TRUNCATE
    };

    enum class PaddingStrategy {
        LONGEST,
        MAX_LENGTH,
        DO_NOT_PAD
    };

    /**
     * Main tokenizer class that handles tokenization, encoding, and decoding
     */
    class AutoTokenizer {
    protected:
        // Configuration
        bool do_lower_case_ = false;
        bool split_special_tokens_ = false;
        int model_max_length_ = 512;
        std::string padding_side_ = "right";
        std::string truncation_side_ = "right";
        bool clean_up_tokenization_spaces_ = true;
        bool add_prefix_space_ = false;

        // Vocabulary and tokens
        std::unordered_map<std::string, int> token_to_id_;
        std::unordered_map<int, std::string> id_to_token_;
        std::unordered_map<std::string, int> added_tokens_encoder_;
        std::unordered_map<int, AddedToken> added_tokens_decoder_;
        int vocab_size_ = 0;
        int total_vocab_size_ = 0;

        // Special tokens
        std::string bos_token_;
        std::string eos_token_;
        std::string pad_token_;
        std::string unk_token_;
        std::string sep_token_;
        std::string cls_token_;
        std::string mask_token_;
        int bos_token_id_ = -1;
        int eos_token_id_ = -1;
        int pad_token_id_ = 0;
        int unk_token_id_ = -1;
        int sep_token_id_ = -1;
        int cls_token_id_ = -1;
        int mask_token_id_ = -1;
        std::vector<std::string> all_special_tokens_;
        std::vector<int> all_special_ids_;

        // Trie for tokenization
        Trie tokens_trie_;

        // Merges for BPE
        std::vector<std::pair<std::string, std::string>> merges_;

        // Internal methods
        virtual std::vector<std::string> _tokenize(const std::string& text);
        int _add_tokens(const std::vector<AddedToken>& tokens, bool special_tokens = false);
        void _update_trie(const std::vector<std::string>& no_split_tokens = {});
        int _convert_token_to_id_with_added_voc(const std::string& token);
        int _convert_token_to_id(const std::string& token);
        std::string _convert_id_to_token(int id);

        // Helper methods
        std::string ltrim(const std::string& s);
        std::string rtrim(const std::string& s);
        std::string trim(const std::string& s);
        void _update_total_vocab_size();

    public:
        // Constructor
        AutoTokenizer();
        virtual ~AutoTokenizer() = default;

        // Factory methods
        static std::shared_ptr<AutoTokenizer> from_pretrained(const std::string& path, X::Runtime* rt);

        // Tokenization methods
        std::vector<std::string> tokenize(const std::string& text, const std::string& text_pair = "",
            bool add_special_tokens = false);

        std::vector<int> encode(const std::string& text, const std::string& text_pair = "",
            bool add_special_tokens = true,
            int max_length = -1,
            bool truncation = false,
            bool padding = false);

        std::string decode(const std::vector<int>& ids, bool skip_special_tokens = false);

        // Conversion methods
        std::vector<int> convert_tokens_to_ids(const std::vector<std::string>& tokens);
        std::vector<std::string> convert_ids_to_tokens(const std::vector<int>& ids, bool skip_special_tokens = false);
        virtual std::string convert_tokens_to_string(const std::vector<std::string>& tokens);

        // Token handling methods
        int add_special_tokens(const std::map<std::string, AddedToken>& special_tokens_dict);
        int add_tokens(const std::vector<AddedToken>& new_tokens, bool special_tokens = false);

        // Input processing methods
        virtual std::vector<int> build_inputs_with_special_tokens(
            const std::vector<int>& token_ids_0,
            const std::vector<int>& token_ids_1 = {});

        virtual std::vector<int> create_token_type_ids_from_sequences(
            const std::vector<int>& token_ids_0,
            const std::vector<int>& token_ids_1 = {});

        virtual std::vector<int> get_special_tokens_mask(
            const std::vector<int>& token_ids_0,
            const std::vector<int>& token_ids_1 = {},
            bool already_has_special_tokens = false);

        // Padding and truncation methods
        std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> truncate_sequences(
            std::vector<int> ids,
            std::vector<int> pair_ids = {},
            int num_tokens_to_remove = 0,
            TruncationStrategy truncation_strategy = TruncationStrategy::LONGEST_FIRST,
            int stride = 0);

        std::map<std::string, std::vector<int>> pad(
            const std::vector<int>& ids,
            bool padding = true,
            int max_length = -1,
            int pad_to_multiple_of = 0,
            const std::string& padding_side = "",
            bool return_attention_mask = true);

        // Utility functions
        virtual int num_special_tokens_to_add(bool pair = false);

        // Getters
        int get_vocab_size() const { return vocab_size_; }
        const std::unordered_map<std::string, int>& get_vocab() const { return token_to_id_; }
        const std::unordered_map<std::string, int>& get_added_vocab() const { return added_tokens_encoder_; }
        const std::vector<std::string>& get_all_special_tokens() const { return all_special_tokens_; }
        std::vector<int> get_all_special_ids() const;

        // X framework specific methods
        void load_tokenizer_config(X::Runtime* rt, const std::string& config_path);
        void load_tokenizer_json(X::Runtime* rt, const std::string& tokenizer_path);
        X::Value to_xlang_value(X::Runtime* rt) const;
        static std::shared_ptr<AutoTokenizer> from_xlang_value(X::Runtime* rt, const X::Value& value);
    };

    /**
     * BPE (Byte-Pair Encoding) Tokenizer
     */
    class BPETokenizer : public AutoTokenizer {
    private:
        std::string continuing_subword_prefix_ = "";
        std::string end_of_word_suffix_ = "";
        std::unordered_map<std::string, std::string> bpe_cache_;

        // Helper methods for BPE
        std::vector<std::string> bpe_tokenize(const std::string& text);
        std::vector<std::pair<std::string, size_t>> get_pairs(const std::vector<std::string>& word);
        std::string apply_bpe_to_word(const std::string& word);

    protected:
        // Override tokenization
        std::vector<std::string> _tokenize(const std::string& text) override;

    public:
        BPETokenizer();

        // Factory method
        static std::shared_ptr<BPETokenizer> from_pretrained(const std::string& path, X::Runtime* rt);

        // Override conversion methods
        std::string convert_tokens_to_string(const std::vector<std::string>& tokens) override;

        // Special token handling
        std::vector<int> build_inputs_with_special_tokens(
            const std::vector<int>& token_ids_0,
            const std::vector<int>& token_ids_1 = {}) override;

        int num_special_tokens_to_add(bool pair = false) override;

        // Load vocabulary and merges using X framework
        void load_vocab_and_merges(X::Runtime* rt, const std::string& vocab_file, const std::string& merges_file);
    };

    /**
     * WordPiece Tokenizer (used by BERT and similar models)
     */
    class WordPieceTokenizer : public AutoTokenizer {
    private:
        std::string wordpiece_prefix_ = "##";

        // Helper methods for WordPiece
        std::vector<std::string> wordpiece_tokenize(const std::string& text);

    protected:
        // Override tokenization
        std::vector<std::string> _tokenize(const std::string& text) override;

    public:
        WordPieceTokenizer();

        // Factory method
        static std::shared_ptr<WordPieceTokenizer> from_pretrained(const std::string& path, X::Runtime* rt);

        // Override conversion methods
        std::string convert_tokens_to_string(const std::vector<std::string>& tokens) override;

        // Special token handling
        std::vector<int> build_inputs_with_special_tokens(
            const std::vector<int>& token_ids_0,
            const std::vector<int>& token_ids_1 = {}) override;

        std::vector<int> create_token_type_ids_from_sequences(
            const std::vector<int>& token_ids_0,
            const std::vector<int>& token_ids_1 = {}) override;

        int num_special_tokens_to_add(bool pair = false) override;

        // Load vocabulary using X framework
        void load_vocab(X::Runtime* rt, const std::string& vocab_file);
    };

} // namespace tokenizer