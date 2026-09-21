// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "AutoTokenizer.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <regex>
#include <cctype>
#include <filesystem>

namespace {
    bool Has(const X::Value& dict, const char* key) {
        for (long long i = 0; i < dict.Size(); ++i) {
            X::Value name, value;
            if (!dict.DictEntry(i, name, value)) throw std::runtime_error("invalid tokenizer dictionary");
            if (name.ToString() == key) return true;
        }
        return false;
    }

    X::Value LoadJson(X::Runtime* runtime, const std::string& path) {
        if (!runtime) throw std::invalid_argument("tokenizer requires a runtime");
        X::Module yaml(*runtime, "yaml", "xlang_yaml");
        X::Value result;
        if (!yaml.Get("load").Call({X::Value::String(runtime->host(), path)}, result))
            throw std::runtime_error(runtime->LastError());
        return result;
    }

    X::Value ReadFile(X::Runtime* runtime, const std::string& path) {
        if (!runtime) throw std::invalid_argument("tokenizer requires a runtime");
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("cannot read tokenizer file: " + path);
        std::string text((std::istreambuf_iterator<char>(file)), {});
        if (file.bad()) throw std::runtime_error("failed reading tokenizer file: " + path);
        return X::Value::String(runtime->host(), text);
    }
}

namespace tokenizer {

    // Helper functions for string manipulation
    std::string AutoTokenizer::ltrim(const std::string& s) {
        std::string result = s;
        result.erase(result.begin(), std::find_if(result.begin(), result.end(), [](unsigned char ch) {
            return !std::isspace(ch);
            }));
        return result;
    }

    std::string AutoTokenizer::rtrim(const std::string& s) {
        std::string result = s;
        result.erase(std::find_if(result.rbegin(), result.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
            }).base(), result.end());
        return result;
    }

    std::string AutoTokenizer::trim(const std::string& s) {
        return ltrim(rtrim(s));
    }

    // Implementation of Trie::add
    void Trie::add(const std::string& word) {
        if (word.empty()) return;

        tokens_.insert(word);
        auto node = root_;
        for (char c : word) {
            if (node->children.find(c) == node->children.end()) {
                node->children[c] = std::make_shared<TrieNode>();
            }
            node = node->children[c];
        }
        node->is_end_of_word = true;
    }

    // Implementation of Trie::update
    void Trie::update(const std::vector<std::string>& words) {
        for (const auto& word : words) {
            add(word);
        }
    }

    // Implementation of Trie::split
    std::vector<std::string> Trie::split(const std::string& text) {
        if (text.empty()) {
            return {};
        }

        // States to track partial matches
        std::map<size_t, std::shared_ptr<TrieNode>> states;
        std::vector<size_t> offsets = { 0 };

        size_t skip = 0;

        // Main loop over the text
        for (size_t current = 0; current < text.size(); current++) {
            if (skip && current < skip) {
                continue;
            }

            std::set<size_t> to_remove;
            bool reset = false;

            // Track best match information
            size_t best_start = 0;
            size_t best_end = 0;
            bool found_match = false;

            // Check existing states for continued matches
            for (auto& state_pair : states) {
                size_t start_pos = state_pair.first;
                auto& trie_pointer = state_pair.second;

                if (trie_pointer->is_end_of_word) {
                    // Found a complete match
                    found_match = true;
                    best_start = start_pos;
                    best_end = current;

                    // Lookahead for longer matches
                    for (auto& lookahead_pair : states) {
                        size_t lookstart = lookahead_pair.first;
                        auto& looktrie_pointer = lookahead_pair.second;

                        if (lookstart > start_pos) {
                            break;
                        }

                        size_t lookahead_index = (lookstart < start_pos) ? current + 1 : current;
                        size_t end = (lookstart < start_pos) ? current + 1 : current;

                        // Check for end of word
                        if (looktrie_pointer->is_end_of_word) {
                            best_start = lookstart;
                            best_end = lookahead_index;
                            skip = lookahead_index;
                        }

                        // Continue lookahead
                        while (lookahead_index < text.size() &&
                            looktrie_pointer->children.find(text[lookahead_index]) != looktrie_pointer->children.end()) {
                            looktrie_pointer = looktrie_pointer->children[text[lookahead_index]];
                            lookahead_index++;

                            if (looktrie_pointer->is_end_of_word) {
                                best_start = lookstart;
                                best_end = lookahead_index;
                                skip = lookahead_index;
                            }
                        }
                    }

                    break;
                }
                else if (trie_pointer->children.find(text[current]) != trie_pointer->children.end()) {
                    // Continue the match
                    trie_pointer = trie_pointer->children[text[current]];
                    // No need to update the map key
                }
                else {
                    // Match failed, remove this state
                    to_remove.insert(start_pos);
                }
            }

            if (found_match) {
                // Store results and reset
                offsets.push_back(best_start);
                offsets.push_back(best_end);
                states.clear();
                reset = true;
            }
            else {
                // Remove failed matches
                for (auto start : to_remove) {
                    states.erase(start);
                }
            }

            // Check for new matches starting at current position
            if (current >= skip && root_->children.find(text[current]) != root_->children.end()) {
                states[current] = root_->children[text[current]];
            }
        }

        // Check final states for matches
        size_t best_final_start = 0;
        bool found_final_match = false;

        for (auto& state_pair : states) {
            size_t start_pos = state_pair.first;
            auto& trie_pointer = state_pair.second;

            if (trie_pointer->is_end_of_word) {
                size_t end = text.size();
                found_final_match = true;
                best_final_start = start_pos;
                break;
            }
        }

        if (found_final_match) {
            offsets.push_back(best_final_start);
            offsets.push_back(text.size());
        }

        // Add final offset and sort
        offsets.push_back(text.size());
        std::sort(offsets.begin(), offsets.end());

        // Remove duplicates
        offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());

        // Create the final splits
        std::vector<std::string> result;
        for (size_t i = 0; i < offsets.size() - 1; i++) {
            size_t start = offsets[i];
            size_t end = offsets[i + 1];

            if (start >= end) {
                continue;
            }

            if (start == end) {
                // Skip zero-width cuts
                continue;
            }

            result.push_back(text.substr(start, end - start));
        }

        return result;
    }
    // AutoTokenizer constructor
    AutoTokenizer::AutoTokenizer() {
        // Initialize with default values
    }

    // Update total vocabulary size
    void AutoTokenizer::_update_total_vocab_size() {
        total_vocab_size_ = vocab_size_ + added_tokens_encoder_.size();
    }

    // Convert token to ID with added vocabulary
    int AutoTokenizer::_convert_token_to_id_with_added_voc(const std::string& token) {
        // Check added tokens first
        auto it = added_tokens_encoder_.find(token);
        if (it != added_tokens_encoder_.end()) {
            return it->second;
        }

        // Then check regular vocabulary
        return _convert_token_to_id(token);
    }

    // Convert token to ID
    int AutoTokenizer::_convert_token_to_id(const std::string& token) {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            return it->second;
        }

        // Return UNK token ID if token not found
        return unk_token_id_;
    }

    // Convert ID to token
    std::string AutoTokenizer::_convert_id_to_token(int id) {
        // Check in added tokens first
        auto it_added = added_tokens_decoder_.find(id);
        if (it_added != added_tokens_decoder_.end()) {
            return it_added->second.content();
        }

        // Then check regular vocabulary
        auto it = id_to_token_.find(id);
        if (it != id_to_token_.end()) {
            return it->second;
        }

        // Return UNK token if ID not found
        return unk_token_;
    }

    // Basic tokenization - to be overridden by derived classes
    std::vector<std::string> AutoTokenizer::_tokenize(const std::string& text) {
        // Simple whitespace tokenization as a fallback
        std::vector<std::string> tokens;
        std::string current_token;

        for (char c : text) {
            if (std::isspace(c)) {
                if (!current_token.empty()) {
                    tokens.push_back(current_token);
                    current_token.clear();
                }
            }
            else {
                current_token += c;
            }
        }

        if (!current_token.empty()) {
            tokens.push_back(current_token);
        }

        return tokens;
    }

    // Update trie for tokenization
    void AutoTokenizer::_update_trie(const std::vector<std::string>& no_split_tokens) {
        // Clear and rebuild the trie
        tokens_trie_ = Trie();

        // Add added tokens to the trie
        for (const auto& [token, _] : added_tokens_encoder_) {
            tokens_trie_.add(token);
        }

        // Add specified no-split tokens
        for (const auto& token : no_split_tokens) {
            tokens_trie_.add(token);
        }
    }

    // Add tokens to the tokenizer
    int AutoTokenizer::_add_tokens(const std::vector<AddedToken>& tokens, bool special_tokens) {
        int added_tokens = 0;

        for (const auto& token : tokens) {
            // Skip empty tokens
            if (token.content().empty()) {
                continue;
            }

            // Skip tokens already in the vocabulary
            if (added_tokens_encoder_.find(token.content()) != added_tokens_encoder_.end()) {
                continue;
            }

            // Add the token
            int new_index = total_vocab_size_;
            added_tokens_encoder_[token.content()] = new_index;
            added_tokens_decoder_[new_index] = token;

            // Add to special tokens if needed
            if (special_tokens) {
                all_special_tokens_.push_back(token.content());
                all_special_ids_.push_back(new_index);
            }

            added_tokens++;
            total_vocab_size_++;
        }

        // Update the trie
        _update_trie();

        return added_tokens;
    }

    // Add special tokens to the tokenizer
    int AutoTokenizer::add_special_tokens(const std::map<std::string, AddedToken>& special_tokens_dict) {
        std::vector<AddedToken> tokens_to_add;

        for (const auto& [key, token] : special_tokens_dict) {
            // Special handling for additional_special_tokens
            if (key == "additional_special_tokens") {
                // TODO: Handle list of tokens
                continue;
            }

            // Set the special token
            if (key == "bos_token") {
                bos_token_ = token.content();
                bos_token_id_ = _convert_token_to_id_with_added_voc(bos_token_);
            }
            else if (key == "eos_token") {
                eos_token_ = token.content();
                eos_token_id_ = _convert_token_to_id_with_added_voc(eos_token_);
            }
            else if (key == "pad_token") {
                pad_token_ = token.content();
                pad_token_id_ = _convert_token_to_id_with_added_voc(pad_token_);
            }
            else if (key == "unk_token") {
                unk_token_ = token.content();
                unk_token_id_ = _convert_token_to_id_with_added_voc(unk_token_);
            }
            else if (key == "sep_token") {
                sep_token_ = token.content();
                sep_token_id_ = _convert_token_to_id_with_added_voc(sep_token_);
            }
            else if (key == "cls_token") {
                cls_token_ = token.content();
                cls_token_id_ = _convert_token_to_id_with_added_voc(cls_token_);
            }
            else if (key == "mask_token") {
                mask_token_ = token.content();
                mask_token_id_ = _convert_token_to_id_with_added_voc(mask_token_);
            }

            // Add to tokens to add
            tokens_to_add.push_back(token);
        }

        // Add the tokens as special tokens
        return _add_tokens(tokens_to_add, true);
    }

    // Add tokens to the tokenizer
    int AutoTokenizer::add_tokens(const std::vector<AddedToken>& new_tokens, bool special_tokens) {
        return _add_tokens(new_tokens, special_tokens);
    }

    // Tokenize text
    std::vector<std::string> AutoTokenizer::tokenize(const std::string& text, const std::string& text_pair, bool add_special_tokens) {
        // Prepare the text for tokenization
        std::string prepared_text = text;

        // Apply lowercasing if needed
        if (do_lower_case_) {
            std::transform(prepared_text.begin(), prepared_text.end(), prepared_text.begin(),
                [](unsigned char c) { return std::tolower(c); });
        }

        // Determine whether to use trie splitting
        std::vector<std::string> tokens;
        if (split_special_tokens_) {
            tokens = { prepared_text };
        }
        else {
            // Use the trie to split the text
            tokens = tokens_trie_.split(prepared_text);
        }

        // Process each token
        std::vector<std::string> result_tokens;
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string token = tokens[i];

            // Check if this is a special token
            auto it = added_tokens_encoder_.find(token);
            if (!split_special_tokens_ && it != added_tokens_encoder_.end()) {
                // This is a special/added token
                result_tokens.push_back(token);
            }
            else {
                // Regular token, apply tokenization
                auto tokenized = _tokenize(token);
                result_tokens.insert(result_tokens.end(), tokenized.begin(), tokenized.end());
            }
        }

        return result_tokens;
    }

    // Convert tokens to IDs
    std::vector<int> AutoTokenizer::convert_tokens_to_ids(const std::vector<std::string>& tokens) {
        std::vector<int> ids;
        ids.reserve(tokens.size());

        for (const auto& token : tokens) {
            ids.push_back(_convert_token_to_id_with_added_voc(token));
        }

        return ids;
    }

    // Convert IDs to tokens
    std::vector<std::string> AutoTokenizer::convert_ids_to_tokens(const std::vector<int>& ids, bool skip_special_tokens) {
        std::vector<std::string> tokens;
        tokens.reserve(ids.size());

        for (int id : ids) {
            // Skip special tokens if requested
            if (skip_special_tokens && std::find(all_special_ids_.begin(), all_special_ids_.end(), id) != all_special_ids_.end()) {
                continue;
            }

            // Convert ID to token
            tokens.push_back(_convert_id_to_token(id));
        }

        return tokens;
    }

    // Convert tokens to string
    std::string AutoTokenizer::convert_tokens_to_string(const std::vector<std::string>& tokens) {
        // Simple implementation - join with spaces
        std::string result;

        for (size_t i = 0; i < tokens.size(); i++) {
            if (i > 0) {
                result += " ";
            }
            result += tokens[i];
        }

        return result;
    }

    // Build inputs with special tokens
    std::vector<int> AutoTokenizer::build_inputs_with_special_tokens(
        const std::vector<int>& token_ids_0,
        const std::vector<int>& token_ids_1
    ) {
        // Default implementation just concatenates the sequences
        if (token_ids_1.empty()) {
            return token_ids_0;
        }
        else {
            std::vector<int> result = token_ids_0;
            result.insert(result.end(), token_ids_1.begin(), token_ids_1.end());
            return result;
        }
    }

    // Create token type IDs from sequences
    std::vector<int> AutoTokenizer::create_token_type_ids_from_sequences(
        const std::vector<int>& token_ids_0,
        const std::vector<int>& token_ids_1
    ) {
        // Default implementation - 0s for first sequence, 1s for second
        std::vector<int> result(token_ids_0.size(), 0);

        if (!token_ids_1.empty()) {
            std::vector<int> second_ids(token_ids_1.size(), 1);
            result.insert(result.end(), second_ids.begin(), second_ids.end());
        }

        return result;
    }

    // Get special tokens mask
    std::vector<int> AutoTokenizer::get_special_tokens_mask(
        const std::vector<int>& token_ids_0,
        const std::vector<int>& token_ids_1,
        bool already_has_special_tokens
    ) {
        if (already_has_special_tokens) {
            // For already formatted sequences, just mark the special tokens
            std::vector<int> mask(token_ids_0.size(), 0);
            for (size_t i = 0; i < token_ids_0.size(); i++) {
                if (std::find(get_all_special_ids().begin(), get_all_special_ids().end(), token_ids_0[i]) != get_all_special_ids().end()) {
                    mask[i] = 1;
                }
            }
            return mask;
        }

        // Create mask - 0 for all tokens
        std::vector<int> mask(token_ids_0.size() + (token_ids_1.empty() ? 0 : token_ids_1.size()), 0);
        return mask;
    }

    // Number of special tokens to add
    int AutoTokenizer::num_special_tokens_to_add(bool pair) {
        // Default implementation - no special tokens
        return 0;
    }

    // Get all special IDs
    std::vector<int> AutoTokenizer::get_all_special_ids() const {
        if (!all_special_ids_.empty()) {
            return all_special_ids_;
        }

        std::vector<int> ids;
        for (const auto& token : all_special_tokens_) {
            auto it = added_tokens_encoder_.find(token);
            if (it != added_tokens_encoder_.end()) {
                ids.push_back(it->second);
            }
        }

        return ids;
    }

    // Encode method - convert text to token IDs
    std::vector<int> AutoTokenizer::encode(
        const std::string& text,
        const std::string& text_pair,
        bool add_special_tokens,
        int max_length,
        bool truncation,
        bool padding
    ) {
        // Tokenize the text
        auto tokens = tokenize(text, "", false);

        // Convert tokens to IDs
        auto ids = convert_tokens_to_ids(tokens);

        // Add special tokens if requested
        if (add_special_tokens) {
            if (text_pair.empty()) {
                ids = build_inputs_with_special_tokens(ids);
            }
            else {
                // Tokenize the pair and add special tokens
                auto pair_tokens = tokenize(text_pair, "", false);
                auto pair_ids = convert_tokens_to_ids(pair_tokens);
                ids = build_inputs_with_special_tokens(ids, pair_ids);
            }
        }

        // Handle truncation
        if (truncation && max_length > 0 && static_cast<int>(ids.size()) > max_length) {
            // Simple truncation - just cut off at max_length
            ids.resize(max_length);
        }

        // Handle padding
        if (padding && max_length > 0 && static_cast<int>(ids.size()) < max_length) {
            // Pad with pad_token_id to max_length
            ids.resize(max_length, pad_token_id_);
        }

        return ids;
    }

    // Decode method - convert token IDs to text
    std::string AutoTokenizer::decode(const std::vector<int>& ids, bool skip_special_tokens) {
        // Convert IDs to tokens
        auto tokens = convert_ids_to_tokens(ids, skip_special_tokens);

        // Convert tokens to string
        auto text = convert_tokens_to_string(tokens);

        // Clean up tokenization spaces if needed
        if (clean_up_tokenization_spaces_) {
            // Simple regex-based cleanup
            std::regex space_before_punct(R"( ([.,!?:;]))");
            text = std::regex_replace(text, space_before_punct, "$1");

            // Replace multiple spaces with a single space
            std::regex multiple_spaces(R"(\s+)");
            text = std::regex_replace(text, multiple_spaces, " ");

            // Trim the result
            text = trim(text);
        }

        return text;
    }

    // Truncate sequences
    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> AutoTokenizer::truncate_sequences(
        std::vector<int> ids,
        std::vector<int> pair_ids,
        int num_tokens_to_remove,
        TruncationStrategy truncation_strategy,
        int stride
    ) {
        // Return early if no truncation needed
        if (num_tokens_to_remove <= 0) {
            return { ids, pair_ids, {} };
        }

        std::vector<int> overflowing_tokens;

        // Handle different truncation strategies
        if (truncation_strategy == TruncationStrategy::ONLY_FIRST ||
            (truncation_strategy == TruncationStrategy::LONGEST_FIRST && pair_ids.empty())) {
            // Truncate first sequence
            if (static_cast<int>(ids.size()) > num_tokens_to_remove) {
                // Collect overflowing tokens if stride > 0
                int window_len = std::min(static_cast<int>(ids.size()), stride + num_tokens_to_remove);

                if (truncation_side_ == "right") {
                    // Take tokens from the end for overflow
                    auto overflow_start = ids.end() - window_len;
                    auto overflow_end = ids.end();
                    overflowing_tokens.insert(overflowing_tokens.end(), overflow_start, overflow_end);

                    // Remove tokens from the end
                    ids.resize(ids.size() - num_tokens_to_remove);
                }
                else {
                    // Take tokens from the beginning for overflow
                    auto overflow_start = ids.begin();
                    auto overflow_end = ids.begin() + window_len;
                    overflowing_tokens.insert(overflowing_tokens.end(), overflow_start, overflow_end);

                    // Remove tokens from the beginning
                    ids.erase(ids.begin(), ids.begin() + num_tokens_to_remove);
                }
            }
            else {
                std::cerr << "Cannot truncate more tokens than sequence length" << std::endl;
            }
        }
        else if (truncation_strategy == TruncationStrategy::LONGEST_FIRST) {
            // Truncate from the longest sequence
            if (ids.size() > pair_ids.size()) {
                auto result = truncate_sequences(
                    ids, {}, num_tokens_to_remove, TruncationStrategy::ONLY_FIRST, stride);
                ids = std::get<0>(result);
                overflowing_tokens = std::get<2>(result);
            }
            else {
                auto result = truncate_sequences(
                    pair_ids, {}, num_tokens_to_remove, TruncationStrategy::ONLY_FIRST, stride);
                pair_ids = std::get<0>(result);
                overflowing_tokens = std::get<2>(result);
            }
        }
        else if (truncation_strategy == TruncationStrategy::ONLY_SECOND && !pair_ids.empty()) {
            // Truncate second sequence
            auto result = truncate_sequences(
                pair_ids, {}, num_tokens_to_remove, TruncationStrategy::ONLY_FIRST, stride);
            pair_ids = std::get<0>(result);
            overflowing_tokens = std::get<2>(result);
        }

        return { ids, pair_ids, overflowing_tokens };
    }

    // Pad sequences
    std::map<std::string, std::vector<int>> AutoTokenizer::pad(
        const std::vector<int>& ids,
        bool padding,
        int max_length,
        int pad_to_multiple_of,
        const std::string& padding_side,
        bool return_attention_mask
    ) {
        std::map<std::string, std::vector<int>> result;
        result["input_ids"] = ids;

        // If no padding, just return the IDs
        if (!padding) {
            if (return_attention_mask) {
                result["attention_mask"] = std::vector<int>(ids.size(), 1);
            }
            return result;
        }

        // Determine padding side
        std::string pad_side = padding_side.empty() ? padding_side_ : padding_side;

        // Determine max length if not provided
        int length = max_length;
        if (length < 0) {
            length = ids.size();
        }

        // Adjust length for pad_to_multiple_of
        if (pad_to_multiple_of > 0) {
            if (length % pad_to_multiple_of != 0) {
                length = ((length / pad_to_multiple_of) + 1) * pad_to_multiple_of;
            }
        }

        // Create padded input_ids
        std::vector<int> padded_ids = ids;
        int padding_length = length - ids.size();

        if (padding_length > 0) {
            if (pad_side == "right") {
                padded_ids.resize(length, pad_token_id_);
            }
            else {
                std::vector<int> padding(padding_length, pad_token_id_);
                padding.insert(padding.end(), padded_ids.begin(), padded_ids.end());
                padded_ids = padding;
            }
        }

        result["input_ids"] = padded_ids;

        // Create attention mask
        if (return_attention_mask) {
            std::vector<int> attention_mask(ids.size(), 1);
            if (padding_length > 0) {
                if (pad_side == "right") {
                    attention_mask.resize(length, 0);
                }
                else {
                    std::vector<int> padding(padding_length, 0);
                    padding.insert(padding.end(), attention_mask.begin(), attention_mask.end());
                    attention_mask = padding;
                }
            }
            result["attention_mask"] = attention_mask;
        }

        return result;
    }

    // Load tokenizer config using X framework
    void AutoTokenizer::load_tokenizer_config(X::Runtime* rt, const std::string& config_path) {
        // Use X::Package to load the JSON file
        X::Value config = LoadJson(rt, config_path);

        // Process configuration if it's a dictionary
        if (config.IsValid() && config.IsDict()) {
            X::Value dict_config(config);

            // Extract configuration parameters
            if (Has(dict_config, "do_lower_case")) {
                do_lower_case_ = (dict_config["do_lower_case"].ToLongLong() != 0);
            }

            if (Has(dict_config, "split_special_tokens")) {
                split_special_tokens_ = (dict_config["split_special_tokens"].ToLongLong() != 0);
            }

            if (Has(dict_config, "model_max_length")) {
                model_max_length_ = static_cast<int>(dict_config["model_max_length"].ToLongLong());
            }

            if (Has(dict_config, "padding_side")) {
                padding_side_ = dict_config["padding_side"].ToString();
            }

            if (Has(dict_config, "truncation_side")) {
                truncation_side_ = dict_config["truncation_side"].ToString();
            }

            if (Has(dict_config, "clean_up_tokenization_spaces")) {
                clean_up_tokenization_spaces_ = (dict_config["clean_up_tokenization_spaces"].ToLongLong() != 0);
            }

            if (Has(dict_config, "add_prefix_space")) {
                add_prefix_space_ = (dict_config["add_prefix_space"].ToLongLong() != 0);
            }

            // Handle special tokens
            if (Has(dict_config, "bos_token")) {
                X::Value bos_token = dict_config["bos_token"];
                if (bos_token.IsDict()) {
                    X::Value bos_dict(bos_token);
                    if (Has(bos_dict, "content")) {
                        bos_token_ = bos_dict["content"].ToString();
                    }
                }
                else {
                    bos_token_ = bos_token.ToString();
                }
            }

            if (Has(dict_config, "eos_token")) {
                X::Value eos_token = dict_config["eos_token"];
                if (eos_token.IsDict()) {
                    X::Value eos_dict(eos_token);
                    if (Has(eos_dict, "content")) {
                        eos_token_ = eos_dict["content"].ToString();
                    }
                }
                else {
                    eos_token_ = eos_token.ToString();
                }
            }

            if (Has(dict_config, "unk_token")) {
                X::Value unk_token = dict_config["unk_token"];
                if (unk_token.IsDict()) {
                    X::Value unk_dict(unk_token);
                    if (Has(unk_dict, "content")) {
                        unk_token_ = unk_dict["content"].ToString();
                    }
                }
                else if (unk_token.IsValid()) {
                    unk_token_ = unk_token.ToString();
                }
            }

            if (Has(dict_config, "pad_token")) {
                X::Value pad_token = dict_config["pad_token"];
                if (pad_token.IsDict()) {
                    X::Value pad_dict(pad_token);
                    if (Has(pad_dict, "content")) {
                        pad_token_ = pad_dict["content"].ToString();
                    }
                }
                else {
                    pad_token_ = pad_token.ToString();
                }
            }

            if (Has(dict_config, "sep_token")) {
                X::Value sep_token = dict_config["sep_token"];
                if (sep_token.IsDict()) {
                    X::Value sep_dict(sep_token);
                    if (Has(sep_dict, "content")) {
                        sep_token_ = sep_dict["content"].ToString();
                    }
                }
                else {
                    sep_token_ = sep_token.ToString();
                }
            }

            if (Has(dict_config, "cls_token")) {
                X::Value cls_token = dict_config["cls_token"];
                if (cls_token.IsDict()) {
                    X::Value cls_dict(cls_token);
                    if (Has(cls_dict, "content")) {
                        cls_token_ = cls_dict["content"].ToString();
                    }
                }
                else {
                    cls_token_ = cls_token.ToString();
                }
            }

            if (Has(dict_config, "mask_token")) {
                X::Value mask_token = dict_config["mask_token"];
                if (mask_token.IsDict()) {
                    X::Value mask_dict(mask_token);
                    if (Has(mask_dict, "content")) {
                        mask_token_ = mask_dict["content"].ToString();
                    }
                }
                else {
                    mask_token_ = mask_token.ToString();
                }
            }
        }
    }

    // Load tokenizer JSON using X framework
    void AutoTokenizer::load_tokenizer_json(X::Runtime* rt, const std::string& tokenizer_path) {
        // Use X::Package to load the JSON file
        X::Value tokenizer_data = LoadJson(rt, tokenizer_path);

        // Process tokenizer data if it's a dictionary
        if (tokenizer_data.IsValid() && tokenizer_data.IsDict()) {
            X::Value dict_tokenizer(tokenizer_data);

            // Load added tokens
            if (Has(dict_tokenizer, "added_tokens") && dict_tokenizer["added_tokens"].IsList()) {
                X::Value added_tokens(dict_tokenizer["added_tokens"]);

                for (long long i = 0; i < added_tokens.Size(); ++i) { X::Value token_value = added_tokens[i];
                    if (token_value.IsDict()) {
                        X::Value token_dict(token_value);
                        int id = static_cast<int>(token_dict["id"].ToLongLong());
                        std::string content = token_dict["content"].ToString();
                        bool is_special = false;

                        if (Has(token_dict, "special")) {
                            is_special = (token_dict["special"].ToLongLong() != 0);
                        }

                        token_to_id_[content] = id;
                        id_to_token_[id] = content;

                        // Create an AddedToken
                        bool lstrip = Has(token_dict, "lstrip") ? (token_dict["lstrip"].ToLongLong() != 0) : false;
                        bool rstrip = Has(token_dict, "rstrip") ? (token_dict["rstrip"].ToLongLong() != 0) : false;
                        bool single_word = Has(token_dict, "single_word") ? (token_dict["single_word"].ToLongLong() != 0) : false;
                        bool normalized = Has(token_dict, "normalized") ? (token_dict["normalized"].ToLongLong() != 0) : true;

                        AddedToken added_token(content, lstrip, rstrip, single_word, normalized, is_special);
                        added_tokens_decoder_[id] = added_token;
                        added_tokens_encoder_[content] = id;

                        if (is_special) {
                            all_special_tokens_.push_back(content);
                            all_special_ids_.push_back(id);
                        }
                    }
                }
            }

            // Load vocabulary
            if (Has(dict_tokenizer, "model") && dict_tokenizer["model"].IsDict()) {
                X::Value model_dict(dict_tokenizer["model"]);

                if (Has(model_dict, "vocab") && model_dict["vocab"].IsDict()) {
                    X::Value vocab_dict(model_dict["vocab"]);

                    for (long long i = 0; i < vocab_dict.Size(); ++i) { X::Value tokenValue, idValue; if (!vocab_dict.DictEntry(i, tokenValue, idValue)) throw std::runtime_error("invalid tokenizer vocabulary");
                        std::string token = tokenValue.ToString();
                        int id = static_cast<int>(idValue.ToLongLong());
                        token_to_id_[token] = id;
                        id_to_token_[id] = token;
                    }

                    vocab_size_ = token_to_id_.size();
                }
            }

            // Load merges for BPE
            if (Has(dict_tokenizer, "merges") && dict_tokenizer["merges"].IsList()) {
                X::Value merges_list(dict_tokenizer["merges"]);

                for (long long i = 0; i < merges_list.Size(); ++i) { X::Value merge_value = merges_list[i];
                    std::string merge_str = merge_value.ToString();
                    size_t space_pos = merge_str.find(' ');

                    if (space_pos != std::string::npos) {
                        std::string first = merge_str.substr(0, space_pos);
                        std::string second = merge_str.substr(space_pos + 1);
                        merges_.push_back({ first, second });
                    }
                }
            }
        }

        // Update total vocab size
        _update_total_vocab_size();
    }

    // Factory method to create tokenizer from pretrained
    std::shared_ptr<AutoTokenizer> AutoTokenizer::from_pretrained(const std::string& path, X::Runtime* rt) {
        std::shared_ptr<AutoTokenizer> tokenizer = std::make_shared<AutoTokenizer>();

        // Load configuration files
        tokenizer->load_tokenizer_config(rt, path + "/tokenizer_config.json");
        tokenizer->load_tokenizer_json(rt, path + "/tokenizer.json");

        // Update trie
        tokenizer->_update_trie();

        return tokenizer;
    }

    //-----------------------------------------------------------------------------------
    // BPETokenizer Implementation
    //-----------------------------------------------------------------------------------

    BPETokenizer::BPETokenizer() : AutoTokenizer() {
        // Initialize with BPE specific defaults
    }

    // Get pairs for BPE merging
    std::vector<std::pair<std::string, size_t>> BPETokenizer::get_pairs(const std::vector<std::string>& word) {
        std::vector<std::pair<std::string, size_t>> pairs;

        for (size_t i = 0; i < word.size() - 1; i++) {
            std::string pair = word[i] + " " + word[i + 1];
            pairs.push_back({ pair, i });
        }

        return pairs;
    }

    // Apply BPE to a single word
    std::string BPETokenizer::apply_bpe_to_word(const std::string& word) {
        // Check cache first
        auto cache_it = bpe_cache_.find(word);
        if (cache_it != bpe_cache_.end()) {
            return cache_it->second;
        }

        // Initial segmentation: treat each character as a separate token
        std::vector<std::string> chars;
        for (size_t i = 0; i < word.size(); ) {
            // Handle multi-byte UTF-8 characters
            size_t char_len = 1;
            if (i < word.size() && (word[i] & 0xE0) == 0xC0) char_len = 2;  // 2-byte char
            else if (i < word.size() && (word[i] & 0xF0) == 0xE0) char_len = 3;  // 3-byte char
            else if (i < word.size() && (word[i] & 0xF8) == 0xF0) char_len = 4;  // 4-byte char

            chars.push_back(word.substr(i, char_len));
            i += char_len;
        }

        // Apply BPE merges
        bool changes = true;
        while (changes && chars.size() > 1) {
            std::vector<std::pair<std::string, size_t>> pairs = get_pairs(chars);
            changes = false;

            // Find the highest-priority merge
            std::pair<std::string, std::string> best_merge = { "", "" };
            size_t best_merge_pos = 0;

            for (const auto& [pair, pos] : pairs) {
                // Separate the pair into first and second tokens
                size_t space_pos = pair.find(' ');
                if (space_pos == std::string::npos) continue;

                std::string first = pair.substr(0, space_pos);
                std::string second = pair.substr(space_pos + 1);

                // Check if this pair is in our merges list
                auto merge_it = std::find(merges_.begin(), merges_.end(), std::make_pair(first, second));
                if (merge_it != merges_.end()) {
                    // We found a merge that applies
                    best_merge = { first, second };
                    best_merge_pos = pos;

                    // Merges are ordered by priority, so we can stop at the first match
                    changes = true;
                    break;
                }
            }

            if (changes) {
                // Apply the best merge
                chars[best_merge_pos] = best_merge.first + best_merge.second;
                chars.erase(chars.begin() + best_merge_pos + 1);
            }
        }

        // Join the tokens with spaces
        std::string result = chars[0];
        for (size_t i = 1; i < chars.size(); i++) {
            result += " " + chars[i];
        }

        // Add to cache and return
        bpe_cache_[word] = result;
        return result;
    }

    // Tokenize with BPE algorithm
    std::vector<std::string> BPETokenizer::bpe_tokenize(const std::string& text) {
        // Pre-tokenize the text (split on whitespace and punctuation)
        std::vector<std::string> pre_tokens;
        std::string current_token;
        bool in_whitespace = true;

        for (size_t i = 0; i < text.size(); i++) {
            char c = text[i];

            if (std::isspace(c)) {
                if (!current_token.empty()) {
                    pre_tokens.push_back(current_token);
                    current_token.clear();
                }
                in_whitespace = true;
            }
            else {
                // If starting a new token after space, add 'Ġ' prefix
                if (in_whitespace && add_prefix_space_) {
                    current_token = "Ġ";
                }
                current_token += c;
                in_whitespace = false;
            }
        }

        if (!current_token.empty()) {
            pre_tokens.push_back(current_token);
        }

        // Apply BPE to each token
        std::vector<std::string> result;
        for (const auto& token : pre_tokens) {
            // Apply BPE on the token
            std::string bpe_result = apply_bpe_to_word(token);

            // Split the BPE result on spaces to get individual tokens
            std::string current;
            for (size_t i = 0; i < bpe_result.size(); i++) {
                if (bpe_result[i] == ' ') {
                    if (!current.empty()) {
                        result.push_back(current);
                        current.clear();
                    }
                }
                else {
                    current += bpe_result[i];
                }
            }

            if (!current.empty()) {
                result.push_back(current);
            }
        }

        return result;
    }

    // Implement _tokenize for BPE
    std::vector<std::string> BPETokenizer::_tokenize(const std::string& text) {
        return bpe_tokenize(text);
    }

    // Convert tokens to string for BPE
    std::string BPETokenizer::convert_tokens_to_string(const std::vector<std::string>& tokens) {
        std::string text;

        for (size_t i = 0; i < tokens.size(); i++) {
            // Handle "Ġ" prefix (space)
            if (!tokens[i].empty() && tokens[i][0] == 'Ġ') {
                text += " " + tokens[i].substr(1);
            }
            else {
                // No space for continuing subwords
                text += tokens[i];
            }
        }

        // Trim and return
        return trim(text);
    }

    // Build inputs with special tokens for BPE
    std::vector<int> BPETokenizer::build_inputs_with_special_tokens(
        const std::vector<int>& token_ids_0,
        const std::vector<int>& token_ids_1
    ) {
        std::vector<int> result;

        // Add BOS token if defined
        if (bos_token_id_ >= 0) {
            result.push_back(bos_token_id_);
        }

        // Add first sequence
        result.insert(result.end(), token_ids_0.begin(), token_ids_0.end());

        // Add separator if we have a second sequence
        if (!token_ids_1.empty() && sep_token_id_ >= 0) {
            result.push_back(sep_token_id_);
        }

        // Add second sequence
        if (!token_ids_1.empty()) {
            result.insert(result.end(), token_ids_1.begin(), token_ids_1.end());
        }

        // Add EOS token if defined
        if (eos_token_id_ >= 0) {
            result.push_back(eos_token_id_);
        }

        return result;
    }

    // Number of special tokens to add for BPE
    int BPETokenizer::num_special_tokens_to_add(bool pair) {
        // Calculate number of special tokens
        int count = 0;

        // BOS token
        if (bos_token_id_ >= 0) count++;

        // SEP token (only if we have a pair)
        if (pair && sep_token_id_ >= 0) count++;

        // EOS token
        if (eos_token_id_ >= 0) count++;

        return count;
    }

    // Load vocabulary and merges for BPE using X framework
    void BPETokenizer::load_vocab_and_merges(X::Runtime* rt, const std::string& vocab_file, const std::string& merges_file) {
        // Load vocabulary using X framework
        X::Value vocab_data = LoadJson(rt, vocab_file);

        // Process vocabulary if it's a dictionary
        if (vocab_data.IsValid() && vocab_data.IsDict()) {
            X::Value vocab_dict(vocab_data);

            for (long long i = 0; i < vocab_dict.Size(); ++i) { X::Value tokenValue, idValue; if (!vocab_dict.DictEntry(i, tokenValue, idValue)) throw std::runtime_error("invalid tokenizer vocabulary");
                std::string token = tokenValue.ToString();
                int id = static_cast<int>(idValue.ToLongLong());
                token_to_id_[token] = id;
                id_to_token_[id] = token;
            }

            vocab_size_ = token_to_id_.size();
        }

        // Load merges file using X Framework
        X::Value merges_content = ReadFile(rt, merges_file);

        if (merges_content.IsValid()) {
            std::string content = merges_content.ToString();
            std::istringstream iss(content);
            std::string line;

            // Skip version header if present
            std::getline(iss, line);
            if (line.find("#version") != std::string::npos) {
                // Skip the version line
            }
            else {
                // Process the first line
                std::istringstream ss(line);
                std::string first, second;
                ss >> first >> second;
                if (!first.empty() && !second.empty()) {
                    merges_.push_back({ first, second });
                }
            }

            // Process remaining merges
            while (std::getline(iss, line)) {
                std::istringstream ss(line);
                std::string first, second;
                ss >> first >> second;
                if (!first.empty() && !second.empty()) {
                    merges_.push_back({ first, second });
                }
            }
        }

        // Update total vocab size
        _update_total_vocab_size();
    }

    // Factory method for BPE tokenizer
    std::shared_ptr<BPETokenizer> BPETokenizer::from_pretrained(const std::string& path, X::Runtime* rt) {
        std::shared_ptr<BPETokenizer> tokenizer = std::make_shared<BPETokenizer>();

        // Load configuration files
        tokenizer->load_tokenizer_config(rt, path + "/tokenizer_config.json");

        // Check for vocab.json and merges.txt
        std::string vocab_file = path + "/vocab.json";
        std::string merges_file = path + "/merges.txt";

        // Check if files exist before trying to load them

        bool vocab_exists = std::filesystem::exists(vocab_file);
        bool merges_exists = std::filesystem::exists(merges_file);

        if (vocab_exists && merges_exists) {
            tokenizer->load_vocab_and_merges(rt, vocab_file, merges_file);
        }
        else {
            // Fall back to loading from tokenizer.json
            tokenizer->load_tokenizer_json(rt, path + "/tokenizer.json");
        }

        // Update trie
        tokenizer->_update_trie();

        return tokenizer;
    }

    //-----------------------------------------------------------------------------------
    // WordPieceTokenizer Implementation
    //-----------------------------------------------------------------------------------

    WordPieceTokenizer::WordPieceTokenizer() : AutoTokenizer() {
        // Initialize with WordPiece specific defaults
    }

    // Tokenize with WordPiece algorithm
    std::vector<std::string> WordPieceTokenizer::wordpiece_tokenize(const std::string& text) {
        // First split on whitespace
        std::vector<std::string> tokens;
        std::string current_word;

        for (char c : text) {
            if (std::isspace(c)) {
                if (!current_word.empty()) {
                    tokens.push_back(current_word);
                    current_word.clear();
                }
            }
            else {
                current_word += c;
            }
        }

        if (!current_word.empty()) {
            tokens.push_back(current_word);
        }

        // Apply WordPiece to each word
        std::vector<std::string> result;

        for (const auto& word : tokens) {
            if (word.empty()) continue;

            // Check if the whole word is in the vocabulary
            if (token_to_id_.find(word) != token_to_id_.end()) {
                result.push_back(word);
                continue;
            }

            // Split the word into subwords
            std::vector<std::string> subwords;
            std::string current_subword;

            for (size_t i = 0; i < word.size(); i++) {
                current_subword += word[i];

                std::string remaining = word.substr(i + 1);
                bool is_good_subword = false;

                // Check if current_subword is in the vocabulary
                if (token_to_id_.find(current_subword) != token_to_id_.end()) {
                    if (remaining.empty()) {
                        // End of word reached with a valid subword
                        subwords.push_back(current_subword);
                        break;
                    }

                    // Try to find a valid continuation
                    std::string test_subword = current_subword;
                    bool found_continuation = false;

                    for (size_t j = i + 1; j <= word.size(); j++) {
                        std::string next_subword = wordpiece_prefix_;
                        next_subword += word.substr(i + 1, j - i - 1);

                        if (token_to_id_.find(next_subword) != token_to_id_.end()) {
                            found_continuation = true;
                            break;
                        }
                    }

                    if (!found_continuation) {
                        // No valid continuation, use current subword
                        subwords.push_back(current_subword);
                        i = i;  // Start next subword from the next character
                        current_subword = "";
                        is_good_subword = true;
                    }
                }

                if (is_good_subword) break;

                // If we reach the end and no valid subword found, use UNK token
                if (i == word.size() - 1) {
                    subwords.clear();
                    subwords.push_back(unk_token_);
                    break;
                }
            }

            // If no subwords found, use the UNK token
            if (subwords.empty()) {
                result.push_back(unk_token_);
            }
            else {
                // Add the first subword
                result.push_back(subwords[0]);

                // Add remaining subwords with prefix
                for (size_t i = 1; i < subwords.size(); i++) {
                    if (subwords[i].find(wordpiece_prefix_) != 0) {
                        result.push_back(wordpiece_prefix_ + subwords[i]);
                    }
                    else {
                        result.push_back(subwords[i]);
                    }
                }
            }
        }

        return result;
    }

    // Implement _tokenize for WordPiece
    std::vector<std::string> WordPieceTokenizer::_tokenize(const std::string& text) {
        return wordpiece_tokenize(text);
    }

    // Convert tokens to string for WordPiece
    std::string WordPieceTokenizer::convert_tokens_to_string(const std::vector<std::string>& tokens) {
        std::string text;

        for (size_t i = 0; i < tokens.size(); i++) {
            if (i > 0 && tokens[i].find(wordpiece_prefix_) != 0) {
                text += " ";
            }

            // Remove WordPiece prefix for subwords
            if (tokens[i].find(wordpiece_prefix_) == 0) {
                text += tokens[i].substr(wordpiece_prefix_.size());
            }
            else {
                text += tokens[i];
            }
        }

        return text;
    }

    // Build inputs with special tokens for WordPiece (BERT style)
    std::vector<int> WordPieceTokenizer::build_inputs_with_special_tokens(
        const std::vector<int>& token_ids_0,
        const std::vector<int>& token_ids_1
    ) {
        std::vector<int> result;

        // Add CLS token
        if (cls_token_id_ >= 0) {
            result.push_back(cls_token_id_);
        }

        // Add first sequence
        result.insert(result.end(), token_ids_0.begin(), token_ids_0.end());

        // Add SEP token
        if (sep_token_id_ >= 0) {
            result.push_back(sep_token_id_);
        }

        // Add second sequence and another SEP token if provided
        if (!token_ids_1.empty()) {
            result.insert(result.end(), token_ids_1.begin(), token_ids_1.end());

            if (sep_token_id_ >= 0) {
                result.push_back(sep_token_id_);
            }
        }

        return result;
    }

    // Create token type IDs for WordPiece (BERT style)
    std::vector<int> WordPieceTokenizer::create_token_type_ids_from_sequences(
        const std::vector<int>& token_ids_0,
        const std::vector<int>& token_ids_1
    ) {
        // Calculate offsets for special tokens
        int cls_offset = (cls_token_id_ >= 0) ? 1 : 0;
        int sep_offset = (sep_token_id_ >= 0) ? 1 : 0;

        // Create token type IDs - 0s for first sequence (including CLS and first SEP)
        std::vector<int> result(token_ids_0.size() + cls_offset + sep_offset, 0);

        // Add 1s for second sequence (including second SEP)
        if (!token_ids_1.empty()) {
            std::vector<int> second_type_ids(token_ids_1.size() + sep_offset, 1);
            result.insert(result.end(), second_type_ids.begin(), second_type_ids.end());
        }

        return result;
    }

    // Number of special tokens to add for WordPiece (BERT style)
    int WordPieceTokenizer::num_special_tokens_to_add(bool pair) {
        // CLS token + SEP token
        int count = ((cls_token_id_ >= 0) ? 1 : 0) + ((sep_token_id_ >= 0) ? 1 : 0);

        // Add another SEP token for pairs
        if (pair && sep_token_id_ >= 0) {
            count += 1;
        }

        return count;
    }

    // Load vocabulary for WordPiece using X framework
    void WordPieceTokenizer::load_vocab(X::Runtime* rt, const std::string& vocab_file) {
        // Try different approaches to load the vocab file

        // First try loading as JSON
        X::Value vocab_data;
        if (std::filesystem::path(vocab_file).extension() != ".txt")
            vocab_data = LoadJson(rt, vocab_file);

        if (vocab_data.IsValid() && vocab_data.IsDict()) {
            // Process the vocab data as a dictionary
            X::Value vocab_dict(vocab_data);

            for (long long i = 0; i < vocab_dict.Size(); ++i) { X::Value tokenValue, idValue; if (!vocab_dict.DictEntry(i, tokenValue, idValue)) throw std::runtime_error("invalid tokenizer vocabulary");
                std::string token = tokenValue.ToString();
                int id = static_cast<int>(idValue.ToLongLong());
                token_to_id_[token] = id;
                id_to_token_[id] = token;
            }
        }
        else {
            // Try loading as a text file with one token per line
            X::Value content = ReadFile(rt, vocab_file);

            if (content.IsValid()) {
                std::string text = content.ToString();
                std::istringstream iss(text);
                std::string line;
                int index = 0;

                while (std::getline(iss, line)) {
                    line = trim(line);
                    if (!line.empty()) {
                        token_to_id_[line] = index;
                        id_to_token_[index] = line;
                        index++;
                    }
                }
            }
        }

        vocab_size_ = token_to_id_.size();
        _update_total_vocab_size();
    }

    // Factory method for WordPiece tokenizer
    std::shared_ptr<WordPieceTokenizer> WordPieceTokenizer::from_pretrained(const std::string& path, X::Runtime* rt) {
        std::shared_ptr<WordPieceTokenizer> tokenizer = std::make_shared<WordPieceTokenizer>();

        // Load configuration files
        tokenizer->load_tokenizer_config(rt, path + "/tokenizer_config.json");

        // Check for vocab.txt or vocab.json
        std::string vocab_file = path + "/vocab.txt";

        // Check if vocab.txt exists

        bool vocab_txt_exists = std::filesystem::exists(vocab_file);

        if (!vocab_txt_exists) {
            // Try vocab.json
            vocab_file = path + "/vocab.json";
            bool vocab_json_exists = std::filesystem::exists(vocab_file);

            if (!vocab_json_exists) {
                // Fall back to tokenizer.json
                tokenizer->load_tokenizer_json(rt, path + "/tokenizer.json");
                return tokenizer;
            }
        }

        // Load vocabulary
        tokenizer->load_vocab(rt, vocab_file);

        // Update trie
        tokenizer->_update_trie();

        return tokenizer;
    }

} // namespace tokenizer
