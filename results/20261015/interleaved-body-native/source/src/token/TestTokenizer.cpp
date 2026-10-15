// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0


#include "AutoTokenizer.h"
#include <iostream>
#include <string>
#include <vector>


int simple_test(X::Runtime& rt) {

    // Load tokenizer
    auto tokenizer = tokenizer::BPETokenizer::from_pretrained("models/deepseek-moe-16b-base", &rt);

    // Tokenize text
    std::string text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is";

    auto tokens = tokenizer->tokenize(text);

    // Convert to token IDs
    auto ids = tokenizer->encode(text, "", true);

    // Decode back to text
    std::string decoded = tokenizer->decode(ids);
    return 0;
}
int Test_Tokenizer(X::Runtime& rt) {

    try {
        std::cout << "Loading tokenizer..." << std::endl;

        // Create a BPE tokenizer from scratch
        auto tokenizer = std::make_shared<tokenizer::BPETokenizer>();

        // Normally, you would load a pre-trained tokenizer like this:
        // auto tokenizer = tokenizer::AutoTokenizer::from_pretrained("/path/to/model", &rt);

        // For this example, we'll manually set up some tokens
        std::vector<tokenizer::AddedToken> special_tokens = {
            tokenizer::AddedToken("<s>", false, false, true, true, true),
            tokenizer::AddedToken("</s>", false, false, true, true, true),
            tokenizer::AddedToken("<unk>", false, false, true, true, true),
            tokenizer::AddedToken("<pad>", false, false, true, true, true)
        };

        tokenizer->add_tokens(special_tokens, true);

        // Add some regular tokens
        std::vector<tokenizer::AddedToken> regular_tokens = {
            tokenizer::AddedToken("hello"),
            tokenizer::AddedToken("world"),
            tokenizer::AddedToken("how"),
            tokenizer::AddedToken("are"),
            tokenizer::AddedToken("you"),
            tokenizer::AddedToken("today"),
            tokenizer::AddedToken("?")
        };

        tokenizer->add_tokens(regular_tokens);

        // Example: Tokenize a simple text
        std::string text = "Hello world, how are you today?";
        std::cout << "\nTokenizing: \"" << text << "\"" << std::endl;

        // Tokenize
        auto tokens = tokenizer->tokenize(text);
        std::cout << "Tokens: ";
        for (const auto& token : tokens) {
            std::cout << "\"" << token << "\" ";
        }
        std::cout << std::endl;

        // Convert to IDs
        auto ids = tokenizer->encode(text, "", true);
        std::cout << "Token IDs: ";
        for (int id : ids) {
            std::cout << id << " ";
        }
        std::cout << std::endl;

        // Decode back to text
        std::string decoded = tokenizer->decode(ids);
        std::cout << "Decoded: \"" << decoded << "\"" << std::endl;

        // Example with special tokens
        std::cout << "\nExample with special tokens:" << std::endl;
        std::string prompt = "Hello, how are you?";
        std::string response = "I'm doing great, thanks for asking!";

        // Add special tokens and encode
        std::cout << "Encoding conversation..." << std::endl;
        auto prompt_ids = tokenizer->encode(prompt);
        auto response_ids = tokenizer->encode(response);

        // Build inputs with special tokens
        auto combined_ids = tokenizer->build_inputs_with_special_tokens(prompt_ids, response_ids);

        std::cout << "Combined IDs: ";
        for (int id : combined_ids) {
            std::cout << id << " ";
        }
        std::cout << std::endl;

        // Decode combined text
        std::string combined_text = tokenizer->decode(combined_ids);
        std::cout << "Decoded conversation: \"" << combined_text << "\"" << std::endl;

        // Example using token type IDs for models like BERT
        auto token_type_ids = tokenizer->create_token_type_ids_from_sequences(prompt_ids, response_ids);
        std::cout << "Token type IDs: ";
        for (int id : token_type_ids) {
            std::cout << id << " ";
        }
        std::cout << std::endl;

        // Demonstrate how to use the X framework with the tokenizer
        std::cout << "\nDemonstrating X framework integration:" << std::endl;

        // Usually, you would load a tokenizer configuration like this
        X::Module yaml(rt, "yaml", "xlang_yaml");
        X::Value tokenizer_config;
        if (!yaml.Get("load").Call({X::Value::String(rt.host(), "tokenizer_config.json")}, tokenizer_config))
            throw std::runtime_error(rt.LastError());

        // Then process that configuration
        if (tokenizer_config.IsDict()) {
            std::cout << "Loaded tokenizer configuration with parameters:" << std::endl;

            for (long long i = 0; i < tokenizer_config.Size(); ++i) {
                X::Value key, value;
                if (!tokenizer_config.DictEntry(i, key, value)) throw std::runtime_error("invalid tokenizer config");
                std::cout << "  " << key.ToString() << ": " << value.ToString() << std::endl;
            }
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
