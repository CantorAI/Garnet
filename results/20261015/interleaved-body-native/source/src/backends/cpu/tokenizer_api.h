// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "xlang3/xlang3.h"
#include "qwen_tokenizer.h"
namespace Garnet {
class CPUTokenizer {
    Tokenization::QwenTokenizer m_tokenizer;
    bool m_ready=false;
    void Require() const {if(!m_ready)throw X::Error("tokenizer must be opened first");}
public:
    bool Open(const std::string& folder) {
        m_ready=false;std::string error;
        if(!m_tokenizer.LoadFromFolder(folder,&error))throw X::Error(error);
        m_ready=true;return true;
    }
    X::Value Encode(const std::string& text) const {
        Require();auto result=X::Value::List(Host());
        for(auto id:m_tokenizer.Encode(text,false))result.Append(X::Value(id));return result;
    }
    std::string Decode(X::Value ids) const {
        Require();if(!ids.IsList())throw X::Error("token IDs must be a list");
        std::vector<int64_t> values;for(uint64_t i=0;i<ids.Size();++i)values.push_back(ids.Get(i).ToInt64());
        return m_tokenizer.Decode(values,false);
    }
    int64_t TokenId(const std::string& token) const {Require();return m_tokenizer.TokenId(token);}
    BEGIN_PACKAGE(CPUTokenizer)
        APISET().AddFunc<1>("open",&CPUTokenizer::Open);
        APISET().AddFunc<1>("encode",&CPUTokenizer::Encode);
        APISET().AddFunc<1>("decode",&CPUTokenizer::Decode);
        APISET().AddFunc<1>("token_id",&CPUTokenizer::TokenId);
    END_PACKAGE
};
}
