// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstdint>
#include <vector>
namespace WideStorage {
inline bool ConsistentWide(const std::vector<uint64_t>& widths) {
    if(widths.empty()||widths.front()<=2048)return false;
    for(auto width:widths)if(width!=widths.front())return false;
    return true;
}
template<class Index> bool Eligible(const Index& index) {
    std::vector<uint64_t> widths;
    for(const char* name:std::array<const char*,3>{"model.embed_tokens.weight","model.language_model.embed_tokens.weight","lm_head.weight"}) {
        const auto* entry=index.Find(name);if(!entry)continue;
        if(entry->shape.size()!=2||entry->shape[0]<=0||entry->shape[1]<=0)return false;
        widths.push_back(uint64_t(entry->shape[1]));
    }
    return ConsistentWide(widths);
}
}
