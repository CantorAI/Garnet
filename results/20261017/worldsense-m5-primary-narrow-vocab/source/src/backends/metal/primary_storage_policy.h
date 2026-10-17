// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <string>
namespace PrimaryPolicy {
inline bool EndsWith(const std::string& value,const std::string& suffix) {
    return value.size()>=suffix.size()&&value.compare(value.size()-suffix.size(),suffix.size(),suffix)==0;
}
inline bool AllowPrimary(const std::string& name,size_t rows,size_t columns,bool rawNarrowVocabulary) {
    const bool vocabulary=name=="lm_head.weight"||EndsWith(name,".lm_head.weight")||EndsWith(name,".embed_tokens.weight");
    return !(rawNarrowVocabulary&&vocabulary&&rows>65536&&columns>0&&columns<=2048);
}
}
