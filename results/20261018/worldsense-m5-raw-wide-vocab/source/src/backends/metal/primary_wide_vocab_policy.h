// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <string>
#include <cstddef>
namespace Garnet::WideVocabularyPolicy {
// An exclusion only: all original shape/dtype/device/packing checks still apply.
inline bool EndsWith(const std::string& name,const std::string& suffix) {
 return name.size()>=suffix.size()&&name.compare(name.size()-suffix.size(),suffix.size(),suffix)==0;
}
inline bool AllowPrimary(const std::string& name,size_t rows,size_t columns,bool wideCheckpoint,bool rawWideVocabulary) {
 const bool vocabulary=name=="lm_head.weight"||EndsWith(name,".lm_head.weight")||EndsWith(name,".embed_tokens.weight");
 return !(rawWideVocabulary&&wideCheckpoint&&vocabulary&&rows>65536&&columns>2048);
}
}
