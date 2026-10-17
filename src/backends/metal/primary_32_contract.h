// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace Primary32 {
constexpr uint64_t MaxRawBytes=uint64_t{1}<<31;
inline bool ShapeFits(uint64_t rows,uint64_t inner,uint64_t bytes) {
    if(rows<=256||!inner||rows%4||inner%128||rows>UINT32_MAX||inner>UINT32_MAX)return false;
    // Divide before multiplying: even invalid caller dimensions cannot overflow.
    if(rows>(MaxRawBytes/2)/inner)return false;
    return bytes==rows*inner*2;
}
inline uint32_t NarrowMetadata(uint64_t value) {
    if(value>UINT32_MAX)throw std::overflow_error("Primary metadata exceeds uint32 range");
    return static_cast<uint32_t>(value);
}
}
