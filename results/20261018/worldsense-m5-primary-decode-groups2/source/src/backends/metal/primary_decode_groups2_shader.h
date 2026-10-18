// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "primary_block_shader.h"
#include <string>
#include <stdexcept>
namespace Garnet {
inline const std::string& PrimaryDecodeGroups2Program() {
    static const std::string program=[] {
        std::string value(kPrimaryBlockShader);
        const std::string before="ulong column=ulong(group)*16+simd*4;";
        const std::string after="ulong column=ulong(group)*8+simd*4;";
        size_t position=0,count=0;
        while((position=value.find(before,position))!=std::string::npos) {
            value.replace(position,before.size(),after);position+=after.size();++count;
        }
        if(count!=2)throw std::logic_error("Packed groups2 requires exactly two original output origins");
        return value;
    }();
    return program;
}
}
