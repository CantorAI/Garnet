#pragma once
namespace Garnet {inline constexpr const char* kPrimaryIsolatedPreamble=R"IPRIMARY(#include <metal_stdlib>
using namespace metal;
struct BF16Params {ulong rows,inner,columns,aStride,oStride;};

constant uint fixed_rows [[function_constant(0)]];
constant uint fixed_inner [[function_constant(1)]];
constant uint fixed_columns [[function_constant(2)]];
constant uint fixed_aStride [[function_constant(3)]];
constant uint fixed_oStride [[function_constant(4)]];
constant bool fixed_no_tail [[function_constant(5)]];
)IPRIMARY";}
