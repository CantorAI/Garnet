// SPDX-License-Identifier: Apache-2.0
// Exact BF16 wire sum/round semantics; expert FP32 partials are ineligible.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace Garnet { namespace GptOssPeer {
constexpr size_t kMaximumPairElements=size_t(2880)*4096;
__device__ __forceinline__ float roundedPair(unsigned a,unsigned b) {
    const float x=__bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(a)));
    const float y=__bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(b)));
    return __bfloat162float(__float2bfloat16_rn(x+y));
}
} }
