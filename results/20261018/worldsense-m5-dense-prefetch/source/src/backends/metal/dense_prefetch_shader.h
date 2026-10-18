// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
// Original per-output BF16 expansion, accumulation and SIMD reduction order.
// Only immutable activation/weight reads are scheduled one iteration ahead.
inline constexpr const char* kDensePrefetchShader=R"DENSELOOK(
struct DenseLookaheadPacket { float4 x; ushort4 b0,b1,b2,b3; };
inline DenseLookaheadPacket dense_lookahead_load(device const float* a,device const ushort* weight,ulong column,ulong k) {
    DenseLookaheadPacket value;
    value.x=*(device const float4*)(a+k);
    value.b0=*(device const ushort4*)(weight+column*ulong(fixed_inner)+k);
    value.b1=ushort4(0);value.b2=ushort4(0);value.b3=ushort4(0);
    if(fixed_no_tail||column+1<ulong(fixed_columns))value.b1=*(device const ushort4*)(weight+(column+1)*ulong(fixed_inner)+k);
    if(fixed_no_tail||column+2<ulong(fixed_columns))value.b2=*(device const ushort4*)(weight+(column+2)*ulong(fixed_inner)+k);
    if(fixed_no_tail||column+3<ulong(fixed_columns))value.b3=*(device const ushort4*)(weight+(column+3)*ulong(fixed_inner)+k);
    return value;
}
kernel void dense_prefetched_bf16_decode(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;
    if(!fixed_no_tail&&column>=ulong(fixed_columns))return;
    ulong k=ulong(lane)*4;
    DenseLookaheadPacket current=dense_lookahead_load(a,weight,column,k);
    float4 sum0=0.0f;
    for(;k<ulong(fixed_inner);k+=128) {
        DenseLookaheadPacket next={float4(0.0f),ushort4(0),ushort4(0),ushort4(0),ushort4(0)};
        if(k+128<ulong(fixed_inner))next=dense_lookahead_load(a,weight,column,k+128);
        for(uint j=0;j<4;++j) {
            float4 w=as_type<float4>(uint4(current.b0[j],current.b1[j],current.b2[j],current.b3[j])<<16);
            sum0+=current.x[j]*w;
        }
        current=next;
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    if(lane==0)for(uint j=0;j<4&&(fixed_no_tail||column+j<ulong(fixed_columns));++j)out[column+j]=sum0[j];
}
)DENSELOOK";
}
