// SPDX-License-Identifier: Apache-2.0
// Private two-output compute; original four-output compact payload unchanged.
#pragma once
namespace Garnet {
inline constexpr const char* kCoalescedTwoBodyShader=R"METAL(
kernel void bf16_body_coalesced_two(device const float* a [[buffer(0)]],device const ushort* fallback [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],device const uchar* payload [[buffer(4)]],
    device const uint* metadata [[buffer(5)]],uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint threads [[threads_per_threadgroup]]) {
    ulong column=ulong(group)*(threads/32)*2+simd*2;if(column>=p.columns)return;
    float2 sum0=0.0f,sum1=0.0f;
    for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
        float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
        ulong block=(column/4)*(p.inner/128)+k/128;uint meta=metadata[block];float2 weights[4];
        if(meta&256u) {
            ulong at=block*768;
            uint4 sm=*(device const uint4*)(payload+at+ulong(lane)*16);
            ushort4 tags=as_type<ushort4>(*(device const uint2*)(payload+at+512+ulong(lane)*8));
            for(uint j=0;j<4;++j) {
                uint2 signMant=uint2(as_type<uchar2>(ushort(sm[j]>>uint((column%4)*8))));
                uint shift=uint((column%4)*4);uint2 exponents=uint2(tags[j]>>shift,tags[j]>>(shift+4))&15u;
                uint2 bits=(signMant&127u)|((signMant&128u)<<8)|((exponents+(meta&255u))<<7);
                weights[j]=as_type<float2>(bits<<16);
            }
        } else {
            ulong at=(column/4)*p.inner*4+k*4;
            uint4 p0=*(device const uint4*)(fallback+at),p1=*(device const uint4*)(fallback+at+8);
            ushort4 words[4]={as_type<ushort4>(p0.xy),as_type<ushort4>(p0.zw),as_type<ushort4>(p1.xy),as_type<ushort4>(p1.zw)};
            for(uint j=0;j<4;++j)weights[j]=as_type<float2>(uint2((column%4)?words[j].zw:words[j].xy)<<16);
        }
        for(uint j=0;j<4;++j){sum0=fma(float2(x[j]),weights[j],sum0);if(p.rows==2)sum1=fma(float2(y[j]),weights[j],sum1);}
    }
    sum0=float2(simd_sum(sum0.x),simd_sum(sum0.y));
    sum1=float2(simd_sum(sum1.x),simd_sum(sum1.y));
    if(lane==0)for(uint j=0;j<2&&column+j<p.columns;++j){out[column+j]=sum0[j];if(p.rows==2)out[p.oStride+column+j]=sum1[j];}
}
)METAL";

}
