// SPDX-License-Identifier: Apache-2.0
// Private eight-output scheduling: two lexical original four-output blocks.
#pragma once
namespace Garnet {
inline constexpr const char* kCoalescedEightBodyShader=R"METAL(
kernel void bf16_body_coalesced_eight(device const float* a [[buffer(0)]],device const ushort* fallback [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],device const uchar* payload [[buffer(4)]],
    device const uint* metadata [[buffer(5)]],uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint threads [[threads_per_threadgroup]]) {
    ulong first=ulong(group)*(threads/32)*8+simd*8;if(first>=p.columns)return;
    float4 sum00=0.0f,sum01=0.0f,sum10=0.0f,sum11=0.0f;
    for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
        float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
        {
            ulong column=first+0;
            if(column<p.columns) {
        ulong block=(column/4)*(p.inner/128)+k/128;uint meta=metadata[block];float4 weights[4];
        if(meta&256u) {
            ulong at=block*768;
            uint4 sm=*(device const uint4*)(payload+at+ulong(lane)*16);
            ushort4 tags=as_type<ushort4>(*(device const uint2*)(payload+at+512+ulong(lane)*8));
            for(uint j=0;j<4;++j) {
                uint4 signMant=uint4(as_type<uchar4>(sm[j]));
                uint4 exponents=uint4(tags[j],tags[j]>>4,tags[j]>>8,tags[j]>>12)&15u;
                uint4 bits=(signMant&127u)|((signMant&128u)<<8)|((exponents+(meta&255u))<<7);
                weights[j]=as_type<float4>(bits<<16);
            }
        } else {
            ulong at=(column/4)*p.inner*4+k*4;
            uint4 p0=*(device const uint4*)(fallback+at),p1=*(device const uint4*)(fallback+at+8);
            ushort4 words[4]={as_type<ushort4>(p0.xy),as_type<ushort4>(p0.zw),as_type<ushort4>(p1.xy),as_type<ushort4>(p1.zw)};
            for(uint j=0;j<4;++j)weights[j]=as_type<float4>(uint4(words[j])<<16);
        }
        for(uint j=0;j<4;++j){sum00+=x[j]*weights[j];if(p.rows==2)sum01+=y[j]*weights[j];}
            }
        }
        {
            ulong column=first+4;
            if(column<p.columns) {
        ulong block=(column/4)*(p.inner/128)+k/128;uint meta=metadata[block];float4 weights[4];
        if(meta&256u) {
            ulong at=block*768;
            uint4 sm=*(device const uint4*)(payload+at+ulong(lane)*16);
            ushort4 tags=as_type<ushort4>(*(device const uint2*)(payload+at+512+ulong(lane)*8));
            for(uint j=0;j<4;++j) {
                uint4 signMant=uint4(as_type<uchar4>(sm[j]));
                uint4 exponents=uint4(tags[j],tags[j]>>4,tags[j]>>8,tags[j]>>12)&15u;
                uint4 bits=(signMant&127u)|((signMant&128u)<<8)|((exponents+(meta&255u))<<7);
                weights[j]=as_type<float4>(bits<<16);
            }
        } else {
            ulong at=(column/4)*p.inner*4+k*4;
            uint4 p0=*(device const uint4*)(fallback+at),p1=*(device const uint4*)(fallback+at+8);
            ushort4 words[4]={as_type<ushort4>(p0.xy),as_type<ushort4>(p0.zw),as_type<ushort4>(p1.xy),as_type<ushort4>(p1.zw)};
            for(uint j=0;j<4;++j)weights[j]=as_type<float4>(uint4(words[j])<<16);
        }
        for(uint j=0;j<4;++j){sum110+=x[j]*weights[j];if(p.rows==2)sum11+=y[j]*weights[j];}
            }
        }
    }
    {
        ulong column=first+0;
        if(column<p.columns) {
    sum00=float4(simd_sum(sum00.x),simd_sum(sum00.y),simd_sum(sum00.z),simd_sum(sum00.w));
    sum01=float4(simd_sum(sum01.x),simd_sum(sum01.y),simd_sum(sum01.z),simd_sum(sum01.w));
    if(lane==0)for(uint j=0;j<4&&column+j<p.columns;++j){out[column+j]=sum00[j];if(p.rows==2)out[p.oStride+column+j]=sum01[j];}
        }
    }
    {
        ulong column=first+4;
        if(column<p.columns) {
    sum110=float4(simd_sum(sum110.x),simd_sum(sum110.y),simd_sum(sum110.z),simd_sum(sum110.w));
    sum11=float4(simd_sum(sum11.x),simd_sum(sum11.y),simd_sum(sum11.z),simd_sum(sum11.w));
    if(lane==0)for(uint j=0;j<4&&column+j<p.columns;++j){out[column+j]=sum110[j];if(p.rows==2)out[p.oStride+column+j]=sum11[j];}
        }
    }
}
)METAL";

}
