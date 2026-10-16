// Private eight-output BF16 body experiment. Original per-output arithmetic order.
#pragma once
inline constexpr const char* kWideBF16BodyShader=R"METAL(
kernel void bf16_body_wide(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint threads [[threads_per_threadgroup]]) {
    ulong column=ulong(group)*(threads/32)*8+simd*8;
    if(column>=p.columns)return;
    float4 lo0=0.0f,hi0=0.0f,lo1=0.0f,hi1=0.0f;
    for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
        float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
        ulong packed=(column/8)*p.inner*8+k*8;
        uint4 words[4]={*(device const uint4*)(weight+packed),*(device const uint4*)(weight+packed+8),
            *(device const uint4*)(weight+packed+16),*(device const uint4*)(weight+packed+24)};
        for(uint j=0;j<4;++j) {
            float4 low=as_type<float4>(uint4(as_type<ushort4>(words[j].xy))<<16);
            float4 high=as_type<float4>(uint4(as_type<ushort4>(words[j].zw))<<16);
            lo0+=x[j]*low;hi0+=x[j]*high;
            if(p.rows==2){lo1+=y[j]*low;hi1+=y[j]*high;}
        }
    }
    lo0=float4(simd_sum(lo0.x),simd_sum(lo0.y),simd_sum(lo0.z),simd_sum(lo0.w));
    hi0=float4(simd_sum(hi0.x),simd_sum(hi0.y),simd_sum(hi0.z),simd_sum(hi0.w));
    lo1=float4(simd_sum(lo1.x),simd_sum(lo1.y),simd_sum(lo1.z),simd_sum(lo1.w));
    hi1=float4(simd_sum(hi1.x),simd_sum(hi1.y),simd_sum(hi1.z),simd_sum(hi1.w));
    if(lane==0)for(uint j=0;j<8&&column+j<p.columns;++j){out[column+j]=j<4?lo0[j]:hi0[j-4];if(p.rows==2)out[p.oStride+column+j]=j<4?lo1[j]:hi1[j-4];}
}
)METAL";
