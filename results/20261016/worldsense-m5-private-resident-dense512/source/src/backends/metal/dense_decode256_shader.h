// Exact per-output SIMD32 arithmetic from body_shape_shader.h; scheduling only.
#pragma once
namespace Garnet { inline constexpr const char* kDenseDecode256Shader=R"DENSE256(
kernel void specialized_bf16_decode_dense256(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*32+simd*4;
    if(!fixed_no_tail&&column>=ulong(fixed_columns))return;
    float4 sum0=0.0f,sum1=0.0f;
    if(ulong(fixed_inner)%4==0) {
        for(ulong k=ulong(lane)*4;k<ulong(fixed_inner);k+=128) {
            float4 x=*(device const float4*)(a+k),y=ulong(fixed_rows)==2?*(device const float4*)(a+ulong(fixed_aStride)+k):float4(0.0f);
            ushort4 b0=*(device const ushort4*)(weight+column*ulong(fixed_inner)+k),b1=ushort4(0),b2=ushort4(0),b3=ushort4(0);
            if(fixed_no_tail||column+1<ulong(fixed_columns))b1=*(device const ushort4*)(weight+(column+1)*ulong(fixed_inner)+k);
            if(fixed_no_tail||column+2<ulong(fixed_columns))b2=*(device const ushort4*)(weight+(column+2)*ulong(fixed_inner)+k);
            if(fixed_no_tail||column+3<ulong(fixed_columns))b3=*(device const ushort4*)(weight+(column+3)*ulong(fixed_inner)+k);
            for(uint j=0;j<4;++j) {
                float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);
                sum0+=x[j]*w;if(ulong(fixed_rows)==2)sum1+=y[j]*w;
            }
        }
    } else {
    for(ulong k=lane;k<ulong(fixed_inner);k+=32) {
        float x=a[k],y=ulong(fixed_rows)==2?a[ulong(fixed_aStride)+k]:0.0f;
        float4 w=0.0f;
        w.x=as_type<float>(uint(weight[column*ulong(fixed_inner)+k])<<16);
        if(fixed_no_tail||column+1<ulong(fixed_columns))w.y=as_type<float>(uint(weight[(column+1)*ulong(fixed_inner)+k])<<16);
        if(fixed_no_tail||column+2<ulong(fixed_columns))w.z=as_type<float>(uint(weight[(column+2)*ulong(fixed_inner)+k])<<16);
        if(fixed_no_tail||column+3<ulong(fixed_columns))w.w=as_type<float>(uint(weight[(column+3)*ulong(fixed_inner)+k])<<16);
        sum0+=x*w;if(ulong(fixed_rows)==2)sum1+=y*w;
    }
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    if(lane==0)for(uint j=0;j<4&&(fixed_no_tail||column+j<ulong(fixed_columns));++j){out[column+j]=sum0[j];if(ulong(fixed_rows)==2)out[ulong(fixed_oStride)+column+j]=sum1[j];}
}
)DENSE256"; }
