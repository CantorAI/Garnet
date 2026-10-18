// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
inline constexpr const char* kSoleDualProjectionShader=R"SOLEDUAL(
struct SoleDualWords {ushort4 words[4];};
inline SoleDualWords sole_dual_words(device const ushort* weight,ulong packed){
 uint4 pair0=*(device const uint4*)(weight+packed),pair1=*(device const uint4*)(weight+packed+8);
 SoleDualWords value;value.words[0]=as_type<ushort4>(pair0.xy);value.words[1]=as_type<ushort4>(pair0.zw);value.words[2]=as_type<ushort4>(pair1.xy);value.words[3]=as_type<ushort4>(pair1.zw);return value;
}
kernel void sole_dual_specialized_decode(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],device const ushort* upweight [[buffer(4)]],device float* upout [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;
    if(column>=ulong(fixed_columns))return;
    float4 sum0=0.0f,sum1=0.0f;float4 up0=0.0f,up1=0.0f;
    if(ulong(fixed_inner)%4==0) {
        for(ulong k=ulong(lane)*4;k<ulong(fixed_inner);k+=128) {
            float4 x=*(device const float4*)(a+k),y=ulong(fixed_rows)==2?*(device const float4*)(a+ulong(fixed_aStride)+k):float4(0.0f);
            ulong packed=(column/4)*ulong(fixed_inner)*4+k*4;
            SoleDualWords gateWords=sole_dual_words(weight,packed),upWords=sole_dual_words(upweight,packed);
            for(uint j=0;j<4;++j) {
                float4 w=as_type<float4>(uint4(gateWords.words[j])<<16);
                sum0+=x[j]*w;if(ulong(fixed_rows)==2)sum1+=y[j]*w;
                float4 uw=as_type<float4>(uint4(upWords.words[j])<<16);up0+=x[j]*uw;if(ulong(fixed_rows)==2)up1+=y[j]*uw;
            }
        }
    } else {
    for(ulong k=lane;k<ulong(fixed_inner);k+=32) {
        float x=a[k],y=ulong(fixed_rows)==2?a[ulong(fixed_aStride)+k]:0.0f;
        float4 w=0.0f;
        w.x=as_type<float>(uint(weight[column*ulong(fixed_inner)+k])<<16);
        if(column+1<ulong(fixed_columns))w.y=as_type<float>(uint(weight[(column+1)*ulong(fixed_inner)+k])<<16);
        if(column+2<ulong(fixed_columns))w.z=as_type<float>(uint(weight[(column+2)*ulong(fixed_inner)+k])<<16);
        if(column+3<ulong(fixed_columns))w.w=as_type<float>(uint(weight[(column+3)*ulong(fixed_inner)+k])<<16);
        float4 uw=0.0f;
        uw.x=as_type<float>(uint(upweight[column*ulong(fixed_inner)+k])<<16);
        if(column+1<ulong(fixed_columns))uw.y=as_type<float>(uint(upweight[(column+1)*ulong(fixed_inner)+k])<<16);
        if(column+2<ulong(fixed_columns))uw.z=as_type<float>(uint(upweight[(column+2)*ulong(fixed_inner)+k])<<16);
        if(column+3<ulong(fixed_columns))uw.w=as_type<float>(uint(upweight[(column+3)*ulong(fixed_inner)+k])<<16);
        sum0+=x*w;if(ulong(fixed_rows)==2)sum1+=y*w;
        up0+=x*uw;if(ulong(fixed_rows)==2)up1+=y*uw;
    }
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    up0=float4(simd_sum(up0.x),simd_sum(up0.y),simd_sum(up0.z),simd_sum(up0.w));
    up1=float4(simd_sum(up1.x),simd_sum(up1.y),simd_sum(up1.z),simd_sum(up1.w));
    if(lane==0)for(uint j=0;j<4&&column+j<ulong(fixed_columns);++j){out[column+j]=sum0[j];if(ulong(fixed_rows)==2)out[ulong(fixed_oStride)+column+j]=sum1[j];upout[column+j]=up0[j];if(ulong(fixed_rows)==2)upout[ulong(fixed_oStride)+column+j]=up1[j];}
}

kernel void sole_dual_read_pair(device const ushort* first [[buffer(0)]],device const ushort* second [[buffer(1)]],device uint* output [[buffer(2)]],constant ulong2& p [[buffer(3)]],uint id [[thread_position_in_grid]]){
 ulong i=id;if(i>=p.y)return;ulong packed=(p.x+i)*16;SoleDualWords g=sole_dual_words(first,packed),u=sole_dual_words(second,packed);
 for(uint k=0;k<4;++k)for(uint c=0;c<4;++c){output[i*32+k*4+c]=uint(g.words[k][c])<<16;output[i*32+16+k*4+c]=uint(u.words[k][c])<<16;}
}
)SOLEDUAL";
}
