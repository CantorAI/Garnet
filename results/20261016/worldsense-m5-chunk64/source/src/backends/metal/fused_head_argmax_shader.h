// SPDX-License-Identifier: Apache-2.0
#pragma once
static constexpr char kFusedHeadArgmaxShader[] = R"FUSED(
#include <metal_stdlib>
using namespace metal;
struct FusedHeadParams {ulong rows,inner,columns,aStride,oStride,partialStride;};
kernel void fused_head(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant FusedHeadParams& p [[buffer(3)]],
    device float* debug [[buffer(4)]],constant uint& debugEnabled [[buffer(5)]],uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;

    float4 sum0=0.0f,sum1=0.0f;
    if(p.inner%4==0) {
        for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
            float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
            ushort4 b0=column<p.columns?*(device const ushort4*)(weight+column*p.inner+k):ushort4(0),b1=ushort4(0),b2=ushort4(0),b3=ushort4(0);
            if(column+1<p.columns)b1=*(device const ushort4*)(weight+(column+1)*p.inner+k);
            if(column+2<p.columns)b2=*(device const ushort4*)(weight+(column+2)*p.inner+k);
            if(column+3<p.columns)b3=*(device const ushort4*)(weight+(column+3)*p.inner+k);
            for(uint j=0;j<4;++j) {
                float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);
                sum0+=x[j]*w;if(p.rows==2)sum1+=y[j]*w;
            }
        }
    } else {
    for(ulong k=lane;k<p.inner;k+=32) {
        float x=a[k],y=p.rows==2?a[p.aStride+k]:0.0f;
        float4 w=0.0f;
        if(column<p.columns)w.x=as_type<float>(uint(weight[column*p.inner+k])<<16);
        if(column+1<p.columns)w.y=as_type<float>(uint(weight[(column+1)*p.inner+k])<<16);
        if(column+2<p.columns)w.z=as_type<float>(uint(weight[(column+2)*p.inner+k])<<16);
        if(column+3<p.columns)w.w=as_type<float>(uint(weight[(column+3)*p.inner+k])<<16);
        sum0+=x*w;if(p.rows==2)sum1+=y*w;
    }
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    threadgroup float maxima[8];threadgroup uint indices[8],invalid[8];
    if(lane==0)for(uint row=0;row<p.rows;++row){
        float best=-INFINITY;uint index=0xffffffff,bad=0;
        for(uint j=0;j<4&&column+j<p.columns;++j){
            float value=row==0?sum0[j]:sum1[j];bad|=!isfinite(value);
            if(debugEnabled)debug[row*p.oStride+column+j]=value;
            if(value>best||(value==best&&column+j<index)){best=value;index=uint(column+j);}
        }
        maxima[row*4+simd]=best;indices[row*4+simd]=index;invalid[row*4+simd]=bad;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(simd==0)for(uint row=0;row<p.rows;++row){
        float best=lane<4?maxima[row*4+lane]:-INFINITY;
        uint index=lane<4?indices[row*4+lane]:0xffffffff,bad=lane<4?invalid[row*4+lane]:0;
        float groupBest=simd_max(best);uint groupIndex=simd_min(best==groupBest?index:0xffffffff),groupBad=simd_max(bad);
        if(lane==0){ulong offset=row*p.partialStride+group*3;out[offset]=groupBest;out[offset+1]=float(groupIndex);out[offset+2]=float(groupBad);}
    }
}
struct ArgmaxParams {ulong rows,columns,iStride,oStride;};
kernel void fused_head_reduce(device const float* input [[buffer(0)]],device float* output [[buffer(1)]],
 constant ArgmaxParams& p [[buffer(2)]],uint row [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 threadgroup float maximum[4];threadgroup uint indices[4],invalid[4];
 float best=-INFINITY;uint index=0xffffffff,bad=0;
 for(ulong j=tid;j<p.columns;j+=128){ulong offset=row*p.iStride+j*3;float value=input[offset];uint original=uint(input[offset+1]);bad|=uint(input[offset+2]);
  if(value>best||(value==best&&original<index)){best=value;index=original;}}
 float groupBest=simd_max(best);uint groupIndex=simd_min(best==groupBest?index:0xffffffff),groupBad=simd_max(bad);
 if(lane==0){maximum[simd]=groupBest;indices[simd]=groupIndex;invalid[simd]=groupBad;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 best=lane<4?maximum[lane]:-INFINITY;index=lane<4?indices[lane]:0xffffffff;bad=lane<4?invalid[lane]:0;
 groupBest=simd_max(best);groupIndex=simd_min(best==groupBest?index:0xffffffff);groupBad=simd_max(bad);
 if(tid==0){output[row*p.oStride]=float(groupIndex);output[row*p.oStride+1]=float(groupBad!=0);}
}
)FUSED";
