#include <metal_stdlib>
using namespace metal;
struct VisionParams {ulong rows,columns,outRows,outColumns,aStride,bStride,cStride,oStride,tokens,hidden,heads,unit,mode;};
kernel void vision_softmax_context(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],device const float* c [[buffer(2)]],
    device float* out [[buffer(3)]],constant VisionParams& p [[buffer(4)]],threadgroup float* scratch [[threadgroup(0)]],
    uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong row=id/p.heads,head=id%p.heads,dim=p.hidden/p.heads,batchBase=(row/p.tokens)*p.tokens;
    if(row>=p.outRows)return;threadgroup float partial[4];
    ulong packedRow=((row/p.tokens)*p.heads+head)*p.tokens+row%p.tokens;
    for(ulong key=tid;key<p.tokens;key+=128)scratch[key]=b[packedRow*p.bStride+key]*rsqrt(float(dim));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;for(ulong key=tid;key<p.tokens;key+=128)maximum=max(maximum,scratch[key]);
    maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=simd_max(lane<4?partial[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=0;for(ulong key=tid;key<p.tokens;key+=128){float v=exp(scratch[key]-maximum);scratch[key]=v;total+=v;}
    total=simd_sum(total);if(lane==0)partial[simd]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total=simd_sum(lane<4?partial[lane]:0.0f);
    for(ulong key=tid;key<p.tokens;key+=128)scratch[key]/=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(ulong d=tid;d<dim;d+=128) {
        float result=0;
        for(ulong key=0;key<p.tokens;++key)result+=scratch[key]*a[(batchBase+key)*p.aStride+2*p.hidden+head*dim+d];
        out[row*p.oStride+head*dim+d]=result;
    }
}
kernel void vision_parallel_context(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],device const float* c [[buffer(2)]],
    device float* out [[buffer(3)]],constant VisionParams& p [[buffer(4)]],threadgroup float* scratch [[threadgroup(0)]],
    uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong row=id/p.heads,head=id%p.heads,dim=p.hidden/p.heads,batchBase=(row/p.tokens)*p.tokens;
    if(row>=p.outRows)return;threadgroup float partial[4];
    ulong packedRow=((row/p.tokens)*p.heads+head)*p.tokens+row%p.tokens;
    for(ulong key=tid;key<p.tokens;key+=128)scratch[key]=b[packedRow*p.bStride+key]*rsqrt(float(dim));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;for(ulong key=tid;key<p.tokens;key+=128)maximum=max(maximum,scratch[key]);
    maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=simd_max(lane<4?partial[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=0;for(ulong key=tid;key<p.tokens;key+=128){float v=exp(scratch[key]-maximum);scratch[key]=v;total+=v;}
    total=simd_sum(total);if(lane==0)partial[simd]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total=simd_sum(lane<4?partial[lane]:0.0f);
    for(ulong key=tid;key<p.tokens;key+=128)scratch[key]/=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float valuePartials[4*128];
    for(ulong d=lane;d<dim;d+=32) {
        float result=0;
        for(ulong key=simd;key<p.tokens;key+=4)result+=scratch[key]*a[(batchBase+key)*p.aStride+2*p.hidden+head*dim+d];
        valuePartials[simd*128+d]=result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(ulong d=tid;d<dim;d+=128) {
        float result=valuePartials[d]+valuePartials[128+d]+valuePartials[256+d]+valuePartials[384+d];
        out[row*p.oStride+head*dim+d]=result;
    }
}
