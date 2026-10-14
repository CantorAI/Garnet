#include <metal_stdlib>
using namespace metal;
struct Params{ulong rows,columns,aStride,bStride,cStride;float epsilon,offset;};
kernel void add_matrix(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],constant Params& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.rows*p.columns)return;
    ulong row=id/p.columns,col=id%p.columns;
    c[row*p.cStride+col]=a[row*p.aStride+col]+b[(p.offset!=0?0:row)*p.bStride+col];
}
kernel void rms_norm(device const float* a [[buffer(0)]],device const float* weight [[buffer(1)]],
    device float* c [[buffer(2)]],constant Params& p [[buffer(3)]],uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=p.rows)return;
    threadgroup float partial[4];
    float sum=0;
    for(ulong j=tid;j<p.columns;j+=128){float x=a[row*p.aStride+j];sum+=x*x;}
    sum=simd_sum(sum);
    if(lane==0)partial[simd]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=simd_sum(lane<4?partial[lane]:0.0f);
    float inverse=rsqrt(sum/float(p.columns)+p.epsilon);
    for(ulong j=tid;j<p.columns;j+=128)c[row*p.cStride+j]=a[row*p.aStride+j]*inverse*(weight[j]+p.offset);
}
kernel void residual_norm(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],device const float* weight [[buffer(2)]],
    device float* residual [[buffer(3)]],device float* c [[buffer(4)]],constant Params& p [[buffer(5)]],uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=p.rows)return;
    threadgroup float partial[4];
    float sum=0;
    for(ulong j=tid;j<p.columns;j+=128){float x=a[row*p.aStride+j]+b[row*p.cStride+j];residual[row*p.cStride+j]=x;sum+=x*x;}
    sum=simd_sum(sum);
    if(lane==0)partial[simd]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
    sum=simd_sum(lane<4?partial[lane]:0.0f);
    float inverse=rsqrt(sum/float(p.columns)+p.epsilon);
    for(ulong j=tid;j<p.columns;j+=128)c[row*p.cStride+j]=residual[row*p.cStride+j]*inverse*(weight[j]+p.offset);
}
