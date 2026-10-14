#include <metal_stdlib>
using namespace metal;
struct Params{ulong rows,columns,aStride,bStride,cStride;float epsilon,offset;};
struct LayoutParams{ulong aRows,aColumns,aStride,bStride,oRows,oColumns,oStride,tokenWidth;};
struct PackedParams{ulong rows,columns,packedStride,normStride,chunk;float epsilon,offset;};
kernel void add_matrix(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],constant Params& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.rows*p.columns)return;
    ulong row=id/p.columns,col=id%p.columns;
    c[row*p.cStride+col]=a[row*p.aStride+col]+b[(p.offset!=0?0:row)*p.bStride+col];
}
kernel void chunk_layout(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
 device float* out [[buffer(2)]],constant LayoutParams& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
 if(id>=p.oRows*p.oColumns)return;
 ulong sourceRow=id/p.aColumns,sourceColumn=id%p.aColumns,row=id/p.oColumns,col=id%p.oColumns;
 float value=a[sourceRow*p.aStride+sourceColumn];
 if(p.tokenWidth&&b[sourceRow*p.bStride+sourceColumn/p.tokenWidth]==0.0f)value=0;
 out[row*p.oStride+col]=value;
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
kernel void packed_residual_norm(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
 device const float* weight [[buffer(2)]],device float* packed [[buffer(3)]],device float* layout [[buffer(4)]],
 device float* out [[buffer(5)]],constant PackedParams& p [[buffer(6)]],uint row [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 if(row>=p.rows)return;
 ulong base=(row/p.chunk)*p.packedStride+(row%p.chunk)*p.columns;
 threadgroup float partial[4];float sum=0;
 for(ulong j=tid;j<p.columns;j+=128){float x=a[base+j]+b[base+j];packed[base+j]=x;layout[row*p.normStride+j]=x;sum+=x*x;}
 sum=simd_sum(sum);if(lane==0)partial[simd]=sum;
 threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
 sum=simd_sum(lane<4?partial[lane]:0.0f);
 float inverse=rsqrt(sum/float(p.columns)+p.epsilon);
 for(ulong j=tid;j<p.columns;j+=128)out[row*p.normStride+j]=layout[row*p.normStride+j]*inverse*(weight[j]+p.offset);
}
