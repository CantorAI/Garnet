#include <metal_stdlib>
using namespace metal;
struct ChunkAttentionParams {ulong batch,chunk,heads,kvHeads,dim,capacity,qStride,kStride,vStride,kcStride,vcStride,sStride,aStride,gStride,oStride,gated;};
kernel void attention_chunk(device const float* query [[buffer(0)]],device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],device const float* keyCache [[buffer(3)]],device const float* valueCache [[buffer(4)]],
    device const float* slots [[buffer(5)]],device const float* active [[buffer(6)]],device const float* gate [[buffer(7)]],
    device float* output [[buffer(8)]],constant ChunkAttentionParams& p [[buffer(9)]],
    threadgroup float* scratch [[threadgroup(0)]],threadgroup int* selected [[threadgroup(1)]],
    uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong flatRow=id/p.heads,head=id%p.heads,row=flatRow/p.chunk,token=flatRow%p.chunk,kv=head/(p.heads/p.kvHeads);
    if(flatRow>=p.batch*p.chunk)return;
    if(active[row*p.aStride+token]==0.0f){for(ulong d=tid;d<p.dim;d+=128)output[flatRow*p.oStride+head*p.dim+d]=0.0f;return;}
    const ulong visible=min(p.capacity,ulong(slots[row*p.sStride+token])+1);
    threadgroup float partial[4];
    for(ulong s=simd;s<visible;s+=4) {
        int latest=-1;for(uint t=0;t<=token;++t)if(active[row*p.aStride+t]!=0.0f&&slots[row*p.sStride+t]==float(s))latest=int(t);
        float score=-1e30f;
        if(float(s)<=slots[row*p.sStride+token]) {
            score=0.0f;
            for(ulong d=lane;d<p.dim;d+=32) {
                float k=latest<0?keyCache[row*p.kcStride+(kv*p.capacity+s)*p.dim+d]:key[(row*p.chunk+ulong(latest))*p.kStride+kv*p.dim+d];
                score+=query[flatRow*p.qStride+head*p.dim+d]*k;
            }
            score=simd_sum(score)*rsqrt(float(p.dim));
        }
        if(lane==0){scratch[s]=score;selected[s]=latest;}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;for(ulong s=tid;s<visible;s+=128)maximum=max(maximum,scratch[s]);
    maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=simd_max(lane<4?partial[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float denominator=0.0f;
    for(ulong s=tid;s<visible;s+=128){float probability=exp(scratch[s]-maximum);scratch[s]=probability;denominator+=probability;}
    denominator=simd_sum(denominator);if(lane==0)partial[simd]=denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    denominator=simd_sum(lane<4?partial[lane]:0.0f);
    for(ulong d=tid;d<p.dim;d+=128) {
        float sum=0.0f;
        for(ulong s=0;s<visible;++s){int latest=selected[s];float v=latest<0?valueCache[row*p.vcStride+(kv*p.capacity+s)*p.dim+d]:value[(row*p.chunk+ulong(latest))*p.vStride+kv*p.dim+d];sum+=scratch[s]/denominator*v;}
        output[flatRow*p.oStride+head*p.dim+d]=p.gated?sum/(1.0f+exp(-gate[flatRow*p.gStride+head*p.dim+d])):sum;
    }
}
kernel void attention_chunk_prefix(device const float* query [[buffer(0)]],device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],device const float* keyCache [[buffer(3)]],device const float* valueCache [[buffer(4)]],
    device const float* slots [[buffer(5)]],device const float* active [[buffer(6)]],device const float* gate [[buffer(7)]],
    device float* output [[buffer(8)]],constant ChunkAttentionParams& p [[buffer(9)]],
    threadgroup float* scratch [[threadgroup(0)]],threadgroup int* selected [[threadgroup(1)]],
    uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong flatRow=id/p.heads,head=id%p.heads,row=flatRow/p.chunk,token=flatRow%p.chunk,kv=head/(p.heads/p.kvHeads);
    if(flatRow>=p.batch*p.chunk)return;
    if(active[row*p.aStride+token]==0.0f){for(ulong d=tid;d<p.dim;d+=128)output[flatRow*p.oStride+head*p.dim+d]=0.0f;return;}
    const ulong visible=min(p.capacity,ulong(slots[row*p.sStride+token])+1);
    threadgroup float partial[4];
    threadgroup int contiguousPrefix;
    if(tid==0) {
        float first=slots[row*p.sStride];
        int valid=isfinite(first)&&first>=0.0f&&first==floor(first);
        for(ulong t=0;t<=token;++t)if(active[row*p.aStride+t]==0.0f||slots[row*p.sStride+t]!=first+float(t))valid=0;
        contiguousPrefix=valid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(ulong s=simd;s<visible;s+=4) {
        int latest=-1;
        if(contiguousPrefix) {
            float first=slots[row*p.sStride];
            if(float(s)>=first)latest=int(float(s)-first);
        } else for(uint t=0;t<=token;++t)if(active[row*p.aStride+t]!=0.0f&&slots[row*p.sStride+t]==float(s))latest=int(t);
        float score=-1e30f;
        if(float(s)<=slots[row*p.sStride+token]) {
            score=0.0f;
            for(ulong d=lane;d<p.dim;d+=32) {
                float k=latest<0?keyCache[row*p.kcStride+(kv*p.capacity+s)*p.dim+d]:key[(row*p.chunk+ulong(latest))*p.kStride+kv*p.dim+d];
                score+=query[flatRow*p.qStride+head*p.dim+d]*k;
            }
            score=simd_sum(score)*rsqrt(float(p.dim));
        }
        if(lane==0){scratch[s]=score;selected[s]=latest;}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;for(ulong s=tid;s<visible;s+=128)maximum=max(maximum,scratch[s]);
    maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=simd_max(lane<4?partial[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float denominator=0.0f;
    for(ulong s=tid;s<visible;s+=128){float probability=exp(scratch[s]-maximum);scratch[s]=probability;denominator+=probability;}
    denominator=simd_sum(denominator);if(lane==0)partial[simd]=denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    denominator=simd_sum(lane<4?partial[lane]:0.0f);
    for(ulong d=tid;d<p.dim;d+=128) {
        float sum=0.0f;
        for(ulong s=0;s<visible;++s){int latest=selected[s];float v=latest<0?valueCache[row*p.vcStride+(kv*p.capacity+s)*p.dim+d]:value[(row*p.chunk+ulong(latest))*p.vStride+kv*p.dim+d];sum+=scratch[s]/denominator*v;}
        output[flatRow*p.oStride+head*p.dim+d]=p.gated?sum/(1.0f+exp(-gate[flatRow*p.gStride+head*p.dim+d])):sum;
    }
}
