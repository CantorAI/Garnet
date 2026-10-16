#include <metal_stdlib>
using namespace metal;
struct DeltaChunkParams {ulong batch,chunk,heads,keyDim,valueDim,pStride,sStride,aStride,oStride,nextStride;};
kernel void gated_delta_chunk_value16(device const float* prepared [[buffer(0)]],device const float* state [[buffer(1)]],
    device const float* active [[buffer(2)]],device float* output [[buffer(3)]],device float* nextState [[buffer(4)]],
    constant DeltaChunkParams& p [[buffer(5)]],uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong tiles=p.valueDim/16,headId=ulong(id)/tiles;
    if(headId>=p.batch*p.heads)return;
    ulong head=headId%p.heads,row=headId/p.heads,column=(ulong(id)%tiles)*16+simd*4;
    ulong stateBase=row*p.sStride+head*p.keyDim*p.valueDim,nextBase=row*p.nextStride+head*p.keyDim*p.valueDim;
    float4 memories[4];
    for(uint j=0;j<4;++j)memories[j]=*(device const float4*)(state+stateBase+(ulong(lane)+32*j)*p.valueDim+column);
    threadgroup float qPartial[4],kPartial[4];
    ulong qw=p.heads*p.keyDim,vw=p.heads*p.valueDim;
    for(ulong token=0;token<p.chunk;++token) {
        ulong base=(row*p.chunk+token)*p.pStride,outputBase=(row*p.chunk+token)*p.oStride+head*p.valueDim;
        if(active[row*p.aStride+token]!=0.0f) {
            float q=prepared[base+head*p.keyDim+tid],k=prepared[base+qw+head*p.keyDim+tid];
            float qs=simd_sum(q*q),ks=simd_sum(k*k);
            if(lane==0){qPartial[simd]=qs;kPartial[simd]=ks;}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            qs=simd_sum(lane<4?qPartial[lane]:0.0f)+1e-6f;ks=simd_sum(lane<4?kPartial[lane]:0.0f)+1e-6f;
            float qi=rsqrt(qs)*rsqrt(float(p.keyDim)),ki=rsqrt(ks),factor=exp(prepared[base+2*qw+vw+head]);
            float4 memory=0.0f;
            for(uint j=0;j<4;++j)memory+=memories[j]*factor*prepared[base+qw+head*p.keyDim+ulong(lane)+32*j]*ki;
            memory=float4(simd_sum(memory.x),simd_sum(memory.y),simd_sum(memory.z),simd_sum(memory.w));
            float4 change=(*(device const float4*)(prepared+base+2*qw+head*p.valueDim+column)-memory)*prepared[base+2*qw+vw+p.heads+head];
            float4 result=0.0f;
            for(uint j=0;j<4;++j) {
                ulong keyIndex=ulong(lane)+32*j;
                memories[j]=memories[j]*factor+prepared[base+qw+head*p.keyDim+keyIndex]*ki*change;
                result+=memories[j]*prepared[base+head*p.keyDim+keyIndex]*qi;
            }
            result=float4(simd_sum(result.x),simd_sum(result.y),simd_sum(result.z),simd_sum(result.w));
            if(lane==0)*(device float4*)(output+outputBase+column)=result;
        }else if(lane==0)*(device float4*)(output+outputBase+column)=float4(0.0f);
        // Every SIMD group must finish reading the partial norms before any
        // group can replace them for the next token.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint j=0;j<4;++j)*(device float4*)(nextState+nextBase+(ulong(lane)+32*j)*p.valueDim+column)=memories[j];
}
kernel void gated_delta_chunk_value32(device const float* prepared [[buffer(0)]],device const float* state [[buffer(1)]],
    device const float* active [[buffer(2)]],device float* output [[buffer(3)]],device float* nextState [[buffer(4)]],
    constant DeltaChunkParams& p [[buffer(5)]],uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong tiles=p.valueDim/32,headId=ulong(id)/tiles;
    if(headId>=p.batch*p.heads)return;
    ulong head=headId%p.heads,row=headId/p.heads,column=(ulong(id)%tiles)*32+simd*4;
    ulong stateBase=row*p.sStride+head*p.keyDim*p.valueDim,nextBase=row*p.nextStride+head*p.keyDim*p.valueDim;
    float4 memories_b0[4];
    float4 memories_b1[4];
    for(uint j=0;j<4;++j)memories_b0[j]=*(device const float4*)(state+stateBase+(ulong(lane)+32*j)*p.valueDim+(column+0));
    for(uint j=0;j<4;++j)memories_b1[j]=*(device const float4*)(state+stateBase+(ulong(lane)+32*j)*p.valueDim+(column+16));
    threadgroup float qPartial[4],kPartial[4];
    ulong qw=p.heads*p.keyDim,vw=p.heads*p.valueDim;
    for(ulong token=0;token<p.chunk;++token) {
        ulong base=(row*p.chunk+token)*p.pStride,outputBase=(row*p.chunk+token)*p.oStride+head*p.valueDim;
        if(active[row*p.aStride+token]!=0.0f) {
            float q=prepared[base+head*p.keyDim+tid],k=prepared[base+qw+head*p.keyDim+tid];
            float qs=simd_sum(q*q),ks=simd_sum(k*k);
            if(lane==0){qPartial[simd]=qs;kPartial[simd]=ks;}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            qs=simd_sum(lane<4?qPartial[lane]:0.0f)+1e-6f;ks=simd_sum(lane<4?kPartial[lane]:0.0f)+1e-6f;
            float qi=rsqrt(qs)*rsqrt(float(p.keyDim)),ki=rsqrt(ks),factor=exp(prepared[base+2*qw+vw+head]);
            float4 memory_b0=0.0f;
            for(uint j=0;j<4;++j)memory_b0+=memories_b0[j]*factor*prepared[base+qw+head*p.keyDim+ulong(lane)+32*j]*ki;
            memory_b0=float4(simd_sum(memory_b0.x),simd_sum(memory_b0.y),simd_sum(memory_b0.z),simd_sum(memory_b0.w));
            float4 change_b0=(*(device const float4*)(prepared+base+2*qw+head*p.valueDim+(column+0))-memory_b0)*prepared[base+2*qw+vw+p.heads+head];
            float4 result_b0=0.0f;
            for(uint j=0;j<4;++j) {
                ulong keyIndex=ulong(lane)+32*j;
                memories_b0[j]=memories_b0[j]*factor+prepared[base+qw+head*p.keyDim+keyIndex]*ki*change_b0;
                result_b0+=memories_b0[j]*prepared[base+head*p.keyDim+keyIndex]*qi;
            }
            result_b0=float4(simd_sum(result_b0.x),simd_sum(result_b0.y),simd_sum(result_b0.z),simd_sum(result_b0.w));
            if(lane==0)*(device float4*)(output+outputBase+(column+0))=result_b0;
            float4 memory_b1=0.0f;
            for(uint j=0;j<4;++j)memory_b1+=memories_b1[j]*factor*prepared[base+qw+head*p.keyDim+ulong(lane)+32*j]*ki;
            memory_b1=float4(simd_sum(memory_b1.x),simd_sum(memory_b1.y),simd_sum(memory_b1.z),simd_sum(memory_b1.w));
            float4 change_b1=(*(device const float4*)(prepared+base+2*qw+head*p.valueDim+(column+16))-memory_b1)*prepared[base+2*qw+vw+p.heads+head];
            float4 result_b1=0.0f;
            for(uint j=0;j<4;++j) {
                ulong keyIndex=ulong(lane)+32*j;
                memories_b1[j]=memories_b1[j]*factor+prepared[base+qw+head*p.keyDim+keyIndex]*ki*change_b1;
                result_b1+=memories_b1[j]*prepared[base+head*p.keyDim+keyIndex]*qi;
            }
            result_b1=float4(simd_sum(result_b1.x),simd_sum(result_b1.y),simd_sum(result_b1.z),simd_sum(result_b1.w));
            if(lane==0)*(device float4*)(output+outputBase+(column+16))=result_b1;
        }else if(lane==0){
            *(device float4*)(output+outputBase+(column+0))=float4(0.0f);
            *(device float4*)(output+outputBase+(column+16))=float4(0.0f);
        }
        // Every SIMD group must finish reading the partial norms before any
        // group can replace them for the next token.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint j=0;j<4;++j)*(device float4*)(nextState+nextBase+(ulong(lane)+32*j)*p.valueDim+(column+0))=memories_b0[j];
    for(uint j=0;j<4;++j)*(device float4*)(nextState+nextBase+(ulong(lane)+32*j)*p.valueDim+(column+16))=memories_b1[j];
}
kernel void gated_delta_chunk_value64(device const float* prepared [[buffer(0)]],device const float* state [[buffer(1)]],
    device const float* active [[buffer(2)]],device float* output [[buffer(3)]],device float* nextState [[buffer(4)]],
    constant DeltaChunkParams& p [[buffer(5)]],uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong tiles=p.valueDim/64,headId=ulong(id)/tiles;
    if(headId>=p.batch*p.heads)return;
    ulong head=headId%p.heads,row=headId/p.heads,column=(ulong(id)%tiles)*64+simd*4;
    ulong stateBase=row*p.sStride+head*p.keyDim*p.valueDim,nextBase=row*p.nextStride+head*p.keyDim*p.valueDim;
    float4 memories_b0[4];
    float4 memories_b1[4];
    float4 memories_b2[4];
    float4 memories_b3[4];
    for(uint j=0;j<4;++j)memories_b0[j]=*(device const float4*)(state+stateBase+(ulong(lane)+32*j)*p.valueDim+(column+0));
    for(uint j=0;j<4;++j)memories_b1[j]=*(device const float4*)(state+stateBase+(ulong(lane)+32*j)*p.valueDim+(column+16));
    for(uint j=0;j<4;++j)memories_b2[j]=*(device const float4*)(state+stateBase+(ulong(lane)+32*j)*p.valueDim+(column+32));
    for(uint j=0;j<4;++j)memories_b3[j]=*(device const float4*)(state+stateBase+(ulong(lane)+32*j)*p.valueDim+(column+48));
    threadgroup float qPartial[4],kPartial[4];
    ulong qw=p.heads*p.keyDim,vw=p.heads*p.valueDim;
    for(ulong token=0;token<p.chunk;++token) {
        ulong base=(row*p.chunk+token)*p.pStride,outputBase=(row*p.chunk+token)*p.oStride+head*p.valueDim;
        if(active[row*p.aStride+token]!=0.0f) {
            float q=prepared[base+head*p.keyDim+tid],k=prepared[base+qw+head*p.keyDim+tid];
            float qs=simd_sum(q*q),ks=simd_sum(k*k);
            if(lane==0){qPartial[simd]=qs;kPartial[simd]=ks;}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            qs=simd_sum(lane<4?qPartial[lane]:0.0f)+1e-6f;ks=simd_sum(lane<4?kPartial[lane]:0.0f)+1e-6f;
            float qi=rsqrt(qs)*rsqrt(float(p.keyDim)),ki=rsqrt(ks),factor=exp(prepared[base+2*qw+vw+head]);
            float4 memory_b0=0.0f;
            for(uint j=0;j<4;++j)memory_b0+=memories_b0[j]*factor*prepared[base+qw+head*p.keyDim+ulong(lane)+32*j]*ki;
            memory_b0=float4(simd_sum(memory_b0.x),simd_sum(memory_b0.y),simd_sum(memory_b0.z),simd_sum(memory_b0.w));
            float4 change_b0=(*(device const float4*)(prepared+base+2*qw+head*p.valueDim+(column+0))-memory_b0)*prepared[base+2*qw+vw+p.heads+head];
            float4 result_b0=0.0f;
            for(uint j=0;j<4;++j) {
                ulong keyIndex=ulong(lane)+32*j;
                memories_b0[j]=memories_b0[j]*factor+prepared[base+qw+head*p.keyDim+keyIndex]*ki*change_b0;
                result_b0+=memories_b0[j]*prepared[base+head*p.keyDim+keyIndex]*qi;
            }
            result_b0=float4(simd_sum(result_b0.x),simd_sum(result_b0.y),simd_sum(result_b0.z),simd_sum(result_b0.w));
            if(lane==0)*(device float4*)(output+outputBase+(column+0))=result_b0;
            float4 memory_b1=0.0f;
            for(uint j=0;j<4;++j)memory_b1+=memories_b1[j]*factor*prepared[base+qw+head*p.keyDim+ulong(lane)+32*j]*ki;
            memory_b1=float4(simd_sum(memory_b1.x),simd_sum(memory_b1.y),simd_sum(memory_b1.z),simd_sum(memory_b1.w));
            float4 change_b1=(*(device const float4*)(prepared+base+2*qw+head*p.valueDim+(column+16))-memory_b1)*prepared[base+2*qw+vw+p.heads+head];
            float4 result_b1=0.0f;
            for(uint j=0;j<4;++j) {
                ulong keyIndex=ulong(lane)+32*j;
                memories_b1[j]=memories_b1[j]*factor+prepared[base+qw+head*p.keyDim+keyIndex]*ki*change_b1;
                result_b1+=memories_b1[j]*prepared[base+head*p.keyDim+keyIndex]*qi;
            }
            result_b1=float4(simd_sum(result_b1.x),simd_sum(result_b1.y),simd_sum(result_b1.z),simd_sum(result_b1.w));
            if(lane==0)*(device float4*)(output+outputBase+(column+16))=result_b1;
            float4 memory_b2=0.0f;
            for(uint j=0;j<4;++j)memory_b2+=memories_b2[j]*factor*prepared[base+qw+head*p.keyDim+ulong(lane)+32*j]*ki;
            memory_b2=float4(simd_sum(memory_b2.x),simd_sum(memory_b2.y),simd_sum(memory_b2.z),simd_sum(memory_b2.w));
            float4 change_b2=(*(device const float4*)(prepared+base+2*qw+head*p.valueDim+(column+32))-memory_b2)*prepared[base+2*qw+vw+p.heads+head];
            float4 result_b2=0.0f;
            for(uint j=0;j<4;++j) {
                ulong keyIndex=ulong(lane)+32*j;
                memories_b2[j]=memories_b2[j]*factor+prepared[base+qw+head*p.keyDim+keyIndex]*ki*change_b2;
                result_b2+=memories_b2[j]*prepared[base+head*p.keyDim+keyIndex]*qi;
            }
            result_b2=float4(simd_sum(result_b2.x),simd_sum(result_b2.y),simd_sum(result_b2.z),simd_sum(result_b2.w));
            if(lane==0)*(device float4*)(output+outputBase+(column+32))=result_b2;
            float4 memory_b3=0.0f;
            for(uint j=0;j<4;++j)memory_b3+=memories_b3[j]*factor*prepared[base+qw+head*p.keyDim+ulong(lane)+32*j]*ki;
            memory_b3=float4(simd_sum(memory_b3.x),simd_sum(memory_b3.y),simd_sum(memory_b3.z),simd_sum(memory_b3.w));
            float4 change_b3=(*(device const float4*)(prepared+base+2*qw+head*p.valueDim+(column+48))-memory_b3)*prepared[base+2*qw+vw+p.heads+head];
            float4 result_b3=0.0f;
            for(uint j=0;j<4;++j) {
                ulong keyIndex=ulong(lane)+32*j;
                memories_b3[j]=memories_b3[j]*factor+prepared[base+qw+head*p.keyDim+keyIndex]*ki*change_b3;
                result_b3+=memories_b3[j]*prepared[base+head*p.keyDim+keyIndex]*qi;
            }
            result_b3=float4(simd_sum(result_b3.x),simd_sum(result_b3.y),simd_sum(result_b3.z),simd_sum(result_b3.w));
            if(lane==0)*(device float4*)(output+outputBase+(column+48))=result_b3;
        }else if(lane==0){
            *(device float4*)(output+outputBase+(column+0))=float4(0.0f);
            *(device float4*)(output+outputBase+(column+16))=float4(0.0f);
            *(device float4*)(output+outputBase+(column+32))=float4(0.0f);
            *(device float4*)(output+outputBase+(column+48))=float4(0.0f);
        }
        // Every SIMD group must finish reading the partial norms before any
        // group can replace them for the next token.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint j=0;j<4;++j)*(device float4*)(nextState+nextBase+(ulong(lane)+32*j)*p.valueDim+(column+0))=memories_b0[j];
    for(uint j=0;j<4;++j)*(device float4*)(nextState+nextBase+(ulong(lane)+32*j)*p.valueDim+(column+16))=memories_b1[j];
    for(uint j=0;j<4;++j)*(device float4*)(nextState+nextBase+(ulong(lane)+32*j)*p.valueDim+(column+32))=memories_b2[j];
    for(uint j=0;j<4;++j)*(device float4*)(nextState+nextBase+(ulong(lane)+32*j)*p.valueDim+(column+48))=memories_b3[j];
}