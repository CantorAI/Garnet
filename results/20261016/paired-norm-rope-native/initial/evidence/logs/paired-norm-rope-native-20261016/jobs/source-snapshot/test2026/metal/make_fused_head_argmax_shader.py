"""Retain native dot-product arithmetic; replace only its output epilogue."""
from pathlib import Path
import sys
text=Path(sys.argv[1]).read_text()
start=text.index('kernel void bf16_decode(');end=text.index('kernel void bf16_decode_small(',start)
native=text[start:end];candidate=native.replace('kernel void bf16_decode(', 'kernel void fused_head(')
candidate=candidate.replace('uint group [[threadgroup_position_in_grid]]', 'device float* debug [[buffer(4)]],constant uint& debugEnabled [[buffer(5)]],uint group [[threadgroup_position_in_grid]]')
candidate=candidate.replace('    if(column>=p.columns)return;', '')
candidate=candidate.replace('ushort4 b0=*(device const ushort4*)(weight+column*p.inner+k)', 'ushort4 b0=column<p.columns?*(device const ushort4*)(weight+column*p.inner+k):ushort4(0)')
candidate=candidate.replace('w.x=as_type<float>(uint(weight[column*p.inner+k])<<16);', 'if(column<p.columns)w.x=as_type<float>(uint(weight[column*p.inner+k])<<16);')
epilogue='    if(lane==0)for(uint j=0;j<4&&column+j<p.columns;++j){out[column+j]=sum0[j];if(p.rows==2)out[p.oStride+column+j]=sum1[j];}'
assert candidate.count(epilogue)==1
candidate=candidate.replace(epilogue,'''    threadgroup float maxima[8];threadgroup uint indices[8],invalid[8];
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
        if(lane==0){ulong offset=(row*((p.columns+15)/16)+group)*3;out[offset]=groupBest;out[offset+1]=float(groupIndex);out[offset+2]=float(groupBad);}
    }''')
argmax=text[text.index('struct ArgmaxParams'):text.index('struct Params { ulong rows')]
reducer='''
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
'''
Path(sys.argv[2]).write_bytes(('#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\n'+native+candidate+argmax+reducer).encode())
