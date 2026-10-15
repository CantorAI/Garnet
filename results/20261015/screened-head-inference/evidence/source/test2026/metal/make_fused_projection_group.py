"""Preserve each original dot product; combine independent projections in one grid."""
import re
from pathlib import Path

def generate(native,original):
    shader='#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\n'+native[native.index('kernel void bf16_decode('):native.index('kernel void bf16_gemm(')]
    functions=shader[shader.index('kernel void bf16_decode('):]
    functions=re.sub(r'\s*\[\[[^\]]+\]\]', '', functions)
    functions=functions.replace('kernel void bf16_decode(', 'inline void grouped_decode(').replace('kernel void bf16_decode_small(', 'inline void grouped_small(')
    shader+=functions+'''
kernel void fused_projection_group(device const float* a [[buffer(0)]],
    device const ushort* w0 [[buffer(1)]],device float* o0 [[buffer(2)]],
    device const ushort* w1 [[buffer(3)]],device float* o1 [[buffer(4)]],
    device const ushort* w2 [[buffer(5)]],device float* o2 [[buffer(6)]],
    constant BF16Params* params [[buffer(7)]],constant uint* cuts [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    uint part=group<cuts[1]?0:(group<cuts[2]?1:2),local=group-cuts[part];
    device const ushort* w=part==0?w0:(part==1?w1:w2);
    device float* o=part==0?o0:(part==1?o1:o2);
    if(params[part].columns<=256) {
        if(simd<2)grouped_small(a,w,o,params[part],local,simd,lane);
    } else grouped_decode(a,w,o,params[part],local,simd,lane);
}
'''
    cpp=original.replace('id<MTLComputePipelineState> kernels[]={pipeline(@"bf16_decode"),pipeline(@"bf16_decode_small")};','id<MTLComputePipelineState> kernels[]={pipeline(@"bf16_decode"),pipeline(@"bf16_decode_small")}; auto fused=pipeline(@"fused_projection_group");')
    begin=cpp.index('   id<MTLComputeCommandEncoder> encoder=nil;');end=cpp.index('auto encoded=Clock::now();',begin)
    cpp=cpp[:begin]+'''   if(mode==1){
    auto encoder=[cb computeCommandEncoder];need(encoder!=nil,"Fused encoder unavailable");
    [encoder setComputePipelineState:fused];[encoder setBuffer:input offset:0 atIndex:0];
    P parameters[3];uint32_t cuts[4]={0,0,0,0};
    for(size_t i=0;i<3;++i){auto& v=projections[std::min(i,projections.size()-1)];parameters[i]=v.p;
     [encoder setBuffer:v.weight offset:0 atIndex:1+2*i];[encoder setBuffer:v.out offset:0 atIndex:2+2*i];
     cuts[i+1]=cuts[i]+(i<projections.size()?(uint32_t)((v.p.columns+(v.p.columns<=256?3:15))/(v.p.columns<=256?4:16)):0);
    }
    [encoder setBytes:parameters length:sizeof(parameters) atIndex:7];[encoder setBytes:cuts length:sizeof(cuts) atIndex:8];
    [encoder dispatchThreadgroups:MTLSizeMake(cuts[3],1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
   }else{
    for(auto& v:projections){auto encoder=[cb computeCommandEncoder];need(encoder!=nil,"Control encoder unavailable");bool small=v.p.columns<=256;
     [encoder setComputePipelineState:kernels[small]];[encoder setBuffer:input offset:0 atIndex:0];[encoder setBuffer:v.weight offset:0 atIndex:1];[encoder setBuffer:v.out offset:0 atIndex:2];[encoder setBytes:&v.p length:sizeof(v.p) atIndex:3];
     [encoder dispatchThreadgroups:MTLSizeMake((v.p.columns+(small?3:15))/(small?4:16),1,1) threadsPerThreadgroup:MTLSizeMake(small?64:128,1,1)];[encoder endEncoding];
    }
   }
   '''+cpp[end:]
    cpp=cpp.replace('mode<3','mode<2').replace('epoch%3','epoch%2').replace('candidate:{1,2}','candidate:{1}')
    assert cpp!=original and 'MTLDispatchTypeConcurrent' not in cpp
    return shader,cpp

if __name__=='__main__':
    import sys
    source=Path(__file__).resolve().parents[2]
    shader,cpp=generate((source/'src/backends/metal/metal_context.mm').read_text(),Path(__file__).with_name('concurrent_bf16_group_gpu.mm').read_text())
    Path(sys.argv[1]).write_text(shader);Path(sys.argv[2]).write_text(cpp)
