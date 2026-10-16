"""Compact-aware independent projections; original per-output accumulation/reduction."""
import re

def generate(native,compact_header,original):
    raw=native[native.index('kernel void bf16_decode('):native.index('kernel void bf16_gemm(')]
    compact=compact_header.split('R"METAL(',1)[1].split(')METAL"',1)[0]
    def inline(source,old,new):
        source=re.sub(r'\s*\[\[[^\]]+\]\]', '',source)
        assert source.count('kernel void '+old+'(')==1
        return source.replace('kernel void '+old+'(','inline void '+new+'(')
    large=raw[:raw.index('kernel void bf16_decode_small(')]
    small=raw[raw.index('kernel void bf16_decode_small('):]
    large=inline(large,'bf16_decode','grouped_raw64')
    assert large.count('ulong(group)*16+simd*4')==1
    large=large.replace('ulong(group)*16+simd*4','ulong(group)*8+simd*4')
    small=inline(small,'bf16_decode_small','grouped_small64')
    packed=inline(compact,'bf16_body_coalesced_block','grouped_compact64')
    shader='#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\n'+raw+compact+large+small+packed+'''
kernel void compact_fused_projection_group(device const float* a [[buffer(0)]],
    device const ushort* w0 [[buffer(1)]],device float* o0 [[buffer(2)]],device const uchar* b0 [[buffer(3)]],device const uint* m0 [[buffer(4)]],
    device const ushort* w1 [[buffer(5)]],device float* o1 [[buffer(6)]],device const uchar* b1 [[buffer(7)]],device const uint* m1 [[buffer(8)]],
    device const ushort* w2 [[buffer(9)]],device float* o2 [[buffer(10)]],device const uchar* b2 [[buffer(11)]],device const uint* m2 [[buffer(12)]],
    constant BF16Params* params [[buffer(13)]],constant uint* cuts [[buffer(14)]],constant uint* kinds [[buffer(15)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    uint part=group<cuts[1]?0:(group<cuts[2]?1:2),local=group-cuts[part];
    device const ushort* w=part==0?w0:(part==1?w1:w2);
    device float* o=part==0?o0:(part==1?o1:o2);
    device const uchar* b=part==0?b0:(part==1?b1:b2);
    device const uint* m=part==0?m0:(part==1?m1:m2);
    if(kinds[part]==2)grouped_compact64(a,w,o,params[part],b,m,local,simd,lane,64);
    else if(kinds[part]==1)grouped_raw64(a,w,o,params[part],local,simd,lane);
    else grouped_small64(a,w,o,params[part],local,simd,lane);
}
'''
    cpp=original
    def replace(old,new):
        nonlocal cpp
        assert cpp.count(old)==1,(old,cpp.count(old));cpp=cpp.replace(old,new)
    replace('auto library=[d newLibraryWithSource:source options:nil error:&error];',
      'auto options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion3_1;auto library=[d newLibraryWithSource:source options:options error:&error];')
    replace('for(NSDictionary* c in cases){uint64_t n=', 'for(NSDictionary* c in cases){auto packStart=Clock::now();uint64_t n=')
    replace('struct Projection{id<MTLBuffer> weight,out;P p;std::vector<float> reference[4];};',
      'struct Projection{id<MTLBuffer> weight,out,payload,metadata;P p;bool compact=false;size_t blocks=0,compactBlocks=0;std::vector<uint16_t> raw;std::vector<float> reference[4];};')
    replace('id<MTLComputePipelineState> kernels[]={pipeline(@"bf16_decode"),pipeline(@"bf16_decode_small")};',
      'id<MTLComputePipelineState> kernels[]={pipeline(@"bf16_decode"),pipeline(@"bf16_decode_small")};auto packed=pipeline(@"bf16_body_coalesced_block"),fused=pipeline(@"compact_fused_projection_group");need(packed.maxTotalThreadsPerThreadgroup>=64&&fused.maxTotalThreadsPerThreadgroup>=64,"64-thread pipeline unavailable");')
    old='v.weight=[d newBufferWithBytes:raw.bytes length:raw.length options:MTLResourceStorageModeShared];'
    replace(old,'''v.raw.resize(k*n);std::memcpy(v.raw.data(),raw.bytes,raw.length);
   v.weight=[d newBufferWithBytes:raw.bytes length:raw.length options:MTLResourceStorageModeShared];
   // Match retained upload admission; dense FP32-key matrices retain raw decode.
   need([c[@"float_key"] boolValue]||n<=256||n>8192||k%4!=0||k<(1048576+n-1)/n||k%128==0,"Interleaved-only K shape needs its retained control, not raw substitution");
   if(![c[@"float_key"] boolValue]&&n>256&&n<=8192&&k%128==0&&k>=(1048576+n-1)/n){
    size_t padded=(n+3)/4*4,blocks=padded/4*(k/128);std::vector<uint16_t> fallback(padded*k,0);
    for(size_t col=0;col<n;++col)for(size_t j=0;j<k;++j)fallback[(col/4)*k*4+j*4+col%4]=v.raw[col*k+j];
    for(size_t col=0;col<padded;++col)for(size_t j=0;j<k;++j)need(fallback[(col/4)*k*4+j*4+col%4]==(col<n?v.raw[col*k+j]:0),"Fallback interleave original-bit roundtrip failed");
    std::vector<uint8_t> payload(blocks*768,0);std::vector<uint32_t> metadata(blocks,0);v.blocks=blocks;
    for(size_t group=0;group<padded/4;++group)for(size_t block=0;block<k/128;++block){
     size_t index=group*(k/128)+block;uint16_t low=255,high=0;
     for(size_t j=0;j<128;++j)for(size_t lane=0;lane<4;++lane){auto e=(fallback[group*k*4+(block*128+j)*4+lane]>>7)&255;low=std::min(low,uint16_t(e));high=std::max(high,uint16_t(e));}
     if(high-low>15)continue;metadata[index]=uint32_t(low)|256u;++v.compactBlocks;
     for(size_t j=0;j<128;++j)for(size_t lane=0;lane<4;++lane){size_t position=j*4+lane,at=index*768;uint16_t bits=fallback[group*k*4+(block*128+j)*4+lane];
      payload[at+position]=uint8_t((bits&127)|((bits>>8)&128));uint8_t tag=uint8_t(((bits>>7)&255)-low);payload[at+512+position/2]|=uint8_t(tag<<((position%2)*4));}
     for(size_t j=0;j<128;++j)for(size_t lane=0;lane<4;++lane){size_t position=j*4+lane,at=index*768;uint16_t sm=payload[at+position],tag=(payload[at+512+position/2]>>((position%2)*4))&15;
      uint16_t bits=uint16_t((sm&127)|((sm&128)<<8)|((tag+low)<<7));need(bits==fallback[group*k*4+(block*128+j)*4+lane],"Compact original-bit roundtrip failed");}
    }
    need(v.compactBlocks>0,"Interleaved-only weight needs its retained control, not raw substitution");
    if(v.compactBlocks){v.compact=true;v.weight=[d newBufferWithBytes:fallback.data() length:fallback.size()*2 options:MTLResourceStorageModeShared];
     v.payload=[d newBufferWithBytes:payload.data() length:payload.size() options:MTLResourceStorageModeShared];v.metadata=[d newBufferWithBytes:metadata.data() length:metadata.size()*4 options:MTLResourceStorageModeShared];need(v.weight&&v.payload&&v.metadata,"Compact allocation failed");}
   }
   if(!v.compact){uint32_t zero=0;v.payload=[d newBufferWithBytes:&zero length:4 options:MTLResourceStorageModeShared];v.metadata=[d newBufferWithBytes:&zero length:4 options:MTLResourceStorageModeShared];}
   emit(@{@"kind":@"packing",@"group":name,@"rows":@(rows),@"matrix":c[@"name"],@"selected":@(v.compact),@"blocks":@(v.blocks),@"compact_blocks":@(v.compactBlocks),@"load_pack_allocate_ms":@(ms(packStart,Clock::now())),@"weight_buffer_bytes":@(v.weight.length),@"payload_buffer_bytes":@(v.payload.length),@"metadata_buffer_bytes":@(v.metadata.length)});''')
    start=cpp.index('   id<MTLComputeCommandEncoder> encoder=nil;');end=cpp.index('auto encoded=Clock::now();',start)
    cpp=cpp[:start]+'''   if(mode==1){
    auto encoder=[cb computeCommandEncoder];need(encoder!=nil,"Fused encoder unavailable");[encoder setComputePipelineState:fused];[encoder setBuffer:input offset:0 atIndex:0];
    P parameters[3];uint32_t cuts[4]={0,0,0,0},kinds[3];
    for(size_t i=0;i<3;++i){auto& v=projections[std::min(i,projections.size()-1)];parameters[i]=v.p;kinds[i]=v.compact?2:v.p.columns>256?1:0;
     [encoder setBuffer:v.weight offset:0 atIndex:1+4*i];[encoder setBuffer:v.out offset:0 atIndex:2+4*i];[encoder setBuffer:v.payload offset:0 atIndex:3+4*i];[encoder setBuffer:v.metadata offset:0 atIndex:4+4*i];
     uint32_t width=kinds[i]?8:4;cuts[i+1]=cuts[i]+(i<projections.size()?uint32_t((v.p.columns+width-1)/width):0);
    }
    [encoder setBytes:parameters length:sizeof(parameters) atIndex:13];[encoder setBytes:cuts length:sizeof(cuts) atIndex:14];[encoder setBytes:kinds length:sizeof(kinds) atIndex:15];
    [encoder dispatchThreadgroups:MTLSizeMake(cuts[3],1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[encoder endEncoding];
   }else{
    // Mode2 is the identical retained pipeline/geometry/bindings null.
    for(auto& v:projections){auto encoder=[cb computeCommandEncoder];need(encoder!=nil,"Control encoder unavailable");bool small=v.p.columns<=256;
     [encoder setComputePipelineState:v.compact?packed:kernels[small]];[encoder setBuffer:input offset:0 atIndex:0];[encoder setBuffer:v.weight offset:0 atIndex:1];[encoder setBuffer:v.out offset:0 atIndex:2];[encoder setBytes:&v.p length:sizeof(v.p) atIndex:3];
     if(v.compact){[encoder setBuffer:v.payload offset:0 atIndex:4];[encoder setBuffer:v.metadata offset:0 atIndex:5];}
     uint32_t width=v.compact?8:small?4:16;[encoder dispatchThreadgroups:MTLSizeMake((v.p.columns+width-1)/width,1,1) threadsPerThreadgroup:MTLSizeMake(v.compact||small?64:128,1,1)];[encoder endEncoding];
    }
   }
   '''+cpp[end:]
    replace('auto weights=static_cast<const uint16_t*>(v.weight.contents);','auto weights=v.raw.data();')
    replace('while(ms(warm,Clock::now())<250);','while(ms(warm,Clock::now())<2000);')
    replace('for(int candidate:{1,2})for(int round=0;round<2;++round)','for(int candidate:{2,1})for(int round=0;round<10;++round)')
    replace('sample<12','sample<5')
    replace('@"Original native BF16/default options; FP32 input, arithmetic and output"','@"Retained compact64/raw128/small64 baseline; grouped64 candidate; original FP32 dot/reduction; exact BF16 payload"')
    assert cpp.count('Projection or padding changed bits')==1 and cpp.count('Original FP64 envelope exceeded')==1
    return shader,cpp
