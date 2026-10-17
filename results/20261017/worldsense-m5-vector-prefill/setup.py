from pathlib import Path
import json,hashlib,shutil
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');base=r/'Garnet';s=r/'out/research/worldsense-vector-prefill/source'
assert not s.exists()
assert hashlib.sha256((w/'native-design-before-implementation.txt').read_bytes()).hexdigest()==json.loads((w/'design-identity.json').read_text())['design_sha256']
manifest=json.loads((w.parent/'worldsense-primary-u32-integration/canonical-source-before-build.json').read_text());assert len(manifest)==1715
for rel,h in manifest.items():
 p=base/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==h,rel
 t=s/rel;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
(w/'seed-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
def replace(rel,a,b):
 p=s/rel;t=p.read_text();assert t.count(a)==1,(rel,a,t.count(a));p.write_text(t.replace(a,b,1))
raw=(s/'src/backends/metal/primary_block_shader.h').read_text().split('R"PRIMARY(',1)[1].split(')PRIMARY"',1)[0]
helper='''
// Admission: columns%4==0, inner%128==0. Original scalar tile values,
// expressed as BF16 bit expansion only; no floating-point arithmetic.
inline void primary_prefill_tile_fill(device const uchar* payload,device const uint* metadata,
 threadgroup float* right,uint colBase,uint kBase,uint columns,uint inner,
 uint tileColumns,uint tileK,uint tid,uint threads) {
 for(uint index=tid;index<(tileColumns/4)*(tileK/4);index+=threads) {
  uint localK=(index/(tileColumns/4))*4,localColumn=(index%(tileColumns/4))*4;
  uint row=colBase+localColumn,k=kBase+localK;
  PrimaryQuad quad;
  if(row<columns&&k<inner)quad=primary_quad(payload,metadata,row,k,inner);
  else {quad.k0=ushort4(0);quad.k1=ushort4(0);quad.k2=ushort4(0);quad.k3=ushort4(0);}
  ushort4 words[4]={quad.k0,quad.k1,quad.k2,quad.k3};
  for(uint j=0;j<4;++j)for(uint c=0;c<4;++c)
   right[(localK+j)*tileColumns+localColumn+c]=as_type<float>(uint(words[j][c])<<16);
 }
}
'''
specs=[('primary_bf16_gemm','primary_vector_bf16_gemm','32*32',32,32,128),('primary_gemm_16_64_64','primary_vector_gemm_16_64_64','4096',64,64,256),('primary_gemm_32_64_32','primary_vector_gemm_32_64_32','2048',64,32,256)]
parts=[];proof=[]
for name,newName,total,c,k,threads in specs:
 start=raw.index('kernel void '+name+'(');end=raw.find('kernel void ',start+1);original=raw[start:end if end>=0 else len(raw)]
 old=f'for(uint i=tid;i<{total};i+={threads}){{ulong k=base+i/{c},column=colBase+i%{c};right[i]=(k<p.inner&&column<p.columns)?as_type<float>(uint(primary_word(payload,metadata,column,k,p.inner))<<16):0.0f;}}'
 new=f'primary_prefill_tile_fill(payload,metadata,right,uint(colBase),uint(base),uint(p.columns),uint(p.inner),{c},{k},tid,{threads});'
 assert original.count(old)==1,name
 candidate=original.replace('kernel void '+name+'(','kernel void '+newName+'(',1).replace(old,new,1)
 assert candidate.replace('kernel void '+newName+'(','kernel void '+name+'(',1).replace(new,old,1)==original
 parts.append(candidate);proof.append({'original':name,'candidate':newName,'tileC':c,'tileK':k,'threads':threads,'inverse_recovers_entire_body':True,'original_sha256':hashlib.sha256(original.encode()).hexdigest(),'candidate_sha256':hashlib.sha256(candidate.encode()).hexdigest()})
diag='''
struct PrimaryPrefillTileProof {uint inner,columns,firstRow,rows,tileColumns,tileK,threads;};
kernel void primary_vector_prefill_expand(device const uchar* payload [[buffer(0)]],device const uint* metadata [[buffer(1)]],
 device uint* output [[buffer(2)]],constant PrimaryPrefillTileProof& p [[buffer(3)]],
 uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 threadgroup float right[4096];
 uint rowBase=p.firstRow+group.x*p.tileColumns,kBase=group.y*p.tileK;
 primary_prefill_tile_fill(payload,metadata,right,rowBase,kBase,p.columns,p.inner,p.tileColumns,p.tileK,tid,p.threads);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint i=tid;i<p.tileColumns*p.tileK;i+=p.threads) {
  uint row=rowBase+i%p.tileColumns,k=kBase+i/p.tileColumns;
  if(row<p.firstRow+p.rows&&k<p.inner)output[(row-p.firstRow)*p.inner+k]=as_type<uint>(right[i]);
 }
}
'''
fragment=helper+'\n'.join(parts)+diag
(s/'src/backends/metal/primary_vector_prefill_shader.h').write_text('// SPDX-License-Identifier: Apache-2.0\n#pragma once\nnamespace Garnet {inline constexpr const char* kPrimaryVectorPrefillShader=R"PVEC('+fragment+')PVEC";}\n')
(w/'shader-body-substitution-proof.json').write_text(json.dumps({'kernels':proof,'whole_original_primary_shader_unchanged':True,'helper':'existing primary_quad unchanged; new filler uses only integer bit expansion and stores','OFF_new_fragment_absent':True},indent=2)+'\n')
(w/'materialized-vector-prefill.metal').write_text(fragment)
replace('src/backends/metal/metal_context.mm','#include "primary_block_shader.h"','#include "primary_block_shader.h"\n#ifdef GARNET_METAL_PRIMARY_VECTOR_PREFILL\n#include "primary_vector_prefill_shader.h"\n#endif')
replace('src/backends/metal/metal_context.mm','primaryGemm32Wide,primaryExpand,primaryExpandVector;','primaryGemm32Wide,primaryExpand,primaryExpandVector,primaryPrefillTileExpand;')
old='        source=[source stringByAppendingString:[NSString stringWithUTF8String:kPrimaryBlockShader]];'
replace('src/backends/metal/metal_context.mm',old,old+'\n#ifdef GARNET_METAL_PRIMARY_VECTOR_PREFILL\n        source=[source stringByAppendingString:[NSString stringWithUTF8String:kPrimaryVectorPrefillShader]];\n#endif')
for member,name,newName in [('primaryGemm','primary_bf16_gemm','primary_vector_bf16_gemm'),('primaryGemm16Wide','primary_gemm_16_64_64','primary_vector_gemm_16_64_64'),('primaryGemm32Wide','primary_gemm_32_64_32','primary_vector_gemm_32_64_32')]:
 old=f'        impl_->kernels->{member}=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"{name}"] error:&error];'
 replace('src/backends/metal/metal_context.mm',old,'#ifdef GARNET_METAL_PRIMARY_VECTOR_PREFILL\n'+old.replace('@"'+name+'"','@"'+newName+'"')+'\n#else\n'+old+'\n#endif')
old='        if(!impl_->kernels->primaryExpandVector)throw std::runtime_error("Cannot create primary vector readback pipeline");'
replace('src/backends/metal/metal_context.mm',old,old+'''
#ifdef GARNET_METAL_PRIMARY_VECTOR_PREFILL
        impl_->kernels->primaryPrefillTileExpand=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"primary_vector_prefill_expand"] error:&error];
        auto requireVectorPrefill=[&](id<MTLComputePipelineState> pipeline,NSUInteger threads) {
            if(!pipeline||pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<threads||pipeline.staticThreadgroupMemoryLength>impl_->device.maxThreadgroupMemoryLength)
                throw std::runtime_error("Unsupported vector prefill pipeline resource limits");
        };
        requireVectorPrefill(impl_->kernels->primaryGemm,128);
        requireVectorPrefill(impl_->kernels->primaryGemm16Wide,256);
        requireVectorPrefill(impl_->kernels->primaryGemm32Wide,256);
        requireVectorPrefill(impl_->kernels->primaryPrefillTileExpand,256);
#endif''')
replace('src/backends/metal/metal_context.h','    std::vector<uint32_t> PrimaryReadVectorWords(const MetalBF16Weight&,size_t start,size_t count);','    std::vector<uint32_t> PrimaryReadVectorWords(const MetalBF16Weight&,size_t start,size_t count);\n    std::vector<uint32_t> PrimaryReadPrefillTileWords(const MetalBF16Weight&,size_t firstRow,size_t rows,size_t tileColumns,size_t tileK);')
with(s/'src/backends/metal/metal_context.mm').open('a')as f:f.write('''
namespace Garnet {
std::vector<uint32_t> MetalContext::PrimaryReadPrefillTileWords(const MetalBF16Weight& weight,size_t firstRow,size_t rows,size_t tileColumns,size_t tileK) {
#ifndef GARNET_METAL_PRIMARY_VECTOR_PREFILL
 (void)weight;(void)firstRow;(void)rows;(void)tileColumns;(void)tileK;throw std::logic_error("Vector prefill tile support disabled");
#else
 @autoreleasepool {
  const bool geometry=(tileColumns==32&&tileK==32)||(tileColumns==64&&(tileK==32||tileK==64));
  if(impl_->command||!weight.PrimaryReady()||!geometry||!rows||firstRow%tileColumns)throw std::invalid_argument("Prefill tile readback requires packed weight, closed batch and aligned geometry");
  auto w=weight.impl_;
  if(w->primaryPayload.device!=impl_->device||w->primaryMetadata.device!=impl_->device||firstRow>w->rows||rows>w->rows-firstRow||rows>8*1024*1024/w->columns||(rows%tileColumns&&firstRow+rows!=w->rows))
      throw std::invalid_argument("Prefill tile readback device/range/count mismatch");
  const size_t count=rows*w->columns,threads=tileColumns==32?128:256;
  id<MTLBuffer> output=[impl_->device newBufferWithLength:count*4 options:MTLResourceStorageModeShared];if(!output)throw std::runtime_error("Prefill tile readback allocation failed");
  auto command=[impl_->queue commandBuffer];auto encoder=[command computeCommandEncoder];if(!encoder)throw std::runtime_error("Prefill tile readback encoder failed");
  const uint32_t params[]={uint32_t(w->columns),uint32_t(w->rows),uint32_t(firstRow),uint32_t(rows),uint32_t(tileColumns),uint32_t(tileK),uint32_t(threads)};
  [encoder setComputePipelineState:impl_->kernels->primaryPrefillTileExpand];[encoder setBuffer:w->primaryPayload offset:0 atIndex:0];[encoder setBuffer:w->primaryMetadata offset:0 atIndex:1];[encoder setBuffer:output offset:0 atIndex:2];[encoder setBytes:params length:sizeof(params) atIndex:3];
  [encoder dispatchThreadgroups:MTLSizeMake((rows+tileColumns-1)/tileColumns,w->columns/tileK,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("Prefill tile readback GPU failed");
  std::vector<uint32_t> result(count);std::memcpy(result.data(),output.contents,count*4);return result;
 }
#endif
}
}
''')
replace('src/entry/metal_main.cpp','        info+="; primary_bf16_blocks; primary_vector_decode; primary_kmajor_decode; primary_u32_metadata";','        info+="; primary_bf16_blocks; primary_vector_decode; primary_kmajor_decode; primary_u32_metadata";\n#ifdef GARNET_METAL_PRIMARY_VECTOR_PREFILL\n        info+="; primary_vector_prefill";\n#endif')
with(s/'cmake/GarnetMetal.cmake').open('a')as f:f.write('''
option(GARNET_METAL_PRIMARY_VECTOR_PREFILL "Vector-load exact primary weights into existing FP32 prefill tiles" OFF)
if(GARNET_METAL_PRIMARY_VECTOR_PREFILL)
 if(NOT GARNET_METAL_PRIMARY_BF16_BLOCKS)
  message(FATAL_ERROR "Vector primary prefill requires primary BF16 weights")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_PRIMARY_VECTOR_PREFILL=1)
endif()
''')
print('WS47 alternate shader kernels and shared diagnostic filler implemented; original shader/codec unchanged. Native tests not run.')
