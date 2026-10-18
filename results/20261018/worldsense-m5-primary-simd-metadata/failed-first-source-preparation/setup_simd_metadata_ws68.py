from pathlib import Path
import json,hashlib,shutil,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');r=Path('/Users/shawn/CantorAI');w=b/'work/worldsense-primary-simd-metadata';seed=r/'out/research/worldsense-cooperative-prefill/source';s=r/'out/research/worldsense-primary-simd-metadata/source';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();a=load(w/'read-only-metadata-consumer-audit.json');protocol=load(w/'protocol.json');assert sha(w/'native-design-before-implementation.txt')==protocol['design_sha256'];manifest=a['seed_manifest'];assert all(sha(seed/k)==v for k,v in manifest.items());s.mkdir(parents=True,exist_ok=False)
for rel in manifest:
 p=s/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(seed/rel,p)
ops={}
def change(rel,x,y,count=1):
 p=s/rel;t=p.read_text();assert t.count(x)==count,(rel,x,t.count(x));p.write_text(t.replace(x,y));ops.setdefault(rel,[]).append(dict(before=x,after=y,count=count))
def append(rel,t):
 p=s/rel;p.write_text(p.read_text()+t);ops.setdefault(rel,[]).append(dict(before='',after=t,append=True,count=1))
# Original shader is preserved byte-for-byte as a separate string. Explicit inverse proves every serving substitution.
original=(s/'src/backends/metal/primary_block_shader.h').read_text().split('R"PRIMARY(',1)[1].split(')PRIMARY"',1)[0];new=original;shader_ops=[]
def sc(x,y,count=1):
 global new
 assert new.count(x)==count,(x,new.count(x));new=new.replace(x,y);shader_ops.append(dict(before=x,after=y,count=count))
word=original.split('inline ushort primary_word(',1)[1].split('inline ushort4 primary_row4(',1)[0].rstrip();word='inline ushort primary_word('+word
quad=original.split('inline PrimaryQuad primary_quad(',1)[1].split('// Correctness-only',1)[0].rstrip();quad='inline PrimaryQuad primary_quad('+quad
metaWord=word.replace('primary_word(device const uchar* payload,device const uint* metadata,uint row,uint k,uint inner)', 'primary_word_from_meta(device const uchar* payload,uint meta,uint row,uint k)').replace('uint block=(row/4)*(inner/128)+k/128,meta=metadata[block],at=', 'uint at=')
metaQuad=quad.replace('primary_quad(device const uchar* payload,device const uint* metadata,uint row,uint k,uint inner)', 'primary_quad_from_meta(device const uchar* payload,uint meta,uint k)').replace('uint meta=metadata[(row/4)*(inner/128)+k/128],at=', 'uint at=')
assert metaWord!=word and metaQuad!=quad
helpers='\n'+metaWord+'\n'+metaQuad+'''
// Packed admission and complete SIMD participation are required: inner%128==0, row%4==0.
inline PrimaryQuad primary_simd_decode_quad(device const uchar* payload,device const uint* metadata,uint row,uint k,uint inner,uint lane){
 uint meta=0;if(lane==0)meta=metadata[(row/4)*(inner/128)+k/128];
 meta=simd_broadcast_first(meta);return primary_quad_from_meta(payload,meta,k);
}
// Four neighboring active lanes have the same metadata address and a valid first-lane source.
inline ushort primary_simd_prefill_word(device const uchar* payload,device const uint* metadata,uint row,uint k,uint inner,uint lane){
 uint meta=0;if((lane&3u)==0)meta=metadata[(row/4)*(inner/128)+k/128];
 meta=simd_shuffle(meta,ushort(lane&~3u));return primary_word_from_meta(payload,meta,row,k);
}
'''
sc('// Correctness-only bounded readback',helpers+'\n// Correctness-only bounded readback')
sc('PrimaryQuad quad=primary_quad(payload,metadata,column,k,p.inner);','PrimaryQuad quad=primary_simd_decode_quad(payload,metadata,column,k,p.inner,lane);')
sc('PrimaryQuad quad=primary_quad(payload,metadata,column,k,ulong(fixed_inner));','PrimaryQuad quad=primary_simd_decode_quad(payload,metadata,column,k,ulong(fixed_inner),lane);')
sc('primary_word(payload,metadata,column,k,p.inner)', 'primary_simd_prefill_word(payload,metadata,column,k,p.inner,tid%32)',3)
reader='''
// Correctness-only complete block readback. Every helper call has 32 active lanes.
kernel void primary_simd_metadata_expand(device const uchar* payload [[buffer(0)]],device const uint* metadata [[buffer(1)]],
 device uint* out [[buffer(2)]],constant ulong& inner [[buffer(3)]],constant ulong& start [[buffer(4)]],constant ulong& count [[buffer(5)]],constant uint& prefill [[buffer(6)]],constant ulong& firstQuad [[buffer(7)]],uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
 uint blocks=uint(inner/128),row=(uint(firstQuad)+group/blocks)*4,base=(group%blocks)*128;
 if(prefill){
  for(uint pass=0;pass<16;++pass){uint rr=row+lane%4,k=base+lane/4+pass*8;uint word=uint(primary_simd_prefill_word(payload,metadata,rr,k,uint(inner),lane))<<16;ulong at=ulong(rr)*inner+k;if(at>=start&&at-start<count)out[at-start]=word;}
 }else{
  uint k=base+lane*4;PrimaryQuad q=primary_simd_decode_quad(payload,metadata,row,k,uint(inner),lane);ushort4 words[4]={q.k0,q.k1,q.k2,q.k3};
  for(uint j=0;j<4;++j)for(uint v=0;v<4;++v){ulong at=ulong(row+v)*inner+k+j;if(at>=start&&at-start<count)out[at-start]=uint(words[j][v])<<16;}
 }
}
'''
new+=reader;inverse=new[:-len(reader)]
for op in reversed(shader_ops):assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
assert inverse==original
(s/'src/backends/metal/primary_simd_metadata_shader.h').write_text('// SPDX-License-Identifier: Apache-2.0\n#pragma once\nnamespace Garnet {inline constexpr const char* kPrimarySimdMetadataShader=R"PSIMD('+new+')PSIMD";}\n')
m='src/backends/metal/metal_context.mm';h='src/backends/metal/metal_context.h';e='src/entry/metal_main.cpp'
change(m,'#include "primary_block_shader.h"','''#include "primary_block_shader.h"
#ifdef GARNET_METAL_PRIMARY_SIMD_METADATA
#include "primary_simd_metadata_shader.h"
#define GARNET_PRIMARY_SELECTED_PROGRAM kPrimarySimdMetadataShader
#else
#define GARNET_PRIMARY_SELECTED_PROGRAM kPrimaryBlockShader
#endif''')
change(m,'stringWithUTF8String:kPrimaryBlockShader','stringWithUTF8String:GARNET_PRIMARY_SELECTED_PROGRAM',2)
change(m,'primaryExpand,primaryExpandVector,primaryPrefillTileExpand;', 'primaryExpand,primaryExpandVector,primaryPrefillTileExpand;\n#ifdef GARNET_METAL_PRIMARY_SIMD_METADATA\n    id<MTLComputePipelineState> primarySIMDMetadataExpand;\n#endif')
needle='        if(!impl_->kernels->primaryExpandVector)throw std::runtime_error("Cannot create primary vector readback pipeline");'
change(m,needle,needle+'''
#ifdef GARNET_METAL_PRIMARY_SIMD_METADATA
        impl_->kernels->primarySIMDMetadataExpand=[impl_->device newComputePipelineStateWithFunction:[primaryLibrary newFunctionWithName:@"primary_simd_metadata_expand"] error:&error];
        auto requireMetadataSIMD=[&](id<MTLComputePipelineState> pipeline,NSUInteger threads){
          if(!pipeline||pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<threads||pipeline.staticThreadgroupMemoryLength>impl_->device.maxThreadgroupMemoryLength)
            throw std::runtime_error("Unsupported primary metadata-sharing pipeline capabilities");
        };
        requireMetadataSIMD(impl_->kernels->primaryDecode,128);requireMetadataSIMD(impl_->kernels->primaryGemm,128);
        requireMetadataSIMD(impl_->kernels->primaryGemm16Wide,256);requireMetadataSIMD(impl_->kernels->primaryGemm32Wide,256);
        requireMetadataSIMD(impl_->kernels->primarySIMDMetadataExpand,32);
#endif''')
change(h,'    std::vector<uint32_t> PrimaryReadVectorWords(const MetalBF16Weight&,size_t start,size_t count);','    std::vector<uint32_t> PrimaryReadVectorWords(const MetalBF16Weight&,size_t start,size_t count);\n    std::vector<uint32_t> PrimaryReadSIMDMetadataWords(const MetalBF16Weight&,size_t start,size_t count,bool prefill);')
append(m,'''
namespace Garnet {
std::vector<uint32_t> MetalContext::PrimaryReadSIMDMetadataWords(const MetalBF16Weight& weight,size_t start,size_t count,bool prefill){
#ifndef GARNET_METAL_PRIMARY_SIMD_METADATA
 (void)weight;(void)start;(void)count;(void)prefill;throw std::logic_error("Primary SIMD metadata support disabled");
#else
 @autoreleasepool {
  if(impl_->command||!weight.PrimaryReady()||!count||count>8*1024*1024)throw std::invalid_argument("Primary SIMD readback requires packed weight, closed batch and bounded count");
  auto w=weight.impl_;if(w->primaryPayload.device!=impl_->device||w->primaryMetadata.device!=impl_->device||start>w->rows*w->columns||count>w->rows*w->columns-start)throw std::invalid_argument("Primary SIMD readback device/range mismatch");
  id<MTLBuffer> output=[impl_->device newBufferWithLength:count*4 options:MTLResourceStorageModeShared];if(!output)throw std::runtime_error("Primary SIMD readback allocation failed");
  auto command=[impl_->queue commandBuffer];auto encoder=[command computeCommandEncoder];if(!encoder)throw std::runtime_error("Primary SIMD readback encoder failed");
  uint64_t inner=w->columns,offset=start,n=count,firstQuad=(start/inner)/4,lastQuad=((start+count-1)/inner)/4;uint32_t mode=prefill?1:0;
  size_t groups=(lastQuad-firstQuad+1)*(inner/128);
  [encoder setComputePipelineState:impl_->kernels->primarySIMDMetadataExpand];[encoder setBuffer:w->primaryPayload offset:0 atIndex:0];[encoder setBuffer:w->primaryMetadata offset:0 atIndex:1];[encoder setBuffer:output offset:0 atIndex:2];[encoder setBytes:&inner length:8 atIndex:3];[encoder setBytes:&offset length:8 atIndex:4];[encoder setBytes:&n length:8 atIndex:5];[encoder setBytes:&mode length:4 atIndex:6];[encoder setBytes:&firstQuad length:8 atIndex:7];
  [encoder dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("Primary SIMD readback GPU failed");
  std::vector<uint32_t> result(count);std::memcpy(result.data(),output.contents,count*4);return result;
 }
#endif
}
}
''')
needle='        info+="; primary_bf16_blocks; primary_vector_decode; primary_kmajor_decode; primary_u32_metadata";'
change(e,needle,needle+'\n#ifdef GARNET_METAL_PRIMARY_SIMD_METADATA\n        info+="; primary_simd_metadata";\n#endif')
append('cmake/GarnetMetal.cmake','''
option(GARNET_METAL_PRIMARY_SIMD_METADATA "Share proven-uniform primary block metadata across SIMD lanes" OFF)
if(GARNET_METAL_PRIMARY_SIMD_METADATA)
 if(NOT GARNET_METAL_PRIMARY_BF16_BLOCKS OR GARNET_METAL_PRIMARY_VECTOR_PREFILL OR GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY)
  message(FATAL_ERROR "Primary SIMD metadata requires primary blocks and ordinary primary prefill/library")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_PRIMARY_SIMD_METADATA=1)
endif()
add_executable(garnet_primary_simd_metadata_test test2026/metal/primary_simd_metadata_test.cpp)
target_link_libraries(garnet_primary_simd_metadata_test PRIVATE garnet_metal)
add_test(NAME metal_primary_simd_metadata_reference COMMAND garnet_primary_simd_metadata_test "${CMAKE_CURRENT_BINARY_DIR}/primary_simd_metadata_words.bin")
''')
test='''// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
using namespace Garnet;
static void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F>static void rejects(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}need(rejected,"Malformed or unavailable metadata readback accepted");}
int main(int argc,char**argv){try{
 need(argc==2,"Usage: primary_simd_metadata_test output.bin");MetalContext gpu;std::ofstream file(argv[1],std::ios::binary);need(bool(file),"Cannot create complete metadata word records");
 auto initial=MetalContext::PrimaryStats();
#ifndef GARNET_METAL_PRIMARY_SIMD_METADATA
 rejects([&]{gpu.PrimaryReadSIMDMetadataWords(MetalBF16Weight(),0,1,false);});
#else
 for(size_t inner:std::vector<size_t>{128,256,512})for(size_t pattern=0;pattern<3;++pattern){
  const size_t rows=260,n=rows*inner;std::vector<uint16_t> bits(n);
  for(size_t i=0;i<n;++i){size_t row=i/inner,k=i%inner,block=(row/4)*(inner/128)+k/128;bits[i]=pattern==0?uint16_t(i):uint16_t(0x3f00u+((i*13)%128));if(pattern==2&&block%3==1)bits[i]=uint16_t(i*19);}
  auto weight=gpu.UploadBF16(bits.data(),bits.size()*2,rows,inner,false,false,true);need(weight.PrimaryReady(),"Metadata fixture not primary packed");auto alias=weight;weight=MetalBF16Weight();auto before=MetalContext::PrimaryStats();
  for(bool prefill:{false,true})for(auto range:std::vector<std::pair<size_t,size_t>>{{0,n},{1,1},{127,3},{inner-1,7},{3*inner+125,517},{n-1,1}}){
   auto actual=gpu.PrimaryReadSIMDMetadataWords(alias,range.first,range.second,prefill);need(actual.size()==range.second,"Metadata range length changed");for(size_t j=0;j<actual.size();++j)need(actual[j]==uint32_t(bits[range.first+j])<<16,"Metadata sharing changed original word");
   uint64_t header[6]={rows,inner,pattern,uint64_t(prefill),range.first,range.second};file.write(reinterpret_cast<const char*>(header),sizeof(header));file.write(reinterpret_cast<const char*>(actual.data()),actual.size()*4);
  }
  auto after=MetalContext::PrimaryStats();need(before.activeWeights==after.activeWeights&&before.allocatedBytes==after.allocatedBytes&&before.decodeEncoded==after.decodeEncoded&&before.prefillEncoded==after.prefillEncoded&&before.headEncoded==after.headEncoded,"Diagnostic altered primary owners or serving counters");
  rejects([&]{gpu.PrimaryReadSIMDMetadataWords(alias,0,0,false);});rejects([&]{gpu.PrimaryReadSIMDMetadataWords(alias,n,1,true);});rejects([&]{gpu.PrimaryReadSIMDMetadataWords(alias,std::numeric_limits<size_t>::max(),1,false);});rejects([&]{gpu.PrimaryReadSIMDMetadataWords(alias,0,8*1024*1024+1,true);});
  gpu.Begin();rejects([&]{gpu.PrimaryReadSIMDMetadataWords(alias,0,1,false);});gpu.Abort();auto recovered=gpu.PrimaryReadSIMDMetadataWords(alias,n-1,1,true);need(recovered[0]==uint32_t(bits.back())<<16,"Abort recovery metadata word changed");
  {MetalContext foreign;rejects([&]{foreign.PrimaryReadSIMDMetadataWords(alias,0,1,false);});}
  alias=MetalBF16Weight();auto released=MetalContext::PrimaryStats();need(released.activeWeights==initial.activeWeights&&released.allocatedBytes==initial.allocatedBytes,"Metadata fixture owner retained");
 }
 rejects([&]{gpu.PrimaryReadSIMDMetadataWords(MetalBF16Weight(),0,1,false);});
#endif
 need(bool(file),"Cannot write complete metadata records");std::cout<<"Original-bit metadata helpers, bounds, owner, Abort and disabled guards PASS\\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\\n";return 1;}}
'''
(s/'test2026/metal/primary_simd_metadata_test.cpp').write_text(test)
for rel,records in ops.items():
 t=(s/rel).read_text()
 for op in reversed(records):
  if op.get('append'):assert t.endswith(op['after']);t=t[:-len(op['after'])]
  else:assert t.count(op['after'])==op['count'];t=t.replace(op['after'],op['before'])
 assert t==(seed/rel).read_text(),rel
out={str(p.relative_to(s)):sha(p)for p in sorted(s.rglob('*'))if p.is_file()};assert len(out)==1726 and all(out[k]==v for k,v in manifest.items()if k not in ops)
proof=dict(original_source=original,original_sha256=hashlib.sha256(original.encode()).hexdigest(),new_source=new,new_sha256=hashlib.sha256(new.encode()).hexdigest(),operations=shader_ops,reader_append=reader,word_helper_original=word,word_helper_supplied_metadata=metaWord,quad_helper_original=quad,quad_helper_supplied_metadata=metaQuad,all_original_float_MAC_K_order_reductions_barriers_geometry_inverse_exact=True)
for name,value in [('seed-source-manifest.json',manifest),('correctness-source-manifest.json',out),('source-transformations.json',ops),('simd-shader-inverse-proof.json',proof)]: (w/name).write_text(json.dumps(value,indent=2)+'\n')
(w/'source-inverse-proof.json').write_text(json.dumps(dict(seed_files=1724,current_files=1726,changed_or_added=sorted(k for k,v in out.items()if manifest.get(k)!=v),old_primary_shader_and_all_original_tests_and_CUDA_engine_runtime_graph_session_image_xModel_byte_exact=True,new_flag_defaultOFF=True,new_reader_no_serving_counters=True),indent=2)+'\n')
shutil.copytree(oldprofile:=b/'work/worldsense-cooperative-prefill/build-profile',w/'build-profile')
print('WS68 source1726: original shader/tests preserved, explicit inverse verified, new helpers/native test added. No build/model yet.')
