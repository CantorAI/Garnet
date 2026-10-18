"""Isolated address-only prefill readers; preserve seed and original arithmetic."""
from pathlib import Path
import json,hashlib,shutil,re,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-direct-interleaved-prefill';old=Path('/Users/shawn/CantorAI/out/research/worldsense-sole-interleaved/source');s=old.parent.parent/'worldsense-direct-interleaved-prefill/source';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((w/'protocol.json').read_text());assert protocol['design_sha256']==sha(w/'native-design-before-implementation.txt');audit=json.loads((w/'read-only-prefill-reader-audit.json').read_text());seed=audit['seed_manifest'];assert len(seed)==1720 and all(sha(old/k)==v for k,v in seed.items());s.mkdir(parents=True,exist_ok=False)
for rel in seed:
 p=s/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/rel,p)
ops={}
def change(name,a,z,count=1):
 p=s/name;t=p.read_text();assert t.count(a)==count,(name,a,t.count(a));p.write_text(t.replace(a,z));ops.setdefault(name,[]).append(dict(before=a,after=z,count=count))
def append(name,z):
 p=s/name;p.write_text(p.read_text()+z);ops.setdefault(name,[]).append(dict(before='',after=z,count=1,append=True))
h='src/backends/metal/metal_context.h';m='src/backends/metal/metal_context.mm';e='src/entry/metal_main.cpp';serv='runtime/metal_serving.py';test='test2026/metal/sole_interleaved_test.cpp'
change(h,'decodeEncoded=0,restoreEncoded=0,scratchBytes=0;','decodeEncoded=0,restoreEncoded=0,scratchBytes=0,prefillEncoded=0;')
change(h,'    std::string SoleShaderProgramSource() const;','''    std::string SoleShaderProgramSource() const;
    std::vector<uint32_t> SoleInterleavedReadPrefillTileWords(const MetalBF16Weight&,size_t firstRow,size_t rows,size_t tileN,size_t tileK);''')
change(m,'#include "sole_interleaved_mlp_shader.h"','''#include "sole_interleaved_mlp_shader.h"
#ifdef GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL
#include "direct_interleaved_prefill_shader.h"
#endif''')
change(m,'restore{0},scratch{0};','restore{0},scratch{0},prefill{0};')
change(m,'    std::string soleProgram;','''    std::string soleProgram;
#ifdef GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL
    id<MTLComputePipelineState> soleGemm=nil,soleGemm16=nil,soleGemm32=nil,soleTileRead=nil;
#endif''')
needle='''        impl_->kernels->soleProgram=kSoleInterleavedMlpShader;
        auto soleLibrary=[impl_->device newLibraryWithSource:[NSString stringWithUTF8String:kSoleInterleavedMlpShader] options:compileOptions error:&error];'''
change(m,needle,'''        impl_->kernels->soleProgram=kSoleInterleavedMlpShader;
#ifdef GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL
        impl_->kernels->soleProgram+=kDirectInterleavedPrefillShader;
#endif
        auto soleLibrary=[impl_->device newLibraryWithSource:[NSString stringWithUTF8String:impl_->kernels->soleProgram.c_str()] options:compileOptions error:&error];''')
needle='        if(!decode||!restore||!read)throw std::runtime_error("Cannot create sole-layout pipelines");'
extra='''
#ifdef GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL
        auto gemm=[impl_->device newComputePipelineStateWithFunction:[soleLibrary newFunctionWithName:@"sole_direct_bf16_gemm"] error:&error];
        auto gemm16=[impl_->device newComputePipelineStateWithFunction:[soleLibrary newFunctionWithName:@"sole_direct_gemm_16_64_64"] error:&error];
        auto gemm32=[impl_->device newComputePipelineStateWithFunction:[soleLibrary newFunctionWithName:@"sole_direct_gemm_32_64_32"] error:&error];
        auto tile=[impl_->device newComputePipelineStateWithFunction:[soleLibrary newFunctionWithName:@"sole_direct_tile_read"] error:&error];
        if(!gemm||!gemm16||!gemm32||!tile)throw std::runtime_error("Cannot create direct sole-prefill pipelines");
        auto fits=[&](id<MTLComputePipelineState> p,size_t threads){return p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=threads&&p.staticThreadgroupMemoryLength<=impl_->device.maxThreadgroupMemoryLength;};
        if(fits(gemm,128)&&fits(gemm16,256)&&fits(gemm32,256)&&fits(tile,256)) {
            impl_->kernels->soleGemm=gemm;impl_->kernels->soleGemm16=gemm16;
            impl_->kernels->soleGemm32=gemm32;impl_->kernels->soleTileRead=tile;
        }
#endif'''
change(m,needle,needle+extra)
needle='        bool soleAvailable=bool(impl_->kernels->soleDecode)&&bool(impl_->kernels->soleRestore)&&bool(impl_->kernels->soleRead);'
change(m,needle,needle+'''
#ifdef GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL
        soleAvailable=soleAvailable&&bool(impl_->kernels->soleGemm)&&bool(impl_->kernels->soleGemm16)&&bool(impl_->kernels->soleGemm32)&&bool(impl_->kernels->soleTileRead);
#endif''')
needle='''        if(w->soleInterleaved&&a.rows>2) {
            if(primary||w->floatPrefill||w->visionPrefill||w->buffer.length!=kSoleScratchBytes)
                throw std::logic_error("Invalid sole-layout prefill representation");'''
change(m,needle,needle+'''
#ifdef GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL
#ifdef GARNET_METAL_PRIMARY_TEST_FAULTS
            SoleMaybeFail(3);
#endif
            pipeline=wide?(a.rows>16?impl_->kernels->soleGemm32:impl_->kernels->soleGemm16):impl_->kernels->soleGemm;
            if(!pipeline)throw std::logic_error("Converted sole weight has no matching direct-prefill reader");
            soleCounters.prefill.fetch_add(1);
#else''')
needle='            storage=impl_->soleScratch;soleCounters.restore.fetch_add(1);'
change(m,needle,needle+'\n#endif')
change(m,'soleCounters.restore.load(),soleCounters.scratch.load()};','soleCounters.restore.load(),soleCounters.scratch.load(),soleCounters.prefill.load()};')
append(m,'''
namespace Garnet {
std::vector<uint32_t> MetalContext::SoleInterleavedReadPrefillTileWords(const MetalBF16Weight& value,size_t firstRow,size_t rows,size_t tileN,size_t tileK) {
#ifndef GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL
 (void)value;(void)firstRow;(void)rows;(void)tileN;(void)tileK;throw std::logic_error("Direct sole-prefill tile readback disabled");
#else
 @autoreleasepool {
 if(impl_->command||impl_->submitted.count)throw std::logic_error("Sole tile readback requires closed drained batch");
 auto w=value.impl_;
 if(!w||!w->soleInterleaved||!w->buffer||w->buffer.device!=impl_->device||w->buffer.length!=kSoleScratchBytes)
  throw std::invalid_argument("Sole tile readback requires same-device sole weight");
 if(!((tileN==32&&tileK==32)||(tileN==64&&(tileK==32||tileK==64)))||!rows||firstRow%tileN||rows%tileN||w->columns%tileK||firstRow>w->rows||rows>w->rows-firstRow||rows>8*1024*1024/w->columns)
  throw std::invalid_argument("Sole tile readback geometry/range invalid");
 size_t count=rows*w->columns;auto output=[impl_->device newBufferWithLength:count*4 options:MTLResourceStorageModeShared];
 if(!output)throw std::runtime_error("Cannot allocate sole tile output");
 auto command=[impl_->queue commandBuffer];auto encoder=[command computeCommandEncoder];
 if(!command||!encoder)throw std::runtime_error("Cannot create sole tile command");
 uint64_t p[6]={w->rows,w->columns,firstRow,rows,tileN,tileK};
 [encoder setComputePipelineState:impl_->kernels->soleTileRead];[encoder setBuffer:w->buffer offset:0 atIndex:0];
 [encoder setBuffer:output offset:0 atIndex:1];[encoder setBytes:p length:sizeof(p) atIndex:2];
 [encoder dispatchThreadgroups:MTLSizeMake(rows/tileN,w->columns/tileK,1) threadsPerThreadgroup:MTLSizeMake(tileN==32?128:256,1,1)];[encoder endEncoding];
 [command commit];[command waitUntilCompleted];
 if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("Sole tile command failed");
 std::vector<uint32_t> result(count);std::memcpy(result.data(),output.contents,count*4);return result;
 }
#endif
}
}
''')
change(e,'{"scratch_bytes",p.scratchBytes}','{"scratch_bytes",p.scratchBytes},{"prefill_encoded",p.prefillEncoded}')
needle='        info+="; sole_interleaved_mlp";\n#endif'
change(e,needle,needle+'''
#ifdef GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL
        info+="; sole_direct_interleaved_prefill";
#endif''')
change(serv,"for key in ['decode_encoded','restore_encoded']","for key in ['decode_encoded','restore_encoded','prefill_encoded']")
append('cmake/GarnetMetal.cmake','''
option(GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL "Read sole BF16 weights directly in unchanged FP32 prefill tiles" OFF)
if(GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL)
 if(NOT GARNET_METAL_SOLE_INTERLEAVED_MLP)
  message(FATAL_ERROR "Direct sole prefill requires sole interleaved MLP storage")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL=1)
endif()
''')
# Keep every63 complete projection/numerical test; add direct counters/tile bits/zero scratch.
change(test,' bool enabled=false;',' bool enabled=false;bool direct=false;\n#ifdef GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL\n direct=true;\n#endif')
needle=' std::vector<float> saved;'
extra=''' if(direct){
 for(auto geometry:std::vector<std::pair<size_t,size_t>>{{32,32},{64,64},{64,32}}){
  size_t tileN=geometry.first,tileK=geometry.second,chunk=(8*1024*1024/inner/tileN)*tileN;
  for(size_t first=0;first<columns;first+=chunk){size_t nr=std::min(chunk,columns-first);auto actual=gpu.SoleInterleavedReadPrefillTileWords(weight,first,nr,tileN,tileK);need(actual.size()==nr*inner,"Native tile word count");for(size_t j=0;j<actual.size();++j)need(actual[j]==uint32_t(raw[first*inner+j])<<16,"Direct prefill tile original bits differ");}
 }
 rejects([&]{gpu.SoleInterleavedReadPrefillTileWords(weight,1,32,32,32);},"Unaligned tile row accepted");
 rejects([&]{gpu.SoleInterleavedReadPrefillTileWords(weight,0,0,32,32);},"Zero tile rows accepted");
 rejects([&]{gpu.SoleInterleavedReadPrefillTileWords(weight,0,32,32,16);},"Wrong tile geometry accepted");
 need(MetalContext::SoleStats().scratchBytes==0,"Direct tile readback allocated restore scratch");
 }
'''
change(test,needle,extra+needle)
change(test,'size_t(enabled&&rows>2),"Wrong sole restore count"','size_t(enabled&&rows>2&&!direct),"Wrong sole restore count"')
needle='  need(gpu.Download(a)==input,"Projection input mutated");'
change(test,needle,'''  need(after.prefillEncoded-before.prefillEncoded==size_t(enabled&&rows>2&&direct),"Wrong direct prefill count");
  if(direct)need(after.scratchBytes==0&&after.restoreEncoded==0,"Direct prefill used restore/scratch");
'''+needle)
needle='  if(enabled)rejects([&]{gpu.SoleInterleavedReadWords(weight,0,1);},"Active-batch readback accepted");'
change(test,needle,needle+'\n  if(direct)rejects([&]{gpu.SoleInterleavedReadPrefillTileWords(weight,0,32,32,32);},"Active-batch tile readback accepted");')
change(test,'"Scratch fault not raised"','"Prefill preparation/dispatch fault not raised"')
change(test,'"Scratch fault recovery differs"','"Prefill fault recovery differs"')
fragment='''inline ushort sole_direct_word(device const ushort* weight,ulong column,ulong k,ulong inner) {
 return weight[(column/4)*inner*4+k*4+column%4];
}
'''
kernelproof={}
for name,item in audit['kernels'].items():
 original=item['source'];new=original.replace('kernel void '+name+'(','kernel void sole_direct_'+name+'(').replace('weight[column*p.inner+k]','sole_direct_word(weight,column,k,p.inner)')
 assert new.replace('kernel void sole_direct_'+name+'(','kernel void '+name+'(').replace('sole_direct_word(weight,column,k,p.inner)','weight[column*p.inner+k]')==original
 fragment+=new+'\n';kernelproof[name]=dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),new_sha256=hashlib.sha256(new.encode()).hexdigest(),only_function_name_and_weight_address_helper_changed=True)
fragment+='''struct SoleDirectTileParams {ulong rows,inner,first,count,tileN,tileK;};
kernel void sole_direct_tile_read(device const ushort* weight [[buffer(0)]],device uint* output [[buffer(1)]],
 constant SoleDirectTileParams& p [[buffer(2)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 threadgroup uint right[4096];
 ulong colBase=p.first+ulong(group.x)*p.tileN,base=ulong(group.y)*p.tileK;
 uint threads=p.tileN==32?128:256;
 for(uint i=tid;i<p.tileN*p.tileK;i+=threads){ulong k=base+i/p.tileN,column=colBase+i%p.tileN;right[i]=(k<p.inner&&column<p.rows)?uint(sole_direct_word(weight,column,k,p.inner))<<16:0u;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint i=tid;i<p.tileN*p.tileK;i+=threads){ulong k=base+i/p.tileN,column=colBase+i%p.tileN;if(k<p.inner&&column<p.first+p.count)output[(column-p.first)*p.inner+k]=right[i];}
}
'''
(s/'src/backends/metal/direct_interleaved_prefill_shader.h').write_text('// SPDX-License-Identifier: Apache-2.0\n#pragma once\nnamespace Garnet {\ninline constexpr const char* kDirectInterleavedPrefillShader=R"DIRECTMLP(\n'+fragment+')DIRECTMLP";\n}\n')
for rel,transforms in ops.items():
 text=(s/rel).read_text()
 for op in reversed(transforms):
  if op.get('append'):assert text.endswith(op['after']);text=text[:-len(op['after'])]
  else:assert text.count(op['after'])==op['count'],(rel,'inverse');text=text.replace(op['after'],op['before'])
 assert text==(old/rel).read_text(),rel
 if rel.endswith('.py'):ast.parse((s/rel).read_text())
pattern=r'R"([A-Za-z_]+)\((.*?)\)\1"';assert re.findall(pattern,(s/m).read_text(),re.S)==re.findall(pattern,(old/m).read_text(),re.S)
manifest={str(p.relative_to(s)):sha(p)for p in sorted(s.rglob('*'))if p.is_file()};assert len(manifest)==1721 and all(manifest[k]==v for k,v in seed.items()if k not in ops)
for name,value in [('seed-source-manifest.json',seed),('correctness-source-manifest.json',manifest),('source-transformations.json',ops),('direct-prefill-kernel-inverse-proof.json',kernelproof)]: (w/name).write_text(json.dumps(value,indent=2)+'\n')
(w/'source-inverse-proof.json').write_text(json.dumps(dict(seed_files=1720,current_files=1721,changed_or_added=sorted(k for k,v in manifest.items()if seed.get(k)!=v),original_shader_literals_and_63_decode_graph_session_xModel_image_CUDA_original_tests_unchanged=True,all_three_prefill_arithmetic_bodies_inverse_exact=True,original63_native_numeric_tests_retained_and_extended=True),indent=2)+'\n')
shutil.copytree(b/'work/worldsense-sole-interleaved/build-profile',w/'build-profile');shutil.copy2(s/test,w/'sole_interleaved_test.cpp');print('WS64 source1721/seven-file delta, original shader literals/arithmetic inverse exact; no build/model/speed yet')
