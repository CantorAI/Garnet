from pathlib import Path
import json,hashlib,shutil,re,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');r=Path('/Users/shawn/CantorAI');w=b/'work/worldsense-cooperative-prefill';old=r/'out/research/worldsense-sole-dual-decode/source';s=r/'out/research/worldsense-cooperative-prefill/source';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();a=json.loads((w/'read-only-prefill-consumer-audit.json').read_text());seed=a['seed_manifest'];assert json.loads((w/'protocol.json').read_text())['design_sha256']==sha(w/'native-design-before-implementation.txt');assert all(sha(old/k)==v for k,v in seed.items());s.mkdir(parents=True,exist_ok=False)
for rel in seed:
 p=s/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/rel,p)
ops={}
def change(rel,oldtext,newtext,count=1):
 p=s/rel;t=p.read_text();assert t.count(oldtext)==count,(rel,oldtext,t.count(oldtext));p.write_text(t.replace(oldtext,newtext));ops.setdefault(rel,[]).append(dict(before=oldtext,after=newtext,count=count))
def append(rel,text):
 p=s/rel;p.write_text(p.read_text()+text);ops.setdefault(rel,[]).append(dict(before='',after=text,count=1,append=True))
h='src/backends/metal/metal_context.h';m='src/backends/metal/metal_context.mm';e='src/entry/metal_main.cpp';test='test2026/metal/sole_interleaved_test.cpp'
change(m,'#include "direct_interleaved_prefill_shader.h"','#include "direct_interleaved_prefill_shader.h"\n#ifdef GARNET_METAL_SOLE_COOPERATIVE_PREFILL\n#include "cooperative_sole_prefill_shader.h"\n#endif')
change(m,'        impl_->kernels->soleProgram+=kDirectInterleavedPrefillShader;','''#ifdef GARNET_METAL_SOLE_COOPERATIVE_PREFILL
        impl_->kernels->soleProgram+=kCooperativeSolePrefillShader;
#else
        impl_->kernels->soleProgram+=kDirectInterleavedPrefillShader;
#endif''')
change(h,'class MetalContext {','''struct MetalSolePrefillPipelineInfo {
 std::string name;uint64_t threadWidth=0,maxThreads=0,staticThreadgroupBytes=0,deviceMaxThreadgroupBytes=0;bool cooperative=false;
};
class MetalContext {''')
change(h,'    std::string SoleShaderProgramSource() const;','    std::string SoleShaderProgramSource() const;\n    std::vector<MetalSolePrefillPipelineInfo> SolePrefillPipelineInfo() const;')
append(m,'''
namespace Garnet {
std::vector<MetalSolePrefillPipelineInfo> MetalContext::SolePrefillPipelineInfo()const{
 std::vector<MetalSolePrefillPipelineInfo> result;
#ifdef GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL
 @autoreleasepool {
 bool cooperative=false;
#ifdef GARNET_METAL_SOLE_COOPERATIVE_PREFILL
 cooperative=true;
#endif
 auto add=[&](const char* name,id<MTLComputePipelineState> pipeline){result.push_back({name,uint64_t(pipeline.threadExecutionWidth),uint64_t(pipeline.maxTotalThreadsPerThreadgroup),uint64_t(pipeline.staticThreadgroupMemoryLength),uint64_t(impl_->device.maxThreadgroupMemoryLength),cooperative});};
 add("base32x32",impl_->kernels->soleGemm);add("wide16_64x64",impl_->kernels->soleGemm16);add("wide32_64x32",impl_->kernels->soleGemm32);add("wordprobe",impl_->kernels->soleTileRead);
 }
#endif
 return result;
}
}
''')
needle='        info+="; sole_dual_projection_decode";\n#endif';change(e,needle,needle+'\n#ifdef GARNET_METAL_SOLE_COOPERATIVE_PREFILL\n        info+="; sole_cooperative_prefill";\n#endif')
# Report-only native extension; every old math/record/limit statement remains identical after inverse.
needle=' MetalContext gpu;std::ofstream shader(std::string(argv[1])+".metal");shader<<gpu.SoleShaderProgramSource();';assert (s/test).read_text().count(needle)==1
change(test,needle,needle+'''
 std::ofstream capabilities(std::string(argv[1])+".prefill-pipelines.tsv");
 for(const auto& p:gpu.SolePrefillPipelineInfo())capabilities<<p.name<<"\\t"<<p.cooperative<<"\\t"<<p.threadWidth<<"\\t"<<p.maxThreads<<"\\t"<<p.staticThreadgroupBytes<<"\\t"<<p.deviceMaxThreadgroupBytes<<"\\n";
 need(bool(capabilities),"Cannot save actual prefill pipeline capabilities");''')
append('cmake/GarnetMetal.cmake','''
option(GARNET_METAL_SOLE_COOPERATIVE_PREFILL "Cooperative physical-order sole BF16 prefill loads with padded logical right tiles" OFF)
if(GARNET_METAL_SOLE_COOPERATIVE_PREFILL)
 if(NOT GARNET_METAL_SOLE_INTERLEAVED_MLP OR NOT GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL)
  message(FATAL_ERROR "Cooperative sole prefill requires sole MLP storage and direct prefill")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_SOLE_COOPERATIVE_PREFILL=1)
endif()
''')
helper='''inline ulong2 sole_cooperative_coordinate(ulong i,ulong tileK){return ulong2((i/4)%tileK,(i/(tileK*4))*4+i%4);}
inline ulong sole_cooperative_slot(ulong k,ulong column,ulong tileN){return k*(tileN+4)+column;}
'''
original=(old/'src/backends/metal/direct_interleaved_prefill_shader.h').read_text().split('R"DIRECTMLP(',1)[1].split(')DIRECTMLP"',1)[0];prefix=original[:original.index('kernel void')];fragment=prefix+helper;proof={}
shapes={'sole_direct_bf16_gemm':(32,32,128,'32*32'),'sole_direct_gemm_16_64_64':(64,64,256,'4096'),'sole_direct_gemm_32_64_32':(64,32,256,'2048')}
for name,item in a['kernels'].items():
 code=item['source'];new=code;records=[]
 def sc(x,z,count=1):
  global new
  assert new.count(x)==count,(name,x,new.count(x));new=new.replace(x,z);records.append(dict(before=x,after=z,count=count))
 if name in shapes:
  n,k,threads,array=shapes[name];sc('right['+array+']',f'right[{k}*({n}+4)]')
  before=f'ulong k=base+i/{n},column=colBase+i%{n};right[i]='
  after=f'ulong2 coordinate=sole_cooperative_coordinate(i,{k});ulong k=base+coordinate.x,column=colBase+coordinate.y;right[sole_cooperative_slot(coordinate.x,coordinate.y,{n})]=';sc(before,after)
  sc(f'simdgroup_load(r,right+offset*{n}+simd*8,{n})',f'simdgroup_load(r,right+offset*({n}+4)+simd*8,{n}+4)')
 else:
  sc('right[4096]','right[64*(64+4)]')
  sc('ulong k=base+i/p.tileN,column=colBase+i%p.tileN;right[i]=','ulong2 coordinate=sole_cooperative_coordinate(i,p.tileK);ulong k=base+coordinate.x,column=colBase+coordinate.y;right[sole_cooperative_slot(coordinate.x,coordinate.y,p.tileN)]=')
  sc('=right[i];','=right[sole_cooperative_slot(i/p.tileN,i%p.tileN,p.tileN)];')
 inverse=new
 for op in reversed(records):assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
 assert inverse==code;fragment+=new+'\n';proof[name]=dict(original_source=code,new_source=new,original_sha256=hashlib.sha256(code.encode()).hexdigest(),new_sha256=hashlib.sha256(new.encode()).hexdigest(),operations=records,all_original_left_float_MAC_store_barrier_geometry_order_inverse_exact=True)
# Probe parameter declaration from original prefix between third kernel and probe, no floating math.
needle='kernel void sole_direct_tile_read';fragment=fragment.replace(needle,'struct SoleDirectTileParams {ulong rows,inner,first,count,tileN,tileK;};\n'+needle)
(s/'src/backends/metal/cooperative_sole_prefill_shader.h').write_text('// SPDX-License-Identifier: Apache-2.0\n#pragma once\nnamespace Garnet {\ninline constexpr const char* kCooperativeSolePrefillShader=R"COOPMLP(\n'+fragment+')COOPMLP";\n}\n')
for rel,records in ops.items():
 t=(s/rel).read_text()
 for op in reversed(records):
  if op.get('append'):assert t.endswith(op['after']);t=t[:-len(op['after'])]
  else:assert t.count(op['after'])==op['count'];t=t.replace(op['after'],op['before'])
 assert t==(old/rel).read_text(),rel
manifest={str(p.relative_to(s)):sha(p)for p in sorted(s.rglob('*'))if p.is_file()};assert len(manifest)==1724 and all(manifest[k]==v for k,v in seed.items()if k not in ops)
pattern=r'R"([A-Za-z_]+)\((.*?)\)\1"';assert re.findall(pattern,(s/m).read_text(),re.S)==re.findall(pattern,(old/m).read_text(),re.S)
for name,value in [('seed-source-manifest.json',seed),('correctness-source-manifest.json',manifest),('source-transformations.json',ops),('cooperative-kernel-inverse-proof.json',proof)]: (w/name).write_text(json.dumps(value,indent=2)+'\n')
(w/'source-inverse-proof.json').write_text(json.dumps(dict(seed_files=1723,current_files=1724,changed_or_added=sorted(k for k,v in manifest.items()if seed.get(k)!=v),all_original_MAIN_primary_decode_dual_scalar_vector_graph_runtime_session_xModel_image_CUDA_engine_unchanged=True,all_original_nine_native_mathematics_limits_and_complete_records_unchanged=True,existing_context_test_report_only_changes_inverse_exact=True,original_three_prefill_float_MAC_barrier_store_order_inverse_exact_after_explicit_integer_shared_layout_normalization=True,cooperative_serving_probe_use_same_coordinate_and_slot_helpers=True),indent=2)+'\n');shutil.copytree(b/'work/worldsense-sole-dual-decode/build-profile',w/'build-profile');print('WS67 source1724/sixdelta, explicit integer/shared-layout inverse proof complete; no build/model/performance yet')
