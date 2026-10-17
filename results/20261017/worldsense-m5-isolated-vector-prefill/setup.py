from pathlib import Path
import json,hashlib,shutil,ast
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');old=w.parent/'worldsense-vector-prefill';before=r/'out/research/worldsense-vector-prefill/source';s=r/'out/research/worldsense-isolated-vector-prefill/source'
assert not s.exists()
assert hashlib.sha256((w/'native-design-before-implementation.txt').read_bytes()).hexdigest()==json.loads((w/'design-identity.json').read_text())['design_sha256']
seed=json.loads((old/'starting-source-manifest.json').read_text());assert len(seed)==1716
for rel,h in seed.items():
 p=before/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==h,rel
 q=s/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
(w/'ws47-seed-source-manifest.json').write_text(json.dumps(seed,indent=2)+'\n')
canonical=json.loads((old/'seed-source-manifest.json').read_text());(w/'seed-source-manifest.json').write_text(json.dumps(canonical,indent=2)+'\n')
def replace(rel,a,b):
 p=s/rel;t=p.read_text();assert t.count(a)==1,(rel,a,t.count(a));p.write_text(t.replace(a,b,1))
rel='src/backends/metal/primary_vector_prefill_shader.h';p=s/rel;t=p.read_text();start=t.index('// Admission:');end=t.index('kernel void primary_vector_bf16_gemm',start)
helper='''// Exact compile-time geometries. Aligned float4 bit stores write the
// same contiguous FP32 right tile; all original barriers/math stay outside.
template<uint tileColumns,uint tileK,uint threads>
inline void primary_prefill_tile_fill(device const uchar* payload,device const uint* metadata,
 threadgroup float* right,uint colBase,uint kBase,uint columns,uint inner,uint tid) {
 for(uint index=tid;index<(tileColumns/4)*(tileK/4);index+=threads) {
  uint localK=(index/(tileColumns/4))*4,localColumn=(index%(tileColumns/4))*4;
  uint row=colBase+localColumn,k=kBase+localK;
  PrimaryQuad quad;
  if(row<columns&&k<inner)quad=primary_quad(payload,metadata,row,k,inner);
  else {quad.k0=ushort4(0);quad.k1=ushort4(0);quad.k2=ushort4(0);quad.k3=ushort4(0);}
  ushort4 words[4]={quad.k0,quad.k1,quad.k2,quad.k3};
  for(uint j=0;j<4;++j)
   *(threadgroup float4*)(right+(localK+j)*tileColumns+localColumn)=as_type<float4>(uint4(words[j])<<16);
 }
}
'''
t=t[:start]+helper+t[end:]
for c,k,threads in [(32,32,128),(64,64,256),(64,32,256)]:
 a=f'primary_prefill_tile_fill(payload,metadata,right,uint(colBase),uint(base),uint(p.columns),uint(p.inner),{c},{k},tid,{threads});';assert t.count(a)==1
 b=f'primary_prefill_tile_fill<{c},{k},{threads}>(payload,metadata,right,uint(colBase),uint(base),uint(p.columns),uint(p.inner),tid);';t=t.replace(a,b,1)
for a,b in [
 ('threadgroup float left[16*32],right[32*32],result[16*32];','threadgroup float left[16*32],result[16*32];\n    threadgroup float4 rightAligned[256];threadgroup float* right=(threadgroup float*)rightAligned;'),
 ('threadgroup float left[1024],right[4096],result[1024];','threadgroup float left[1024],result[1024];\nthreadgroup float4 rightAligned[1024];threadgroup float* right=(threadgroup float*)rightAligned;'),
 ('threadgroup float left[1024],right[2048],result[2048];','threadgroup float left[1024],result[2048];\nthreadgroup float4 rightAligned[512];threadgroup float* right=(threadgroup float*)rightAligned;'),
 (' threadgroup float right[4096];',' threadgroup float4 rightAligned[1024];threadgroup float* right=(threadgroup float*)rightAligned;')]:
 assert t.count(a)==1,a;t=t.replace(a,b,1)
a=' primary_prefill_tile_fill(payload,metadata,right,rowBase,kBase,p.columns,p.inner,p.tileColumns,p.tileK,tid,p.threads);';assert t.count(a)==1
t=t.replace(a,''' if(p.tileColumns==32)primary_prefill_tile_fill<32,32,128>(payload,metadata,right,rowBase,kBase,p.columns,p.inner,tid);
 else if(p.tileK==32)primary_prefill_tile_fill<64,32,256>(payload,metadata,right,rowBase,kBase,p.columns,p.inner,tid);
 else primary_prefill_tile_fill<64,64,256>(payload,metadata,right,rowBase,kBase,p.columns,p.inner,tid);''',1);p.write_text(t)
replace('src/backends/metal/metal_context.mm','    id<MTLLibrary> library;','    id<MTLLibrary> library;\n    id<MTLLibrary> vectorPrefillLibrary=nil;\n    std::string mainProgram,prefillProgram;')
a='        source=[source stringByAppendingString:[NSString stringWithUTF8String:kPrimaryVectorPrefillShader]];'
replace('src/backends/metal/metal_context.mm',a,'#ifndef GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY\n'+a+'\n#endif')
a='        impl_->kernels->library=library;'
replace('src/backends/metal/metal_context.mm',a,a+'''
        impl_->kernels->mainProgram=source.UTF8String;
#ifdef GARNET_METAL_PRIMARY_VECTOR_PREFILL
        id<MTLLibrary> prefillLibrary=library;
#ifdef GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY
        NSString* prefillSource=[source stringByAppendingString:[NSString stringWithUTF8String:kPrimaryVectorPrefillShader]];
        auto isolated=[impl_->device newLibraryWithSource:prefillSource options:compileOptions error:&error];
        if(!isolated)throw std::runtime_error(std::string(error.localizedDescription.UTF8String));
        impl_->kernels->vectorPrefillLibrary=isolated;
        impl_->kernels->prefillProgram=prefillSource.UTF8String;
        prefillLibrary=isolated;
#endif
#endif''')
for name in ['primary_vector_bf16_gemm','primary_vector_gemm_16_64_64','primary_vector_gemm_32_64_32','primary_vector_prefill_expand']:
 replace('src/backends/metal/metal_context.mm','[library newFunctionWithName:@"'+name+'"]','[prefillLibrary newFunctionWithName:@"'+name+'"]')
a='    std::vector<uint32_t> PrimaryReadPrefillTileWords(const MetalBF16Weight&,size_t firstRow,size_t rows,size_t tileColumns,size_t tileK);'
replace('src/backends/metal/metal_context.h',a,a+'\n    std::vector<std::string> ShaderProgramSources() const;')
with(s/'src/backends/metal/metal_context.mm').open('a')as f:f.write('''
namespace Garnet {
std::vector<std::string> MetalContext::ShaderProgramSources() const {
 if(!impl_->kernels)throw std::logic_error("Shader program sources require ready kernels");
 return {impl_->kernels->mainProgram,impl_->kernels->prefillProgram};
}
}
''')
a='        info+="; primary_vector_prefill";'
replace('src/entry/metal_main.cpp',a,a+'\n#ifdef GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY\n        info+="; primary_prefill_isolated_library";\n#endif')
with(s/'cmake/GarnetMetal.cmake').open('a')as f:f.write('''
option(GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY "Keep the legacy main Metal program invariant and isolate vector prefill" OFF)
if(GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY)
 if(NOT GARNET_METAL_PRIMARY_VECTOR_PREFILL OR NOT GARNET_METAL_PRIMARY_BF16_BLOCKS)
  message(FATAL_ERROR "Isolated primary prefill requires vector prefill and primary weights")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY=1)
endif()
''')
replace('test2026/metal/primary_storage_test.cpp','#include <iostream>','#include <iostream>\n#include <fstream>')
a=' MetalContext gpu;auto initial=MetalContext::PrimaryStats();'
replace('test2026/metal/primary_storage_test.cpp',a,a+'''
 {
  auto programs=gpu.ShaderProgramSources();need(programs.size()==2&&!programs[0].empty(),"Actual main shader source missing");
  for(size_t i=0;i<2;++i){std::ofstream out(i?"prefill_shader_program.metal":"main_shader_program.metal",std::ios::binary);out<<programs[i];need(bool(out),"Actual shader program dump failed");}
#ifdef GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY
  need(!programs[1].empty()&&programs[0].find("kernel void primary_vector_bf16_gemm")==std::string::npos,"Legacy main program contains vector prefill");
#else
  need(programs[1].empty(),"Unexpected separate prefill program");
#endif
  std::cout<<"Actual shader program bytes main="<<programs[0].size()<<" prefill="<<programs[1].size()<<std::endl;
 }
''')
manifest={str(p.relative_to(s)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(s.rglob('*'))if p.is_file()};changes=[k for k,h in manifest.items()if canonical.get(k)!=h]
assert len(manifest)==1716 and len(changes)==6
for rel,h in canonical.items():
 if '/cuda/'in rel or rel in ['src/backends/metal/primary_block_shader.h','src/backends/metal/body_shape_shader.h','src/backends/metal/primary_block_codec.h','runtime/metal_serving.py','test2026/metal/primary_projection_test.cpp']:assert manifest[rel]==h
for name in ['correctness-source-manifest.json','starting-source-manifest.json']:(w/name).write_text(json.dumps(manifest,indent=2)+'\n')
(w/'implementation-audit.json').write_text(json.dumps({'source_files':1716,'changed_files':changes,'original_shaders_codec_wrapper_projection_tests_CUDA_unchanged':True,'scope':'Newjoint isolated-library/static-geometry/alignedfloat4 implementation. No build/native/program/GPU/model/performance evidence yet.'},indent=2)+'\n')
shutil.copytree(old/'build-profile',w/'build-profile');shutil.copy2(old/'build-profile-identity.json',w/'build-profile-identity.json')
for name in ['build.py','wait_cantor.py','restore_normal.py']:
 t=(old/name).read_text().replace('worldsense-vector-prefill','worldsense-isolated-vector-prefill')
 if name=='build.py':
  a="'-DGARNET_METAL_PRIMARY_VECTOR_PREFILL='+vector,";assert t.count(a)==1;t=t.replace(a,a+"'-DGARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY='+vector,",1)
 ast.parse(t);(w/name).write_text(t)
for name in ['before-libgarnet.dylib','cases.json']:shutil.copy2(old/name,w/name)
print('WS48 source1716/sixoverlay frozen; original shader/codec/wrapper/CUDA untouched. Newprogram proof, isolatedrouting/staticfloat4 tileloader implemented before firstbuild.')
