"""Prepare WS43 full-primary seed and bounded value16 fold, before execution."""
from pathlib import Path
import ast,hashlib,json,shutil
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');old=r/'Garnet';s=r/'out/research/worldsense-primary-value16/source'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not s.exists()
assert sha(w/'native-design-before-implementation.txt')==json.loads((w/'design-identity.json').read_text())['design_sha256']
manifest=json.loads((w.parent/'worldsense-primary-u32-integration/canonical-source-before-build.json').read_text())
for rel,h in manifest.items():
    assert sha(old/rel)==h,rel
    p=s/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/rel,p)
(w/'seed-source-manifest.json').write_text(json.dumps(manifest,indent=2))

p=s/'src/backends/metal/primary_block_shader.h';t=p.read_text();start=t.index('inline ushort4 primary_quad_kword(');end=t.index('\ninline PrimaryQuad primary_quad(',start);legacy=t[start:end]
candidate='''inline ushort4 primary_quad_kword(uint sm,ushort tags,uint base) {
    ushort4 signMant=ushort4(as_type<uchar4>(sm));
    ushort4 exponents=ushort4(tags,tags>>4,tags>>8,tags>>12)&ushort(15);
    ushort4 placed=(signMant*ushort(257))&ushort(0x807f);
    return placed|((exponents+ushort(base))<<ushort(7));
}'''
replacement=')PRIMARY"\n#ifdef GARNET_METAL_PRIMARY_VALUE16_FOLD\nR"PRIMARY('+candidate+')PRIMARY"\n#else\nR"PRIMARY('+legacy+')PRIMARY"\n#endif\nR"PRIMARY('
p.write_text(t[:start]+replacement+t[end:])
(w/'legacy-helper.txt').write_text(legacy);(w/'candidate-helper.txt').write_text(candidate)
(w/'reference-shader.metal').write_text(t.split('R"PRIMARY(',1)[1].rsplit(')PRIMARY"',1)[0])

def replace(rel,a,b):
    p=s/rel;t=p.read_text();assert t.count(a)==1,(rel,a);p.write_text(t.replace(a,b,1))
replace('cmake/GarnetMetal.cmake','option(GARNET_METAL_PRIMARY_TEST_FAULTS "Correctness-only primary weight allocation failure" OFF)','''option(GARNET_METAL_PRIMARY_TEST_FAULTS "Correctness-only primary weight allocation failure" OFF)
option(GARNET_METAL_PRIMARY_VALUE16_FOLD "Bounded ushort compact-value reconstruction and sign-mantissa fold" OFF)
if(GARNET_METAL_PRIMARY_VALUE16_FOLD)
 if(NOT GARNET_METAL_PRIMARY_BF16_BLOCKS)
  message(FATAL_ERROR "Value16 fold requires primary BF16 weights")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_PRIMARY_VALUE16_FOLD=1)
endif()''')
replace('src/entry/metal_main.cpp','#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS\n        info+="; lazy_resident_defaults";','#ifdef GARNET_METAL_PRIMARY_VALUE16_FOLD\n        info+="; primary_value16_fold";\n#endif\n#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS\n        info+="; lazy_resident_defaults";')
replace('test2026/metal/primary_storage_test.cpp','#include <vector>','#include <vector>\n#include <algorithm>')
replace('test2026/metal/primary_storage_test.cpp',' MetalContext gpu;auto initial=MetalContext::PrimaryStats();',''' MetalContext gpu;auto initial=MetalContext::PrimaryStats();
 {
  // Every valid finite exponent base and every sign/mantissa byte visits all
  // sixteen tags across eight entirely compact blocks. Read bits, not floats.
  constexpr size_t bases=255,blocksPerBase=8,wordsPerBlock=512,inner=128;
  std::vector<uint16_t> bits(bases*blocksPerBase*wordsPerBlock);
  for(size_t base=0;base<bases;++base)for(size_t block=0;block<blocksPerBase;++block)for(size_t i=0;i<wordsPerBlock;++i) {
   const unsigned sm=unsigned(i%256),tag=(sm+unsigned(2*block+i/256))%16;
   const unsigned exponent=std::min(254u,unsigned(base)+tag);
   bits[(base*blocksPerBase+block)*wordsPerBlock+i]=uint16_t((exponent<<7)|(sm&127u)|((sm&128u)<<8));
  }
  auto compact=gpu.UploadBF16(bits.data(),bits.size()*2,bits.size()/inner,inner);
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
  need(compact.PrimaryReady(),"Compact finite-pattern case was not packed");
  auto info=MetalContext::WeightBufferInfo(compact);
  need(info.size()==2&&info[0].lengthBytes==bases*blocksPerBase*768&&info[1].lengthBytes==bases*blocksPerBase*4,"Exhaustive pattern did not remain entirely compact");
  auto scalar=gpu.PrimaryReadWords(compact,0,bits.size()),vector=gpu.PrimaryReadVectorWords(compact,0,bits.size());
  need(scalar==vector,"Compact base/tag/signmant scalar/vector mismatch");
  for(size_t i=0;i<bits.size();++i)need(vector[i]==uint32_t(bits[i])<<16,"Compact finite-pattern GPU bits changed");
  std::cout<<"Compact base/tag/signmant1044480words BOTH GPU readers exact PASS"<<std::endl;
#else
  need(!compact.PrimaryReady(),"Disabled primary compact case was packed");
#endif
 }
 need(MetalContext::PrimaryStats().activeWeights==initial.activeWeights&&MetalContext::PrimaryStats().allocatedBytes==initial.allocatedBytes,"Compact finite-pattern owner leaked");''')

seed=w.parent/'worldsense-primary-u32'
for rel in ['build.py','build_fault.py','build_corpus.py','primary_corpus_test.cpp','full_correctness.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','make_resident_fixtures.py','wait_cantor.py']:
    (w/rel).write_text((seed/rel).read_text().replace('worldsense-primary-u32','worldsense-primary-value16'))
for name in ['fixture-1-8','fixture-2-64']:shutil.copytree(seed/name,w/name)
for name in ['build.py','build_fault.py']:
    p=w/name;t=p.read_text();needle="'-DGARNET_METAL_CACHED_MATRIX_METADATA=OFF',";assert t.count(needle)==1
    value="'-DGARNET_METAL_PRIMARY_VALUE16_FOLD='+mode.upper(),"if name=='build.py'else"'-DGARNET_METAL_PRIMARY_VALUE16_FOLD=ON',"
    p.write_text(t.replace(needle,needle+value,1))
p=w/'api_state_guard.py';t=p.read_text();needle="assert ('primary_u32_metadata' in call.get('backend',''))==bool(expected);";assert t.count(needle)==1
p.write_text(t.replace(needle,needle+"assert ('primary_value16_fold' in call.get('backend',''))==bool(expected);",1))
(w/'dump_shader.cpp').write_text('#include "primary_block_shader.h"\n#include <iostream>\nint main(){std::cout<<Garnet::kPrimaryBlockShader;}\n')
changed=[rel for rel,h in manifest.items()if sha(s/rel)!=h];assert len(changed)==4,changed
cuda=[rel for rel in manifest if 'cuda'in rel.lower()or Path(rel).suffix=='.cu'];assert all(sha(s/rel)==manifest[rel]for rel in cuda)
assert(s/'src/backends/metal/metal_context.mm').read_bytes()==(old/'src/backends/metal/metal_context.mm').read_bytes()
assert(s/'src/backends/metal/primary_block_codec.h').read_bytes()==(old/'src/backends/metal/primary_block_codec.h').read_bytes()
(w/'implementation-audit.json').write_text(json.dumps(dict(changed_files=changed,source_files=len(manifest),CUDA_unchanged_files=len(cuda),native_upload_dispatch_compute_and_CPU_codec_unchanged=True,only_shader_helper_selected_by_flag=True,scope='Static invariants only; materialized shader/source and actualGPU proof still required'),indent=2))
(w/'correctness-source-manifest.json').write_text(json.dumps({rel:sha(s/rel)for rel in sorted(manifest)},indent=2))
for p in w.glob('*.py'):ast.parse(p.read_text(),str(p))
print('WS43 four-file value16fold delta prepared; not compiled/GPUtested yet',flush=True)
