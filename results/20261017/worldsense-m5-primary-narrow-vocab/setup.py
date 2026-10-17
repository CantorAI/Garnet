"""Apply the predeclared WS41 semantic storage policy to an isolated source."""
from pathlib import Path
import ast, hashlib, json, re, shutil
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');base=r/'Garnet'
s=r/'out/research/worldsense-primary-narrow-vocab/source'
assert not s.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(w/'native-design-before-implementation.txt')==json.loads((w/'design-identity.json').read_text())['design_sha256']
manifest=json.loads((w.parent/'worldsense-primary-u32-integration/canonical-source-before-build.json').read_text())
for rel,digest in manifest.items():
    assert sha(base/rel)==digest,rel
    t=s/rel;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/rel,t)
(w/'seed-source-manifest.json').write_text(json.dumps(manifest,indent=2))

(s/'src/backends/metal/primary_storage_policy.h').write_text('''// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <string>
namespace PrimaryPolicy {
inline bool EndsWith(const std::string& value,const std::string& suffix) {
    return value.size()>=suffix.size()&&value.compare(value.size()-suffix.size(),suffix.size(),suffix)==0;
}
inline bool AllowPrimary(const std::string& name,size_t rows,size_t columns,bool rawNarrowVocabulary) {
    const bool vocabulary=name=="lm_head.weight"||EndsWith(name,".lm_head.weight")||EndsWith(name,".embed_tokens.weight");
    return !(rawNarrowVocabulary&&vocabulary&&rows>65536&&columns>0&&columns<=2048);
}
}
''')

def replace(path,old,new):
    p=s/path;t=p.read_text();assert t.count(old)==1,(path,old,t.count(old));p.write_text(t.replace(old,new,1))
replace('src/backends/metal/metal_context.h','bool visionPrefill=false);','bool visionPrefill=false,bool allowPrimary=true);')
replace('src/backends/metal/metal_context.mm','size_t columns,bool floatPrefill,bool visionPrefill) {','size_t columns,bool floatPrefill,bool visionPrefill,bool allowPrimary) {')
replace('src/backends/metal/metal_context.mm','if(!floatPrefill&&!visionPrefill&&Primary32::ShapeFits(rows,columns,bytes))','if(allowPrimary&&!floatPrefill&&!visionPrefill&&Primary32::ShapeFits(rows,columns,bytes))')
replace('src/backends/metal/compiled_graph.cpp','#include "compiled_graph.h"','#include "compiled_graph.h"\n#include "primary_storage_policy.h"')
replace('src/backends/metal/compiled_graph.cpp','    auto weight=gpu_->UploadBF16(data,entry->dataSize,size_t(entry->shape[0]),size_t(entry->shape[1]),floatPrefill,visionPrefill);','''    bool allowPrimary=true;
#ifdef GARNET_METAL_PRIMARY_RAW_NARROW_VOCAB
    allowPrimary=PrimaryPolicy::AllowPrimary(name,size_t(entry->shape[0]),size_t(entry->shape[1]),true);
#endif
    auto weight=gpu_->UploadBF16(data,entry->dataSize,size_t(entry->shape[0]),size_t(entry->shape[1]),floatPrefill,visionPrefill,allowPrimary);''')
replace('src/entry/metal_main.cpp','#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS\n        info+="; lazy_resident_defaults";','#ifdef GARNET_METAL_PRIMARY_RAW_NARROW_VOCAB\n        info+="; primary_raw_narrow_vocab";\n#endif\n#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS\n        info+="; lazy_resident_defaults";')
replace('cmake/GarnetMetal.cmake','option(GARNET_METAL_PRIMARY_TEST_FAULTS "Correctness-only primary weight allocation failure" OFF)','''option(GARNET_METAL_PRIMARY_TEST_FAULTS "Correctness-only primary weight allocation failure" OFF)
option(GARNET_METAL_PRIMARY_RAW_NARROW_VOCAB "Preserve raw BF16 storage for semantic narrow vocabulary weights" OFF)
if(GARNET_METAL_PRIMARY_RAW_NARROW_VOCAB)
 if(NOT GARNET_METAL_PRIMARY_BF16_BLOCKS)
  message(FATAL_ERROR "Narrow vocabulary storage policy requires primary weights")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_PRIMARY_RAW_NARROW_VOCAB=1)
endif()''')
replace('test2026/metal/primary_storage_test.cpp','#include "primary_32_contract.h"','#include "primary_32_contract.h"\n#include "primary_storage_policy.h"')
replace('test2026/metal/primary_storage_test.cpp',' // Pure integer boundary checks require no multi-GiB allocation.',''' // Semantic storage boundaries: matrix geometry alone never defines vocabulary.
 for(const std::string name:{"lm_head.weight","model.embed_tokens.weight","model.language_model.embed_tokens.weight","model.lm_head.weight"}) {
  need(!PrimaryPolicy::AllowPrimary(name,151936,2048,true),"Narrow vocabulary was packed");
  need(PrimaryPolicy::AllowPrimary(name,151936,2048,false),"Disabled semantic policy changed admission");
  need(PrimaryPolicy::AllowPrimary(name,248320,2560,true),"Wide vocabulary unexpectedly excluded");
  need(PrimaryPolicy::AllowPrimary(name,65536,2048,true),"Vocabulary row boundary changed");
  need(PrimaryPolicy::AllowPrimary(name,65540,2049,true),"Vocabulary column boundary changed");
  need(PrimaryPolicy::AllowPrimary(name,65540,0,true),"Zero width should retain ordinary shape validation");
 }
 need(PrimaryPolicy::AllowPrimary("model.layers.0.self_attn.q_proj.weight",151936,2048,true),"Ordinary matrix mistaken for vocabulary");
 need(PrimaryPolicy::AllowPrimary("not_lm_head.weight",151936,2048,true),"Unrecognized role changed admission");
 // Pure integer boundary checks require no multi-GiB allocation.''')
replace('test2026/metal/primary_storage_test.cpp',' MetalContext gpu;auto initial=MetalContext::PrimaryStats();',''' MetalContext gpu;auto initial=MetalContext::PrimaryStats();
 {
  const size_t n=260,k=2048;std::vector<uint16_t> bits(n*k);
  for(size_t i=0;i<bits.size();++i)bits[i]=uint16_t(((120u+unsigned(i%4))<<7)|unsigned(i%127)|unsigned((i&1)<<15));
  auto packed=gpu.UploadBF16(bits.data(),bits.size()*2,n,k);
  auto raw=gpu.UploadBF16(bits.data(),bits.size()*2,n,k,false,false,false);
  need(!raw.PrimaryReady(),"Explicit raw storage request was ignored");
  auto info=MetalContext::WeightBufferInfo(raw);
  need(info.size()==1&&info[0].representation=="bf16"&&info[0].lengthBytes==bits.size()*2,"Raw storage duplicated/changed");
  for(size_t batch:{1,2,16}) {
   std::vector<float> values(batch*k);for(size_t i=0;i<values.size();++i)values[i]=float(int(i%29)-14)*.017f;
   auto input=gpu.Upload(values,batch,k),a=gpu.Allocate(batch,n),b=gpu.Allocate(batch,n);
   gpu.Begin();gpu.EncodeBF16MatVec(input,packed,a);gpu.SubmitAndWait();auto expected=gpu.Download(a);
   gpu.Begin();gpu.EncodeBF16MatVec(input,raw,b);gpu.SubmitAndWait();auto actual=gpu.Download(b);
   need(expected.size()==actual.size()&&!std::memcmp(expected.data(),actual.data(),actual.size()*4),"Explicit raw/packed complete projection FP32 words differ");
  }
 }
 need(MetalContext::PrimaryStats().activeWeights==initial.activeWeights,"Raw-policy test owner leak");''')

# Reuse the original complete correctness drivers, never old output artifacts.
drivers=['build.py','build_fault.py','build_corpus.py','primary_corpus_test.cpp','full_correctness.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','make_resident_fixtures.py','wait_cantor.py']
for name in drivers:
    (w/name).write_text((w.parent/'worldsense-primary-u32'/name).read_text().replace('worldsense-primary-u32','worldsense-primary-narrow-vocab'))
for name in ['fixture-1-8','fixture-2-64']:shutil.copytree(w.parent/'worldsense-primary-u32'/name,w/name)
for name in ['build.py','build_fault.py']:
    p=w/name;t=p.read_text();needle="'-DGARNET_METAL_CACHED_MATRIX_METADATA=OFF',";assert t.count(needle)==1
    value="'-DGARNET_METAL_PRIMARY_RAW_NARROW_VOCAB='+mode.upper(),"if name=='build.py'else"'-DGARNET_METAL_PRIMARY_RAW_NARROW_VOCAB=ON',"
    p.write_text(t.replace(needle,needle+value,1))
p=w/'primary_corpus_test.cpp';t=p.read_text().replace('#include "metal_context.h"','#include "metal_context.h"\n#include "primary_storage_policy.h"',1)
t=t.replace('auto w=gpu.UploadBF16(data,entry->dataSize,rows,inner);bool expected=rows>256&&rows%4==0&&inner%128==0;', 'bool allow=PrimaryPolicy::AllowPrimary(name,rows,inner,true);auto w=gpu.UploadBF16(data,entry->dataSize,rows,inner,false,false,allow);bool expected=allow&&rows>256&&rows%4==0&&inner%128==0;')
t=t.replace('packedMatrices==398&&rawMatrices==48','packedMatrices==397&&rawMatrices==49').replace('wordsChecked==5921308672ULL','wordsChecked==5610143744ULL').replace('primaryBytes==9033678080ULL','primaryBytes==9182203392ULL').replace('Full398packed/48raw','Full397packed/49raw').replace('Raw small fallback changed','Raw fallback changed')
t=t.replace('{{"model",model},{"tensor",name}', '{{"semantic_primary_allowed",allow},{"model",model},{"tensor",name}')
p.write_text(t)
p=w/'api_state_guard.py';t=p.read_text();t=t.replace("assert ('primary_u32_metadata' in call.get('backend',''))==bool(expected);", "assert ('primary_u32_metadata' in call.get('backend',''))==bool(expected);assert ('primary_raw_narrow_vocab' in call.get('backend',''))==bool(expected);")
t=t.replace("assert all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded'])", "assert all(delta[k]>0 for k in ['decode_encoded','prefill_encoded'])\n  assert (delta['head_encoded']==0)if name=='Qwen3-1.7B'else(delta['head_encoded']>0)")
p.write_text(t)
p=w/'full_correctness.py';t=p.read_text().replace('==5921308672','==5610143744').replace('ALL398PACKED','ALL397PACKED').replace('all398actualpacked_weights','all397actualpacked_weights');p.write_text(t)

changed=[rel for rel in manifest if sha(s/rel)!=manifest[rel]]+['src/backends/metal/primary_storage_policy.h']
assert len(changed)==7,changed
cuda=[rel for rel in manifest if 'cuda'in rel.lower()or Path(rel).suffix=='.cu']
assert all(sha(s/rel)==manifest[rel]for rel in cuda)
assert (s/'src/backends/metal/primary_block_shader.h').read_bytes()==(base/'src/backends/metal/primary_block_shader.h').read_bytes()
frozen={str(p.relative_to(s)):sha(p)for p in sorted(s.rglob('*'))if p.is_file()}
(w/'correctness-source-manifest.json').write_text(json.dumps(frozen,indent=2))
(w/'implementation-audit.json').write_text(json.dumps(dict(changed_files=changed,seed_frozen_files=len(manifest),candidate_frozen_files=len(frozen),CUDA_files_unchanged=len(cuda),entire_primary_shader_unchanged=True,scope='Static delta, not numerical or performance proof'),indent=2))
for p in w.glob('*.py'):ast.parse(p.read_text(),str(p))
print('WS41 seven-file semantic policy prepared; source frozen; no build/test yet',flush=True)
