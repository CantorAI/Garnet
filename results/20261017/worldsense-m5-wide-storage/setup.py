"""Apply the predeclared checkpoint-scoped storage policy to isolated f21."""
from pathlib import Path
import ast,hashlib,json,shutil
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');base=r/'Garnet';s=r/'out/research/worldsense-wide-storage/source'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not s.exists()
assert sha(w/'native-design-before-implementation.txt')==json.loads((w/'design-identity.json').read_text())['design_sha256']
manifest=json.loads((w.parent/'worldsense-primary-u32-integration/canonical-source-before-build.json').read_text())
for rel,h in manifest.items():
    assert sha(base/rel)==h,rel
    p=s/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/rel,p)
(w/'seed-source-manifest.json').write_text(json.dumps(manifest,indent=2))
(s/'src/backends/metal/wide_checkpoint_policy.h').write_text('''// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstdint>
#include <vector>
namespace WideStorage {
inline bool ConsistentWide(const std::vector<uint64_t>& widths) {
    if(widths.empty()||widths.front()<=2048)return false;
    for(auto width:widths)if(width!=widths.front())return false;
    return true;
}
template<class Index> bool Eligible(const Index& index) {
    std::vector<uint64_t> widths;
    for(const char* name:std::array<const char*,3>{"model.embed_tokens.weight","model.language_model.embed_tokens.weight","lm_head.weight"}) {
        const auto* entry=index.Find(name);if(!entry)continue;
        if(entry->shape.size()!=2||entry->shape[0]<=0||entry->shape[1]<=0)return false;
        widths.push_back(uint64_t(entry->shape[1]));
    }
    return ConsistentWide(widths);
}
}
''')
def replace(rel,a,b):
    p=s/rel;t=p.read_text();assert t.count(a)==1,(rel,a,t.count(a));p.write_text(t.replace(a,b,1))
replace('src/backends/metal/metal_context.h','bool visionPrefill=false);','bool visionPrefill=false,bool allowPrimary=true);')
replace('src/backends/metal/metal_context.mm','size_t columns,bool floatPrefill,bool visionPrefill) {','size_t columns,bool floatPrefill,bool visionPrefill,bool allowPrimary) {')
replace('src/backends/metal/metal_context.mm','if(!floatPrefill&&!visionPrefill&&Primary32::ShapeFits(rows,columns,bytes))','if(allowPrimary&&!floatPrefill&&!visionPrefill&&Primary32::ShapeFits(rows,columns,bytes))')
replace('src/backends/metal/compiled_graph.h','    SafeTensorsIndex index_;','''    SafeTensorsIndex index_;
#ifdef GARNET_METAL_WIDE_CHECKPOINT_STORAGE
    bool wideCheckpoint_=false;
#endif''')
replace('src/backends/metal/compiled_graph.h','std::unordered_map<std::string,SharedWeight>,bool bf16Projections=false);','std::unordered_map<std::string,SharedWeight>,bool bf16Projections=false,const MetalCompiledGraph* sharedSource=nullptr);')
replace('src/backends/metal/compiled_graph.h','    size_t LazyRetiredBytes() const;','    bool WideStorageEligible() const;\n    size_t LazyRetiredBytes() const;')
replace('src/backends/metal/compiled_graph.h','        APISET().AddProp("lazy_default_retired_bytes",&MetalCompiledGraph::LazyRetiredBytes);','        APISET().AddProp("wide_storage_eligible",&MetalCompiledGraph::WideStorageEligible);\n        APISET().AddProp("lazy_default_retired_bytes",&MetalCompiledGraph::LazyRetiredBytes);')
replace('src/backends/metal/compiled_graph.cpp','#include "compiled_graph.h"','#include "compiled_graph.h"\n#include "wide_checkpoint_policy.h"')
replace('src/backends/metal/compiled_graph.cpp','    auto weight=gpu_->UploadBF16(data,entry->dataSize,size_t(entry->shape[0]),size_t(entry->shape[1]),floatPrefill,visionPrefill);','''    bool allowPrimary=true;
#ifdef GARNET_METAL_WIDE_CHECKPOINT_STORAGE
    allowPrimary=wideCheckpoint_;
#endif
    auto weight=gpu_->UploadBF16(data,entry->dataSize,size_t(entry->shape[0]),size_t(entry->shape[1]),floatPrefill,visionPrefill,allowPrimary);''')
replace('src/backends/metal/compiled_graph.cpp','void MetalCompiledGraph::Clear() {','''void MetalCompiledGraph::Clear() {
#ifdef GARNET_METAL_WIDE_CHECKPOINT_STORAGE
    wideCheckpoint_=false;
#endif''')
replace('src/backends/metal/compiled_graph.cpp','return BuildInternal(graph,symbols,checkpoint,peer->weightRecords_,peer->bf16Projections_);','return BuildInternal(graph,symbols,checkpoint,peer->weightRecords_,peer->bf16Projections_,peer);')
replace('src/backends/metal/compiled_graph.cpp','std::unordered_map<std::string,SharedWeight> shared,bool bf16Projections) {','std::unordered_map<std::string,SharedWeight> shared,bool bf16Projections,const MetalCompiledGraph* sharedSource) {')
replace('src/backends/metal/compiled_graph.cpp','        std::string error;if(!index_.Open(checkpoint,error))throw std::runtime_error(error);','''        std::string error;if(!index_.Open(checkpoint,error))throw std::runtime_error(error);
#ifdef GARNET_METAL_WIDE_CHECKPOINT_STORAGE
        wideCheckpoint_=WideStorage::Eligible(index_);
        if(sharedSource&&sharedSource->wideCheckpoint_!=wideCheckpoint_)
            throw std::invalid_argument("Shared Metal storage scope mismatch");
#else
        (void)sharedSource;
#endif''')
replace('src/backends/metal/compiled_graph.cpp','void MetalCompiledGraph::RetireResidentDefaults() {\n#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS','''void MetalCompiledGraph::RetireResidentDefaults() {
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
#ifdef GARNET_METAL_WIDE_CHECKPOINT_STORAGE
    if(!wideCheckpoint_)return;
#endif''')
replace('src/backends/metal/compiled_graph.cpp','size_t MetalCompiledGraph::LazyRetiredBytes() const {','''bool MetalCompiledGraph::WideStorageEligible() const {
#ifdef GARNET_METAL_WIDE_CHECKPOINT_STORAGE
    std::lock_guard<std::mutex> lock(mutex_);return ready_&&wideCheckpoint_;
#else
    return false;
#endif
}
size_t MetalCompiledGraph::LazyRetiredBytes() const {''')
replace('src/entry/metal_main.cpp','#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS\n        info+="; lazy_resident_defaults";','#ifdef GARNET_METAL_WIDE_CHECKPOINT_STORAGE\n        info+="; wide_checkpoint_storage";\n#endif\n#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS\n        info+="; lazy_resident_defaults";')
replace('cmake/GarnetMetal.cmake','option(GARNET_METAL_PRIMARY_TEST_FAULTS "Correctness-only primary weight allocation failure" OFF)','''option(GARNET_METAL_PRIMARY_TEST_FAULTS "Correctness-only primary weight allocation failure" OFF)
option(GARNET_METAL_WIDE_CHECKPOINT_STORAGE "Scope primary and lazy default storage to consistent wide checkpoints" OFF)
if(GARNET_METAL_WIDE_CHECKPOINT_STORAGE)
 if(NOT GARNET_METAL_PRIMARY_BF16_BLOCKS OR NOT GARNET_METAL_LAZY_RESIDENT_DEFAULTS)
  message(FATAL_ERROR "Wide checkpoint storage requires primary weights and lazy defaults")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_WIDE_CHECKPOINT_STORAGE=1)
endif()''')
replace('test2026/metal/primary_storage_test.cpp','#include "primary_32_contract.h"','#include "primary_32_contract.h"\n#include "wide_checkpoint_policy.h"\n#include <map>\n#include <string>')
replace('test2026/metal/primary_storage_test.cpp',' // Pure integer boundary checks require no multi-GiB allocation.',''' // Eligibility is checkpoint evidence, not individual weight geometry.
 need(!WideStorage::ConsistentWide({})&&!WideStorage::ConsistentWide({0})&&!WideStorage::ConsistentWide({2048}),"Narrow/unknown width admitted");
 need(WideStorage::ConsistentWide({2049})&&WideStorage::ConsistentWide({2560,2560}),"Wide consistent width rejected");
 need(!WideStorage::ConsistentWide({2048,2560})&&!WideStorage::ConsistentWide({2560,0}),"Conflicting/malformed widths admitted");
 struct Metadata {std::vector<int64_t> shape;};
 struct Index {std::map<std::string,Metadata> entries;const Metadata* Find(const std::string& name)const{auto it=entries.find(name);return it==entries.end()?nullptr:&it->second;}};
 Index index;need(!WideStorage::Eligible(index),"Unknown checkpoint admitted");
 index.entries["model.embed_tokens.weight"]={{10,2048}};need(!WideStorage::Eligible(index),"Narrow checkpoint admitted");
 index.entries["model.embed_tokens.weight"]={{10,2560}};need(WideStorage::Eligible(index),"Wide checkpoint rejected");
 index.entries["lm_head.weight"]={{10,2048}};need(!WideStorage::Eligible(index),"Conflicting canonical metadata admitted");
 index.entries["lm_head.weight"]={{10,2560}};need(WideStorage::Eligible(index),"Consistent canonical metadata rejected");
 for(const std::vector<int64_t> shape:std::vector<std::vector<int64_t>>{{2560},{0,2560},{10,0},{10,-1},{1,1,2560}}){index.entries["lm_head.weight"]={shape};need(!WideStorage::Eligible(index),"Malformed canonical metadata admitted");}
 // Pure integer boundary checks require no multi-GiB allocation.''')
replace('test2026/metal/primary_storage_test.cpp',' MetalContext gpu;auto initial=MetalContext::PrimaryStats();',''' MetalContext gpu;auto initial=MetalContext::PrimaryStats();
 {
  const size_t n=260,k=2048;std::vector<uint16_t> bits(n*k);
  for(size_t i=0;i<bits.size();++i)bits[i]=uint16_t(((120u+unsigned(i%4))<<7)|unsigned(i%127)|unsigned((i&1)<<15));
  auto packed=gpu.UploadBF16(bits.data(),bits.size()*2,n,k),raw=gpu.UploadBF16(bits.data(),bits.size()*2,n,k,false,false,false);
  need(!raw.PrimaryReady(),"Explicit eager raw request ignored");auto info=MetalContext::WeightBufferInfo(raw);
  need(info.size()==1&&info[0].representation=="bf16"&&info[0].lengthBytes==bits.size()*2,"Raw ownership changed/duplicated");
  for(size_t batch:{1,2,16}) {
   std::vector<float> values(batch*k);for(size_t i=0;i<values.size();++i)values[i]=float(int(i%29)-14)*.017f;
   auto input=gpu.Upload(values,batch,k),a=gpu.Allocate(batch,n),b=gpu.Allocate(batch,n);
   gpu.Begin();gpu.EncodeBF16MatVec(input,packed,a);gpu.SubmitAndWait();auto wanted=gpu.Download(a);
   gpu.Begin();gpu.EncodeBF16MatVec(input,raw,b);gpu.SubmitAndWait();auto actual=gpu.Download(b);
   need(wanted.size()==actual.size()&&!std::memcmp(wanted.data(),actual.data(),actual.size()*4),"Scoped raw/packed FP32 projection differs");
  }
 }
 need(MetalContext::PrimaryStats().activeWeights==initial.activeWeights&&MetalContext::PrimaryStats().allocatedBytes==initial.allocatedBytes,"Scoped raw/packed owner leaked");''')

seed=w.parent/'worldsense-primary-u32'
for rel in ['build.py','build_fault.py','build_corpus.py','primary_corpus_test.cpp','full_correctness.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','make_resident_fixtures.py','wait_cantor.py']:
    (w/rel).write_text((seed/rel).read_text().replace('worldsense-primary-u32','worldsense-wide-storage'))
for rel in ['fixture-1-8','fixture-2-64']:shutil.copytree(seed/rel,w/rel)
for name in ['build.py','build_fault.py']:
    p=w/name;t=p.read_text();needle="'-DGARNET_METAL_LAZY_RESIDENT_DEFAULTS=OFF',";assert t.count(needle)==1
    value="'-DGARNET_METAL_LAZY_RESIDENT_DEFAULTS='+mode.upper(),'-DGARNET_METAL_WIDE_CHECKPOINT_STORAGE='+mode.upper(),"if name=='build.py'else"'-DGARNET_METAL_LAZY_RESIDENT_DEFAULTS=ON','-DGARNET_METAL_WIDE_CHECKPOINT_STORAGE=ON',"
    p.write_text(t.replace(needle,value,1))
p=w/'primary_corpus_test.cpp';t=p.read_text().replace('#include "metal_context.h"','#include "metal_context.h"\n#include "wide_checkpoint_policy.h"',1)
t=t.replace('auto w=gpu.UploadBF16(data,entry->dataSize,rows,inner);bool expected=rows>256&&rows%4==0&&inner%128==0;','bool wide=WideStorage::Eligible(*indexes.at(model));need(wide==(model=="Qwen3.5-4B"),"Checkpoint scope differs from plan");auto w=gpu.UploadBF16(data,entry->dataSize,rows,inner,false,false,wide);bool expected=wide&&rows>256&&rows%4==0&&inner%128==0;',1)
t=t.replace('{{"model",model},{"tensor",name}','{{"wide_scope",wide},{"model",model},{"tensor",name}',1).replace('packedMatrices==398&&rawMatrices==48','packedMatrices==201&&rawMatrices==245',1).replace('wordsChecked==5921308672ULL','wordsChecked==4200857600ULL',1).replace('primaryBytes==9033678080ULL','primaryBytes==9850930176ULL',1).replace('"raw_small_matrices",rawMatrices','"raw_fallback_matrices",rawMatrices',1).replace('Full398packed/48raw','Full201packed/245raw').replace('Raw small fallback changed','Scoped raw fallback changed').replace('raw small fallback remains original.','all narrow checkpoint weights remain raw; no primary-reader bit claim for raw buffers.')
p.write_text(t)
p=w/'full_correctness.py';p.write_text(p.read_text().replace('==5921308672','==4200857600').replace('ALL398PACKED','ALL201PACKED').replace('all398actualpacked_weights','all201actualpacked_weights'))
p=w/'api_state_guard.py';t=p.read_text();needle="assert ('primary_u32_metadata' in call.get('backend',''))==bool(expected);";assert t.count(needle)==1
t=t.replace(needle,needle+"assert ('wide_checkpoint_storage' in call.get('backend',''))==bool(expected);assert ('lazy_resident_defaults' in call.get('backend',''))==bool(expected);",1)
t=t.replace("timings=call['timings'];assert timings['lazy_default_retired_bytes']==0 and timings['lazy_default_rehydrated_bytes']==0", "timings=call['timings'];wide=bool(expected)and name=='Qwen3.5-4B'\n assert (timings['lazy_default_retired_bytes']>0)==wide and timings['lazy_default_rehydrated_bytes']==0\n graphs=[layer['graph']for layer in metal_serving._models[name].decoder.layers]+list(metal_serving._models[name].decoder.prefill_graphs.values())\n assert all(bool(graph.wide_storage_eligible)==wide for graph in graphs)",1)
t=t.replace(' if expected:\n  assert stats', ' if expected and wide:\n  assert stats',1)
t=t.replace(" else:assert all(v==0 for v in stats.values()) and all(v==0 for v in delta.values())", " elif expected:\n  assert stats['active_weights']==stats['original_bytes']==stats['payload_bytes']==stats['metadata_bytes']==stats['allocated_bytes']==0 and all(v==0 for v in delta.values())\n else:assert all(v==0 for v in stats.values()) and all(v==0 for v in delta.values())",1)
p.write_text(t)
p=w/'linear_guard.py';t=p.read_text();needle=' graph.build_bf16(forward(*symbols),symbols,checkpoint)';assert t.count(needle)==2
t=t.replace(needle,needle+'\n assert bool(graph.wide_storage_eligible)==enabled',1)
t=t.replace("  assert graph.run(control).tolist()==want", "  assert (graph.lazy_default_retired_bytes>0)==enabled\n  assert graph.run(control).tolist()==want\n  assert (graph.lazy_default_rehydrated_bytes>0)==enabled",1)
needle=' # Shared-weight construction retains its own owners after closing source graph.';assert t.count(needle)==1
check=''' if enabled:
  rejected=False;before=state(a);bad=garnet.compiled_graph()
  try:bad.build_shared(forward(*symbols),symbols,'/Users/shawn/CantorAI/out/models/Qwen3-1.7B/model.safetensors.index.json',graph)
  except Exception as error:rejected='storage scope mismatch'in str(error)
  finally:bad.close()
  assert rejected and before==state(a)
'''
t=t.replace(needle,check+needle,1);p.write_text(t)
changed=[rel for rel,h in manifest.items()if sha(s/rel)!=h]+['src/backends/metal/wide_checkpoint_policy.h'];assert len(changed)==8,changed
cuda=[rel for rel in manifest if 'cuda'in rel.lower()or Path(rel).suffix=='.cu'];assert all(sha(s/rel)==manifest[rel]for rel in cuda)
assert(s/'src/backends/metal/primary_block_shader.h').read_bytes()==(base/'src/backends/metal/primary_block_shader.h').read_bytes()
assert(s/'src/backends/metal/primary_block_codec.h').read_bytes()==(base/'src/backends/metal/primary_block_codec.h').read_bytes()
frozen={str(p.relative_to(s)):sha(p)for p in sorted(s.rglob('*'))if p.is_file()}
(w/'correctness-source-manifest.json').write_text(json.dumps(frozen,indent=2))
(w/'implementation-audit.json').write_text(json.dumps(dict(changed_files=changed,source_files=len(frozen),CUDA_unchanged_files=len(cuda),whole_primary_shader_CPUcodec_FP32_and_retirement_plan_algorithm_unchanged=True,scope='Eligibility scope changes; native/full correctness and resource/speed not executed'),indent=2))
for p in w.rglob('*.py'):ast.parse(p.read_text(),str(p))
print('WS44 eight-file scoped policy prepared; source1716 frozen; no build/test yet',flush=True)
