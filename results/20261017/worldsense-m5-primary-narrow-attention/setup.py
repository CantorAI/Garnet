"""Apply predeclared WS42 to a fresh isolated copy of terminal WS41 source."""
from pathlib import Path
import ast,hashlib,json,shutil
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI')
old=r/'out/research/worldsense-primary-narrow-vocab/source';s=r/'out/research/worldsense-primary-narrow-attention/source'
seed=w.parent/'worldsense-primary-narrow-vocab'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not s.exists()
assert sha(w/'native-design-before-implementation.txt')==json.loads((w/'design-identity.json').read_text())['design_sha256']
assert json.loads((seed/'decision.json').read_text())['stage_outcomes']=={'resource':True,'speed':False}
manifest=json.loads((seed/'starting-source-manifest.json').read_text())
for rel,h in manifest.items():
    assert sha(old/rel)==h,rel
    t=s/rel;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/rel,t)
(w/'seed-source-manifest.json').write_text(json.dumps(manifest,indent=2))
def replace(rel,a,b):
    p=s/rel;t=p.read_text();assert t.count(a)==1,(rel,a);p.write_text(t.replace(a,b,1))
replace('src/backends/metal/primary_storage_policy.h','inline bool AllowPrimary(const std::string& name,size_t rows,size_t columns,bool rawNarrowVocabulary) {','''inline bool NarrowAttention(const std::string& name,size_t columns) {
    const bool projection=EndsWith(name,".q_proj.weight")||EndsWith(name,".k_proj.weight")||EndsWith(name,".v_proj.weight")||EndsWith(name,".o_proj.weight");
    return columns>0&&columns<=2048&&name.find(".self_attn.")!=std::string::npos&&projection;
}
inline bool AllowPrimary(const std::string& name,size_t rows,size_t columns,bool rawNarrowVocabulary,bool rawNarrowAttention=false) {''')
replace('src/backends/metal/primary_storage_policy.h','    return !(rawNarrowVocabulary&&vocabulary&&rows>65536&&columns>0&&columns<=2048);','    return !(rawNarrowVocabulary&&vocabulary&&rows>65536&&columns>0&&columns<=2048)&&!(rawNarrowAttention&&NarrowAttention(name,columns));')
replace('src/backends/metal/compiled_graph.cpp','''    bool allowPrimary=true;
#ifdef GARNET_METAL_PRIMARY_RAW_NARROW_VOCAB
    allowPrimary=PrimaryPolicy::AllowPrimary(name,size_t(entry->shape[0]),size_t(entry->shape[1]),true);
#endif''','''    bool rawNarrowVocabulary=false,rawNarrowAttention=false;
#ifdef GARNET_METAL_PRIMARY_RAW_NARROW_VOCAB
    rawNarrowVocabulary=true;
#endif
#ifdef GARNET_METAL_PRIMARY_RAW_NARROW_ATTENTION
    rawNarrowAttention=true;
#endif
    const bool allowPrimary=PrimaryPolicy::AllowPrimary(name,size_t(entry->shape[0]),size_t(entry->shape[1]),rawNarrowVocabulary,rawNarrowAttention);''')
replace('src/entry/metal_main.cpp','#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS\n        info+="; lazy_resident_defaults";','#ifdef GARNET_METAL_PRIMARY_RAW_NARROW_ATTENTION\n        info+="; primary_raw_narrow_attention";\n#endif\n#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS\n        info+="; lazy_resident_defaults";')
replace('cmake/GarnetMetal.cmake','option(GARNET_METAL_PRIMARY_RAW_NARROW_VOCAB "Preserve raw BF16 storage for semantic narrow vocabulary weights" OFF)','''option(GARNET_METAL_PRIMARY_RAW_NARROW_VOCAB "Preserve raw BF16 storage for semantic narrow vocabulary weights" OFF)
option(GARNET_METAL_PRIMARY_RAW_NARROW_ATTENTION "Preserve raw BF16 storage for semantic narrow attention projections" OFF)
if(GARNET_METAL_PRIMARY_RAW_NARROW_ATTENTION)
 if(NOT GARNET_METAL_PRIMARY_BF16_BLOCKS)
  message(FATAL_ERROR "Narrow attention storage policy requires primary weights")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_PRIMARY_RAW_NARROW_ATTENTION=1)
endif()''')
replace('test2026/metal/primary_storage_test.cpp',' // Pure integer boundary checks require no multi-GiB allocation.',''' // Preserve MLP/ordinary/wide storage while selecting only semantic narrow attention.
 for(const std::string suffix:{"q_proj.weight","k_proj.weight","v_proj.weight","o_proj.weight"}) {
  const std::string name="model.layers.0.self_attn."+suffix;
  need(!PrimaryPolicy::AllowPrimary(name,2048,2048,true,true),"Narrow attention stayed packed");
  need(PrimaryPolicy::AllowPrimary(name,2048,2048,true,false),"Disabled narrow attention policy changed storage");
  need(PrimaryPolicy::AllowPrimary(name,2560,2560,true,true),"Wide attention excluded");
  need(PrimaryPolicy::AllowPrimary(name,2048,2049,true,true),"Attention width boundary changed");
  need(PrimaryPolicy::AllowPrimary(name,2048,0,true,true),"Zero attention width bypassed ordinary validation");
 }
 for(const std::string name:{"model.layers.0.mlp.gate_proj.weight","model.layers.0.mlp.up_proj.weight","model.layers.0.mlp.down_proj.weight","other.q_proj.weight","model.layers.0.self_attn.q_norm.weight"})
  need(PrimaryPolicy::AllowPrimary(name,2048,2048,true,true),"Non-attention projection incorrectly excluded");
 need(!PrimaryPolicy::AllowPrimary("model.embed_tokens.weight",151936,2048,true,true),"Combined policy lost raw vocabulary");
 // Pure integer boundary checks require no multi-GiB allocation.''')

files=['build.py','build_fault.py','build_corpus.py','primary_corpus_test.cpp','full_correctness.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','make_resident_fixtures.py','wait_cantor.py','freeze.py','prepare_cohort.py','restore_normal.py','run_resource.py','analyze_resource.py','audit_gates.py','dependency_closure.py','speed/prepare.py','speed/run_stage.py','speed/analyze.py','confirmation/prepare.py','confirmation/run_stage.py','confirmation/analyze.py','production-speed/prepare.py','production-speed/run_stage.py','production-speed/analyze.py']
for rel in files:
    p=w/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text((seed/rel).read_text().replace('worldsense-primary-narrow-vocab','worldsense-primary-narrow-attention'))
for rel in ['fixture-1-8','fixture-2-64']:shutil.copytree(seed/rel,w/rel)
for rel in ['build.py','build_fault.py']:
    p=w/rel;t=p.read_text();needle="'-DGARNET_METAL_CACHED_MATRIX_METADATA=OFF',";assert t.count(needle)==1
    value="'-DGARNET_METAL_PRIMARY_RAW_NARROW_ATTENTION='+mode.upper(),"if rel=='build.py'else"'-DGARNET_METAL_PRIMARY_RAW_NARROW_ATTENTION=ON',"
    p.write_text(t.replace(needle,needle+value,1))
p=w/'primary_corpus_test.cpp';t=p.read_text();t=t.replace('PrimaryPolicy::AllowPrimary(name,rows,inner,true)','PrimaryPolicy::AllowPrimary(name,rows,inner,true,true)',1)
t=t.replace('primaryBytes=0,rawVocabulary=0;','primaryBytes=0,rawVocabulary=0,rawAttention=0;',1)
needle='if(!allow){need(model=="Qwen3-1.7B"&&name=="model.embed_tokens.weight"&&rows==151936&&inner==2048,"Unexpected semantic exclusion");++rawVocabulary;}'
assert t.count(needle)==1
t=t.replace(needle,'if(!allow){need(model=="Qwen3-1.7B","VLM semantic exclusion");if(name=="model.embed_tokens.weight"){need(rows==151936&&inner==2048,"Vocabulary dimensions changed");++rawVocabulary;}else{need(PrimaryPolicy::NarrowAttention(name,inner),"Unexpected semantic exclusion");++rawAttention;}}',1)
t=t.replace('{"raw_vocabulary_owners",rawVocabulary}', '{"raw_vocabulary_owners",rawVocabulary},{"raw_attention_owners",rawAttention}',1)
t=t.replace('rawVocabulary==1&&packedMatrices==397&&rawMatrices==49','rawVocabulary==1&&rawAttention==112&&packedMatrices==285&&rawMatrices==161',1)
t=t.replace('wordsChecked==5610143744ULL','wordsChecked==5257822208ULL',1).replace('primaryBytes==9182203392ULL','primaryBytes==9349231104ULL',1).replace('Full397packed/49raw','Full285packed/161raw').replace('single explicitly excluded vocabulary owner','explicitly excluded vocabulary and112attention owners')
p.write_text(t)
p=w/'full_correctness.py';t=p.read_text().replace('==5610143744','==5257822208').replace('ALL397PACKED','ALL285PACKED').replace('all397actualpacked_weights','all285actualpacked_weights');p.write_text(t)
p=w/'api_state_guard.py';t=p.read_text();needle="assert ('primary_raw_narrow_vocab' in call.get('backend',''))==bool(expected);";assert t.count(needle)==1
p.write_text(t.replace(needle,needle+"assert ('primary_raw_narrow_attention' in call.get('backend',''))==bool(expected);",1))
oldtuples='196:(2818572288,2149845504,11010048),201:(8401715200,6402163712,32819200),397:(11220287488,8552009216,43829248)'
newtuples='84:(2113929216,1612230144,8257536),201:(8401715200,6402163712,32819200),285:(10515644416,8014393856,41076736)'
for rel in ['analyze_resource.py','speed/analyze.py','confirmation/analyze.py','production-speed/analyze.py']:
    p=w/rel;t=p.read_text();assert t.count(oldtuples)==1;t=t.replace(oldtuples,newtuples,1)
    needle=" if ('primary_raw_narrow_vocab'in call['backend'])!=enabled:return False";assert t.count(needle)==1
    t=t.replace(needle,needle+"\n if ('primary_raw_narrow_attention'in call['backend'])!=enabled:return False",1)
    if rel.startswith('production-speed/'):
        needle="'primary_u32_metadata','primary_raw_narrow_vocab']";assert t.count(needle)==1;t=t.replace(needle,"'primary_u32_metadata','primary_raw_narrow_vocab','primary_raw_narrow_attention']",1)
    p.write_text(t)
p=w/'prepare_cohort.py';p.write_text(p.read_text().replace('all397actualpacked_weights','all285actualpacked_weights'))
p=w/'freeze.py';t=p.read_text();needle="'GARNET_METAL_PRIMARY_RAW_NARROW_VOCAB':('OFF','ON')}";assert t.count(needle)==1
t=t.replace(needle,"'GARNET_METAL_PRIMARY_RAW_NARROW_VOCAB':('OFF','ON'),'GARNET_METAL_PRIMARY_RAW_NARROW_ATTENTION':('OFF','ON')}",1).replace('and semantic narrow-vocabulary storage flags differ','and semantic narrow-vocabulary/narrow-attention storage flags differ');p.write_text(t)
p=w/'audit_gates.py';t=p.read_text().replace("old=w.parent/'worldsense-primary-u32'","old=w.parent/'worldsense-primary-narrow-vocab'").replace(".replace('worldsense-primary-narrow-attention','worldsense-primary-u32')",".replace('worldsense-primary-narrow-attention','worldsense-primary-narrow-vocab')")
needle="   bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace('primary_raw_narrow_vocab','')if mode=='on'else bad['backend']+' primary_raw_narrow_vocab';assert not namespace['native_route'](bad,mode=='on')";assert t.count(needle)==1
t=t.replace(needle,needle+"\n   bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace('primary_raw_narrow_attention','')if mode=='on'else bad['backend']+' primary_raw_narrow_attention';assert not namespace['native_route'](bad,mode=='on')",1);p.write_text(t)

changed=[rel for rel,h in manifest.items()if sha(s/rel)!=h]
assert len(changed)==5,changed
cuda=[rel for rel in manifest if 'cuda'in rel.lower()or Path(rel).suffix=='.cu'];assert all(sha(s/rel)==manifest[rel]for rel in cuda)
assert(s/'src/backends/metal/primary_block_shader.h').read_bytes()==(old/'src/backends/metal/primary_block_shader.h').read_bytes()
assert(s/'src/backends/metal/metal_context.mm').read_bytes()==(old/'src/backends/metal/metal_context.mm').read_bytes()
frozen={rel:sha(s/rel)for rel in sorted(manifest)}
(w/'correctness-source-manifest.json').write_text(json.dumps(frozen,indent=2))
(w/'implementation-audit.json').write_text(json.dumps(dict(changed_files_from_WS41=changed,source_files=len(frozen),CUDA_unchanged_files=len(cuda),entire_primary_shader_and_native_upload_compute_unchanged=True,scope='Static invariants only; no correctness or performance claim'),indent=2))
for p in w.rglob('*.py'):ast.parse(p.read_text(),str(p))
print('WS42 five-file delta prepared and frozen; source/nativeimages/math/CUDA invariants pass; not built/tested yet',flush=True)
