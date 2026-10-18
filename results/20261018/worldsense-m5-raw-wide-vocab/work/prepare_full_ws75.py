"""Preserve original201 compatibility proof and add actual200/raw-head coverage."""
from pathlib import Path
import ast,json,hashlib,shutil
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-raw-wide-vocab';old=w.parent/'worldsense-minimal-selective-defaults';f=b/'work/worldsense-opt';seed='worldsense-minimal-selective-defaults';slug='worldsense-raw-wide-vocab';proof={};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['build_corpus.py','build_direct_corpus.py','make_resident_fixtures.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','primary_corpus_test.cpp','direct_float_corpus_test.cpp','request_session_guard.py','request_full_session_guard.py']:
 original=(old/name).read_text();t=original.replace(seed,slug);ops=[]
 def change(before,after,count=1):
  global t
  assert t.count(before)==count,(name,before,t.count(before));t=t.replace(before,after);ops.append(dict(before=before,after=after,count=count))
 if name=='build_corpus.py':
  change("garnet-metal-worldsense-raw-wide-vocab-on'","garnet-metal-worldsense-raw-wide-vocab-compat56'")
 if name=='api_state_guard.py':
  change("selective_expected=mode in ['primary','on']","selective_expected=mode in ['primary','on'];raw_wide_expected=mode=='on'")
  change(" assert ('primary_full_isolated_library'in call['backend'])==vector_expected"," assert ('primary_full_isolated_library'in call['backend'])==vector_expected\n assert ('primary_raw_wide_vocab'in call['backend'])==raw_wide_expected")
  change("  assert all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded'])", """  wanted=(200,7130316800,5433064960,27852800)if raw_wide_expected else(201,8401715200,6402163712,32819200)
  assert (stats['active_weights'],stats['original_bytes'],stats['payload_bytes']+stats['metadata_bytes'],stats['metadata_bytes'])==wanted
  if raw_wide_expected:
   assert delta['decode_encoded']>0 and delta['prefill_encoded']>0 and delta['head_encoded']==delta['vector_head_encoded']==0
  else:assert all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded'])""")
  change('raw_text_retirement_expected=compat_expected,cached_metadata_expected=compat_expected', 'raw_text_retirement_expected=compat_expected,raw_wide_vocabulary_expected=raw_wide_expected,cached_metadata_expected=compat_expected')
 if name.endswith('.py'):ast.parse(t)
 inverse=t
 for op in reversed(ops):assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
 assert inverse.replace(slug,seed)==original and not(w/name).exists();(w/name).write_text(t);proof[name]=dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),current_sha256=hashlib.sha256(t.encode()).hexdigest(),operations=ops,original_math_oracles_bounds_control_flow_inverse_exact=True)
for name in ['direct-upload-selection.json','model-assets.json']:shutil.copy2(old/name,w/name)
(w/'learned-guard-adaptation-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
# The original C++ corpus remains byte-exact and runs against COMPAT56. The
# supplemental candidate corpus changes admission/counts and adds, never removes,
# the complete original-bit comparison of the one newly raw head on the GPU.
original=(w/'primary_corpus_test.cpp').read_text();t=original;ops=[]
def change(before,after,count=1):
 global t
 assert t.count(before)==count,(before,t.count(before));t=t.replace(before,after);ops.append(dict(before=before,after=after,count=count))
change('#include "wide_checkpoint_policy.h"','#include "wide_checkpoint_policy.h"\n#include "primary_wide_vocab_policy.h"\n#ifndef GARNET_METAL_PRIMARY_RAW_WIDE_VOCAB\n#error Supplemental candidate corpus requires raw-wide-vocabulary policy\n#endif')
change('all245rawbuffers separately HOST-word checked, not GPU kernel proof.','all246rawbuffers separately HOST-word checked plus full newlyraw widevocabulary GPU words. Original201 compatibility is a separate mandatory subprocess.')
change('std::map<std::string,uint64_t> allTileWords', 'uint64_t rawHeadGPUWords=0,rawHeadMatrices=0;\n std::map<std::string,uint64_t> allTileWords')
change('const bool allow=WideStorage::Eligible(*indexes.at(model));', 'const bool baseAllow=WideStorage::Eligible(*indexes.at(model));const bool roleAllow=WideVocabularyPolicy::AllowPrimary(name,rows,inner,baseAllow,true);const bool rawHeadExcluded=baseAllow&&!roleAllow;const bool allow=baseAllow&&roleAllow;')
change('uint64_t rawChecked=0,checked=0;','uint64_t rawChecked=0,checked=0,rawHeadChecked=0;')
change('  rawBytes+=entry->dataSize;', '''  if(rawHeadExcluded){
   need(!expected&&!w.PrimaryReady()&&rows==248320&&inner==2560,"New raw vocabulary identity/admission differs");
   for(size_t start=0;start<rows*inner;start+=8*1024*1024){const size_t count=std::min(size_t(8*1024*1024),rows*inner-start);auto actual=gpu.RawBF16ReadGPUWords(w,start,count);need(actual.size()==count,"Raw vocabulary GPU word count differs");for(size_t i=0;i<count;++i){uint16_t word;std::memcpy(&word,data+2*(start+i),2);need(actual[i]==uint32_t(word)<<16,"Raw vocabulary original GPU BF16 word differs");}rawHeadChecked+=count;rawHeadGPUWords+=count;}
   need(rawHeadChecked==rows*inner,"Incomplete actual raw vocabulary GPU words");++rawHeadMatrices;
  }
  rawBytes+=entry->dataSize;''')
change('{{"HOST_raw_FP32_words_checked",rawChecked},{"model",model}', '{{"HOST_raw_FP32_words_checked",rawChecked},{"GPU_raw_wide_vocabulary_FP32_words_checked",rawHeadChecked},{"model",model}')
change('report["totals"]={{"HOST_raw_FP32_words_checked",rawWordsChecked}', 'report["totals"]={{"GPU_raw_wide_vocabulary_FP32_words_checked",rawHeadGPUWords},{"raw_wide_vocabulary_matrices",rawHeadMatrices},{"HOST_raw_FP32_words_checked",rawWordsChecked}')
change('packedMatrices==201&&rawMatrices==245&&rawBytes==11850481664ULL&&wordsChecked==4200857600ULL&&rawWordsChecked==1724383232ULL&&primaryBytes==9850930176ULL', 'packedMatrices==200&&rawMatrices==246&&rawBytes==11850481664ULL&&wordsChecked==3565158400ULL&&rawWordsChecked==2360082432ULL&&primaryBytes==10153229824ULL')
change('entry.second==4200857600ULL','entry.second==3565158400ULL')
change(' report["status"]=', ' need(rawHeadMatrices==1&&rawHeadGPUWords==635699200ULL&&wordsChecked+rawHeadGPUWords==4200857600ULL,"Original word-family coverage missing after representation change");\n report["status"]=')
change('Full201packed/245raw actual checkpoint GPU word and owner proof PASS','Actualcandidate200packed/246raw plus fullrawwidevocabulary GPU word and owner proof PASS')
inverse=t
for op in reversed(ops):assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
assert inverse==original;(w/'raw_wide_corpus_test.cpp').write_text(t);(w/'supplemental-corpus-adaptation-proof.json').write_text(json.dumps(dict(original_name='primary_corpus_test.cpp',original_sha256=sha(w/'primary_corpus_test.cpp'),candidate_sha256=sha(w/'raw_wide_corpus_test.cpp'),operations=ops,original_per_word_bit_comparisons_and_owner_controls_inverse_exact=True,COMPAT_full_original_corpus_still_mandatory=True),indent=2)+'\n')
t=(w/'build_corpus.py').read_text().replace('garnet-metal-worldsense-raw-wide-vocab-compat56','garnet-metal-worldsense-raw-wide-vocab-on').replace("str(w/'primary_corpus_test.cpp')","str(w/'raw_wide_corpus_test.cpp')").replace("str(b/'primary_corpus_test')","str(b/'raw_wide_corpus_test')").replace("'primary_corpus_test.cpp'","'raw_wide_corpus_test.cpp'").replace("'primary_corpus_test'","'raw_wide_corpus_test'").replace("'build-corpus.log'","'build-raw-wide-corpus.log'").replace("'corpus-build-identity.json'","'raw-wide-corpus-build-identity.json'")
t=t.replace("'-O3',","'-O3','-DGARNET_METAL_PRIMARY_RAW_WIDE_VOCAB=1',")
anchor="cmd=['/usr/bin/c++'";assert t.count(anchor)==1;t=t.replace(anchor,"assert 'GARNET_METAL_PRIMARY_RAW_WIDE_VOCAB:BOOL=ON'in(b/'CMakeCache.txt').read_text()\n"+anchor);ast.parse(t);(w/'build_raw_wide_corpus.py').write_text(t)
t=(old/'full_correctness.py').read_text().replace(seed,slug).replace('WS74','WS75').replace('len(manifest)==1721','len(manifest)==1723').replace("['native_tests_pass']==54","['native_tests_pass']==60").replace('source1721_and_explicit_inputs_unchanged','source1723_and_explicit_inputs_unchanged')
anchor="[('primary_corpus_test',candidate,selection,'corpus-GPU-bits'),";assert t.count(anchor)==1;t=t.replace(anchor,"[('primary_corpus_test',compat,selection,'corpus-GPU-bits'),('raw_wide_corpus_test',candidate,selection,'raw-wide-corpus-GPU-bits'),")
t=t.replace('original_five_full_readers_in_actual_candidate_raw_direct_pass','original_COMPAT201_and_actual_candidate200_plus_rawhead_GPU_readers_pass');ast.parse(t);(w/'full_correctness.py').write_text(t)
# Independent audit first retains every original201/full learned assertion;
# supplemental coverage and actual loader admission are checked in addition.
v=(f/'verify_learned_ws74.py').read_text().replace(seed,slug).replace("old=w.parent/'worldsense-minimal-full-prefill'","old=w.parent/'worldsense-minimal-selective-defaults'").replace("text.replace('worldsense-raw-wide-vocab','worldsense-minimal-full-prefill')","text.replace('worldsense-raw-wide-vocab','worldsense-minimal-selective-defaults')").replace("len(frozen['source'])==1721","len(frozen['source'])==1723").replace('len(journal)==16','len(journal)==17').replace("['native_tests_pass']==54","['native_tests_pass']==60").replace('source1721_and_all_explicit_inputs_unchanged','source1723_and_all_explicit_inputs_unchanged')
anchor="assert load(w/'direct-upload-corpus.json')['totals']==";assert v.count(anchor)==1
supplement='''supplement=load(w/'supplemental-corpus-adaptation-proof.json');value=(w/'raw_wide_corpus_test.cpp').read_text();assert sha(w/'primary_corpus_test.cpp')==supplement['original_sha256']and sha(w/'raw_wide_corpus_test.cpp')==supplement['candidate_sha256']
for op in reversed(supplement['operations']):assert value.count(op['after'])==op['count'];value=value.replace(op['after'],op['before'])
assert value==(w/'primary_corpus_test.cpp').read_text()
candidate=load(w/'raw-wide-corpus-GPU-bits.json');ct=candidate['totals'];assert ct['packed_matrices']==200 and ct['raw_small_matrices']==246 and ct['HOST_raw_FP32_words_checked']==2360082432 and ct['retained_primary_and_small_raw_bytes']==10153229824 and ct['original_bytes']==11850481664
assert ct['GPU_original_FP32_words_checked']==ct['GPU_vector_original_FP32_words_checked']==3565158400 and ct['GPU_prefill_tile_FP32_words_checked']=={k:3565158400 for k in ['32x32','64x32','64x64']}
assert ct['GPU_raw_wide_vocabulary_FP32_words_checked']==635699200 and ct['raw_wide_vocabulary_matrices']==1 and ct['GPU_original_FP32_words_checked']+ct['GPU_raw_wide_vocabulary_FP32_words_checked']==4200857600
oldrows={(x['model'],x['tensor']):x for x in corpus['matrices']};assert len(candidate['matrices'])==446;head_rows=0
for row in candidate['matrices']:
 oldrow=oldrows[(row['model'],row['tensor'])];assert row['owner_accounting_restored']and all(row[k]==oldrow[k]for k in ['model','tensor','rows','inner','original_bytes']);count=row['rows']*row['inner'];is_head=(row['model'],row['tensor'])==('Qwen3.5-4B','model.language_model.embed_tokens.weight');assert row['packed']==(oldrow['packed']and not is_head)
 if row['packed']:assert row['GPU_original_FP32_words_checked']==row['GPU_vector_original_FP32_words_checked']==count and row['GPU_prefill_tile_FP32_words_checked']=={k:count for k in ['32x32','64x32','64x64']}and row['GPU_raw_wide_vocabulary_FP32_words_checked']==0
 else:
  assert row['HOST_raw_FP32_words_checked']==count and row['GPU_original_FP32_words_checked']==row['GPU_vector_original_FP32_words_checked']==0 and all(v==0 for v in row['GPU_prefill_tile_FP32_words_checked'].values())
  assert row['GPU_raw_wide_vocabulary_FP32_words_checked']==(635699200 if is_head else 0)
 if is_head:assert(row['rows'],row['inner'])==(248320,2560);head_rows+=1
assert head_rows==1
''';v=v.replace(anchor,supplement+anchor)
anchor="  assert('primary_core_isolated_library'in call['backend'])==isolated";assert v.count(anchor)==1;v=v.replace(anchor,anchor+"\n  raw_wide=role=='on';assert ('primary_raw_wide_vocab'in call['backend'])==raw_wide")
x="   assert stats['active_weights']==201 and stats['original_bytes']==8401715200 and stats['payload_bytes']+stats['metadata_bytes']==6402163712 and stats['metadata_bytes']==32819200";y="   wanted=(200,7130316800,5433064960,27852800)if raw_wide else(201,8401715200,6402163712,32819200)\n   assert (stats['active_weights'],stats['original_bytes'],stats['payload_bytes']+stats['metadata_bytes'],stats['metadata_bytes'])==wanted";assert v.count(x)==1;v=v.replace(x,y)
x="   assert all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded'])and delta['vector_decode_encoded']==delta['decode_encoded']and delta['vector_head_encoded']==delta['head_encoded']";y="   if raw_wide:assert delta['decode_encoded']>0 and delta['prefill_encoded']>0 and delta['head_encoded']==delta['vector_head_encoded']==0\n   else:assert all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded'])\n   assert delta['vector_decode_encoded']==delta['decode_encoded']and delta['vector_head_encoded']==delta['head_encoded']";assert v.count(x)==1;v=v.replace(x,y)
v=v.replace('original_five_full_GPU_readers_EACH_words=4200857600,all_five_readers_in_actual_FULL_candidate=True','original_COMPAT201_five_full_GPU_readers_EACH_words=4200857600,actual_candidate200_five_full_GPU_readers_EACH_words=3565158400,actual_raw_wide_head_GPU_words=635699200,all_five_readers_in_actual_FULL_candidate=True,actual_loader200_head_zero_and_original_goldens_pass=True')
v=v.replace('raw_HOST_words=1724383232','original_COMPAT_raw_HOST_words=1724383232,actual_candidate_raw_HOST_words=2360082432')
anchor="for name,info in load(w/'learned-guard-adaptation-proof.json').items():"
builder_audit='''for filename,role,cpp,exe in [('corpus-build-identity.json','compat56','primary_corpus_test.cpp','primary_corpus_test'),('raw-wide-corpus-build-identity.json','on','raw_wide_corpus_test.cpp','raw_wide_corpus_test'),('direct-corpus-build-identity.json','compat56','direct_float_corpus_test.cpp','direct_float_corpus_test')]:
 record=load(w/filename);build=r/('out/build/garnet-metal-worldsense-raw-wide-vocab-'+role);command=record['command']
 assert record['source_sha256']==sha(w/cpp)and record['library_sha256']==sha(build/'libgarnet_metal.a')and record['executable_sha256']==sha(build/exe)
 assert str(w/cpp)in command and str(build/'libgarnet_metal.a')in command and command[-2:]==['-o',str(build/exe)]
 assert ('-DGARNET_METAL_PRIMARY_RAW_WIDE_VOCAB=1'in command)==(role=='on')
'''
assert v.count(anchor)==1;v=v.replace(anchor,builder_audit+anchor)
ast.parse(v);(f/'verify_learned_ws75.py').write_text(v)
t=(f/'prepare_learned_ws74.py').read_text().replace(seed,slug).replace('ws74.py','ws75.py').replace('WS74','WS75').replace("['native_tests_pass']==54","['native_tests_pass']==60").replace('len(source)==1721','len(source)==1723').replace("w.parent/'worldsense-minimal-full-prefill'/name","w.parent/'worldsense-minimal-selective-defaults'/name")
t=t.replace("['build_corpus.py','build_direct_corpus.py','make_resident_fixtures.py']","['build_corpus.py','build_raw_wide_corpus.py','build_direct_corpus.py','make_resident_fixtures.py']")
t=t.replace("'immutable57-policy-guard.txt',","'prospective-byte-budget.json','pre-native-fixture-repair.json',")
t=t.replace("'build_corpus.py','build_direct_corpus.py'","'build_corpus.py','build_raw_wide_corpus.py','raw_wide_corpus_test.cpp','supplemental-corpus-adaptation-proof.json','build_direct_corpus.py'")
t=t.replace("'raw_wide_corpus_test.cpp','supplemental-corpus-adaptation-proof.json',","'raw_wide_corpus_test.cpp','supplemental-corpus-adaptation-proof.json','corpus-build-identity.json','raw-wide-corpus-build-identity.json','direct-corpus-build-identity.json',")
t=t.replace("'design_selective_ws75.py','setup_selective_ws75.py'","'design_raw_wide_vocab_ws75.py','setup_raw_wide_vocab_ws75.py','repair_native_fixture_ws75.py'")
t=t.replace("[('compat56',['direct_float_corpus_test']),('on',['primary_corpus_test'])]","[('compat56',['direct_float_corpus_test','primary_corpus_test']),('on',['raw_wide_corpus_test'])]")
ast.parse(t);(f/'prepare_learned_ws75.py').write_text(t)
print('WS75 original12guards inverse-exact, full COMPAT201five readers retained PLUSactualcandidate200/rawheadGPU coverage.17checks/64private steps/20APIcalls planned; no learned invocation.')
