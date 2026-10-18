"""Retain original learned checks; change role admission and actual reader build only."""
from pathlib import Path
import json,hashlib,ast,shutil
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-minimal-full-prefill';old=w.parent/'worldsense-primary-core-isolation';proof={}
names=['build_corpus.py','build_direct_corpus.py','make_resident_fixtures.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','primary_corpus_test.cpp','direct_float_corpus_test.cpp','request_session_guard.py','request_full_session_guard.py']
for name in names:
 original=(old/name).read_text();text=original.replace('worldsense-primary-core-isolation','worldsense-minimal-full-prefill');ops=[]
 def change(x,y,count=1):
  global text
  assert text.count(x)==count,(name,x,text.count(x));text=text.replace(x,y);ops.append(dict(before=x,after=y,count=count))
 if name=='build_corpus.py':change('garnet-metal-worldsense-minimal-full-prefill-compat56','garnet-metal-worldsense-minimal-full-prefill-on')
 if name=='api_state_guard.py':
  change("vector_expected=mode=='compat56';core_expected=mode=='on'","vector_expected=mode in ['on','compat56'];core_expected=False;compat_expected=mode=='compat56'")
  change("('request_scope_weight_residency'in call['backend'])==vector_expected","('request_scope_weight_residency'in call['backend'])==compat_expected")
  change("('cached_matrix_metadata'in call['backend'])==vector_expected","('cached_matrix_metadata'in call['backend'])==compat_expected")
  change("if vector_expected:\n  q=timings['request_weight_residency']","if compat_expected:\n  q=timings['request_weight_residency']")
  change('cached_metadata_expected=vector_expected','cached_metadata_expected=compat_expected')
 if name.endswith('.py'):ast.parse(text)
 inverse=text
 for op in reversed(ops):assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
 assert inverse.replace('worldsense-minimal-full-prefill','worldsense-primary-core-isolation')==original
 assert not(w/name).exists();(w/name).write_text(text);proof[name]=dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),current_sha256=hashlib.sha256(text.encode()).hexdigest(),operations=ops,original_math_oracles_bounds_control_flow_inverse_exact=True)
for name in ['direct-upload-selection.json','model-assets.json']:shutil.copy2(old/name,w/name)
(w/'learned-guard-adaptation-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
t=(old/'full_correctness.py').read_text().replace('worldsense-primary-core-isolation','worldsense-minimal-full-prefill')
t=t.replace("('primary_corpus_test',compat,selection,'corpus-GPU-bits'),('core_library_corpus_test',candidate,selection,'core-library-corpus'),","('primary_corpus_test',candidate,selection,'corpus-GPU-bits'),")
t=t.replace('original_five_full_readers_raw_direct_and_actual_core_two_readers_pass','original_five_full_readers_in_actual_candidate_raw_direct_pass').replace('actual_core_counter_deltas_match_original_primary_and_zero_text','actual_core_counter_deltas_zero_all_modes')
t=t.replace('WS70 full learned controller PASS','WS71 full learned controller PASS');ast.parse(t);(w/'full_correctness.py').write_text(t)
v=(b/'work/worldsense-opt/verify_learned_ws70.py').read_text().replace('worldsense-primary-core-isolation','worldsense-minimal-full-prefill')
v=v.replace("old=w.parent/'worldsense-request-weight-residency'","old=w.parent/'worldsense-primary-core-isolation'")
v=v.replace("text.replace('worldsense-minimal-full-prefill','worldsense-request-weight-residency')","text.replace('worldsense-minimal-full-prefill','worldsense-primary-core-isolation')")
v=v.replace('len(journal)==17','len(journal)==16')
a=v.index("core=load(w/'core-library-corpus.json')");z=v.index("assert load(w/'direct-upload-corpus.json')",a);v=v[:a]+v[z:]
v=v.replace("isolated=role=='on'","isolated=False;vector=role in ['on','compat56']")
before="  for cap in ['primary_vector_prefill','primary_prefill_isolated_library','primary_full_isolated_library','cached_matrix_metadata','request_scope_weight_residency']:assert(cap in call['backend'])==compat"
after="  for cap in ['primary_vector_prefill','primary_prefill_isolated_library','primary_full_isolated_library']:assert(cap in call['backend'])==vector\n  for cap in ['cached_matrix_metadata','request_scope_weight_residency']:assert(cap in call['backend'])==compat"
assert v.count(before)==1;v=v.replace(before,after)
v=v.replace('both_original_GPU_readers_in_actual_CORE_EACH_words=4200857600,','all_five_readers_in_actual_FULL_candidate=True,').replace('actual_core_deltas_equal_original_VLM_and_zero_text=True','actual_core_deltas_zero_all_modes=True')
ast.parse(v);(b/'work/worldsense-opt/verify_learned_ws71.py').write_text(v)
t=(b/'work/worldsense-opt/prepare_learned_ws70.py').read_text().replace('worldsense-primary-core-isolation','worldsense-minimal-full-prefill').replace('ws70.py','ws71.py').replace('WS70','WS71')
for part in ["'build_core_library_corpus.py',","'core_library_corpus_test.cpp',","'diagnostic-api-amendment-before-native.txt',","'source-transformations.json',"]:t=t.replace(part,'')
t=t.replace("w.parent/'worldsense-request-weight-residency'/name","w.parent/'worldsense-primary-core-isolation'/name")
t=t.replace("['prepare_full_ws71.py','prepare_learned_ws71.py','verify_learned_ws71.py','verify_native_ws71.py']","['prepare_full_ws71.py','prepare_learned_ws71.py','verify_learned_ws71.py','verify_native_ws71.py','prepare_native_ws71.py']")
before="for role,names in [('compat56',['primary_corpus_test','direct_float_corpus_test']),('on',['core_library_corpus_test'])]"
assert t.count(before)==1;t=t.replace(before,"for role,names in [('compat56',['direct_float_corpus_test']),('on',['primary_corpus_test'])]")
ast.parse(t);(b/'work/worldsense-opt/prepare_learned_ws71.py').write_text(t)
print('WS71 full learned proof prepared: actual candidate all5original GPUreaders; original state/token/lifecycle guards inverse-exact;16subprocess checks planned, no learned execution.')
