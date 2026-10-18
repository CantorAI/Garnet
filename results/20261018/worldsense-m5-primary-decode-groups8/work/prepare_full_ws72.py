"""Prepare all original learned guards plus actual groups8 per-call admission."""
from pathlib import Path
import json,hashlib,ast,shutil
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');d=b/'work/worldsense-opt';w=b/'work/worldsense-primary-decode-groups8';old=w.parent/'worldsense-minimal-full-prefill';proof={}
names=['build_corpus.py','build_direct_corpus.py','make_resident_fixtures.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','primary_corpus_test.cpp','direct_float_corpus_test.cpp','request_session_guard.py','request_full_session_guard.py']
for name in names:
 original=(old/name).read_text();text=original.replace('worldsense-minimal-full-prefill','worldsense-primary-decode-groups8');ops=[]
 def change(x,y,count=1):
  global text
  assert text.count(x)==count,(name,x,text.count(x));text=text.replace(x,y);ops.append(dict(before=x,after=y,count=count))
 if name=='api_state_guard.py':
  change("compat_expected=mode=='compat56'","compat_expected=mode=='compat56';groups8_expected=mode=='on'")
  before=" assert ('primary_core_isolated_library'in call['backend'])==core_expected"
  after=""" assert ('primary_decode_groups8'in call['backend'])==groups8_expected
 for key,oldkey in [('decode_groups8_encoded','decode_encoded'),('head_groups8_encoded','head_encoded')]:
  before_group=timings['primary_storage_before'][key];after_group=timings['primary_storage_after'][key]
  assert type(before_group)is int and type(after_group)is int and 0<=before_group<=after_group
  assert after_group-before_group==(delta[oldkey]if groups8_expected else 0)
  if not groups8_expected:assert before_group==after_group==0
  if name=='Qwen3-1.7B':assert before_group==after_group
 assert ('primary_core_isolated_library'in call['backend'])==core_expected"""
  change(before,after)
 if name.endswith('.py'):ast.parse(text)
 inverse=text
 for op in reversed(ops):assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
 assert inverse.replace('worldsense-primary-decode-groups8','worldsense-minimal-full-prefill')==original
 assert not(w/name).exists();(w/name).write_text(text);proof[name]=dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),current_sha256=hashlib.sha256(text.encode()).hexdigest(),operations=ops,original_math_oracles_bounds_control_flow_inverse_exact=True)
for name in ['direct-upload-selection.json','model-assets.json']:shutil.copy2(old/name,w/name)
(w/'learned-guard-adaptation-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
t=(old/'full_correctness.py').read_text().replace('worldsense-minimal-full-prefill','worldsense-primary-decode-groups8').replace('len(manifest)==1721','len(manifest)==1723').replace("['native_tests_pass']==54","['native_tests_pass']==60").replace('source1721_and_explicit_inputs_unchanged','source1723_and_explicit_inputs_unchanged').replace('actual_core_counter_deltas_zero_all_modes=True','actual_core_counter_deltas_zero_all_modes=True,actual_groups8_deltas_match_primary_and_zero_text=True').replace('WS71','WS72');ast.parse(t);(w/'full_correctness.py').write_text(t)
v=(d/'verify_learned_ws71.py').read_text().replace('worldsense-minimal-full-prefill','worldsense-primary-decode-groups8').replace("old=w.parent/'worldsense-primary-core-isolation'","old=w.parent/'worldsense-minimal-full-prefill'").replace("text.replace('worldsense-primary-decode-groups8','worldsense-primary-core-isolation')","text.replace('worldsense-primary-decode-groups8','worldsense-minimal-full-prefill')").replace("len(frozen['source'])==1721","len(frozen['source'])==1723").replace("['native_tests_pass']==54","['native_tests_pass']==60")
before="  assert('primary_core_isolated_library'in call['backend'])==isolated"
after="""  assert('primary_decode_groups8'in call['backend'])==(role=='on')
  for key,oldkey in [('decode_groups8_encoded','decode_encoded'),('head_groups8_encoded','head_encoded')]:
   before_group=timings['primary_storage_before'][key];after_group=timings['primary_storage_after'][key]
   assert type(before_group)is int and type(after_group)is int and 0<=before_group<=after_group
   assert after_group-before_group==(timings['primary_encoded_delta'][oldkey]if role=='on'else 0)
   if role!='on':assert before_group==after_group==0
   if index>=2:assert before_group==after_group
  assert('primary_core_isolated_library'in call['backend'])==isolated"""
assert v.count(before)==1;v=v.replace(before,after).replace('source1721_and_all_explicit_inputs_unchanged','source1723_and_all_explicit_inputs_unchanged').replace('actual_core_deltas_zero_all_modes=True','actual_core_deltas_zero_all_modes=True,actual_groups8_deltas_match_original_primary_and_zero_text=True');ast.parse(v);(d/'verify_learned_ws72.py').write_text(v)
t=(d/'prepare_learned_ws71.py').read_text().replace('worldsense-minimal-full-prefill','worldsense-primary-decode-groups8').replace("w.parent/'worldsense-primary-core-isolation'/name","w.parent/'worldsense-minimal-full-prefill'/name").replace('ws71.py','ws72.py').replace('WS71','WS72').replace("['native_tests_pass']==54","['native_tests_pass']==60").replace('len(source)==1721','len(source)==1723')
t=t.replace("'source-inverse-proof.json']","'source-inverse-proof.json','source-transformations.json','primary_decode_groups8_shader.h','primary_decode_groups8_test.cpp','pre-native-fixture-amendment.txt']")
ast.parse(t);(d/'prepare_learned_ws72.py').write_text(t)
print('WS72 full original learned proofs prepared inverse-exact to71; actual body/head8delta parity andzero text added, same16subprocess checks. No learned execution.')
