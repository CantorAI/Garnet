"""Prospective original learned guards with inverse-exact role/core-route adaptations."""
from pathlib import Path
import ast,json,hashlib,shutil
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-primary-core-isolation';old=w.parent/'worldsense-request-weight-residency';proof={}
names=['build_corpus.py','build_direct_corpus.py','make_resident_fixtures.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','primary_corpus_test.cpp','direct_float_corpus_test.cpp','request_session_guard.py','request_full_session_guard.py']
for name in names:
 original=(old/name).read_text();text=original.replace('worldsense-request-weight-residency','worldsense-primary-core-isolation');ops=[]
 def change(x,y,count=1):
  global text
  assert text.count(x)==count,(name,x,text.count(x));text=text.replace(x,y);ops.append(dict(before=x,after=y,count=count))
 if name in ['build_corpus.py','build_direct_corpus.py','request_session_guard.py','request_full_session_guard.py']:change('garnet-metal-worldsense-primary-core-isolation-on','garnet-metal-worldsense-primary-core-isolation-compat56')
 if name in ['request_session_guard.py','request_full_session_guard.py']:
  out='request-session-guard'if name=='request_session_guard.py'else'request-full-session-guard';change(out+'.json',out+'-compat56.json')
  if name=='request_session_guard.py':change('request-session-guard-progress.json','request-session-guard-progress-compat56.json')
 if name=='resident_controller.py':change("['off','primary','on']","['off','primary','on','compat56']")
 if name=='resident_guard.py':change("endswith('-on')","endswith('-compat56')")
 if name=='api_state_guard.py':
  change("['off','primary','on'];expected=0 if mode=='off'else 1;vector_expected=mode=='on'","['off','primary','on','compat56'];expected=0 if mode=='off'else 1;vector_expected=mode=='compat56';core_expected=mode=='on'")
  change(" else:assert 'request_weight_residency'not in timings",""" else:assert 'request_weight_residency'not in timings
 assert ('primary_core_isolated_library'in call['backend'])==core_expected
 before=timings['primary_storage_before'];after=timings['primary_storage_after']
 for key,oldkey in [('core_library_decode_encoded','decode_encoded'),('core_library_prefill_encoded','prefill_encoded'),('core_library_head_encoded','head_encoded')]:
  assert type(before[key])is int and type(after[key])is int and 0<=before[key]<=after[key]
  core_delta=after[key]-before[key]
  assert core_delta==(delta[oldkey]if core_expected else 0)
  if not core_expected:assert before[key]==after[key]==0
  if name=='Qwen3-1.7B':assert core_delta==0
""")
 if name.endswith('.py'):ast.parse(text)
 inverse=text
 for op in reversed(ops):assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
 assert inverse.replace('worldsense-primary-core-isolation','worldsense-request-weight-residency')==original,name
 assert not(w/name).exists(),name;(w/name).write_text(text);proof[name]=dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),current_sha256=hashlib.sha256(text.encode()).hexdigest(),operations=ops,original_math_oracles_bounds_control_flow_inverse_exact=True)
for name in ['direct-upload-selection.json','model-assets.json']:shutil.copy2(old/name,w/name)
(w/'learned-guard-adaptation-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print('WS70 original learned guard adaptations inverse exact; CORE counters derived from unchanged serving snapshots. No model execution.')
