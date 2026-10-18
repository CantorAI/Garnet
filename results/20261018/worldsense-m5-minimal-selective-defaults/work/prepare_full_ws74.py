"""Adapt storage admission only; inverse preserve every original numerical guard."""
from pathlib import Path
import ast,json,hashlib,shutil
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-minimal-selective-defaults';old=w.parent/'worldsense-minimal-full-prefill';proof={};slug='worldsense-minimal-selective-defaults';seed='worldsense-minimal-full-prefill'
for name in ['build_corpus.py','build_direct_corpus.py','make_resident_fixtures.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','primary_corpus_test.cpp','direct_float_corpus_test.cpp','request_session_guard.py','request_full_session_guard.py']:
 original=(old/name).read_text();t=original.replace(seed,slug);ops=[]
 def change(before,after,count=1):
  global t
  assert t.count(before)==count,(name,before,t.count(before));t=t.replace(before,after);ops.append(dict(before=before,after=after,count=count))
 if name=='resident_guard.py':
  change("lazy_enabled='lazy_resident_defaults'in garnet.backend_info();", "assert not f['vl']and f['checkpoint'].endswith('/Qwen3-1.7B/model.safetensors.index.json')\nlazy_enabled='lazy_resident_defaults'in garnet.backend_info()and 'primary_only_lazy_defaults'not in garnet.backend_info();")
  change("assert (graph.lazy_default_retired_bytes>0)==lazy_enabled", "assert (graph.lazy_default_retired_bytes>0)==lazy_enabled and (graph.lazy_default_planned_slots>0)==lazy_enabled")
 if name=='api_state_guard.py':
  change("compat_expected=mode=='compat56'", "compat_expected=mode=='compat56';selective_expected=mode in ['primary','on']")
  change("timings=call['timings'];assert (timings['lazy_default_retired_bytes']>0)==bool(expected) and (timings['lazy_default_planned_slots']>0)==bool(expected) and timings['lazy_default_rehydrated_bytes']==0", "retirement_expected=bool(expected)and (not selective_expected or name=='Qwen3.5-4B');assert ('primary_only_lazy_defaults'in call['backend'])==selective_expected\n timings=call['timings'];assert (timings['lazy_default_retired_bytes']>0)==retirement_expected and (timings['lazy_default_planned_slots']>0)==retirement_expected and timings['lazy_default_rehydrated_bytes']==0")
  change('lazy_defaults_expected=bool(expected),cached_metadata_expected=compat_expected', 'lazy_defaults_expected=bool(expected),selective_defaults_expected=selective_expected,raw_text_retirement_expected=compat_expected,cached_metadata_expected=compat_expected')
 if name.endswith('.py'):ast.parse(t)
 inverse=t
 for op in reversed(ops):assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
 assert inverse.replace(slug,seed)==original and not(w/name).exists();(w/name).write_text(t);proof[name]=dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),current_sha256=hashlib.sha256(t.encode()).hexdigest(),operations=ops,original_math_oracles_bounds_control_flow_inverse_exact=True)
for name in ['direct-upload-selection.json','model-assets.json']:shutil.copy2(old/name,w/name)
(w/'learned-guard-adaptation-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
t=(old/'full_correctness.py').read_text().replace(seed,slug).replace('WS71','WS74');ast.parse(t);(w/'full_correctness.py').write_text(t)
v=(b/'work/worldsense-opt/verify_learned_ws71.py').read_text().replace(seed,slug).replace("old=w.parent/'worldsense-primary-core-isolation'","old=w.parent/'worldsense-minimal-full-prefill'").replace("text.replace('worldsense-minimal-selective-defaults','worldsense-primary-core-isolation')","text.replace('worldsense-minimal-selective-defaults','worldsense-minimal-full-prefill')")
x="  assert timings['lazy_default_rehydrated_bytes']==0 and(timings['lazy_default_retired_bytes']>0)==enabled and(timings['lazy_default_planned_slots']>0)==enabled";y="  selective=role in ['primary','on'];retirement=enabled and (not selective or index<2)\n  assert ('primary_only_lazy_defaults'in call['backend'])==selective\n  assert timings['lazy_default_rehydrated_bytes']==0 and(timings['lazy_default_retired_bytes']>0)==retirement and(timings['lazy_default_planned_slots']>0)==retirement";assert v.count(x)==1;v=v.replace(x,y)
v=v.replace('actual_core_deltas_zero_all_modes=True,old_raw_compatibility_lifecycles_pass=True','actual_core_deltas_zero_all_modes=True,selective_raw_text_retirement_zero_and_VLM_positive_all_calls=True,old_raw_compatibility_lifecycles_pass=True');ast.parse(v);(b/'work/worldsense-opt/verify_learned_ws74.py').write_text(v)
t=(b/'work/worldsense-opt/prepare_learned_ws71.py').read_text().replace(seed,slug).replace('ws71.py','ws74.py').replace('WS71','WS74').replace("w.parent/'worldsense-primary-core-isolation'/name","w.parent/'worldsense-minimal-full-prefill'/name")
x="'source-inverse-proof.json']";assert t.count(x)==1;t=t.replace(x,"'source-inverse-proof.json','source-transformations.json','immutable57-policy-guard.txt','read-only-consumer-prior-audit.json']")
x="'prepare_native_ws74.py']";assert t.count(x)==1;t=t.replace(x,"'prepare_native_ws74.py','design_selective_ws74.py','setup_selective_ws74.py']");ast.parse(t);(b/'work/worldsense-opt/prepare_learned_ws74.py').write_text(t)
print('WS74 all12original learned guards inverse exact to71; only strict graphpolicy admission changes.16checks/64private steps/all5readers prepared, no learned execution.')
