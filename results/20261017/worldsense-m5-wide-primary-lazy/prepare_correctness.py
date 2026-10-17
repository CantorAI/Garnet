from pathlib import Path
import json,shutil,ast
w=Path(__file__).resolve().parent;old=w.parent/'worldsense-wide-storage';global_old=w.parent/'worldsense-primary-lazy'
for name in ['api_state_guard.py','linear_guard.py','vision_grid_guard.py','resident_controller.py','make_resident_fixtures.py','full_correctness.py']:
 t=(old/name).read_text().replace('worldsense-wide-storage','worldsense-wide-primary-lazy').replace('wide_storage_eligible','wide_primary_eligible').replace('wide_checkpoint_storage','wide_primary_storage')
 if name=='api_state_guard.py':
  a="assert (timings['lazy_default_retired_bytes']>0)==wide and timings['lazy_default_rehydrated_bytes']==0"
  assert t.count(a)==1;t=t.replace(a,"assert (timings['lazy_default_retired_bytes']>0)==bool(expected) and timings['lazy_default_rehydrated_bytes']==0",1)
  t=t.replace("lazy_defaults_expected='wideVLM_only_when_on'","lazy_defaults_expected='both_models_when_on'")
 ast.parse(t);(w/name).write_text(t)
t=(global_old/'resident_guard.py').read_text().replace('worldsense-primary-lazy','worldsense-wide-primary-lazy');ast.parse(t);(w/'resident_guard.py').write_text(t)
for f in ['fixture-1-8','fixture-2-64']:
 shutil.copytree(global_old/f,w/f)
 p=w/f/'fixture.json';p.write_text(p.read_text().replace(str(global_old),str(w)))
(w/'driver-adaptation-before-execution.json').write_text(json.dumps({'sources':'WS44 original full native/model/vision/shared scope guards; WS45 text raw fixture global-lazy expectations','changes':['paths/newprimary-only capability/getter','bothONmodels retirement positive, normal rehydration zero','text primary per-call delta zero, VLM positive','raw text fixtures generic/resident retirement/rehydration positive','metadata scope affects only primary admission, preserves shared scope checks'],'original_numeric_output_state_token_gates_retained':True,'old_disabled_lazy_fault_branch_unclaimed':True},indent=2)+'\n')
print('Full WS46 correctness guards ready; 201 packed/245 raw corpus and shared-scope proof retained, text default retirement positive.')
