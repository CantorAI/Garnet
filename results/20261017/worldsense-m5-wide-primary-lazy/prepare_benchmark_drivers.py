"""Prepare unchanged quantitative gates with primary-only scope/native global-retirement predicates."""
from pathlib import Path
import ast,json
w=Path(__file__).resolve().parent;seed=w.parent/'worldsense-wide-storage'
files=['freeze.py','prepare_cohort.py','run_resource.py','analyze_resource.py','audit_gates.py','dependency_closure.py','speed/prepare.py','speed/run_stage.py','speed/analyze.py','confirmation/prepare.py','confirmation/run_stage.py','confirmation/analyze.py','production-speed/prepare.py','production-speed/run_stage.py','production-speed/analyze.py']
for rel in files:
 p=w/rel;assert not p.exists(),rel;p.parent.mkdir(parents=True,exist_ok=True)
 t=(seed/rel).read_text().replace('worldsense-wide-storage','worldsense-wide-primary-lazy').replace('GARNET_METAL_WIDE_CHECKPOINT_STORAGE','GARNET_METAL_WIDE_PRIMARY_STORAGE').replace('wide_checkpoint_storage','wide_primary_storage')
 if rel in ['analyze_resource.py','speed/analyze.py','confirmation/analyze.py','production-speed/analyze.py']:
  old="return t['lazy_default_retired_bytes']==t['lazy_default_planned_slots']==0 and all(v==0 for v in delta.values())"
  assert t.count(old)==1;t=t.replace(old,"return t['lazy_default_retired_bytes']>0 and t['lazy_default_planned_slots']>0 and all(v==0 for v in delta.values())",1)
 if rel=='freeze.py':t=t.replace('actualtext staysraw/eager andwide4Busesunchangedstoragealgorithms','actualtext weightsstayraw;safe defaultretirementglobal;wide4Bprimary unchanged algorithms')
 if rel=='prepare_cohort.py':t=t.replace('actual_wide_VLM_and_raw_eager_text_scope_proven','actual_wide_primary_and_global_retirement_scope_proven')
 if rel=='audit_gates.py':t=t.replace("bad['timings']['lazy_default_retired_bytes']=0 if c['model_id']=='Qwen3.5-4B'else 1","bad['timings']['lazy_default_retired_bytes']=0")
 ast.parse(t);p.write_text(t)
(w/'benchmark-preparation-scope.json').write_text(json.dumps({'no_worker_dispatched':True,'native_scope':'VLM201packed/textallraw;safe retirementpositive bothONmodels','original_golden_metric_AST_to_be_independently_audited':True,'all_original_resource_speed_and_both_pairs_required':True,'extra_actual_WS10_20_required':True},indent=2)+'\n')
print('WS46 resource8/speed20/extraactual20/full48 drivers ready; no worker dispatched.')
