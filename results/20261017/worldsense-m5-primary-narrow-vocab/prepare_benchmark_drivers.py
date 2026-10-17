"""Prepare future WS41 gates. Does not dispatch a worker or change frozen code."""
from pathlib import Path
import ast,json
w=Path(__file__).resolve().parent;old=w.parent/'worldsense-primary-u32'
files=['freeze.py','prepare_cohort.py','restore_normal.py','run_resource.py','analyze_resource.py','audit_gates.py','dependency_closure.py','speed/prepare.py','speed/run_stage.py','speed/analyze.py','confirmation/prepare.py','confirmation/run_stage.py','confirmation/analyze.py']
for rel in files:
    p=w/rel;assert not p.exists(),rel;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text((old/rel).read_text().replace('worldsense-primary-u32','worldsense-primary-narrow-vocab'))
old_expected='197:(3440902144,2623650048,13441024),201:(8401715200,6402163712,32819200),398:(11842617344,9025813760,46260224)'
new_expected='196:(2818572288,2149845504,11010048),201:(8401715200,6402163712,32819200),397:(11220287488,8552009216,43829248)'
for rel in ['analyze_resource.py','speed/analyze.py','confirmation/analyze.py']:
    p=w/rel;t=p.read_text();assert old_expected in t;t=t.replace(old_expected,new_expected,1)
    needle=" if ('primary_u32_metadata'in call['backend'])!=enabled:return False";assert t.count(needle)==1
    t=t.replace(needle,needle+"\n if ('primary_raw_narrow_vocab'in call['backend'])!=enabled:return False",1)
    positive="all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded'])";assert t.count(positive)==1
    t=t.replace(positive,"all(delta[k]>0 for k in ['decode_encoded','prefill_encoded']) and ((delta['head_encoded']==0)if call['model_id']=='Qwen3-1.7B'else(delta['head_encoded']>0))",1)
    p.write_text(t)
p=w/'freeze.py';t=p.read_text();needle="=={'GARNET_METAL_PRIMARY_BF16_BLOCKS':('OFF','ON')}";assert needle in t
t=t.replace(needle,"=={'GARNET_METAL_PRIMARY_BF16_BLOCKS':('OFF','ON'),'GARNET_METAL_PRIMARY_RAW_NARROW_VOCAB':('OFF','ON')}")
t=t.replace('only default-OFF primaryBF16 storage flag differs','only default-OFF primaryBF16 and semantic narrow-vocabulary storage flags differ')
p.write_text(t)
p=w/'prepare_cohort.py';t=p.read_text().replace('all398actualpacked_weights','all397actualpacked_weights');p.write_text(t)
p=w/'audit_gates.py';t=p.read_text().replace("old=w.parent/'worldsense-primary-kmajor'","old=w.parent/'worldsense-primary-u32'").replace(".replace('worldsense-primary-narrow-vocab','worldsense-primary-kmajor')",".replace('worldsense-primary-narrow-vocab','worldsense-primary-u32')")
needle="   bad=copy.deepcopy(c);bad['timings']['primary_storage_after']['metadata_bytes']+=4;assert not namespace['native_route'](bad,mode=='on')";assert t.count(needle)==1
t=t.replace(needle,needle+"\n   bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace('primary_raw_narrow_vocab','')if mode=='on'else bad['backend']+' primary_raw_narrow_vocab';assert not namespace['native_route'](bad,mode=='on')\n   if mode=='on'and c['model_id']=='Qwen3-1.7B':\n    bad=copy.deepcopy(c);bad['timings']['primary_encoded_delta']['head_encoded']=1;bad['timings']['primary_encoded_delta']['vector_head_encoded']=1;assert not namespace['native_route'](bad,True)",1)
p.write_text(t)
p=w/'confirmation/prepare.py';t=p.read_text();needle="assert json.loads((p/'speed/analysis.json').read_text())['passed']";assert needle in t
p.write_text(t.replace(needle,needle+"\nassert json.loads((p/'production-speed/analysis.json').read_text())['passed']",1))
for p in w.rglob('*.py'):ast.parse(p.read_text(),str(p))
(w/'benchmark-preparation-scope.json').write_text(json.dumps(dict(no_worker_dispatched=True,original_three_metric_and_golden_analyzers_unchanged_except_native_route=True,additional_actual_WS10_speed_screen_required_before_full_confirmation=True,original_memory_and_two_pair_gates_retained=True),indent=2))
print('Original8/20/48 drivers prepared; additionalactualWS10speed screen still to prepare; no dispatch',flush=True)
