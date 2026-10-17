"""Prepare predeclared wide-scoped gates without dispatching a worker."""
from pathlib import Path
import ast,json
w=Path(__file__).resolve().parent;seed=w.parent/'worldsense-primary-u32'
files=['freeze.py','prepare_cohort.py','restore_normal.py','run_resource.py','analyze_resource.py','audit_gates.py','dependency_closure.py','speed/prepare.py','speed/run_stage.py','speed/analyze.py','confirmation/prepare.py','confirmation/run_stage.py','confirmation/analyze.py']
for rel in files:
    p=w/rel;assert not p.exists(),rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text((seed/rel).read_text().replace('worldsense-primary-u32','worldsense-wide-storage'))
route='''def native_route(call,enabled):
 t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']
 for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_checkpoint_storage','lazy_resident_defaults']:
  if (cap in call['backend'])!=enabled:return False
 if t['memory_after_decode']['task_info_status']!=0 or t['lazy_default_rehydrated_bytes']!=0:return False
 if not enabled:return t['lazy_default_retired_bytes']==0 and t['lazy_default_planned_slots']==0 and all(v==0 for v in s.values())and all(v==0 for v in delta.values())
 owner=['active_weights','original_bytes','payload_bytes','metadata_bytes','allocated_bytes']
 if s['active_weights']==201:
  if s['original_bytes']!=8401715200 or s['payload_bytes']+s['metadata_bytes']!=6402163712 or s['metadata_bytes']!=32819200 or not(0<s['allocated_bytes']<s['original_bytes']):return False
 elif s['active_weights']==0:
  if not all(s[k]==0 for k in owner):return False
 else:return False
 if call['model_id']=='Qwen3.5-4B':
  return s['active_weights']==201 and t['lazy_default_retired_bytes']>0 and t['lazy_default_planned_slots']>0 and all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded'])and delta['vector_decode_encoded']==delta['decode_encoded']and delta['vector_head_encoded']==delta['head_encoded']
 if call['model_id']=='Qwen3-1.7B':
  return t['lazy_default_retired_bytes']==t['lazy_default_planned_slots']==0 and all(v==0 for v in delta.values())
 return False'''
for rel in ['analyze_resource.py','speed/analyze.py','confirmation/analyze.py']:
    p=w/rel;t=p.read_text();fn=next(n for n in ast.parse(t).body if isinstance(n,ast.FunctionDef)and n.name=='native_route');old=ast.get_source_segment(t,fn);assert t.count(old)==1;p.write_text(t.replace(old,route,1))
p=w/'freeze.py';t=p.read_text();needle="=={'GARNET_METAL_PRIMARY_BF16_BLOCKS':('OFF','ON')}";assert t.count(needle)==1
t=t.replace(needle,"=={'GARNET_METAL_PRIMARY_BF16_BLOCKS':('OFF','ON'),'GARNET_METAL_LAZY_RESIDENT_DEFAULTS':('OFF','ON'),'GARNET_METAL_WIDE_CHECKPOINT_STORAGE':('OFF','ON')}",1)
needle="assert all(flags[m]['GARNET_METAL_LAZY_RESIDENT_DEFAULTS']=='OFF'and flags[m]['GARNET_METAL_CACHED_MATRIX_METADATA']=='OFF'for m in ['off','on'])";assert t.count(needle)==1
t=t.replace(needle,"assert all(flags[m]['GARNET_METAL_CACHED_MATRIX_METADATA']=='OFF'for m in ['off','on'])",1)
t=t.replace("(w/'starting-source-manifest.json').write_text","assert manifest==json.loads((w/'correctness-source-manifest.json').read_text()),'Source changed since native correctness'\n(w/'starting-source-manifest.json').write_text",1)
t=t.replace('only default-OFF primaryBF16 storage flag differs','only default-OFF primary/lazy/wide-checkpoint-scope flags differ; actualtext staysraw/eager andwide4Busesunchangedstoragealgorithms');p.write_text(t)
p=w/'prepare_cohort.py';t=p.read_text().replace('all398actualpacked_weights','all201actualpacked_weights')
needle="assert all(v==0 for v in json.loads((w/'native-results.json').read_text()).values())";assert t.count(needle)==1
t=t.replace(needle,needle+"\nassert json.loads((w/'correctness-terminal.json').read_text())['actual_wide_VLM_and_raw_eager_text_scope_proven']",1);p.write_text(t)
p=w/'audit_gates.py';t=p.read_text().replace("old=w.parent/'worldsense-primary-kmajor'","old=w.parent/'worldsense-primary-u32'").replace(".replace('worldsense-wide-storage','worldsense-primary-kmajor')",".replace('worldsense-wide-storage','worldsense-primary-u32')")
needle="   bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace('primary_u32_metadata','')if mode=='on'else bad['backend']+' primary_u32_metadata';assert not namespace['native_route'](bad,mode=='on')";assert t.count(needle)==1
extra="""
   bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace('wide_checkpoint_storage','')if mode=='on'else bad['backend']+' wide_checkpoint_storage';assert not namespace['native_route'](bad,mode=='on')
   bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace('lazy_resident_defaults','')if mode=='on'else bad['backend']+' lazy_resident_defaults';assert not namespace['native_route'](bad,mode=='on')
   if mode=='on':
    bad=copy.deepcopy(c);bad['timings']['lazy_default_retired_bytes']=0 if c['model_id']=='Qwen3.5-4B'else 1;assert not namespace['native_route'](bad,True)
    if c['model_id']=='Qwen3-1.7B':
     bad=copy.deepcopy(c);bad['timings']['primary_encoded_delta']['head_encoded']=1;assert not namespace['native_route'](bad,True)
"""
t=t.replace(needle,needle+extra,1);p.write_text(t)
p=w/'confirmation/prepare.py';t=p.read_text();needle="assert json.loads((p/'speed/analysis.json').read_text())['passed']";assert t.count(needle)==1;p.write_text(t.replace(needle,needle+"\nassert json.loads((p/'production-speed/analysis.json').read_text())['passed']",1))
d=w/'production-speed';d.mkdir()
for name in ['prepare.py','run_stage.py','analyze.py']:
    t=(w.parent/'worldsense-primary-narrow-attention/production-speed'/name).read_text().replace('worldsense-primary-narrow-attention','worldsense-wide-storage');(d/name).write_text(t)
guard=" if not enabled:\n  return 'dense_prefix_cache'in call['backend']and not any(k in call['backend']for k in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_checkpoint_storage','lazy_resident_defaults'])and not any(k in call['timings']for k in ['primary_storage_after','primary_encoded_delta','memory_after_decode'])\n"
extra_route=route.replace(" t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']",guard+" t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']",1)
p=d/'analyze.py';t=p.read_text();fn=next(n for n in ast.parse(t).body if isinstance(n,ast.FunctionDef)and n.name=='native_route');old=ast.get_source_segment(t,fn);assert t.count(old)==1;p.write_text(t.replace(old,extra_route,1))
for p in w.rglob('*.py'):ast.parse(p.read_text(),str(p))
(w/'benchmark-preparation-scope.json').write_text(json.dumps(dict(no_worker_dispatched=True,quantitative_golden_model_input_config_source_rules_unchanged=True,wide_native_scope_guard_only=True,original_memory_and_speed_preservation_gates_retained=True,additional_actualWS10_speed20_required_before48=True),indent=2))
print('Scoped-route8/20/extraactual20/48drivers ready; no worker dispatched',flush=True)
