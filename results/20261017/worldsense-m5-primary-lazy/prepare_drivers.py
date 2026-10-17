"""Prepare the WS45 joint-policy cohort drivers, never dispatch workers."""
from pathlib import Path
import ast,json
w=Path(__file__).resolve().parent;seed=w.parent/'worldsense-wide-storage'
files=['freeze.py','prepare_cohort.py','run_resource.py','analyze_resource.py','dependency_closure.py','speed/prepare.py','speed/run_stage.py','speed/analyze.py','confirmation/prepare.py','confirmation/run_stage.py','confirmation/analyze.py','production-speed/prepare.py','production-speed/run_stage.py','production-speed/analyze.py']
for rel in files:
 p=w/rel;assert not p.exists(),rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text((seed/rel).read_text().replace('worldsense-wide-storage','worldsense-primary-lazy'))
route='''def native_route(call,enabled):
 t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']
 for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','lazy_resident_defaults']:
  if (cap in call['backend'])!=enabled:return False
 if 'wide_checkpoint_storage'in call['backend']:return False
 if t['memory_after_decode']['task_info_status']!=0 or t['lazy_default_rehydrated_bytes']!=0:return False
 if not enabled:return t['lazy_default_retired_bytes']==0 and t['lazy_default_planned_slots']==0 and all(v==0 for v in s.values())and all(v==0 for v in delta.values())
 tuples={197:(3440902144,2623650048,13441024),201:(8401715200,6402163712,32819200),398:(11842617344,9025813760,46260224)}
 if s['active_weights']not in tuples:return False
 if (s['original_bytes'],s['payload_bytes']+s['metadata_bytes'],s['metadata_bytes'])!=tuples[s['active_weights']]:return False
 if not(0<s['allocated_bytes']<s['original_bytes']):return False
 if call['model_id']=='Qwen3.5-4B':
  if s['active_weights']not in [201,398]:return False
 elif call['model_id']=='Qwen3-1.7B':
  if s['active_weights']not in [197,398]:return False
 else:return False
 return t['lazy_default_retired_bytes']>0 and t['lazy_default_planned_slots']>0 and all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded'])and delta['vector_decode_encoded']==delta['decode_encoded']and delta['vector_head_encoded']==delta['head_encoded']'''
for rel in ['analyze_resource.py','speed/analyze.py','confirmation/analyze.py','production-speed/analyze.py']:
 p=w/rel;text=p.read_text();fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='native_route');old=ast.get_source_segment(text,fn)
 new=route
 if rel.startswith('production-speed/'):
  guard=" if not enabled:\n  return 'dense_prefix_cache'in call['backend']and not any(k in call['backend']for k in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','lazy_resident_defaults','wide_checkpoint_storage'])and not any(k in call['timings']for k in ['primary_storage_after','primary_encoded_delta','memory_after_decode'])\n"
  new=new.replace(" t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']",guard+" t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']",1)
 p.write_text(text.replace(old,new,1))
p=w/'freeze.py';t=p.read_text();t=t.replace(",'GARNET_METAL_WIDE_CHECKPOINT_STORAGE':('OFF','ON')",'');t=t.replace('only default-OFF primary/lazy/wide-checkpoint-scope flags differ; actualtext staysraw/eager andwide4Busesunchangedstoragealgorithms','only default-OFF primary/lazy flags differ; both models use existing primary and retirement algorithms');p.write_text(t)
p=w/'prepare_cohort.py';t=p.read_text().replace('all201actualpacked_weights','all398actualpacked_weights').replace('actual_wide_VLM_and_raw_eager_text_scope_proven','joint_primary_and_lazy_scope_proven');p.write_text(t)
t=(seed/'audit_gates.py').read_text().replace('worldsense-wide-storage','worldsense-primary-lazy')
t=t.replace("   bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace('wide_checkpoint_storage','')if mode=='on'else bad['backend']+' wide_checkpoint_storage';assert not namespace['native_route'](bad,mode=='on')","   bad=copy.deepcopy(c);bad['backend']+=' wide_checkpoint_storage';assert not namespace['native_route'](bad,mode=='on')")
start=t.index("   if mode=='on':\n")
end=t.index("   bad=copy.deepcopy(c);bad['timings']['primary_storage_after']['metadata_bytes']",start)
t=t[:start]+"   if mode=='on':\n    bad=copy.deepcopy(c);bad['timings']['lazy_default_retired_bytes']=0;assert not namespace['native_route'](bad,True)\n    bad=copy.deepcopy(c);bad['timings']['primary_encoded_delta']['head_encoded']=0;assert not namespace['native_route'](bad,True)\n\n"+t[end:]
(w/'audit_gates.py').write_text(t)
for p in w.rglob('*.py'):ast.parse(p.read_text(),str(p))
(w/'benchmark-preparation-scope.json').write_text(json.dumps({'no_worker_dispatched':True,'both_models_primary_and_lazy':True,'original_numeric_golden_metric_AST_to_be_audited':True,'actual_control_extra20_and_both_full_pairs_required':True},indent=2)+'\n')
print('WS45 8/20/extra actual20/full48 gates prepared; no workers dispatched.')
