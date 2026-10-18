"""Independent fresh-worker screen audit; no within-process lifetime-policy mixing."""
from pathlib import Path
import json,statistics,hashlib

def native_route(call,enabled,primary_control=False):
 if not enabled:
  if any(cap in call['backend']for cap in ['sole_interleaved_mlp','sole_direct_interleaved_prefill','sole_dual_projection_decode','sole_cooperative_prefill','native_text_worker'])or any(k in call['timings']for k in ['sole_storage_before','sole_storage_after','sole_encoded_delta','dual_projection_before','dual_projection_after','dual_projection_delta','native_gateway']):return False
  return 'dense_prefix_cache'in call['backend']and not any(k in call['backend']for k in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','primary_vector_prefill','primary_prefill_isolated_library','lazy_resident_defaults','cached_matrix_metadata','wide_primary_storage','direct_float_weight_upload'])and not any(k in call['timings']for k in ['primary_storage_after','primary_encoded_delta','memory_after_decode'])
 packed=enabled or primary_control;t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']
 if call['model_id']not in ['Qwen3.5-4B','Qwen3-1.7B']:return False
 if 'native_text_worker'in call['backend']or'native_gateway'in t:return False
 if ('sole_interleaved_mlp'in call['backend'])!=enabled:return False
 if ('sole_direct_interleaved_prefill'in call['backend'])!=enabled:return False
 if ('sole_dual_projection_decode'in call['backend'])!=enabled:return False
 if ('sole_cooperative_prefill'in call['backend'])!=enabled:return False
 fields_dual={'encoded_pairs','represented_projections'};dual_groups=[t.get(key)for key in ['dual_projection_before','dual_projection_after','dual_projection_delta']]
 if any(not isinstance(group,dict)or set(group)!=fields_dual for group in dual_groups):return False
 if any(not isinstance(value,int)or isinstance(value,bool)or value<0 for group in dual_groups for value in group.values()):return False
 db,da,dd=dual_groups
 if any(dd[k]!=da[k]-db[k]for k in dd)or any(group['represented_projections']!=2*group['encoded_pairs']for group in dual_groups):return False
 if not enabled:
  if any(value for group in dual_groups for value in group.values()):return False
 elif call['model_id']=='Qwen3-1.7B':
  if dd['encoded_pairs']<=0 or dd['encoded_pairs']<28*(call['generated_tokens']-1):return False
 elif any(dd.values()):return False

 fields={'active_weights','allocated_bytes','uploads','decode_encoded','restore_encoded','scratch_bytes','prefill_encoded'}
 sb=t.get('sole_storage_before');ss=t.get('sole_storage_after');sd=t.get('sole_encoded_delta')
 if not isinstance(sb,dict)or not isinstance(ss,dict)or not isinstance(sd,dict):return False
 if set(sb)!=fields or set(ss)!=fields or set(sd)!={'decode_encoded','restore_encoded','prefill_encoded'}:return False
 if any(not isinstance(v,int)or isinstance(v,bool)or v<0 for v in list(sb.values())+list(ss.values())+list(sd.values())):return False
 if any(sd[k]!=ss[k]-sb[k]for k in sd):return False
 if not enabled:
  if any(sb.values())or any(ss.values())or any(sd.values()):return False
 else:
  if ss['active_weights']not in [0,84]or ss['allocated_bytes']!=ss['active_weights']*25165824 or ss['uploads']<ss['active_weights']:return False
  if ss['scratch_bytes']!=0 or ss['restore_encoded']!=0 or sd['restore_encoded']!=0:return False
  if call['model_id']=='Qwen3-1.7B':
   if ss['active_weights']!=84 or sd['decode_encoded']<=0:return False
   if not t.get('prefix_cache_hit',False)and sd['prefill_encoded']<=0:return False
  elif any(sd.values()):return False

 for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_primary_storage','lazy_resident_defaults']:
  if (cap in call['backend'])!=packed:return False
 for cap in ['primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','direct_float_weight_upload','primary_vector_prefill_k32_only','cached_matrix_metadata','primary_full_isolated_library']:
  if (cap in call['backend'])!=enabled:return False
 if 'wide_checkpoint_storage'in call['backend']:return False
 if t['lazy_default_rehydrated_bytes']!=0 or t['memory_after_decode']['task_info_status']!=0:return False
 if packed:
  if t['lazy_default_retired_bytes']<=0 or t['lazy_default_planned_slots']<=0:return False
 elif t['lazy_default_retired_bytes']!=0 or t['lazy_default_planned_slots']!=0:return False
 if not packed:return all(v==0 for v in s.values())and all(v==0 for v in delta.values())
 active=s['active_weights'];model=call['model_id']
 if model=='Qwen3.5-4B':
  if active!=201:return False
  if not all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded']):return False
  if delta['vector_decode_encoded']!=delta['decode_encoded']or delta['vector_head_encoded']!=delta['head_encoded']:return False
 elif model=='Qwen3-1.7B':
  if active not in [0,201]or any(v!=0 for v in delta.values()):return False
 else:return False
 if active==0:return all(s[k]==0 for k in ['active_weights','original_bytes','payload_bytes','metadata_bytes','allocated_bytes'])
 return s['original_bytes']==8401715200 and s['payload_bytes']+s['metadata_bytes']==6402163712 and s['metadata_bytes']==32819200 and 0<s['allocated_bytes']<8401715200



w=Path(__file__).resolve().parent;labels=['control-1','candidate-1'];data={name:json.loads((w/(name+'.json')).read_text())for name in labels}
gold=json.loads((Path('/Users/shawn/CantorAI/Garnet/results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json')).read_text());refs={}
for item in gold['items']:
 if item['policy']!='candidate':continue
 for phase,c in item['result'].get('model_response',{}).items():
  if isinstance(c,dict)and c.get('status')=='ok':refs[(item['observation_id'],phase)]=(item,c)
manifest={m['label']:m for m in json.loads((w/'cohort-manifest.json').read_text())};environment=json.loads((w/'environment.json').read_text());source=Path('/Users/shawn/CantorAI/out/research/worldsense-cooperative-prefill/source');source_hashes=json.loads((w/'starting-source-manifest.json').read_text());assert all(hashlib.sha256((source/rel).read_bytes()).hexdigest()==h for rel,h in source_hashes.items())
rows={};checks=[]
for label,dataset in data.items():
 expected=1 if label.startswith('candidate')else 0;assert dataset['valid']and dataset['assignments_restored']and len(dataset['items'])==5 and len(dataset['search_plans'])==0
 assert [(i['repetition'],i['warmup'],i['policy'])for i in dataset['items']]==[(i,i==0,'default')for i in range(5)]
 assert dataset['request']==json.loads((w/'cases.json').read_text())
 assert manifest[label]['library_sha256']==environment['libraries']['on'if expected else'off']and manifest[label]['serving_sha256']==(source_hashes['runtime/metal_serving.py']if expected else environment['control_serving_sha256'])
 rows[label]=[]
 for item in dataset['items']:
  golden_item=next(ri for (oid,phase),(ri,rc) in refs.items()if oid==item['observation_id'])
  stable=lambda r:{k:v for k,v in r.items()if k not in ['model_response','model_response_json']}
  assert stable(item['result'])==stable(golden_item['result']),'Skill semantic output changed'
  for phase,c in item['result'].get('model_response',{}).items():
   if not isinstance(c,dict)or c.get('status')!='ok':continue
   ri,ref=refs[(item['observation_id'],phase)]
   exact=c['generated_token_ids']==ref['generated_token_ids']and c['text']==ref['text']and c['input_prompt_sha256']==ref['input_prompt_sha256']and c['serving_configuration']==ref['serving_configuration']and c['timings']['input_tokens']==ref['timings']['input_tokens']and item['image_sha256']==ri['image_sha256']and item['skill_sha256']==ri['skill_sha256']
   timings=c['timings'];native=native_route(c,bool(expected))
   assert Path(c['timings']['serving_source']).resolve()==(source if expected else Path(environment['control_source']))/'runtime/metal_serving.py'
   checks.append(dict(worker=label,phase=phase,exact=exact,native_lazy_defaults_valid=native))
   rows[label].append(dict(call=c,warmup=item['warmup']))
models={'Qwen3.5-4B','Qwen3-1.7B'}
metrics={}
for label,rr in rows.items():
 assert len(rr)==10 and {r['call']['model_id']for r in rr}==models;metrics[label]={}
 for model in sorted(models):
  cs=[r['call']for r in rr if not r['warmup']and r['call']['model_id']==model];assert len(cs)==4
  metrics[label][model]=dict(samples=len(cs),ttft=statistics.median(c['ttft_seconds']for c in cs),total=statistics.median(c['total_seconds']for c in cs),decode_tps=statistics.median((c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds'])for c in cs))
comparisons=[]
for model in sorted(models):
 a,b=metrics['control-1'][model],metrics['candidate-1'][model];comparisons.append(dict(model=model,ttft_change=b['ttft']/a['ttft']-1,total_change=b['total']/a['total']-1,decode_gain=b['decode_tps']/a['decode_tps']-1))
mem={label:(max(r['call']['timings']['memory_after_decode']['metal_allocated_bytes']for r in rr)if label.startswith('candidate')else None)for label,rr in rows.items()}
wall={label:statistics.median(i['native_call_seconds']for i in d['items']if not i['warmup'])for label,d in data.items()}
memory_reduction=None;wall_gain=1-wall['candidate-1']/wall['control-1']
passed=all(c['exact']and c['native_lazy_defaults_valid']for c in checks)and wall_gain>=-.02 and all(c['ttft_change']<=.02 and c['total_change']<=.02and c['decode_gain']>=-.02 for c in comparisons)
result=dict(status='passed_screen_requires_fresh_full_confirmation'if passed else'rejected',passed=passed,checks=checks,metrics=metrics,comparisons=comparisons,sampled_max_post_request_metal_bytes=mem,sampled_metal_reduction=memory_reduction,native_skill_wall_p50=wall,native_skill_wall_gain=wall_gain,scope='Additional predeclared actualWS10screen, same correctness/budget/2%latency/decode/Skillwall gates. Oldcontrol memory API unavailable/null. Separate instrumented matched resource and confirmation memory gates remain required. First/warm separate; no promotion from this screen.')
(w/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k not in ['metrics','checks']},indent=2))
