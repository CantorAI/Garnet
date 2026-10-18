"""Independent fresh-worker screen audit; no within-process lifetime-policy mixing."""
from pathlib import Path
import json,statistics,hashlib

def native_route(call,enabled,primary_control=False):

 if call['model_id']not in ['Qwen3.5-4B','Qwen3-1.7B']:return False
 t0=call['timings']
 if 'native_text_worker'in call['backend']or'native_gateway'in t0:return False
 for cap in ['primary_decode_groups8','request_scope_weight_residency','packed_request_weight_residency','primary_simd_metadata','sole_interleaved_mlp','sole_direct_interleaved_prefill','sole_dual_projection_decode','sole_cooperative_prefill']:
  if cap in call['backend']:return False
 if any(key in t0 for key in ['request_weight_residency','packed_request_weight_residency','packed_request_eligible','memory_after_packed_request_cleanup']):return False
 for cap in ['primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','primary_full_isolated_library']:
  if (cap in call['backend'])!=enabled:return False
 for cap in ['direct_float_weight_upload','primary_vector_prefill_k32_only','cached_matrix_metadata']:
  if cap in call['backend']:return False
 if ('primary_decode_groups2'in call['backend'])!=enabled:return False
 if 'primary_core_isolated_library'in call['backend']:return False
 if 'primary_storage_after'in t0:
  before=t0.get('primary_storage_before');after=t0.get('primary_storage_after');encoded=t0.get('primary_encoded_delta')
  fields={'active_weights','original_bytes','payload_bytes','metadata_bytes','allocated_bytes','uploads','decode_encoded','prefill_encoded','head_encoded','vector_decode_encoded','vector_head_encoded','core_library_decode_encoded','core_library_prefill_encoded','core_library_head_encoded','decode_groups2_encoded','head_groups2_encoded'}
  expected={'decode_encoded','prefill_encoded','head_encoded','vector_decode_encoded','vector_head_encoded'}
  if not isinstance(before,dict)or not isinstance(after,dict)or set(before)!=fields or set(after)!=fields:return False
  if not isinstance(encoded,dict)or set(encoded)!=expected:return False
  if any(type(v)is not int or v<0 for obj in [before,after,encoded]for v in obj.values()):return False
  if any(after[k]<before[k]or encoded[k]!=after[k]-before[k]for k in expected):return False
  for key,oldkey in [('core_library_decode_encoded','decode_encoded'),('core_library_prefill_encoded','prefill_encoded'),('core_library_head_encoded','head_encoded')]:
   if after[key]<before[key]:return False
   if after[key]-before[key]!=0:return False
   if before[key]!=0 or after[key]!=0:return False
   if call['model_id']=='Qwen3-1.7B'and after[key]!=before[key]:return False
  for key,oldkey in [('decode_groups2_encoded','decode_encoded'),('head_groups2_encoded','head_encoded')]:
   if after[key]<before[key]or after[key]-before[key]!=(encoded[oldkey]if enabled else 0):return False
   if not enabled and(before[key]!=0 or after[key]!=0):return False
   if call['model_id']=='Qwen3-1.7B'and after[key]!=before[key]:return False
 elif enabled or any(key in t0 for key in ['primary_storage_before','primary_encoded_delta']):return False
 packed=enabled or primary_control;t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']
 for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_primary_storage','lazy_resident_defaults']:
  if (cap in call['backend'])!=packed:return False
 for cap in ['primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','direct_float_weight_upload','primary_vector_prefill_k32_only','cached_matrix_metadata','primary_full_isolated_library']:
  if cap in ['primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','primary_full_isolated_library']:
   if (cap in call['backend'])!=enabled:return False
  elif cap in call['backend']:return False
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
manifest={m['label']:m for m in json.loads((w/'cohort-manifest.json').read_text())};environment=json.loads((w/'environment.json').read_text());source=Path('/Users/shawn/CantorAI/out/research/worldsense-primary-decode-groups2/source');source_hashes=json.loads((w/'starting-source-manifest.json').read_text());assert all(hashlib.sha256((source/rel).read_bytes()).hexdigest()==h for rel,h in source_hashes.items())
rows={};checks=[]
for label,dataset in data.items():
 expected=1 if label.startswith('candidate')else 0;assert dataset['valid']and dataset['assignments_restored']and len(dataset['items'])==5 and len(dataset['search_plans'])==0
 assert [(i['repetition'],i['warmup'],i['policy'])for i in dataset['items']]==[(i,i==0,'default')for i in range(5)]
 assert dataset['request']==json.loads((w/'cases.json').read_text())
 assert manifest[label]['library_sha256']==environment['libraries']['on'if expected else'off']and manifest[label]['serving_sha256']==source_hashes['runtime/metal_serving.py']
 rows[label]=[]
 for item in dataset['items']:
  golden_item=next(ri for (oid,phase),(ri,rc) in refs.items()if oid==item['observation_id'])
  stable=lambda r:{k:v for k,v in r.items()if k not in ['model_response','model_response_json']}
  assert stable(item['result'])==stable(golden_item['result']),'Skill semantic output changed'
  for phase,c in item['result'].get('model_response',{}).items():
   if not isinstance(c,dict)or c.get('status')!='ok':continue
   ri,ref=refs[(item['observation_id'],phase)]
   exact=c['generated_token_ids']==ref['generated_token_ids']and c['text']==ref['text']and c['input_prompt_sha256']==ref['input_prompt_sha256']and c['serving_configuration']==ref['serving_configuration']and c['timings']['input_tokens']==ref['timings']['input_tokens']and item['image_sha256']==ri['image_sha256']and item['skill_sha256']==ri['skill_sha256']
   timings=c['timings'];native=native_route(c,bool(expected),True)
   assert Path(c['timings']['serving_source']).resolve()==source/'runtime/metal_serving.py'
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
mem={label:max(r['call']['timings']['memory_after_decode']['metal_allocated_bytes']for r in rr)for label,rr in rows.items()}
wall={label:statistics.median(i['native_call_seconds']for i in d['items']if not i['warmup'])for label,d in data.items()}
memory_reduction=1-mem['candidate-1']/mem['control-1'];wall_gain=1-wall['candidate-1']/wall['control-1']
foot={label:max(r['call']['timings']['memory_after_decode']['task_physical_footprint_bytes']for r in rr)for label,rr in rows.items()}
matched_memory_preserved=mem['candidate-1']/mem['control-1']-1<=.02 and foot['candidate-1']/foot['control-1']-1<=.02
vlm_ttft_gain=1-metrics['candidate-1']['Qwen3.5-4B']['ttft']/metrics['control-1']['Qwen3.5-4B']['ttft']
passed=all(c['exact']and c['native_lazy_defaults_valid']for c in checks)and matched_memory_preserved and vlm_ttft_gain>=.05 and wall_gain>=-.02 and all(c['ttft_change']<=.02 and c['total_change']<=.02and c['decode_gain']>=-.02 for c in comparisons)
result=dict(matched_primary=True,matched_memory_preserved=matched_memory_preserved,sampled_max_footprint_bytes=foot,minimum_VLM_TTFT_gain=.05,observed_VLM_TTFT_gain=vlm_ttft_gain,status='passed_screen_requires_fresh_full_confirmation'if passed else'rejected',passed=passed,checks=checks,metrics=metrics,comparisons=comparisons,sampled_max_post_request_metal_bytes=mem,sampled_metal_reduction=memory_reduction,native_skill_wall_p50=wall,native_skill_wall_gain=wall_gain,scope='Primary-STORAGE-control/policy screen: shared primaryformat; lifetimes/metadata/prefill/directupload differ;5%VLMwarmTTFT improvement and2%model/wall/memory preservation required. Not isolated GPU timing or confirmed causal evidence. Warm metrics separate from first; memory maxima sampled after decode over all returned calls, not peaks. Native Skill wall includes complete native request; memory snapshots count Metal resources, not free RAM or a peak.')
(w/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k not in ['metrics','checks']},indent=2))
