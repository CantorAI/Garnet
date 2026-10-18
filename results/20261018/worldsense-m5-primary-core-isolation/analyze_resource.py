"""Prospective WorldSense resource feasibility gate; no speed promotion."""
from pathlib import Path
import json,hashlib

def native_route(call,enabled,primary_control=False):

 if call['model_id']not in ['Qwen3.5-4B','Qwen3-1.7B']:return False
 t0=call['timings']
 if 'native_text_worker'in call['backend']or'native_gateway'in t0:return False
 for cap in ['request_scope_weight_residency','packed_request_weight_residency','primary_simd_metadata','sole_interleaved_mlp','sole_direct_interleaved_prefill','sole_dual_projection_decode','sole_cooperative_prefill']:
  if cap in call['backend']:return False
 if any(key in t0 for key in ['request_weight_residency','packed_request_weight_residency','packed_request_eligible','memory_after_packed_request_cleanup']):return False
 if ('primary_core_isolated_library'in call['backend'])!=enabled:return False
 if 'primary_storage_after'in t0:
  before=t0.get('primary_storage_before');after=t0.get('primary_storage_after');encoded=t0.get('primary_encoded_delta')
  fields={'active_weights','original_bytes','payload_bytes','metadata_bytes','allocated_bytes','uploads','decode_encoded','prefill_encoded','head_encoded','vector_decode_encoded','vector_head_encoded','core_library_decode_encoded','core_library_prefill_encoded','core_library_head_encoded'}
  expected={'decode_encoded','prefill_encoded','head_encoded','vector_decode_encoded','vector_head_encoded'}
  if not isinstance(before,dict)or not isinstance(after,dict)or set(before)!=fields or set(after)!=fields:return False
  if not isinstance(encoded,dict)or set(encoded)!=expected:return False
  if any(type(v)is not int or v<0 for obj in [before,after,encoded]for v in obj.values()):return False
  if any(after[k]<before[k]or encoded[k]!=after[k]-before[k]for k in expected):return False
  for key,oldkey in [('core_library_decode_encoded','decode_encoded'),('core_library_prefill_encoded','prefill_encoded'),('core_library_head_encoded','head_encoded')]:
   if after[key]<before[key]:return False
   if after[key]-before[key]!=(encoded[oldkey]if enabled else 0):return False
   if not enabled and(before[key]!=0 or after[key]!=0):return False
   if call['model_id']=='Qwen3-1.7B'and after[key]!=before[key]:return False
 elif enabled or any(key in t0 for key in ['primary_storage_before','primary_encoded_delta']):return False
 packed=enabled or primary_control;t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']
 for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_primary_storage','lazy_resident_defaults']:
  if (cap in call['backend'])!=packed:return False
 for cap in ['primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','direct_float_weight_upload','primary_vector_prefill_k32_only','cached_matrix_metadata','primary_full_isolated_library']:
  if cap in call['backend']:return False
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



w=Path(__file__).resolve().parent;root=Path('/Users/shawn/CantorAI');source=root/'out/research/worldsense-primary-core-isolation/source';hashes=json.loads((w/'starting-source-manifest.json').read_text());assert all(hashlib.sha256((source/k).read_bytes()).hexdigest()==h for k,h in hashes.items())
gold=json.loads((root/'Garnet/results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json').read_text());refs={}
for item in gold['items']:
 if item['policy']=='candidate':
  for phase,c in item['result'].get('model_response',{}).items():
   if isinstance(c,dict)and c.get('status')=='ok':refs[(item['observation_id'],phase)]=(item,c)
manifest={m['label']:m for m in json.loads((w/'cohort-manifest.json').read_text())};env=json.loads((w/'environment.json').read_text());rows=[];maximum={};checks=[]
for label,mode in [('control-1','off'),('candidate-1','on')]:
 d=json.loads((w/(label+'.json')).read_text());assert d['valid']and d['assignments_restored']and len(d['items'])==2 and not d['search_plans'];assert d['request']==json.loads((w/'cases.json').read_text());assert [(i['repetition'],i['warmup'],i['policy'])for i in d['items']]==[(0,True,'default'),(1,False,'default')]
 assert manifest[label]['library_sha256']==env['libraries'][mode]and manifest[label]['serving_sha256']==hashes['runtime/metal_serving.py'];calls=[]
 for item in d['items']:
  for phase,c in item['result']['model_response'].items():
   if not isinstance(c,dict)or c.get('status')!='ok':continue
   ri,ref=refs[(item['observation_id'],phase)];stable=lambda r:{k:v for k,v in r.items()if k not in ['model_response','model_response_json']};exact=stable(item['result'])==stable(ri['result'])and all(c[k]==ref[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])and c['timings']['input_tokens']==ref['timings']['input_tokens']and item['image_sha256']==ri['image_sha256']and item['skill_sha256']==ri['skill_sha256']
   assert Path(c['timings']['serving_source']).resolve()==source/'runtime/metal_serving.py'
   t=c['timings'];native=native_route(c,mode=='on');memory=t['memory_after_decode'];assert memory['task_info_status']==0
   checks.append(dict(worker=label,phase=phase,exact=exact,native_admission=native));calls.append(c);rows.append(dict(worker=label,model=c['model_id'],warmup=item['warmup'],retired_default_bytes=t['lazy_default_retired_bytes'],metal_allocated_bytes=memory['metal_allocated_bytes'],task_physical_footprint_bytes=memory['task_physical_footprint_bytes'],task_compressed_bytes=memory['task_compressed_bytes']))
 assert len(calls)==4 and {c['model_id']for c in calls}=={'Qwen3.5-4B','Qwen3-1.7B'}
 maximum[label]={key:max(c['timings']['memory_after_decode'][key]for c in calls)for key in ['metal_allocated_bytes','task_physical_footprint_bytes']}
reduction={key:1-maximum['candidate-1'][key]/maximum['control-1'][key]for key in maximum['control-1']};passed=all(c['exact']and c['native_admission']for c in checks)and reduction['metal_allocated_bytes']>=.10 and reduction['task_physical_footprint_bytes']>=.05;result=dict(status='resource_feasibility_pass_requires_fresh_speed_screen'if passed else'rejected_resource_feasibility',passed=passed,checks=checks,sampled_maximum_bytes=maximum,reduction=reduction,calls=rows,scope='Eight actual WorldSense/Galaxy calls over two fresh workers and one warmup+one followup each. Maximum post-decode snapshots, not peaks/free system RAM. No timing or speed promotion from this resource feasibility check.')
(w/'resource-analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k not in ['checks','calls']},indent=2))
