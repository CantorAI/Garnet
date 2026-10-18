"""Prospective WorldSense resource feasibility gate; no speed promotion."""
from pathlib import Path
import json,hashlib

def native_route(call,enabled,primary_control=False):
 import math
 try:
  packed=enabled or primary_control;t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta'];model=call['model_id'];remote=packed and model=='Qwen3-1.7B'
  primary_caps=['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_primary_storage','lazy_resident_defaults']
  vector_caps=['primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','direct_float_weight_upload','primary_vector_prefill_k32_only','cached_matrix_metadata','primary_full_isolated_library']
  if model not in ['Qwen3.5-4B','Qwen3-1.7B']:return False
  if model=='Qwen3.5-4B'and call['image_preprocess_backend']!='native-cpp':return False
  if 'wide_checkpoint_storage'in call['backend']:return False
  for cap in primary_caps:
   if(cap in call['backend'])!=(packed and not remote):return False
  for cap in vector_caps:
   if(cap in call['backend'])!=(enabled and not remote):return False
  if('native_text_worker'in call['backend'])!=(packed and not remote):return False
  if t['lazy_default_rehydrated_bytes']!=0 or t['memory_after_decode']['task_info_status']!=0:return False
  if packed and not remote:
   if t['lazy_default_retired_bytes']<=0 or t['lazy_default_planned_slots']<=0:return False
  elif t['lazy_default_retired_bytes']!=0 or t['lazy_default_planned_slots']!=0:return False
  if not packed:
   return 'native_gateway'not in t and all(v==0 for v in s.values())and all(v==0 for v in delta.values())
  g=t['native_gateway'];parent=g['parent_backend'];pid=g['parent_pid'];memory=t['memory_after_decode'];parts=memory['process_samples'];child=g['child']
  if g['route']!=('native_lrpc_raw_text'if remote else'local')or'native_text_worker'not in parent:return False
  if 'native_text_worker_validated_ws10'not in parent:return False
  if any((cap in parent)!=True for cap in primary_caps)or any((cap in parent)!=enabled for cap in vector_caps):return False
  if g['model_memory_after_decode']['task_info_status']!=0:return False
  if not all(math.isfinite(g[k])for k in ['gateway_started_monotonic','model_started_monotonic','first_token_monotonic','raw_model_ttft_seconds','raw_model_total_seconds']):return False
  if not(g['gateway_started_monotonic']<=g['model_started_monotonic']<=g['first_token_monotonic']<=g['gateway_started_monotonic']+call['total_seconds']+.0001):return False
  if abs(call['ttft_seconds']-(g['first_token_monotonic']-g['gateway_started_monotonic']))>1e-9:return False
  if min(g['raw_model_ttft_seconds'],g['raw_model_total_seconds'])<0 or g['raw_model_ttft_seconds']>call['ttft_seconds']+.0001 or g['raw_model_total_seconds']>call['total_seconds']+.0001:return False
  if not math.isfinite(call['total_seconds'])or call['total_seconds']<call['ttft_seconds']:return False
  expected_pids={pid}
  if child is not None:
   support=child['support'];expected_pids.add(support['pid'])
   if support['parent_pid']!=pid or support['pid']==pid or not support['watchdog_active']or support['watch_parent_pid']!=pid:return False
   if child['protocol']!='garnet-native-text-v1'or not child['birth_id']or not child['clock_epoch_metadata_enabled']:return False
   role='on'if enabled else'primary'
   if support['library_path']!='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-validated-text-worker-'+role+'/libgarnet_worker_host.dylib'or support['source_root']!='/Users/shawn/CantorAI/out/research/worldsense-validated-text-worker/source':return False
   backend=child['model_backend']
   if backend['library_path']!='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib'or backend['library_sha256']!='dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72'or backend['loaded_garnet_images']!=1 or backend['package_abi']<=0:return False
   if backend['expected_original_source_root']!='/Users/shawn/CantorAI/out/research/worldsense-prefix-cache/source':return False
   if child['decoder_source']!='/Users/shawn/CantorAI/out/research/worldsense-prefix-cache/source/xModel/qwen/metal_dense/decoder.py'or child['serving_source']!='/Users/shawn/CantorAI/out/research/worldsense-validated-text-worker/source/runtime/validated_ws10/metal_serving.py':return False
   if child['raw_counter_policy']!='pinned_ws10_structural_absence'or not isinstance(child['raw_counter_origin'],str)or'Structural zeros'not in child['raw_counter_origin']:return False
   if remote:
    if t['serving_source']!=child['serving_source']or t['counter_origin']!=child['raw_counter_origin']:return False
    for key in ['primary_storage_before','primary_storage_after','primary_encoded_delta']:
     if any(not isinstance(v,int)or isinstance(v,bool)or v!=0 for v in t[key].values()):return False
    for key in ['lazy_default_planned_slots','lazy_default_retired_bytes','lazy_default_rehydrated_bytes']:
     if not isinstance(t[key],int)or isinstance(t[key],bool)or t[key]!=0:return False
   if any(cap in child['backend']for cap in primary_caps+vector_caps+['native_text_worker']):return False
   if remote and(child['backend']!=call['backend']or child['models']!=[model]or child['infer_calls']!=call['native_worker_call_index']):return False
  elif remote:return False
  if len(parts)!=len(expected_pids)or {p['pid']for p in parts}!=expected_pids or memory['cohort_process_count']!=len(parts)or g['process_count']!=len(parts):return False
  if parts[0]['pid']!=pid:return False
  for p in parts:
   if p['task_info_status']!=0 or p['metal_registry_id']!=memory['metal_registry_id']:return False
   if not math.isfinite(p['snapshot_monotonic'])or not(g['first_token_monotonic']<=p['snapshot_monotonic']<=g['gateway_started_monotonic']+call['total_seconds']+.0001):return False
   if p['pid']!=pid and p['birth_id']!=child['birth_id']:return False
  for key in ['metal_allocated_bytes','task_physical_footprint_bytes','task_resident_bytes','task_compressed_bytes','task_internal_bytes','task_virtual_bytes']:
   if any(not isinstance(p[key],int)or isinstance(p[key],bool)or p[key]<0 for p in parts)or memory[key]!=sum(p[key]for p in parts):return False
  if memory['metal_recommended_working_set_bytes']!=parts[0]['metal_recommended_working_set_bytes']:return False
  before=g['parent_primary_storage_before'];after=g['parent_primary_storage_after'];parent_delta=g['parent_primary_encoded_delta']
  for key in delta:
   if after[key]<before[key]or parent_delta[key]!=after[key]-before[key]:return False
  if remote:
   if before!=after or any(v!=0 for v in parent_delta.values())or any(v!=0 for v in s.values())or any(v!=0 for v in delta.values()):return False
  elif after!=s or parent_delta!=delta:return False
  active=after['active_weights']
  if model=='Qwen3.5-4B':
   if active!=201 or not all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded']):return False
   if delta['vector_decode_encoded']!=delta['decode_encoded']or delta['vector_head_encoded']!=delta['head_encoded']:return False
  elif active not in [0,201]:return False
  if active==0:return all(after[k]==0 for k in ['active_weights','original_bytes','payload_bytes','metadata_bytes','allocated_bytes'])
  return after['original_bytes']==8401715200 and after['payload_bytes']+after['metadata_bytes']==6402163712 and after['metadata_bytes']==32819200 and 0<after['allocated_bytes']<8401715200
 except (KeyError,ValueError,TypeError,OverflowError):return False



w=Path(__file__).resolve().parent;root=Path('/Users/shawn/CantorAI');source=root/'out/research/worldsense-validated-text-worker/source';hashes=json.loads((w/'starting-source-manifest.json').read_text());assert all(hashlib.sha256((source/k).read_bytes()).hexdigest()==h for k,h in hashes.items())
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
   assert Path(c['timings']['serving_source']).resolve()==source/('runtime/validated_ws10/metal_serving.py'if mode=='on' and c['model_id']=='Qwen3-1.7B'else'runtime/metal_serving.py')
   t=c['timings'];native=native_route(c,mode=='on');memory=t['memory_after_decode'];assert memory['task_info_status']==0
   checks.append(dict(worker=label,phase=phase,exact=exact,native_admission=native));calls.append(c);rows.append(dict(worker=label,model=c['model_id'],warmup=item['warmup'],retired_default_bytes=t['lazy_default_retired_bytes'],metal_allocated_bytes=memory['metal_allocated_bytes'],task_physical_footprint_bytes=memory['task_physical_footprint_bytes'],task_compressed_bytes=memory['task_compressed_bytes']))
 assert len(calls)==4 and {c['model_id']for c in calls}=={'Qwen3.5-4B','Qwen3-1.7B'}
 maximum[label]={key:max(c['timings']['memory_after_decode'][key]for c in calls)for key in ['metal_allocated_bytes','task_physical_footprint_bytes']}
reduction={key:1-maximum['candidate-1'][key]/maximum['control-1'][key]for key in maximum['control-1']};passed=all(c['exact']and c['native_admission']for c in checks)and reduction['metal_allocated_bytes']>=.10 and reduction['task_physical_footprint_bytes']>=.05;result=dict(status='resource_feasibility_pass_requires_fresh_speed_screen'if passed else'rejected_resource_feasibility',passed=passed,checks=checks,sampled_maximum_bytes=maximum,reduction=reduction,calls=rows,scope='Eight actual WorldSense/Galaxy calls over two fresh workers and one warmup+one followup each. Maximum post-decode snapshots, not peaks/free system RAM. No timing or speed promotion from this resource feasibility check.')
(w/'resource-analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k not in ['checks','calls']},indent=2))
