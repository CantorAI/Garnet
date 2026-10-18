"""Independent fresh-worker screen audit; no within-process lifetime-policy mixing."""
from pathlib import Path
import json,statistics,hashlib
import sys
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




w=Path(sys.argv[1]).resolve();labels=sys.argv[2:4];assert len(labels)==2 and labels[0].startswith('control') and labels[1].startswith('candidate');data={name:json.loads((w/(name+'.json')).read_text())for name in labels}
gold=json.loads((Path('/Users/shawn/CantorAI/Garnet/results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json')).read_text());refs={}
for item in gold['items']:
 if item['policy']!='candidate':continue
 for phase,c in item['result'].get('model_response',{}).items():
  if isinstance(c,dict)and c.get('status')=='ok':refs[(item['observation_id'],phase)]=(item,c)
manifest={m['label']:m for m in json.loads((w/'cohort-manifest.json').read_text())};environment=json.loads((w.parent/'environment.json').read_text());source=Path('/Users/shawn/CantorAI/out/research/worldsense-validated-text-worker/source');source_hashes=json.loads((w.parent/'starting-source-manifest.json').read_text());assert all(hashlib.sha256((source/rel).read_bytes()).hexdigest()==h for rel,h in source_hashes.items())
rows={};checks=[]
for label,dataset in data.items():
 expected=1 if label.startswith('candidate')else 0;assert dataset['valid']and dataset['assignments_restored']and len(dataset['items'])==6 and len(dataset['search_plans'])==2
 assert [(i['repetition'],i['warmup'],i['policy'])for i in dataset['items']]==[(r,r==0,'default')for r in range(2)for _ in range(3)]
 assert dataset['request']==json.loads((w/'cases.json').read_text())
 assert [(q['repetition'],q['warmup'],q['policy'])for q in dataset['search_plans']]==[(0,True,'default'),(1,False,'default')]
 assert all(q['query']==json.loads((w/'cases.json').read_text())['queries'][0]for q in dataset['search_plans'])
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
   timings=c['timings'];native=native_route(c,bool(expected))
   assert Path(c['timings']['serving_source']).resolve()==source/('runtime/validated_ws10/metal_serving.py'if bool(expected) and c['model_id']=='Qwen3-1.7B'else'runtime/metal_serving.py')
   checks.append(dict(worker=label,phase=phase,exact=exact,native_lazy_defaults_valid=native))
   rows[label].append(dict(call=c,warmup=item['warmup']))
 for query in dataset['search_plans']:
  c=query['result']['benchmark_model_response'];refq=next(q for q in gold['search_plans']if q['policy']=='candidate'and q['query']==query['query']);ref=refq['result']['benchmark_model_response']
  exact=all(c[k]==ref[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])and c['timings']['input_tokens']==ref['timings']['input_tokens']
  exact=exact and {k:v for k,v in query['result'].items()if k!='benchmark_model_response'}=={k:v for k,v in refq['result'].items()if k!='benchmark_model_response'}
  timings=c['timings'];native=native_route(c,bool(expected))
  assert Path(c['timings']['serving_source']).resolve()==source/('runtime/validated_ws10/metal_serving.py'if bool(expected) and c['model_id']=='Qwen3-1.7B'else'runtime/metal_serving.py')
  checks.append(dict(worker=label,phase='query_plan',exact=exact,native_lazy_defaults_valid=native));rows[label].append(dict(call=c,warmup=query['warmup']))
models={'Qwen3.5-4B','Qwen3-1.7B'}
metrics={}
for label,rr in rows.items():
 assert len(rr)==12 and {r['call']['model_id']for r in rr}==models;metrics[label]={}
 for model in sorted(models):
  cs=[r['call']for r in rr if not r['warmup']and r['call']['model_id']==model];assert len(cs)==3
  metrics[label][model]=dict(samples=len(cs),ttft=statistics.median(c['ttft_seconds']for c in cs),total=statistics.median(c['total_seconds']for c in cs),decode_tps=statistics.median((c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds'])for c in cs))
comparisons=[]
for model in sorted(models):
 a,b=metrics[labels[0]][model],metrics[labels[1]][model];comparisons.append(dict(model=model,ttft_change=b['ttft']/a['ttft']-1,total_change=b['total']/a['total']-1,decode_gain=b['decode_tps']/a['decode_tps']-1))
mem={label:max(r['call']['timings']['memory_after_decode']['metal_allocated_bytes']for r in rr)for label,rr in rows.items()}
wall={label:statistics.median(i['native_call_seconds']for i in d['items']if not i['warmup'])for label,d in data.items()}
memory_reduction=1-mem[labels[1]]/mem[labels[0]];wall_gain=1-wall[labels[1]]/wall[labels[0]]
passed=all(c['exact']and c['native_lazy_defaults_valid']for c in checks)and memory_reduction>=.10 and wall_gain>=-.02 and all(c['ttft_change']<=.02 and c['total_change']<=.02and c['decode_gain']>=-.02 for c in comparisons)
result=dict(status='passed_individual_confirmation_pair'if passed else'rejected',passed=passed,checks=checks,metrics=metrics,comparisons=comparisons,sampled_max_post_request_metal_bytes=mem,sampled_metal_reduction=memory_reduction,native_skill_wall_p50=wall,native_skill_wall_gain=wall_gain,scope='Fresh three-frame and query-plan confirmation pair. Warm metrics separate from first; memory maxima sampled after decode over all returned calls, not peaks. Native Skill wall includes complete native request; memory snapshots count Metal resources, not free RAM or a peak.')
(w/('analysis-'+labels[0]+'-'+labels[1]+'.json')).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k not in ['metrics','checks']},indent=2))
