"""Independent fresh-worker screen audit; no within-process lifetime-policy mixing."""
from pathlib import Path
import json,statistics,hashlib

def native_route(call,enabled,primary_control=False):
 packed=enabled or primary_control;t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']
 for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata']:
  if (cap in call['backend'])!=packed:return False
 if ('primary_vector_prefill'in call['backend'])!=enabled:return False
 if ('primary_prefill_isolated_library'in call['backend'])!=enabled:return False
 if ('primary_duo_prefill'in call['backend'])!=enabled:return False
 if ('primary_minimal_prefill_library'in call['backend'])!=enabled:return False
 if ('direct_float_weight_upload'in call['backend'])!=enabled:return False
 if ('primary_vector_prefill_k32_only'in call['backend'])!=enabled:return False
 if any(cap in call['backend']for cap in ['lazy_resident_defaults','wide_primary_storage','wide_checkpoint_storage']):return False
 if t['lazy_default_retired_bytes']!=0 or t['lazy_default_rehydrated_bytes']!=0 or t['lazy_default_planned_slots']!=0:return False
 if t['memory_after_decode']['task_info_status']!=0:return False
 if not packed:return all(v==0 for v in s.values())and all(v==0 for v in delta.values())
 expected={197:(3440902144,2623650048,13441024),201:(8401715200,6402163712,32819200),398:(11842617344,9025813760,46260224)}
 if s['active_weights']not in expected:return False
 if call['model_id']=='Qwen3.5-4B':
  if s['active_weights']not in [201,398]:return False
 elif call['model_id']=='Qwen3-1.7B':
  if s['active_weights']not in [197,398]:return False
 else:return False
 original,retained,meta=expected[s['active_weights']]
 return s['original_bytes']==original and s['payload_bytes']+s['metadata_bytes']==retained and s['metadata_bytes']==meta and 0<s['allocated_bytes']<original and all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded'])and delta['vector_decode_encoded']==delta['decode_encoded']and delta['vector_head_encoded']==delta['head_encoded']



w=Path(__file__).resolve().parent;labels=['control-1','candidate-1'];data={name:json.loads((w/(name+'.json')).read_text())for name in labels}
gold=json.loads((Path('/Users/shawn/CantorAI/Garnet/results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json')).read_text());refs={}
for item in gold['items']:
 if item['policy']!='candidate':continue
 for phase,c in item['result'].get('model_response',{}).items():
  if isinstance(c,dict)and c.get('status')=='ok':refs[(item['observation_id'],phase)]=(item,c)
manifest={m['label']:m for m in json.loads((w/'cohort-manifest.json').read_text())};environment=json.loads((w/'environment.json').read_text());source=Path('/Users/shawn/CantorAI/out/research/worldsense-k32-prefill/source');source_hashes=json.loads((w/'starting-source-manifest.json').read_text());assert all(hashlib.sha256((source/rel).read_bytes()).hexdigest()==h for rel,h in source_hashes.items())
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
   timings=c['timings'];native=native_route(c,bool(expected))
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
passed=all(c['exact']and c['native_lazy_defaults_valid']for c in checks)and memory_reduction>=.10 and wall_gain>=-.02 and all(c['ttft_change']<=.02 and c['total_change']<=.02and c['decode_gain']>=-.02 for c in comparisons)
result=dict(status='passed_screen_requires_fresh_full_confirmation'if passed else'rejected',passed=passed,checks=checks,metrics=metrics,comparisons=comparisons,sampled_max_post_request_metal_bytes=mem,sampled_metal_reduction=memory_reduction,native_skill_wall_p50=wall,native_skill_wall_gain=wall_gain,scope='Fresh single-car screen. Warm metrics separate from first; memory maxima sampled after decode over all returned calls, not peaks. Native Skill wall includes complete native request; memory snapshots count Metal resources, not free RAM or a peak.')
(w/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k not in ['metrics','checks']},indent=2))
