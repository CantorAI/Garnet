"""Independent fresh-worker screen audit; no within-process lifetime-policy mixing."""
from pathlib import Path
import json,statistics,hashlib
w=Path(__file__).resolve().parent;labels=['control-1','candidate-1'];data={name:json.loads((w/(name+'.json')).read_text())for name in labels}
gold=json.loads((w.parent/'worldsense-prefix-cache-confirm/paired-1.json').read_text());refs={}
for item in gold['items']:
 if item['policy']!='candidate':continue
 for phase,c in item['result'].get('model_response',{}).items():
  if isinstance(c,dict)and c.get('status')=='ok':refs[(item['observation_id'],phase)]=(item,c)
manifest={m['label']:m for m in json.loads((w/'cohort-manifest.json').read_text())};environment=json.loads((w/'environment.json').read_text());source=Path('/Users/shawn/CantorAI/out/research/worldsense-request-pool/source');source_hashes=json.loads((w/'starting-source-manifest.json').read_text());assert all(hashlib.sha256((source/rel).read_bytes()).hexdigest()==h for rel,h in source_hashes.items())
rows={};checks=[]
for label,dataset in data.items():
 expected=1 if label.startswith('candidate')else 0;assert dataset['valid']and dataset['assignments_restored']and len(dataset['items'])==5 and len(dataset['search_plans'])==0
 assert [(i['repetition'],i['warmup'],i['policy'])for i in dataset['items']]==[(i,i==0,'default')for i in range(5)]
 assert dataset['request']==json.loads((w/'cases.json').read_text())
 assert manifest[label]['library_sha256']==environment['libraries']['on'if expected else'off']and manifest[label]['serving_sha256']==source_hashes['runtime/metal_serving.py']
 rows[label]=[]
 for item in dataset['items']:
  for phase,c in item['result'].get('model_response',{}).items():
   if not isinstance(c,dict)or c.get('status')!='ok':continue
   ri,ref=refs[(item['observation_id'],phase)]
   exact=c['generated_token_ids']==ref['generated_token_ids']and c['text']==ref['text']and c['input_prompt_sha256']==ref['input_prompt_sha256']and c['serving_configuration']==ref['serving_configuration']and c['timings']['input_tokens']==ref['timings']['input_tokens']and item['image_sha256']==ri['image_sha256']and item['skill_sha256']==ri['skill_sha256']
   scope=c['native_request_scope'];native=scope['invocations']==expected and scope['memory_after_scope']['task_info_status']==0
   checks.append(dict(worker=label,phase=phase,exact=exact,native_scope_valid=native))
   rows[label].append(dict(call=c,warmup=item['warmup']))
models={'Qwen3.5-4B','Qwen3-1.7B'}
metrics={}
for label,rr in rows.items():
 assert len(rr)==10 and {r['call']['model_id']for r in rr}==models;metrics[label]={}
 for model in sorted(models):
  cs=[r['call']for r in rr if not r['warmup']and r['call']['model_id']==model];assert len(cs)==4
  metrics[label][model]=dict(samples=len(cs),ttft=statistics.median(c['ttft_seconds']for c in cs),total=statistics.median(c['total_seconds']for c in cs),decode_tps=statistics.median((c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds'])for c in cs),native_api_wall=statistics.median(c['native_request_scope']['api_wall_seconds_before_final_dump']for c in cs))
comparisons=[]
for model in sorted(models):
 a,b=metrics['control-1'][model],metrics['candidate-1'][model];comparisons.append(dict(model=model,ttft_change=b['ttft']/a['ttft']-1,total_change=b['total']/a['total']-1,decode_gain=b['decode_tps']/a['decode_tps']-1,native_api_wall_change=b['native_api_wall']/a['native_api_wall']-1))
mem={label:max(r['call']['native_request_scope']['memory_after_scope']['metal_allocated_bytes']for r in rr)for label,rr in rows.items()}
wall={label:statistics.median(i['native_call_seconds']for i in d['items']if not i['warmup'])for label,d in data.items()}
memory_reduction=1-mem['candidate-1']/mem['control-1'];wall_gain=1-wall['candidate-1']/wall['control-1']
passed=all(c['exact']and c['native_scope_valid']for c in checks)and memory_reduction>=.10 and wall_gain>=.05 and all(c['ttft_change']<=.02 and c['total_change']<=(-.03 if c['model']=='Qwen3.5-4B'else .02)and c['decode_gain']>=-.02 for c in comparisons)
result=dict(status='passed_screen_requires_fresh_full_confirmation'if passed else'rejected',passed=passed,checks=checks,metrics=metrics,comparisons=comparisons,sampled_max_post_request_metal_bytes=mem,sampled_metal_reduction=memory_reduction,native_skill_wall_p50=wall,native_skill_wall_gain=wall_gain,scope='Fresh single-car screen. Warm metrics separate from first; memory maxima sampled after scope over all returned calls, not peaks. Model total excludes native pool cleanup; native Skill/API wall includes it.')
(w/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k not in ['metrics','checks']},indent=2))
