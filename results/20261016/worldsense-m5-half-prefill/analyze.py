import json,pathlib,statistics,hashlib
w=pathlib.Path(__file__).resolve().parent
labels=['control-1','candidate-1','candidate-2','control-2']
data={label:json.loads((w/(label+'.json')).read_text()) for label in labels}
def records(d):
 r=[]
 for sample in d['items']:
  for phase,call in sample['result'].get('model_response',{}).items():
   if isinstance(call,dict)and call.get('status')=='ok':
    r.append({'key':[sample['observation_id'],sample['skill_ref'],phase,sample['repetition']], 'warmup':sample['warmup'],'image_sha256':sample['image_sha256'],'skill_sha256':sample['skill_sha256'],'call':call,'native_call_seconds':sample['native_call_seconds']})
 for sample in d.get('search_plans',[]):
  call=sample['result'].get('benchmark_model_response',{})
  r.append({'key':[sample['query'],'search_plan',sample['repetition']], 'warmup':sample['warmup'],'call':call,'native_call_seconds':sample['native_call_seconds']})
 return r
rows={label:records(d)for label,d in data.items()}
reference={tuple(r['key']):r for r in rows['control-1']}
checks=[]
for label,rr in rows.items():
 assert data[label]['valid']and data[label]['assignments_restored'],label
 assert set(tuple(r['key'])for r in rr)==set(reference),label
 for r in rr:
  prior=reference[tuple(r['key'])];a,b=prior['call'],r['call']
  checks.append({'label':label,'key':r['key'],'tokens_equal':a['generated_token_ids']==b['generated_token_ids'],'text_equal':a['text']==b['text'],'inputs_equal':a['timings']['input_tokens']==b['timings']['input_tokens'] and prior.get('image_sha256')==r.get('image_sha256') and prior.get('skill_sha256')==r.get('skill_sha256')})
def metric(rr,model):
 calls=[r['call']for r in rr if not r['warmup']and r['call']['model_id']==model]
 med=lambda field:statistics.median([c[field]for c in calls])
 return {'n':len(calls),'total_seconds_p50':med('total_seconds'),'ttft_seconds_p50':med('ttft_seconds'),'decode_tps_p50':statistics.median([(c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds'])for c in calls]),'stages_p50':{key:statistics.median([c['timings'].get(key,0)for c in calls])for key in ['reset_seconds','vision_seconds','host_input_seconds','prefill_seconds','decode_seconds']},'fused_decode_dispatches':sum(c['timings']['decode_fused_head_dispatches']for c in calls)}
models=sorted({r['call']['model_id']for r in rows['control-1']});metrics={label:{model:metric(rr,model)for model in models}for label,rr in rows.items()};comparisons=[]
for control,candidate in [('control-1','candidate-1'),('control-2','candidate-2')]:
 for model in models:
  a,b=metrics[control][model],metrics[candidate][model]
  comparisons.append({'control':control,'candidate':candidate,'model':model,'decode_gain':b['decode_tps_p50']/a['decode_tps_p50']-1,'ttft_change':b['ttft_seconds_p50']/a['ttft_seconds_p50']-1,'total_change':b['total_seconds_p50']/a['total_seconds_p50']-1})
exact=all(c['tokens_equal']and c['text_equal']and c['inputs_equal']for c in checks)
passed=exact and all(c['decode_gain']>=-.02 and (c['ttft_change']<=-.10 and c['total_change']<=-.05 if c['model']=='Qwen3.5-4B'else c['ttft_change']<=.02 and c['total_change']<=.02)for c in comparisons)
result={'status':'screen_pass_needs_confirmation'if passed else'rejected','passed_screen':passed,'exact_outputs_and_inputs':exact,'checks':checks,'metrics':metrics,'comparisons':comparisons,'note':'Two warm requests per fixed frame/query in each fresh worker. Screening only; no confidence/significance or general throughput claim.'}
(w/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k]for k in ['status','exact_outputs_and_inputs','comparisons']},indent=2))
