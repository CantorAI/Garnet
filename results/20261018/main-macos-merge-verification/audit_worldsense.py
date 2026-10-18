from pathlib import Path
import json,statistics,hashlib
w=Path(__file__).resolve().parent;repo=Path('/Users/shawn/CantorAI/Garnet');data=json.loads((w/'worldsense-main.json').read_text());gold=json.loads((repo/'results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json').read_text())
assert data['valid']and data['assignments_restored']and not data['error']
refs={i['observation_id']:i for i in gold['items']if i['policy']=='candidate'and not i['warmup']}
counts={};groups={};details=[]
for i in data['items']:
 ref=refs[i['observation_id']]
 for k in ['image_sha256','skill_ref','skill_sha256','skill']:assert i[k]==ref[k],k
 r=i['result'];r0=ref['result']
 for k in ['kind','status','skill_id','event','search','state_updates','state_changes','event_condition','event_enabled','direct_event','verify_event','search_enabled','world_state_scope','tool_trace']:
  if k in r or k in r0:assert r.get(k)==r0.get(k),(i['observation_id'],k)
 for phase in ['phase1','phase3']:
  if phase not in r['model_response']:continue
  call=r['model_response'][phase];expected=r0['model_response'][phase]
  for k in ['backend','status','generated_token_ids','generated_tokens','input_prompt_sha256','serving_configuration','model_id','text','image_preprocess_backend']:
   assert call[k]==expected[k],(i['observation_id'],phase,k)
  t=call['timings'];assert t['serving_source']==str(repo/'runtime/metal_serving.py');assert all(v==0 for v in t['primary_encoded_delta'].values());assert len(t['primary_storage_before'])==11 and len(t['primary_storage_after'])==11
  assert t['input_tokens']==t['executed_prefill_tokens']+t['prefix_cache_restored_tokens']
  if phase=='phase1':assert call['image_preprocess_backend']=='native-cpp'and max(t['prefill_chunk_plan'])==64
  else:assert call['image_preprocess_backend']==''and max(t['prefill_chunk_plan'])==16
  if not i['warmup']:
   assert phase=='phase1'or(t['prefix_cache_hit']and t['prefix_cache_restored_tokens']==64)
   groups.setdefault(call['model_id'],[]).append(dict(total_seconds=call['total_seconds'],ttft_seconds=call['ttft_seconds'],decode_tokens_per_second=(call['generated_tokens']-1)/(call['total_seconds']-call['ttft_seconds'])))
  counts[phase]=counts.get(phase,0)+1
 details.append(dict(observation_id=i['observation_id'],skill_ref=i['skill_ref'],warmup=i['warmup'],event=r['event'],frame_sha256=i['image_sha256']))
assert len(refs)==3 and set(refs)=={i['observation_id']for i in data['items']}
assert any(i['event']['matched']for i in details)and any(not i['event']['matched']for i in details)
# Search planning is an actual LLM route; retain full response in raw evidence.
assert data['search_plans'] and {p['query']for p in data['search_plans']}=={'a passenger car'}
for plan in data['search_plans']:
 reference=next(p for p in gold['search_plans']if p['query']==plan['query']and p['policy']=='candidate'and not p['warmup'])
 result=plan['result'];expected=reference['result'];assert result['model_route']=='llm'
 for k in expected:
  if k!='benchmark_model_response':assert result[k]==expected[k],('search',k)
 a=result['benchmark_model_response'];b=expected['benchmark_model_response']
 for k in ['generated_token_ids','text','model_id','input_prompt_sha256','serving_configuration']:assert a[k]==b[k],('search',k)
 assert a['timings']['serving_source']==str(repo/'runtime/metal_serving.py')
 assert all(v==0 for v in a['timings']['primary_encoded_delta'].values())
summary=dict(passed=True,scope='Mac main integration correctness; descriptive timings, no speedup claim',frames=3,items=len(data['items']),native_calls=counts,frozen_WS10_full_tokens_prompts_configs_skill_results_match=True,main_source_route=True,native_cpp_images=True,experimental_primary_encoded_deltas_zero=True,search_plans=data['search_plans'],details=details,steady_metrics={m:{k:statistics.median(c[k]for c in calls)for k in ['total_seconds','ttft_seconds','decode_tokens_per_second']}for m,calls in groups.items()})
(w/'worldsense-audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items()if k not in ['search_plans','details']},indent=2))
