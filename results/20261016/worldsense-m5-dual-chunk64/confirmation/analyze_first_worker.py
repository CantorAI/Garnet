"""Independent raw-result audit for the prospectively declared paired screen."""
import json,pathlib,statistics,hashlib
w=pathlib.Path(__file__).resolve().parent
source=pathlib.Path('/Users/shawn/CantorAI/out/research/worldsense-dual-chunk64/source');baseline=pathlib.Path('/Users/shawn/CantorAI/Garnet')
decode_source_equal=all((source/rel).read_bytes()==(baseline/rel).read_bytes()for rel in ['src/backends/metal/metal_context.mm','src/backends/metal/decoder_session.cpp','src/backends/metal/body_shape_shader.h'])
labels=['paired-1'];datasets={label:json.loads((w/(label+'.json')).read_text())for label in labels}
def records(d):
 rows=[]
 for item in d['items']:
  for phase,call in item['result'].get('model_response',{}).items():
   if isinstance(call,dict)and call.get('status')=='ok':rows.append(dict(key=[item['observation_id'],item['skill_ref'],phase],policy=item.get('policy'),warmup=item['warmup'],call=call,image_sha256=item['image_sha256'],skill_sha256=item['skill_sha256']))
 for item in d['search_plans']:
  rows.append(dict(key=[item['query'],'search_plan'],policy=item.get('policy'),warmup=item['warmup'],call=item['result']['benchmark_model_response']))
 return rows
old=json.loads((w.parent/'worldsense-shared-half/control-1.json').read_text());reference={tuple(row['key']):row for row in records(old)};rows={label:records(d)for label,d in datasets.items()};checks=[]
for label,rr in rows.items():
 assert datasets[label]['valid']and datasets[label]['assignments_restored']
 assert len(datasets[label]['items'])==18 and len(datasets[label]['search_plans'])==6
 for row in rr:
  ref=reference[tuple(row['key'])];a,b=ref['call'],row['call'];timing=b['timings'];count=max(timing['prefill_chunk_plan']);policy=row['policy'];expected_chunk=64 if policy=='candidate' else 16
  checks.append(dict(worker=label,key=row['key'],policy=policy,tokens_equal=a['generated_token_ids']==b['generated_token_ids'],text_equal=a['text']==b['text'],inputs_equal=a['timings']['input_tokens']==timing['input_tokens']and ref.get('image_sha256')==row.get('image_sha256')and ref.get('skill_sha256')==row.get('skill_sha256'),correct_prefill_route=count==expected_chunk,decode_untouched=decode_source_equal,serving_source=timing.get('serving_source')))
models=sorted({r['call']['model_id']for r in rows['paired-1']})
def metric(rr,model,policy):
 cs=[r['call']for r in rr if not r['warmup']and r['policy']==policy and r['call']['model_id']==model]
 assert len(cs)==6
 return dict(samples=len(cs),total_seconds_p50=statistics.median(c['total_seconds']for c in cs),ttft_seconds_p50=statistics.median(c['ttft_seconds']for c in cs),decode_tps_p50=statistics.median((c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds'])for c in cs),stages_p50={key:statistics.median(c['timings'].get(key,0)for c in cs)for key in ['reset_seconds','vision_seconds','host_input_seconds','prefill_seconds','decode_seconds']})
metrics={label:{model:{policy:metric(rr,model,policy)for policy in ['control','candidate']}for model in models}for label,rr in rows.items()};comparisons=[]
for label,models_data in metrics.items():
 for model,groups in models_data.items():
  a,b=groups['control'],groups['candidate'];comparisons.append(dict(worker=label,model=model,decode_gain=b['decode_tps_p50']/a['decode_tps_p50']-1,ttft_change=b['ttft_seconds_p50']/a['ttft_seconds_p50']-1,total_change=b['total_seconds_p50']/a['total_seconds_p50']-1))
exact=all(c['tokens_equal']and c['text_equal']and c['inputs_equal']for c in checks);routes=all(c['correct_prefill_route']and c['decode_untouched']for c in checks)
passed=exact and routes and all(c['decode_gain']>=-.02 and (c['ttft_change']<=-.10 and c['total_change']<=-.03 if c['model']=='Qwen3.5-4B'else c['ttft_change']<=.02 and c['total_change']<=.02)for c in comparisons)
wall={label:{policy:statistics.median(item['native_call_seconds']for item in d['items']if not item['warmup']and item['policy']==policy)for policy in ['control','candidate']}for label,d in datasets.items()}
passed=passed and all(group['candidate']<group['control']for group in wall.values())
r=dict(status='interim_first_worker_pass_not_full_confirmation'if passed else'rejected',passed=passed,exact_outputs_and_inputs=exact,actual_routes_valid=routes,checks=checks,metrics=metrics,comparisons=comparisons,current_skill_native_wall_p50=wall,scope='Paired WorldSense/Galaxy current-Skill and text planning; no decode/storage/event-scheduler end-to-end claim')
(w/'worker-1-analysis.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:r[k]for k in ['status','exact_outputs_and_inputs','actual_routes_valid','comparisons','current_skill_native_wall_p50']},indent=2))
