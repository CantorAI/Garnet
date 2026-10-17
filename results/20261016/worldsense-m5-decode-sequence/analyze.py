"""New sequence diagnostic; no promotion or timing pass/fail selection."""
from pathlib import Path
import json,hashlib,math,sys
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-vision-workspace/source';labels=['control-diag-1','candidate-diag-1','candidate-diag-2','control-diag-2'];env=json.loads((w/'environment.json').read_text());hashes=json.loads((w/'starting-source-manifest.json').read_text());assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==v for k,v in hashes.items());manifest=json.loads((w/'cohort-manifest.json').read_text());partial='--partial'in sys.argv
if partial:
 assert manifest and [m['label']for m in manifest]==labels[:len(manifest)];labels=labels[:len(manifest)]
else:assert [m['label']for m in manifest]==labels
gold=json.loads((r/'Garnet/results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json').read_text());refs={}
for i in gold['items']:
 if i['policy']=='candidate':
  for phase,c in i['result']['model_response'].items():
   if isinstance(c,dict)and c.get('status')=='ok':refs[(i['observation_id'],phase)]=(i,c)
qref=next(q for q in gold['search_plans']if q['policy']=='candidate'and q['query']=='a passenger car');rows=[]
def stable(x,exclude):return {k:v for k,v in x.items()if k not in exclude}
def add(label,kind,repetition,warmup,position,c,ref):
 assert c['status']=='ok'and all(c[k]==ref[k]for k in ['model_id','text','generated_token_ids','input_prompt_sha256','serving_configuration'])and c['timings']['input_tokens']==ref['timings']['input_tokens']
 t=c['timings'];candidate=label.startswith('candidate');pooled=candidate and c['model_id']=='Qwen3.5-4B';private=candidate and c['model_id']=='Qwen3-1.7B'
 assert (t['vision_workspace_fullwrite_slots']>0)==pooled and (t['vision_workspace_allocated_bytes']>0)==pooled and (t['workspace_allocated_bytes']>0)==pooled and (t['workspace_bound_slots']>0)==pooled
 assert (not pooled or t['workspace_buffer_count']<t['workspace_bound_slots'])and (t['resident_avoided_storage_bytes']>0)==pooled and (t['lazy_default_planned_slots']>0)==pooled and t['lazy_default_retired_bytes']==0 and t['lazy_default_rehydrated_bytes']==0
 assert t['decoder_private_storage_graphs']==(28 if private else 0)and (t['prefill_private_storage_graphs']>0)==private
 assert c['generated_tokens']>1 and c['total_seconds']>c['ttft_seconds']>0
 row=dict(worker=label,kind=kind,repetition=repetition,warmup=warmup,query_position=position,model=c['model_id'],input_tokens=t['input_tokens'],generated_tokens=c['generated_tokens'],prompt_sha256=c['input_prompt_sha256'],ttft=c['ttft_seconds'],total=c['total_seconds'],decode_tps=(c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds']),prefill_seconds=t['prefill_seconds'],decode_seconds=t['decode_seconds'],prefix_hit=t['prefix_cache_hit'],executed_prefill_tokens=t['executed_prefill_tokens'],decode_profile=t['decode_profile'],prefill_profile=t['prefill_profile'],memory=t['memory_after_decode'],exact=True,native_policy=True);row.update(serving_tail_seconds=c['total_seconds']-c['ttft_seconds'],decode_loop_tps=(c['generated_tokens']-1)/t['decode_seconds'],post_loop_seconds=c['total_seconds']-c['ttft_seconds']-t['decode_seconds'],loop_minus_reported_wait_seconds=t['decode_seconds']-t['decode_profile']['wait_ms']/1000);assert math.isfinite(row['decode_tps']);rows.append(row)
for label,m in zip(labels,manifest):
 expected='on'if label.startswith('candidate')else'off';assert m['client_exit']==0 and m['library_sha256']==env['libraries'][expected]and m['serving_sha256']==hashes['runtime/metal_serving.py']and m['galaxy_library_sha256']==env['dependencies']['galaxy_library_sha256']
 d=json.loads((w/(label+'.json')).read_text());assert d['valid']and d['assignments_restored']and d['request']==json.loads((w/'cases.json').read_text())and len(d['items'])==2 and len(d['search_plans'])==4
 assert [(i['repetition'],i['warmup'],i['policy'])for i in d['items']]==[(0,True,'default'),(1,False,'default')]
 assert [(q['repetition'],q['warmup'],q['policy'],q['query'])for q in d['search_plans']]==[(j,j==0,'default','a passenger car')for j in range(2)for _ in range(2)]
 for i in d['items']:
  for phase,c in i['result']['model_response'].items():
   if not isinstance(c,dict)or c.get('status')!='ok':continue
   ri,ref=refs[(i['observation_id'],phase)];assert stable(i['result'],['model_response','model_response_json'])==stable(ri['result'],['model_response','model_response_json'])and i['image_sha256']==ri['image_sha256']and i['skill_sha256']==ri['skill_sha256'];add(label,'skill_'+phase,i['repetition'],i['warmup'],None,c,ref)
 for index,q in enumerate(d['search_plans']):
  assert stable(q['result'],['benchmark_model_response'])==stable(qref['result'],['benchmark_model_response']);add(label,'query_plan',q['repetition'],q['warmup'],index%2+1,q['result']['benchmark_model_response'],qref['result']['benchmark_model_response'])
assert len(rows)==8*len(labels)
query_pairs=[]
for label in labels:
 for repetition in [0,1]:
  a,b=[v for v in rows if v['worker']==label and v['kind']=='query_plan'and v['repetition']==repetition];assert a['prompt_sha256']==b['prompt_sha256']and a['input_tokens']==b['input_tokens']and a['generated_tokens']==b['generated_tokens'];query_pairs.append(dict(worker=label,repetition=repetition,warmup=a['warmup'],first_decode_tps=a['decode_tps'],second_decode_tps=b['decode_tps'],second_vs_first_decode_change=b['decode_tps']/a['decode_tps']-1,first_prefix_hit=a['prefix_hit'],second_prefix_hit=b['prefix_hit']))
result=dict(status='valid_partial_sequence_diagnostic_no_promotion_decision'if partial else'valid_sequence_diagnostic_no_promotion_decision',rows=rows,identical_query_pairs=query_pairs,scope=str(len(rows))+' actual WorldSense/Galaxy calls over '+str(len(labels))+' completed fresh workers (4 planned),onewarm+onefollow each; repeated identical PlanSearch after fixedcarSkill. Descriptive sequence data only. No independent-sample/stats/clock/thermal/VM/allocator causality claim; WS21 rejection unchanged.')
(w/('partial-analysis.json'if partial else'analysis.json')).write_text(json.dumps(result,indent=2));print(json.dumps(dict(status=result['status'],calls=len(rows),query_pairs=query_pairs),indent=2))
