"""Post-terminal recomputation of every real pipeline output and the rejection."""
from pathlib import Path
import json,hashlib,statistics,ast,datetime,math
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-cooperative-prefill';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-cooperative-prefill/source';load=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
assert load(w/'production-speed/analysis.json')['status']=='passed_screen_requires_fresh_full_confirmation'
assert all(sha(s/k)==v for k,v in load(w/'correctness-source-manifest.json').items())
for folder in [w,w/'production-speed']:
 identity=load(folder/'driver-identities-before-dispatch.json');assert all(sha(Path(k))==v for k,v in identity['files'].items())
assert load(w/'dependencies-after-production-speed.json')['all_aliases_and_file_bytes_unchanged']
dep=load(w/'build-host-dependencies-before-dispatch.json');assert len(dep['files'])==1881
assert all(str(Path(k).resolve())==v for k,v in dep['aliases'].items())
assert all(Path(k).stat().st_size==v['bytes']and sha(Path(k))==v['sha256']for k,v in dep['files'].items())
outer=load(w/'outer-method-verifier-identities-before-resource.json');assert len(outer['files'])==15 and all(sha(Path(k))==v for k,v in outer['files'].items())
assert load(w/'dual-pair-corpus.json')['totals']=={'pairs':28,'matrices':56,'original_FP32_words_checked':704643072}
assert load(w/'dual-word-comparison.json')['record_count']==36 and load(w/'dual-word-comparison.json')['all_seven_builds_complete_FP32_words_identical']
assets=load(w/'production-speed/learned-assets-before-dispatch.json');assert len(assets)==19
assert all(Path(k).stat().st_size==v['bytes']and sha(Path(k))==v['sha256']for k,v in assets.items())
control=r/'out/research/worldsense-prefix-cache/source';assert all(sha(control/k)==v for k,v in load(w/'production-speed/control-source-before-dispatch.json').items())
frozen=load(w/'learned-identities-before.json');assert all(sha(Path(k))==v for k,v in frozen['external'].items())
gold=load(r/'Garnet/results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json');refs={(i['observation_id'],phase):(i,c)for i in gold['items']if i['policy']=='candidate'for phase,c in i['result']['model_response'].items()if isinstance(c,dict)and c.get('status')=='ok'}
all_calls=[];all_parents=[];metrics={};wall={};phase_data={}
for stage,folder,labels,count in [('resource',w,['control-1','candidate-1'],2),('production-speed',w/'production-speed',['control-1','candidate-1'],5),('functional-correctness-only',w/'functional-smoke',['candidate-1'],2)]:
 analyzer=w/'production-speed/analyze.py'if stage=='production-speed'else w/'analyze_resource.py';fn=next(n for n in ast.parse(analyzer.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='native_route');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'native admission','exec'),ns)
 for label in labels:
  data=load(folder/(label+'.json'));assert data['valid']and data['assignments_restored']and len(data['items'])==count and not data['search_plans'];calls=[]
  for i,item in enumerate(data['items']):
   assert(item['repetition'],item['warmup'],item['policy'])==(i,i==0,'default');all_parents.append(dict(stage=stage,label=label,repetition=i,warmup=item['warmup'],native_call_seconds=item['native_call_seconds']))
   for phase,c in item['result']['model_response'].items():
    if not isinstance(c,dict)or c.get('status')!='ok':continue
    ri,ref=refs[(item['observation_id'],phase)];stable=lambda v:{k:x for k,x in v.items()if k not in ['model_response','model_response_json']};assert stable(item['result'])==stable(ri['result'])
    assert all(c[k]==ref[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])and c['timings']['input_tokens']==ref['timings']['input_tokens']
    assert item['image_sha256']==ri['image_sha256']and item['skill_sha256']==ri['skill_sha256']and ns['native_route'](c,label=='candidate-1')
    row=dict(stage=stage,label=label,repetition=i,warmup=item['warmup'],phase=phase,model=c['model_id'],ttft=c['ttft_seconds'],total=c['total_seconds'],decode_tps=(c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds']),generated_tokens=c['generated_tokens'],timings=c['timings']);calls.append(row);all_calls.append(row)
  assert len(calls)==2*count
  if stage=='production-speed':
   metrics[label]={};phase_data[label]={}
   for model in ['Qwen3-1.7B','Qwen3.5-4B']:
    rows=[c for c in calls if c['model']==model and not c['warmup']];assert len(rows)==4
    metrics[label][model]=dict(samples=4,**{key:statistics.median(c[key]for c in rows)for key in ['ttft','total','decode_tps']})
    individual=[]
    for c in rows:
     t=c['timings'];individual.append(dict(repetition=c['repetition'],decode_seconds=t['decode_seconds'],prefill_seconds=t['prefill_seconds'],decode_interval=c['total']-c['ttft'],post_decode_residual=(c['total']-c['ttft'])-t['decode_seconds'],decode_gpu_ms=t['decode_profile']['gpu_ms'],prefill_gpu_ms=t['prefill_profile']['gpu_ms'],sole_encoded_delta=t.get('sole_encoded_delta'),sole_storage_after=t.get('sole_storage_after')))
    phase_data[label][model]=dict(per_call=individual,medians={key:statistics.median(row[key]for row in individual)for key in ['decode_seconds','prefill_seconds','decode_interval','post_decode_residual','decode_gpu_ms','prefill_gpu_ms']})
   wall[label]=statistics.median(item['native_call_seconds']for item in data['items']if not item['warmup'])
analysis=load(w/'production-speed/analysis.json');assert analysis['metrics']==metrics and analysis['native_skill_wall_p50']==wall
comparisons=[dict(model=model,ttft_change=metrics['candidate-1'][model]['ttft']/metrics['control-1'][model]['ttft']-1,total_change=metrics['candidate-1'][model]['total']/metrics['control-1'][model]['total']-1,decode_gain=metrics['candidate-1'][model]['decode_tps']/metrics['control-1'][model]['decode_tps']-1)for model in sorted(metrics['control-1'])]
assert comparisons==analysis['comparisons'];wall_gain=1-wall['candidate-1']/wall['control-1'];assert wall_gain==analysis['native_skill_wall_gain']
passed=wall_gain>=-.02 and all(v['ttft_change']<=.02 and v['total_change']<=.02 and v['decode_gain']>=-.02 for v in comparisons);assert passed==analysis['passed']==True
assert len(all_calls)==32 and len(all_parents)==16 and load(w/'restoration-verification.json')['benchmark_valid_request_http404']
(w/'production-speed/posthoc-screen-phase-diagnostics.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Retained warm calls only, posthoc descriptive phase data; no new inference, exclusions, causal attribution or speed eligibility.',workers=phase_data,limits='Differences formed per call before medians. GPU/wait/encode spans overlap/include gaps and separate medians are nonadditive. Phase/counter data do not establish isolated pairing/prefill-layout cost versus64 or54. Derived decode includes host/stop/postloop/possible EOS/snapshots; not pureGPU. No cross-trial causal ratio or reclassification of rejection.'),indent=2)+'\n')
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_speed_session=22117,observed_exit_code=0,decision='provisional-actualWS10-screen-pass',promotion=False,all32_calls_and16_requests_exact_original_goldens_and_native_routes=True,all_source_library_runtime_driver_model_and1881dependencies_unchanged=True,source_files=1724,learned_assets=19,metrics=metrics,comparisons=comparisons,whole_skill_wall=wall,wall_gain=wall_gain,resource_passed=True,speed_passed=True,reason='ActualWS10 screen20 passes original exact/native/2%per-model/wholeSkill gates; matchedPRIMARY/RAW/mirrored/canonical stages still required.',original_additional_confirmation_and_canonical_stages_required=True)
(w/'production-speed/independent-screen-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['metrics','comparisons']}))
