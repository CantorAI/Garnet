"""Retained-record diagnostics and post-screen identity audit; no inference."""
from pathlib import Path
import json, hashlib, statistics, datetime
root=Path(__file__).resolve().parent.parent;w=root/'worldsense-validated-text-worker';r=Path('/Users/shawn/CantorAI');p=w/'production-speed'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb')as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
for name in ['driver-identities-before-dispatch.json','common-drivers-before-dispatch.json']:
 d=json.loads((p/name).read_text());files=d.get('files',d)
 assert all(sha(Path(k))==v for k,v in files.items()),name
assert json.loads((w/'dependencies-after-production-speed.json').read_text())['all_aliases_and_file_bytes_unchanged']
frozen=json.loads((w/'learned-input-identities-before.json').read_text())
for key,source in [('parent_source',r/'out/research/worldsense-validated-text-worker/source'),('legacy_source',r/'out/research/worldsense-prefix-cache/source')]:
 assert all(sha(source/k)==v for k,v in frozen[key].items())
assets=json.loads((w/'model-assets.json').read_text());count=0
for model,files in assets.items():
 for rel,info in files.items():
  path=r/'out/models'/model/rel;assert path.stat().st_size==info['bytes']and sha(path)==info['sha256'];count+=1
(w/'model-assets-after-screen.json').write_text(json.dumps(dict(all_recorded_model_assets_rehashed_exact=True,files=count,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
workers={}
for label in ['control-1','candidate-1']:
 data=json.loads((p/(label+'.json')).read_text());assert data['valid']and data['assignments_restored']and len(data['items'])==5
 rows=[]
 for item in data['items']:
  if item['warmup']:continue
  call=next(c for c in item['result']['model_response'].values()if isinstance(c,dict)and c.get('model_id')=='Qwen3-1.7B')
  t=call['timings'];g=t.get('native_gateway');raw_total=g['raw_model_total_seconds']if g else call['total_seconds'];raw_first=g['raw_model_ttft_seconds']if g else call['ttft_seconds']
  row=dict(repetition=item['repetition'],generated_tokens=call['generated_tokens'],gateway_total=call['total_seconds'],gateway_ttft=call['ttft_seconds'],raw_model_total=raw_total,raw_model_ttft=raw_first,decode_phase_seconds=t['decode_seconds'],decode_profile=t['decode_profile'],public_after_first_seconds=call['total_seconds']-call['ttft_seconds'],raw_after_first_seconds=raw_total-raw_first,scope='Per-call differences before aggregation; nested/overlapping clocks, not additive disjoint phases.')
  if g:
   row.update(pre_model_route_seconds=g['model_started_monotonic']-g['gateway_started_monotonic'],total_minus_raw_seconds=call['total_seconds']-raw_total,after_first_minus_raw_after_first_seconds=(call['total_seconds']-call['ttft_seconds'])-(raw_total-raw_first))
  rows.append(row)
 keys=['decode_phase_seconds','public_after_first_seconds','raw_after_first_seconds']+(['pre_model_route_seconds','total_minus_raw_seconds','after_first_minus_raw_after_first_seconds']if label=='candidate-1'else[])
 medians={key:statistics.median(row[key]for row in rows)for key in keys}
 medians['decode_profile']={key:statistics.median(row['decode_profile'][key]for row in rows)for key in rows[0]['decode_profile']}
 workers[label]=dict(rows=rows,medians=medians)
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),workers=workers,
 interpretation='The same pinned old library in the child still shows larger decode-loop and GPU spans. Gateway overhead alone does not explain the regression. GPU wait/spans overlap; medians are nonadditive. No measured clock/thermal/VM/residency/register/occupancy cause, significance or cross-trial ranking. Preserve valid rejection; no timing retry or later stages.',new_inference=False)
(w/'posthoc-text-decode-boundaries.json').write_text(json.dumps(result,indent=2)+'\n')
analysis=json.loads((p/'analysis.json').read_text());assert not analysis['passed']and analysis['status']=='rejected'and len(analysis['checks'])==20 and all(c['exact']and c['native_lazy_defaults_valid']for c in analysis['checks'])
assert next(c for c in analysis['comparisons']if c['model']=='Qwen3-1.7B')['decode_gain']<-.02
assert not any((w/stage/'analysis.json').exists()for stage in ['primary-speed','speed','confirmation'])
(w/'production-speed-terminal.json').write_text(json.dumps(dict(controller_session=66771,observed_exit_code=0,all20outputs_native_exact=True,all_frozen_drivers_sources_models_libraries_and_dependencies_unchanged=True,passed=False,no_later_stages_or_unchanged_retry=True),indent=2)+'\n')
print('WS61 valid speed rejection, full identity audit, 19 asset hashes and retained text timing boundaries recorded')
