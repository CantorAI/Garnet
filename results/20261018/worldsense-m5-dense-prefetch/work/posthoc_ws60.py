"""Retained-record diagnostics only, after terminal failure; no new inference."""
from pathlib import Path
import json,statistics,datetime,re
w=Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-dense-prefetch');folder=w/'production-speed';result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Posthoc descriptive paired retained calls, no exclusions/causal attribution or new speed eligibility.',workers={})
for label in ['control-1','candidate-1']:
 dataset=json.loads((folder/(label+'.json')).read_text());rows=[]
 for item in dataset['items']:
  if item['warmup']:continue
  for phase,c in item['result']['model_response'].items():
   if not isinstance(c,dict)or c.get('model_id')!='Qwen3-1.7B':continue
   t=c['timings'];row=dict(repetition=item['repetition'],phase=phase,generated_tokens=c['generated_tokens'],ttft_seconds=c['ttft_seconds'],total_seconds=c['total_seconds'],decode_interval=c['total_seconds']-c['ttft_seconds'],decode_phase_seconds=t['decode_seconds'],post_decode_residual=(c['total_seconds']-c['ttft_seconds'])-t['decode_seconds'],dense_prefetch_encoded_delta=t.get('dense_prefetch_encoded_delta'),decode_profile=t['decode_profile'])
   if 'native_gateway'in t:
    g=t['native_gateway'];row.update(raw_decode_interval=g['raw_model_total_seconds']-g['raw_model_ttft_seconds'],gateway_extra_decode_interval=(c['total_seconds']-c['ttft_seconds'])-(g['raw_model_total_seconds']-g['raw_model_ttft_seconds']),gateway_extra_total=c['total_seconds']-g['raw_model_total_seconds'],gateway_extra_ttft=c['ttft_seconds']-g['raw_model_ttft_seconds'])
   rows.append(row)
 assert len(rows)==4
 summary={key:statistics.median(x[key]for x in rows)for key in rows[0]if key not in ['repetition','phase','decode_profile']and all(isinstance(x[key],(int,float))for x in rows)};summary['decode_profile']={key:statistics.median(x['decode_profile'][key]for x in rows)for key in rows[0]['decode_profile']}
 result['workers'][label]=dict(per_call=rows,medians=summary)
result['limits']='Per-call differences computed before medians. GPU/wait/encode/prepare spans may overlap/include gaps; do not add separate medians or treat these as pure kernel costs. WS60 uses one native process, no IPC gateway. Post-decode residual is formed per call before its median. No register-allocation, shader, scheduling, clock, VM or residency cause is established by these descriptive retained-record timings.'
(w/'posthoc-text-decode-boundaries.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v['medians']for k,v in result['workers'].items()},indent=2))
