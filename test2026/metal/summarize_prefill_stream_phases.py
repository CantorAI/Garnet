"""Observed native phases: GPU spans overlap encoding/waiting; never add them."""
import json,statistics,sys
from pathlib import Path
folder=Path(sys.argv[1]);outer=json.loads((folder/'summary.json').read_text());rows=[];assert len(outer['profiles'])==6
for p in outer['profiles']:
 records=json.loads((folder/Path(p['evidence']).name/'summary.json').read_text())['records'];assert len(records)==12
 for policy in ('control','candidate'):
  warm=[r for r in records if r['build_policy']==policy and r['request']>0];first=[r for r in records if r['build_policy']==policy and r['request']==0];assert len(warm)==4 and len(first)==2
  value=dict(model=p['model'],batch=p['batch'],policy=policy,ttft_ms=statistics.mean(r['ttft_seconds'] for r in warm)*1000,
   decode_tps_per_row=statistics.mean(r['decode_wall_steps_per_second'] for r in warm),
   first_request_after_init_ttft_ms=statistics.mean(r['ttft_seconds'] for r in first)*1000,
   initialization_s=statistics.mean(r['initialization_seconds'] for r in first),
   prefill_inference_ms=statistics.mean(r['prefill_inference_ms'] for r in warm),
   reset_ms=statistics.mean(r['reset_seconds'] for r in warm)*1000,
   prefill_phases={k:statistics.mean(r['prefill_profile'][k] for r in warm) for k in ('prepare_ms','encode_ms','gpu_ms','wait_ms')},
   observed_prefill_partial_submissions=[r['prefill_profile'].get('prefill_partial_submissions',0) for r in warm])
  rows.append(value)
result=dict(scope='Observed clocks; multi-command GPU span includes gaps and overlaps CPU encoding/waiting; no isolated kernel attribution; init/first request separate and caches unflushed',profiles=rows)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
