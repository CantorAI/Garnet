"""Separate logical reset time from materialized GPU clearing inside prefill."""
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);outer=json.loads((root/'summary.json').read_text());profiles=[]
assert len(outer['profiles'])==6
for p in outer['profiles']:
    records=json.loads((root/Path(p['evidence']).name/'summary.json').read_text())['records'];assert len(records)==12
    for policy in ('control','candidate'):
        warm=[r for r in records if r['build_policy']==policy and r['request']>0];first=[r for r in records if r['build_policy']==policy and r['request']==0];assert len(warm)==4 and len(first)==2
        value=dict(model=p['model'],batch=p['batch'],policy=policy,ttft_ms=statistics.mean(r['ttft_seconds'] for r in warm)*1000,
            reset_call_ms=statistics.mean(r['reset_seconds'] for r in warm)*1000,
            prefill_inference_ms=statistics.mean(r['prefill_inference_ms'] for r in warm),
            prefill_observed_clocks={key:statistics.mean(r['prefill_profile'][key] for r in warm) for key in ('prepare_ms','encode_ms','gpu_ms','wait_ms')},
            materialized_clear_bytes=statistics.mean(r['prefill_profile'].get('deferred_zero_bytes',0) for r in warm),
            materialized_clear_states=statistics.mean(r['prefill_profile'].get('deferred_zero_states',0) for r in warm),
            decode_tps_per_row=statistics.mean(r['decode_wall_steps_per_second'] for r in warm),
            initialization_s=statistics.mean(r['initialization_seconds'] for r in first),
            first_request_after_init_ttft_ms=statistics.mean(r['ttft_seconds'] for r in first)*1000)
        profiles.append(value)
result=dict(scope='Observed clocks only. Candidate reset_call_ms is logical marking; physical clearing is included in prefill, not eliminated. Encode/GPU/wait clocks overlap and must not be added. Model initialization and first request after init are separate; caches unflushed.',profiles=profiles)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps([{k:p[k] for k in ('model','batch','policy','reset_call_ms','prefill_inference_ms','materialized_clear_states')} for p in profiles],indent=2))
