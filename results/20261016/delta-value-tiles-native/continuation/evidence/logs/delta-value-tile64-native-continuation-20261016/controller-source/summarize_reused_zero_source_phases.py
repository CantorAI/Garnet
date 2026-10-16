"""Recompute measured reset/prefill clocks and retained scratch from raw records."""
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);outer=json.loads((root/'summary.json').read_text());profiles=[]
assert len(outer['profiles'])==6
for p in outer['profiles']:
    records=json.loads((root/Path(p['evidence']).name/'summary.json').read_text())['records'];assert len(records)==12
    for policy in ('control','candidate'):
        warm=[r for r in records if r['build_policy']==policy and r['request']>0]
        first=[r for r in records if r['build_policy']==policy and r['request']==0]
        assert len(warm)==4 and len(first)==2
        mean=lambda key:statistics.mean(r[key] for r in warm)
        profiles.append(dict(model=p['model'],batch=p['batch'],policy=policy,
            ttft_ms=mean('ttft_seconds')*1000,reset_call_ms=mean('reset_seconds')*1000,
            prefill_inference_ms=mean('prefill_inference_ms'),
            prefill_observed_clocks={key:statistics.mean(r['prefill_profile'][key] for r in warm) for key in ('prepare_ms','encode_ms','gpu_ms','wait_ms')},
            source_bytes=mean('reused_zero_source_bytes'),copied_bytes=mean('reused_zero_copy_bytes'),copied_states=mean('reused_zero_copy_states'),
            decode_tps_per_row=mean('decode_wall_steps_per_second'),
            initialization_s=statistics.mean(r['initialization_seconds'] for r in first),
            first_request_after_init_ttft_ms=statistics.mean(r['ttft_seconds'] for r in first)*1000))
result=dict(scope='Actual reset copies complete before prefill. Retained source allocation is included in initialization. Encode/GPU/wait clocks overlap and must not be added. Init and first request after init are separate; caches unflushed. No new vLLM measurement.',profiles=profiles)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps([{k:p[k] for k in ('model','batch','policy','reset_call_ms','prefill_inference_ms','decode_tps_per_row','source_bytes')} for p in profiles],indent=2))
