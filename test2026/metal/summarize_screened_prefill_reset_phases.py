"""Recompute separate clocks, initialization and actual reset/submission counts."""
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);outer=json.loads((root/'summary.json').read_text());profiles=[]
assert len(outer['profiles'])==6
for p in outer['profiles']:
    records=json.loads((root/Path(p['evidence']).name/'summary.json').read_text())['records'];assert len(records)==12
    for policy in ('control','candidate'):
        warm=[r for r in records if r['build_policy']==policy and r['request']>0]
        first=[r for r in records if r['build_policy']==policy and r['request']==0];assert len(warm)==4 and len(first)==2
        mean=lambda key:statistics.mean(r[key] for r in warm)
        v=dict(model=p['model'],batch=p['batch'],policy=policy,warm_ttft_ms=mean('ttft_seconds')*1000,
            reset_ms=mean('reset_seconds')*1000,prefill_inference_ms=mean('prefill_inference_ms'),
            warm_decode_tps_per_row=mean('decode_wall_steps_per_second'),
            prefill_observed_clocks={k:statistics.mean(r['prefill_profile'][k] for r in warm) for k in ('prepare_ms','encode_ms','gpu_ms','wait_ms')},
            actual_dirty_reset_bytes=sorted({r['dirty_prefix_reset_bytes'] for r in warm}),
            actual_prefill_submissions=sorted({r['prefill_profile'].get('prefill_partial_submissions',0) for r in warm}),
            initialization_s=statistics.mean(r['initialization_seconds'] for r in first),
            first_request_after_init_ttft_ms=statistics.mean(r['ttft_seconds'] for r in first)*1000,
            peak_individual_rss_bytes=mean('peak_individual_rss_bytes'))
        if all(r.get('decode_phases') for r in warm):
            v['decode_observed_clocks']={k:statistics.mean(d[k] for r in warm for d in r['decode_phases']) for k in ('prepare_ms','encode_ms','gpu_ms','wait_ms')}
            v['decode_position_gpu_ms']={label:statistics.mean(d['gpu_ms'] for r in warm for d in r['decode_phases'][part]) for label,part in [('first_four',slice(0,4)),('last_four',slice(-4,None))]}
        if all('common_preprocessing_seconds' in r for r in warm):v['image_observed_clocks_ms']={k:mean(k)*1000 for k in ('common_preprocessing_seconds','worker_ttft_seconds','vision_seconds','assembly_seconds','prefill_wall_seconds')}
        profiles.append(v)
result=dict(scope='Native paired observations only. Init/first after init separate; caches unflushed. GPU/encode/wait clocks overlap; vision includes reset. Individual child RSS is not aggregate/unique GPU memory. No fresh vLLM ratio.',profiles=profiles)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps([dict(model=p['model'],batch=p['batch'],policy=p['policy'],decode=p.get('decode_observed_clocks')) for p in profiles]))
