"""Recompute eager initialization, first use, warm clocks and body-route telemetry."""
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);summary=json.loads((root/'summary.json').read_text());result=[]
assert len(summary['profiles'])==6
for p in summary['profiles']:
    records=json.loads((root/Path(p['evidence']).name/'summary.json').read_text())['records'];assert len(records)==12
    for policy in ('control','candidate'):
        warm=[r for r in records if r['build_policy']==policy and r['request']>0]
        first=[r for r in records if r['build_policy']==policy and r['request']==0]
        assert len(warm)==4 and len(first)==2
        v=dict(model=p['model'],batch=p['batch'],policy=policy,
            initialization_s=statistics.mean(r['initialization_seconds'] for r in first),
            first_after_init_ttft_ms=statistics.mean(r['ttft_seconds'] for r in first)*1000,
            warm_ttft_ms=statistics.mean(r['ttft_seconds'] for r in warm)*1000,
            reset_ms=statistics.mean(r['reset_seconds'] for r in warm)*1000,
            prefill_inference_ms=statistics.mean(r['prefill_inference_ms'] for r in warm),
            prefill_observed_clocks={k:statistics.mean(r['prefill_profile'][k] for r in warm) for k in ('prepare_ms','encode_ms','gpu_ms','wait_ms')},
            warm_decode_tps_per_row=statistics.mean(r['decode_wall_steps_per_second'] for r in warm),
            pipeline_counts=sorted({r['body_shape_pipeline_count'] for r in records if r['build_policy']==policy}),
            prefill_dispatch_counts=sorted({r['body_shape_counts'][0] for r in warm}),
            decode_dispatch_counts=sorted({count for r in warm for count in r['body_shape_counts'][1:]}),
            fallback_misses=sum(sum(r['body_shape_misses']) for r in records if r['build_policy']==policy))
        if all(r.get('decode_phases') for r in warm):
            v['decode_observed_clocks']={k:statistics.mean(d[k] for r in warm for d in r['decode_phases']) for k in ('prepare_ms','encode_ms','gpu_ms','wait_ms')}
        if all('common_preprocessing_seconds' in r for r in warm):
            v['image_observed_clocks']={k:statistics.mean(r[k] for r in warm)*1000 for k in ('common_preprocessing_seconds','worker_ttft_seconds','vision_seconds','assembly_seconds','prefill_wall_seconds')}
        result.append(v)
output=dict(scope='Native control/candidate only. Eager pipeline compilation is included in init. Init and first request after init are separate; caches unflushed. GPU/encode/wait clocks overlap. Image vision_seconds includes reset; do not add nested clocks. No fresh vLLM ratio.',profiles=result)
Path(sys.argv[2]).write_bytes((json.dumps(output,indent=2)+'\n').encode());print(json.dumps(dict(profiles=len(result))))
