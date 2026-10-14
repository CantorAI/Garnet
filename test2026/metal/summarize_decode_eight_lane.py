"""Independent component GPU/wall ratios by batch; no serving-speed attribution."""
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();folder=root/'evidence/logs/decode-eight-lane-20261015/jobs'
summary=json.loads((folder/'summary.json').read_text());groups=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    logs=[json.loads(v) for v in (folder/(model+'-gpu.log')).read_text().splitlines() if v.startswith('{')]
    for batch in (1,2):
        for variant in ('bf16_decode_s8_t64','bf16_decode_s8_t128'):
            cases=[r for r in summary['results'] if r['model']==model and r['rows']==batch and r['variant']==variant and r['columns']>256 and not r['tensor'].startswith('synthetic')]
            assert cases
            times={policy:{clock:[] for clock in ('gpu_ms_median','wall_ms_median')} for policy in ('control','candidate')}
            for r in cases:
                entries=[x for x in logs if x['kind']=='timing' and x['matrix']==r['name'] and x['rows']==batch and x['variant']==variant]
                assert len(entries)==8
                for policy in times:
                    for clock in times[policy]:times[policy][clock].append(statistics.median(x[clock] for x in entries if x['policy']==policy))
            groups.append(dict(model=model,batch=batch,variant=variant,large_trained_matrices=len(cases),
                unweighted_gpu_component_speedup=sum(times['control']['gpu_ms_median'])/sum(times['candidate']['gpu_ms_median']),
                unweighted_wall_component_speedup=sum(times['control']['wall_ms_median'])/sum(times['candidate']['wall_ms_median'])))
result=dict(scope='Sum of per-matrix medians, unweighted across distinct matrices; not a layer-weighted or full-model speedup',profiles=groups,
    fp64_worst_ratio=max(x['fp64_worst_ratio'] for x in summary['checks']),
    comparison_worst_ratio=max(x['comparison_worst_ratio'] for x in summary['checks']))
(root/'component-by-batch.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(result,indent=2))
