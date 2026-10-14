"""Verify the archived real-weight component evidence and derived ratios."""
import hashlib,json,statistics,sys
from pathlib import Path
r=Path(sys.argv[1]).resolve();e=r/'evidence';files=json.loads((e/'files.json').read_text());count=0;pooled=[]
for name,item in files.items():
    raw=(r/name).read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],name
proof=json.loads((e/'omitted-matrix-proof.json').read_text())
for folder in sorted((e/'logs').glob('decode-subsimd-probe*')):
    assert json.loads((folder/'exit.json').read_text())['exit_code']==0
    job=folder/'jobs';metadata=json.loads((job/'metadata.json').read_text())
    for name,expected in metadata['source_sha256'].items():assert hashlib.sha256((job/'source-snapshot'/name).read_bytes()).hexdigest()==expected,name
    statuses=json.loads((job/'statuses.json').read_text());assert len(statuses)==9 and all(s['exit_code']==0 for s in statuses)
    summary=json.loads((job/'summary.json').read_text());assert len(summary['checks'])==136 and len(summary['results'])==136
    for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
        assets=job/(model+'-assets');manifest=json.loads((assets/'manifest.json').read_text())
        for item in manifest['cases']:
            name='logs/'+folder.name+'/jobs/'+model+'-assets/'+item['name']+'.bin'
            assert proof[name]['sha256']==item['sha256'] and proof[name]['bytes']==item['columns']*item['inner']*2
            assert proof[name]['checkpoint_sha256']==manifest['checkpoint_sha256']
        rows=[json.loads(v) for v in (job/(model+'-gpu.log')).read_text().splitlines() if v.startswith('{')]
        checks=[v for v in rows if v['kind']=='check'];assert len(checks)==len(manifest['cases'])*4
        assert json.loads((job/(model+'-gpu-monitor.json')).read_text())
        for q in checks:
            assert q['padding_preserved'] and all(0<=q[k]<=1 for k in ('fp64_worst_ratio','control_fp64_worst_ratio','comparison_worst_ratio'))
            times=[v for v in rows if v['kind']=='timing' and all(v[k]==q[k] for k in ('matrix','rows','variant'))];assert len(times)==8
            control=statistics.median(v['gpu_ms_median'] for v in times if v['policy']=='control')
            candidate=statistics.median(v['gpu_ms_median'] for v in times if v['policy']=='candidate')
            result=next(v for v in summary['results'] if v['model']==model and all(v[k]==q[k] for k in ('rows','variant')) and v['name']==q['matrix'])
            assert result['control_gpu_ms']==control and result['candidate_gpu_ms']==candidate
            assert abs(result['component_speedup']-control/candidate)<1e-12
        for variant in ('bf16_decode_s16_t64','bf16_decode_s16_t128'):
            large=[v for v in summary['results'] if v['model']==model and v['variant']==variant and v['columns']>256]
            pooled.append(dict(run=folder.name,model=model,variant=variant,unweighted_large_matrix_ratio=sum(v['control_gpu_ms'] for v in large)/sum(v['candidate_gpu_ms'] for v in large)))
        count+=len(checks)
(r/'component-pooled.json').write_bytes((json.dumps(pooled,indent=2)+'\n').encode())
print(json.dumps(dict(files=len(files),omitted_matrices=len(proof),checks=count,scope='Matrix components only; no full-model claim')))
