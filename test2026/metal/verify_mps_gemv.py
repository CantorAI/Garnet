"""Verify actual dtype rejection and valid-route numerical/timing evidence."""
import hashlib,json,statistics,sys
from pathlib import Path
r=Path(sys.argv[1]).resolve();e=r/'evidence';files=json.loads((e/'files.json').read_text())
for name,item in files.items():
    raw=(r/name).read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],name
compat=e/'logs/mps-gemv-compat-20261013';assert json.loads((compat/'exit.json').read_text())['exit_code']==0
statuses=json.loads((compat/'jobs/statuses.json').read_text());assert [s['exit_code'] for s in statuses]==[0,-6,0]
assert 'Both inputs must use the same type' in (compat/'jobs/fp16.log').read_text()
valid=[json.loads(v) for v in (compat/'jobs/fp32.log').read_text().splitlines() if v.startswith('{')]
assert len(valid)==1 and valid[0]['reference_exact'] and valid[0]['outputs']==5
proof=json.loads((e/'omitted-matrix-proof.json').read_text());count=0;pooled=[]
for folder in sorted((e/'logs').glob('mps-gemv-*')):
    metadata=json.loads((folder/'jobs/metadata.json').read_text())
    snapshot=folder/'jobs/source-snapshot'
    for name,expected in metadata['source_sha256'].items():
        path=snapshot/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    if 'compat' in folder.name:continue
    assert json.loads((folder/'exit.json').read_text())['exit_code']==0
    job=folder/'jobs';statuses=json.loads((job/'statuses.json').read_text())
    assert len(statuses)==9 and all(s['exit_code']==0 for s in statuses)
    summary=json.loads((job/'summary.json').read_text());assert len(summary['checks'])==68 and len(summary['results'])==68
    for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
        assets=job/(model+'-assets');manifest=json.loads((assets/'manifest.json').read_text())
        for item in manifest['cases']:
            key='logs/'+folder.name+'/jobs/'+model+'-assets/'+item['name']+'.bin'
            assert proof[key]['sha256']==item['sha256'] and proof[key]['bytes']==item['columns']*item['inner']*2
            assert proof[key]['checkpoint_sha256']==manifest['checkpoint_sha256']
        rows=[json.loads(v) for v in (job/(model+'-gpu.log')).read_text().splitlines() if v.startswith('{')]
        checks=[v for v in rows if v['kind']=='check'];assert len(checks)==len(manifest['cases'])*2
        assert json.loads((job/(model+'-gpu-monitor.json')).read_text())
        for q in checks:
            assert q['variant']=='mps_gemv_fp32' and q['padding_preserved']
            assert all(0<=q[k]<=1 for k in ('fp64_worst_ratio','control_fp64_worst_ratio','comparison_worst_ratio'))
            times=[v for v in rows if v['kind']=='timing' and all(v[k]==q[k] for k in ('matrix','rows','variant'))];assert len(times)==8
            control=statistics.median(v['gpu_ms_median'] for v in times if v['policy']=='control')
            candidate=statistics.median(v['gpu_ms_median'] for v in times if v['policy']=='candidate')
            result=next(v for v in summary['results'] if v['model']==model and v['name']==q['matrix'] and v['rows']==q['rows'])
            assert result['control_gpu_ms']==control and result['candidate_gpu_ms']==candidate
            assert abs(result['component_speedup']-control/candidate)<1e-12
        for batch in (1,2):
            large=[v for v in summary['results'] if v['model']==model and v['rows']==batch and v['columns']>256]
            pooled.append(dict(model=model,batch=batch,unweighted_large_matrix_ratio=sum(v['control_gpu_ms'] for v in large)/sum(v['candidate_gpu_ms'] for v in large)))
        count+=len(checks)
(r/'pooled.json').write_bytes((json.dumps(pooled,indent=2)+'\n').encode())
print(json.dumps(dict(files=len(files),real_matrix_checks=count,mixed_dtype_rejected=True,fp32_compatibility_outputs=5,scope='Component only; no full-model claim')))
