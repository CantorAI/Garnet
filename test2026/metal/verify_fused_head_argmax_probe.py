"""Recompute every archived complete-head bit/greedy gate and ABBA mean."""
import hashlib,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();trial=sys.argv[3] if len(sys.argv)>3 else 'fused-head-argmax-20261015'
proof=json.loads((root/'evidence/manifest.json').read_text())
for name,info in proof['files'].items():
    p=root/name;assert p.resolve().is_relative_to(root) and p.stat().st_size==info['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']
assert proof['jobs'][trial]['exit_code']==0
jobs=root/'evidence/logs'/trial/'jobs';meta=json.loads((jobs/'metadata.json').read_text())
for name,sha in meta['source_sha256'].items():assert hashlib.sha256((jobs/'source-snapshot'/name).read_bytes()).hexdigest()==sha
assert hashlib.sha256((jobs/'combined.metal').read_bytes()).hexdigest()==meta['shader_sha256']
assert proof['omitted_binaries']['logs/'+trial+'/jobs/fused_head_argmax_gpu']['sha256']==meta['binary_sha256']
stages=json.loads((jobs/'statuses.json').read_text());assert len(stages)==5 and all(s['exit_code']==0 for s in stages)
for s in stages:assert json.loads((jobs/(s['name']+'-monitor.json')).read_text())
reported=json.loads((jobs/'summary.json').read_text())['profiles'];rows=[];captured=0;edges=0;timed=0
for model,assets in meta['assets'].items():
    records=[json.loads(line) for line in (jobs/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
    checks=[v for v in records if v['kind']=='check'];assert len(checks)==len(assets['queries'])+32
    assert {v['query'] for v in checks if v['timed']}=={v['name'] for v in assets['queries']}
    edge_cases={(v['columns'],v['batch'],int(v['query'].split('-f')[-1])) for v in checks if not v['timed']}
    assert edge_cases=={(n,b,f) for n in (1,3,17,33) for b in (1,2) for f in range(4)}
    for check in checks:
        assert all(check[k] for k in ('projection_bits_exact','greedy_exact','padding_preserved'))
        if not check['timed']:assert not check['captured_reference_checked'];edges+=1;continue
        assert check['captured_reference_checked'];captured+=1
        windows=[v for v in records if v['kind']=='timing' and v['query']==check['query']];assert len(windows)==8
        for round in (0,1):
            group=sorted((v for v in windows if v['round']==round),key=lambda v:v['window'])
            assert [v['window'] for v in group]==list(range(4)) and [v['candidate'] for v in group]==[False,True,True,False]
        assert all(len(v[k])==8 and all(x>0 for x in v[k]) for v in windows for k in ('gpu_ms','wall_ms'));timed+=64
        means={name:{metric:statistics.mean(x for v in windows if v['candidate']==candidate for x in v[metric]) for metric in ('gpu_ms','wall_ms')} for name,candidate in (('control',False),('candidate',True))}
        row=next(v for v in reported if v['model']==model and v['query']==check['query'])
        for name in means:
            for metric in means[name]:assert abs(row['means'][name][metric]-means[name][metric])<1e-12
        assert abs(row['gpu_fraction']-means['candidate']['gpu_ms']/means['control']['gpu_ms'])<1e-12
        assert abs(row['wall_fraction']-means['candidate']['wall_ms']/means['control']['wall_ms'])<1e-12
        rows.append(row)
assert captured==51 and edges==96 and timed==3264 and len(reported)==51
pools=[]
for model in meta['assets']:
    for batch in (1,2):
        group=[v for v in rows if v['model']==model and v['batch']==batch]
        pools.append(dict(model=model,batch=batch,captured_queries=len(group),gpu_fraction=sum(v['means']['candidate']['gpu_ms'] for v in group)/sum(v['means']['control']['gpu_ms'] for v in group),wall_fraction=sum(v['means']['candidate']['wall_ms'] for v in group)/sum(v['means']['control']['wall_ms'] for v in group)))
result=dict(trial=trial,verified_files=len(proof['files']),captured_head_queries=captured,edge_checks=edges,timed_executions=timed,pools=pools,scope='Whole-head component only; captured complete logits, synthetic ties/nonfinite/tails; no new full-model or comparator gain',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
