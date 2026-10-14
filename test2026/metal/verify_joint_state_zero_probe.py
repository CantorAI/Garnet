"""Recompute config-derived sizes, all clear/read gates and balanced windows."""
import hashlib,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();proof=json.loads((root/'manifest.json').read_text());results=[]
for name,info in proof['files'].items():
    p=root/name;assert p.resolve().is_relative_to(root) and p.stat().st_size==info['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']
for name,status in proof['jobs'].items():
    assert status['exit_code']==0;jobs=root/'logs'/name/'jobs';meta=json.loads((jobs/'metadata.json').read_text())
    for src,sha in meta['source_sha256'].items():assert hashlib.sha256((jobs/'source-snapshot'/src).read_bytes()).hexdigest()==sha
    stages=json.loads((jobs/'statuses.json').read_text());assert len(stages)==2 and all(v['exit_code']==0 for v in stages)
    for stage in stages:assert json.loads((jobs/(stage['name']+'-monitor.json')).read_text())
    profiles=[]
    for model,record in meta['configs'].items():
        raw=(root/'models'/model/'config.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==record['sha256'] and json.loads(raw)==record['config'];c=record['config'].get('text_config',record['config']);columns=[]
        for kind in c.get('layer_types',['full_attention']*c['num_hidden_layers']):
            if kind=='linear_attention':
                channels=2*c['linear_num_key_heads']*c['linear_key_head_dim']+c['linear_num_value_heads']*c['linear_value_head_dim'];columns.extend([channels*c['linear_conv_kernel_dim'],c['linear_num_value_heads']*c['linear_key_head_dim']*c['linear_value_head_dim']])
            else:
                width=c['num_key_value_heads']*256*c.get('head_dim',c['hidden_size']//c['num_attention_heads']);columns.extend([width,width])
        for batch in (1,2):profiles.append(dict(name=model+'-b'+str(batch),batch=batch,columns=columns,logical_bytes=batch*sum(columns)*4))
    profiles.append(dict(name='padded',batch=3,columns=[1,3,17,65],logical_bytes=1032));assert profiles==meta['profiles']
    expected='\n'.join(p['name']+' '+str(p['batch'])+' '+str(len(p['columns']))+' '+' '.join(map(str,p['columns'])) for p in profiles)+'\n';assert (jobs/'profiles.tsv').read_text()==expected
    records=[json.loads(line) for line in (jobs/'gpu.log').read_text().splitlines() if line.startswith('{')];checks=[v for v in records if v['kind']=='check'];assert len(checks)==63
    summary=json.loads((jobs/'summary.json').read_text())['profiles'];observed=[]
    for p in profiles:
        group=[v for v in checks if v['profile']==p['name']];assert {(v['policy'],v['generation']) for v in group}=={(policy,generation) for policy in range(3) for generation in range(3)}
        assert all(v['bytes']==p['logical_bytes'] and v['positive_zero_cpu_gpu'] and v['padding_preserved'] for v in group)
        windows=[v for v in records if v['kind']=='timing' and v['profile']==p['name']];assert len(windows)==12
        for round in (0,1):
            part=sorted((v for v in windows if v['round']==round),key=lambda v:v['window']);assert [v['window'] for v in part]==list(range(6)) and [v['policy'] for v in part]==[0,1,2,2,1,0]
        assert all(len(v[k])==8 and all(x>0 for x in v[k]) for v in windows for k in ('wall_ms','gpu_ms'))
        means={str(policy):{metric:statistics.mean(x for v in windows if v['policy']==policy for x in v[metric]) for metric in ('wall_ms','gpu_ms')} for policy in range(3)}
        r=next(v for v in summary if v['profile']==p['name'])
        for policy in means:
            for metric in means[policy]:assert abs(means[policy][metric]-r['means'][policy][metric])<1e-12
        assert abs(r['joint_over_legacy_wall']-means['2']['wall_ms']/means['0']['wall_ms'])<1e-12 and abs(r['joint_over_direct_cpu_wall']-means['2']['wall_ms']/means['1']['wall_ms'])<1e-12
        observed.append(r)
    results.append(dict(trial=name,checks=63,timed_executions=672,profiles=observed))
audit=dict(verified_files=len(proof['files']),trials=results,scope='Same-command reset plus synthetic dependent read; GPU conditioning excluded; no model latency or fresh comparator claim',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(audit,indent=2)+'\n').encode());print(json.dumps(dict(verified_files=audit['verified_files'],trials=len(results),checks=sum(t['checks'] for t in results),timed_executions=sum(t['timed_executions'] for t in results),full_goal_met=False)))
