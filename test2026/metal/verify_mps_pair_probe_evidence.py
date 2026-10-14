"""Validate archived pair-probe identities, exact outputs and timing coverage."""
import hashlib,json,sys
from pathlib import Path
base=Path(sys.argv[1]).resolve();manifest=json.loads((base/'manifest.json').read_text())
for e in manifest['files']:
    p=base/e['path'];assert p.is_relative_to(base)
    assert p.stat().st_size==e['bytes'] and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==e['sha256']
assert len(manifest['jobs'])==4 and all(v['exit_code']==0 for v in manifest['jobs'].values())
models=('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B');proof={};shape_checks=0;generations=0
for name in manifest['jobs']:
    folder=base/'logs'/name/'checks';summary=json.loads((folder/'summary.json').read_text());checks=summary['checks']
    assert len(checks)==24 and len(summary['results'])==24 and all(c['bits_exact'] and c['padding_preserved'] and c['worst_ratio']==0 for c in checks)
    dependency=name!='concurrent-mps-pair-20261013'
    for model in models:
        raw=[json.loads(line) for line in (folder/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
        assert len([v for v in raw if v['kind']=='check'])==8
        for rows in (22,24,26,32,44,52,64,80):
            windows=[v for v in raw if v['kind']=='timing' and v['rows']==rows]
            key='joined' if 'joined' in name else 'concurrent'
            assert len(windows)==8 and sum(v[key] for v in windows)==4
            assert all(v['gpu_completion_span_ms']>0 and v['host_wall_ms']>0 for v in windows)
        if dependency:
            quality=[v for v in raw if v['kind']=='check'];assert all(v['generation_checks']==16 and v['dependent_consumer_bits_exact'] for v in quality)
            if 'joined' in name:assert all(v['consumer_worst_ratio']==0 for v in quality)
            generations+=sum(v['generation_checks'] for v in quality)
    shape_checks+=len(checks);proof[name]=summary['results']
result=dict(verified_files=len(manifest['files']),shape_checks=shape_checks,alternating_generation_comparisons=generations,all_checked_outputs_bit_exact=True,serving_changed=False,full_goal_met=False,results=proof)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in result.items() if k!='results'}))
