"""Summarize cast coverage and rejected/admitted unchanged numerical gates."""
from pathlib import Path
import json,sys
base=Path(sys.argv[1]).resolve();audit=json.loads((base/'component/verification.json').read_text())
jobs=base/'component/evidence/logs/bf16-feed-mlp-component-20261015/jobs'
meta=json.loads((jobs/'metadata.json').read_text());casts=[];checks=[]
for model,assets in meta['assets'].items():
    rows=[json.loads(v) for v in (jobs/(model+'-gpu.log')).read_text().splitlines() if v.startswith('{')]
    learned={c['tensor']:c['rows']*c['inner'] for c in assets['cases'] if '.mlp.' in c['tensor']}
    actual=[v for v in rows if v['kind']=='weight_cast']
    assert len(actual)==len(learned) and {v['tensor'] for v in actual}==set(learned)
    assert all(v['bitwise_equal'] and v['values']==learned[v['tensor']] for v in actual)
    casts.extend(dict(model=model,**v) for v in actual)
    checks.extend(dict(model=model,**v) for v in rows if v['kind']=='check')
assert len(casts)==12 and len(checks)==audit['numeric_checks']==48
assert all(v['comparison_worst']<=1 and v['finite'] for v in checks)
assert len(audit['rejected_profiles'])==6 and len(audit['profiles'])==6 and audit['timed_executions']==384
assert all(all(not v['passed'] or v['control_fp64_worst']<=1 for v in p['checks']) for p in audit['rejected_profiles'])
assert all(p['wall_fraction']>1 for p in audit['profiles'])
result=dict(learned_tensor_gpu_roundtrips=12,exact_learned_values=sum(v['values'] for v in casts),casts=casts,
    numerical_records=48,candidate_native_comparisons_passed=48,rejected_profiles=audit['rejected_profiles'],
    admitted_profiles=6,timed_executions=384,profiles=audit['profiles'],
    minimum_wall_fraction=min(p['wall_fraction'] for p in audit['profiles']),maximum_wall_fraction=max(p['wall_fraction'] for p in audit['profiles']),
    decision='Reject BF16-feed graph promotion: every admitted wall profile regresses. Original six native/reference failures remain excluded; no tolerance relaxation. No complete-model or memory saving claim.',
    scope='Single sequential synchronous per-MLP component cohort; exact original learned-bit casts, unchanged FP32 arithmetic and native/reference gates. No native integration or vLLM ratio.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in result.items() if k not in ('casts','rejected_profiles','profiles')}))
