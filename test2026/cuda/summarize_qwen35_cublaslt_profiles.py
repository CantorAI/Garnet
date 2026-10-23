"""Shortlist isolated matrix candidates without forecasting model speed."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('output');a=p.parse_args()
root=Path(a.root);output=Path(a.output);assert not output.exists(),'Preserve matrix-screen summary'
profiles=[];sources={}
for mode,expected in [('decode',20),('prefill',16)]:
    path=root/f'cublaslt-profile-{mode}.log';blob=path.read_bytes()
    assert json.loads((root/f'cublaslt-profile-{mode}-exit.json').read_text())['exit_code']==0
    lines=blob.decode('utf-8-sig').splitlines()
    assert 'BF16_CUBLASLT_LAYOUT_COMPLETE' in lines
    records=[(i+1,json.loads(line)) for i,line in enumerate(lines) if line.startswith('{')]
    metadata=[r for _,r in records if r.get('event')=='diagnostic_metadata']
    assert len(metadata)==1 and metadata[0]['mode']==mode and metadata[0]['sm']==89
    sources[mode]={'file':path.name,'sha256':hashlib.sha256(blob).hexdigest(),'metadata':metadata[0]}
    groups={}
    for line,r in records:
        if 'cold_median_us' not in r:continue
        assert r['math']['fp64_samples']==128 and r['math']['max_fixed_bound_fraction']<=1
        groups.setdefault((r['m'],r['k'],r['n']),[]).append(dict(r,source_line=line))
    assert len(groups)==expected
    for (m,k,n),rows in groups.items():
        baselines=[r for r in rows if r['provider']=='GemmEx' and r['layout']=='NK/TN']
        assert len(baselines)==1
        baseline=baselines[0]
        same=[r for r in rows if r['layout']=='NK/TN']
        zero=[r for r in same if r.get('workspace_limit_bytes',r.get('workspace_bytes',0))==0]
        def select(candidates):
            best=min(candidates,key=lambda r:r['cold_median_us'])
            return {'record':best,'sweep_speed_ratio':baseline['cold_median_us']/best['cold_median_us'],
                'reuse_speed_ratio_for_same_candidate':baseline['warm_median_us']/best['warm_median_us']}
        profiles.append({'mode':mode,'m':m,'k':k,'n':n,'baseline':baseline,
            'best_same_layout_zero_workspace':select(zero),'best_same_layout':select(same),'best_any_layout':select(rows)})
result={'sources':sources,'profiles':profiles,'goal_met':False,'model_quality_accepted':False,
    'scope':'Synthetic isolated matrices, nine event samples per candidate. Candidates selected on256MiB-sweep median; this does not establish L2 residency. 128 unchanged FP64 sample bounds per result, not a complete model quality gate. Multiple-candidate selection needs repeat and runtime validation. Algorithm IDs alone do not encode complete heuristic configurations.'}
output.write_text(json.dumps(result,indent=2))
print(json.dumps({'output':str(output),'profiles':len(profiles),'goal_met':False}))
