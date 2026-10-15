"""Separate active selected-kernel workloads from unchanged-route observations."""
from pathlib import Path
import json,sys
base=Path(sys.argv[1]).resolve();audit=json.loads((base/'comparison/verification.json').read_text())
assert audit['full_output_requests']==72 and audit['actual_selected_two_routes']
jobs=base/'comparison/evidence/logs/shape-selected-two-native-20261015/jobs'
summary=json.loads((jobs/'summary.json').read_text());profiles=[]
for p in audit['profiles']:
    raw=next(v for v in summary['profiles'] if (v['model'],v['batch'])==(p['model'],p['batch']))
    folder=jobs/Path(raw['evidence']).name
    records=json.loads((folder/'summary.json').read_text())['records']
    counts={}
    for policy in ('control','candidate'):
        rows=[v for v in records if v['build_policy']==policy]
        assert len(rows)==6
        seen={x for v in rows for x in v['two_body_counts'][1:]}
        assert len(seen)==1 and all(v['two_body_counts'][0]==0 for v in rows)
        counts[policy]=seen.pop()
    assert counts['control']==0
    active=counts['candidate']>0
    profiles.append(dict(model=p['model'],batch=p['batch'],selected_dispatches_per_decode_step=counts['candidate'],
        selected_kernel_active=active,decode_ratio=p['decode_ratio'],ttft_fraction=p['ttft_fraction'],
        decode_gain=p['decode_ratio']>1,ttft_gain=p['ttft_fraction']<1,
        both_directions_improve=p['decode_ratio']>1 and p['ttft_fraction']<1))
assert sum(p['selected_kernel_active'] for p in profiles)==3
result=dict(full_output_requests=72,profiles=profiles,
    active_profiles=3,active_profiles_improving_both=sum(p['selected_kernel_active'] and p['both_directions_improve'] for p in profiles),
    promoted=False,decision='Reject promotion: none of the three workloads executing selected kernels improves both native decode and warm TTFT. Retain c186. Inactive-route changes cannot establish benefit from executing the selected kernel.',
    scope='Single native ABBA cohort only; no fresh vLLM ratio or statistically confirmed end-to-end gain.',full_goal_met=False)
assert result['active_profiles_improving_both']==0
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
