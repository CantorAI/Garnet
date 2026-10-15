"""Record every remaining matched target from both independently audited cohorts."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();trials=[];by_profile={}
for label,base,job in [('initial',root,'wide-batch-vllm-20261015'),('repeat',root/'confirmation','wide-batch-vllm-confirm-20261015')]:
    audit=json.loads((root/('verification.json' if label=='initial' else 'confirmation-verification.json')).read_text())
    assert audit['complete_matched_outputs']==72 and audit['actual_dtype_audits']==3
    jobs=base/'evidence/logs'/job/'jobs';summary=json.loads((jobs/'summary.json').read_text());profiles=[]
    for p in summary['profiles']:
        s=p['summary'];folder=jobs/Path(p['evidence']).name;raw=json.loads((folder/'summary.json').read_text());key=(p['model'],p['batch'])
        for r in raw['records']:
            assert len(r['normalized_token_ids'])==p['batch'] and len({tuple(v) for v in r['normalized_token_ids']})==1
        previous=by_profile.setdefault(key,[])
        record=dict(trial=label,decode_ratio=s['decode_ratio'],ttft_fraction=s['ttft_fraction'],means=s['means'],additional_decode_fraction=max(0,2/s['decode_ratio']-1),additional_ttft_reduction_fraction=max(0,1-.5/s['ttft_fraction']))
        previous.append(record);profiles.append(dict(model=key[0],batch=key[1],**record))
    trials.append(dict(trial=label,profiles=profiles))
assert len(by_profile)==6 and all(len(v)==2 for v in by_profile.values())
result=dict(scope='Two frozen matched-FP32 cohorts; warm means only; init/first-after-init separate in raw evidence; no hardware-cold sum',complete_matched_outputs=144,actual_gpu_dtype_audits=6,decode_profiles_met_in_both=sum(all(v['decode_ratio']>=2 for v in p) for p in by_profile.values()),ttft_profiles_met_in_both=sum(all(v['ttft_fraction']<=.5 for v in p) for p in by_profile.values()),both_targets_met_in_both=sum(all(v['decode_ratio']>=2 and v['ttft_fraction']<=.5 for v in p) for p in by_profile.values()),aggregate_decode_throughput_batch_multiplier_verified=True,trials=trials,full_goal_met=False)
(root/'remaining-budgets.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='trials'}))
