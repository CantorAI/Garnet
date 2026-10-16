"""Apply predeclared >=5% GPU/wall screen with per-cohort native-null calibration."""
from pathlib import Path
import json,sys
base=Path(sys.argv[1]).resolve();cohorts={};metadata={}
for label in ('repaired','confirmation'):
    folder=base/label
    if not (folder/'verification.json').exists():continue
    audit=json.loads((folder/'verification.json').read_text())
    assert audit['numeric_checks']==72 and not audit['full_goal_met']
    cohorts[label]=audit
    proof=json.loads((folder/'evidence/manifest.json').read_text());trial=next(iter(proof['jobs']))
    assert proof['jobs'][trial]['exit_code']==0
    metadata[label]=json.loads((folder/'evidence/logs'/trial/'jobs/metadata.json').read_text())
assert 'repaired' in cohorts
if 'confirmation' in cohorts:
    for key in ('source_sha256','context_generated_sha256','private_header_sha256','harness_generated_sha256','context_macro_defines','reference_static_library_sha256','sdk_headers','hardware','versions','assets'):
        assert metadata['repaired'][key]==metadata['confirmation'][key],key
observations=[];qualified=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    for batch in (1,2):
        for policy in (1,2):
            row=dict(model=model,batch=batch,policy=policy,cohorts={})
            for label,audit in cohorts.items():
                null=next(v for v in audit['profiles'] if (v['model'],v['batch'],v['policy'])==(model,batch,3))
                candidate=next((v for v in audit['profiles'] if (v['model'],v['batch'],v['policy'])==(model,batch,policy)),None)
                calibrated=all(abs(null[k]-1)<=.05 for k in ('gpu_fraction','wall_fraction'))
                passed=candidate is not None and calibrated and all(candidate[k]<=.95 for k in ('gpu_fraction','wall_fraction'))
                row['cohorts'][label]=dict(admitted=candidate is not None,calibrated=calibrated,qualifies=passed,
                    gpu_fraction=None if candidate is None else candidate['gpu_fraction'],
                    wall_fraction=None if candidate is None else candidate['wall_fraction'],
                    null_gpu_fraction=null['gpu_fraction'],null_wall_fraction=null['wall_fraction'])
            row['qualifies_in_all_available']=all(v['qualifies'] for v in row['cohorts'].values())
            if row['qualifies_in_all_available']:qualified.append({k:row[k] for k in ('model','batch','policy')})
            observations.append(row)
result=dict(cohorts=list(cohorts),observations=observations,qualified=qualified,confirmation_complete=len(cohorts)==2,
    rule='Original numeric gates; candidate GPU and wall fraction <=0.95, own native-null GPU/wall within5%, in both frozen cohorts before integration.',
    scope='Trained projection sets only, not full-model or matched vLLM result. No statistical confidence interval.',full_goal_met=False)
(base/'decision-analysis.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='observations'}))
