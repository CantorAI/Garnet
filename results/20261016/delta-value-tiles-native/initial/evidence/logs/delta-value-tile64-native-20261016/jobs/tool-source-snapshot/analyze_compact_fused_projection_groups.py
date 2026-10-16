"""Predeclared learned-group both-batch gain/null gate; all groups retained."""
from pathlib import Path
import json,sys
base=Path(sys.argv[1]).resolve();labels=sys.argv[2:] or ['initial'];assert labels in (['initial'],['initial','repeat'])
trials=[];identity=None
for label in labels:
    folder=base/label;proof=json.loads((folder/'evidence/manifest.json').read_text());assert len(proof['jobs'])==1
    trial=next(iter(proof['jobs']));assert proof['jobs'][trial]['exit_code']==0
    meta=json.loads((folder/'evidence/logs'/trial/'jobs/metadata.json').read_text())
    current={k:meta[k] for k in ('source_sha256','native_shader_sha256','generated_harness_sha256','policies','timing_design','hardware','versions','assets')}
    if identity is None:identity=current
    else:assert current==identity,'Repeat must preserve complete source, groups, environment and timing design'
    audit=json.loads((folder/'verification.json').read_text());assert audit['group_batch_checks']==16 and len(audit['profiles'])==32 and audit['timed_executions']==6400
    groups=[]
    for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
        for group in sorted({p['group'] for p in audit['profiles'] if p['model']==model}):
            serving=[]
            for batch in (1,2):
                rows={p['candidate_mode']:p for p in audit['profiles'] if (p['model'],p['group'],p['batch'])==(model,group,batch)};assert set(rows)=={1,2}
                fused,null=rows[1],rows[2]
                calibrated=all(.95<=null[k+'_fraction']<=1.05 for k in ('gpu','wall'))
                gain=all(fused[k+'_fraction']<=.97 for k in ('gpu','wall'))
                serving.append(dict(batch=batch,gpu_fraction=fused['gpu_fraction'],wall_fraction=fused['wall_fraction'],null_gpu_fraction=null['gpu_fraction'],null_wall_fraction=null['wall_fraction'],screen_pass=calibrated and gain))
            groups.append(dict(model=model,group=group,serving=serving,learned=group!='irregular-readonly-pair',qualified=group!='irregular-readonly-pair' and all(p['screen_pass'] for p in serving)))
    trials.append(dict(label=label,groups=groups))
initial={(g['model'],g['group']) for g in trials[0]['groups'] if g['qualified']};qualified=[]
if len(trials)==2:
    qualified=[dict(model=m,group=g) for m,g in sorted(initial) if all(next(r for r in t['groups'] if (r['model'],r['group'])==(m,g))['qualified'] for t in trials)]
result=dict(trials=trials,repeat_admitted=bool(initial),native_integration_groups=qualified,promote=False,fresh_vllm_run=False,target_count_update=False,full_goal_met=False,
 scope='Learned group needs >=3% GPU AND wall gain BOTH batches, own null within5%, all original numerical groups, unchanged complete confirmation. Component only; no full-model or vLLM gain.')
(base/('decision-'+labels[-1]+'.json')).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='trials'}))
