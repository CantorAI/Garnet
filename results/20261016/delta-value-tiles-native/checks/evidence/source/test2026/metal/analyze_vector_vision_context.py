"""Apply predeclared all-shape exactness, both-serving-shape gain and own-null calibration."""
from pathlib import Path
import json,math,sys
base=Path(sys.argv[1]).resolve();labels=sys.argv[2:] or ['initial'];assert labels in (['initial'],['initial','repeat'])
trials=[];identities=None
for label in labels:
    proof=json.loads((base/label/'evidence/manifest.json').read_text());assert len(proof['jobs'])==1
    trial=next(iter(proof['jobs']));assert proof['jobs'][trial]['exit_code']==0
    meta=json.loads((base/label/'evidence/logs'/trial/'jobs/metadata.json').read_text())
    current={k:meta[k] for k in ('source_sha256','shader_sha256','harness_sha256','variants','hardware','versions')}
    if identities is None:identities=current
    else:assert current==identities,'Confirmation must preserve original complete sources, variants and environment'
    audit=json.loads((base/label/'verification.json').read_text());assert audit['shape_variant_checks']==72
    checks=audit['checks'];profiles=audit['profiles'];groups=[]
    for variant in (1,2):
        selected=[c for c in checks if c['variant']==variant];assert len(selected)==24
        numerical=all(c['passed'] for c in selected);serving=[]
        for batch in (1,2):
            match=lambda p:p['check']['batch']==batch and (p['check']['tokens'],p['check']['heads'],p['check']['dim'])==(216,12,64)
            null=next(p for p in profiles if match(p) and p['check']['variant']==0)
            candidate=next((p for p in profiles if match(p) and p['check']['variant']==variant),None)
            calibrated=all(.95<=null[k+'_fraction']<=1.05 for k in ('gpu','wall'))
            gain=candidate is not None and all(candidate[k+'_fraction']<=.95 for k in ('gpu','wall'))
            serving.append(dict(batch=batch,gpu_fraction=None if candidate is None else candidate['gpu_fraction'],
                wall_fraction=None if candidate is None else candidate['wall_fraction'],null_gpu_fraction=null['gpu_fraction'],
                null_wall_fraction=null['wall_fraction'],screen_pass=numerical and calibrated and gain))
        groups.append(dict(variant=variant,vector_width=2 if variant==1 else 4,all24_numerical_pass=numerical,
            serving=serving,qualified=numerical and all(p['screen_pass'] for p in serving)))
    trials.append(dict(label=label,candidates=groups))
initial=[c['variant'] for c in trials[0]['candidates'] if c['qualified']]
qualified=[]
if len(trials)==2:
    for variant in initial:
        candidates=[next(c for c in t['candidates'] if c['variant']==variant) for t in trials]
        if all(c['qualified'] for c in candidates):
            values=[s['wall_fraction'] for c in candidates for s in c['serving']]
            qualified.append(dict(variant=variant,vector_width=candidates[0]['vector_width'],geomean_serving_wall_fraction=math.prod(values)**.25))
qualified.sort(key=lambda c:(c['geomean_serving_wall_fraction'],c['vector_width']))
result=dict(trials=trials,repeat_admitted=bool(initial),qualified=qualified,
    native_integration_vector_width=qualified[0]['vector_width'] if qualified else None,
    promote=False,fresh_vllm_run=False,target_count_update=False,full_goal_met=False,
    scope='Predeclared synthetic component screen only. All24 original numerical checks per width; >=5% GPU AND wall gain at both serving batches, own null within5%, repeated unchanged. No trained/full-model speed claim.')
(base/('decision-'+labels[-1]+'.json')).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in result.items() if k!='trials'}))
