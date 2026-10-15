"""Select conservative shape candidates from two independently audited probes.

This is training/selection evidence, not an independent performance confirmation.
Require a >=2% GPU gain and a wall gain in every observed matrix/cohort for a
shape/policy; keep the original kernel elsewhere. Full inference must confirm it.
"""
from pathlib import Path
import hashlib,json,statistics,sys

base=Path(sys.argv[1]).resolve();audits=[];metadata=[]
for name in ('initial','confirmation'):
    folder=base/name;a=json.loads((folder/'verification.json').read_text())
    assert a['selected_original_matrices']==20 and a['numeric_checks']==480
    assert a['safety_profiles']==189 and a['timed_executions']==7680
    assert not a['rejected_profiles'] and len(a['profiles'])==120
    trial=a['profiles'][0]['trial'];path=folder/'evidence/logs'/trial/'jobs/metadata.json'
    proof=json.loads((folder/'evidence/manifest.json').read_text())
    raw=path.read_bytes();info=proof['files'][path.relative_to(folder).as_posix()]
    assert len(raw)==info['bytes'] and hashlib.sha256(raw).hexdigest()==info['sha256']
    metadata.append(json.loads(raw));audits.append(a)
for key in ('source_sha256','context_generated_sha256','private_header_sha256','harness_generated_sha256',
            'context_macro_defines','assets','hardware','versions','reference_static_library_sha256','binary_sha256'):
    assert metadata[0][key]==metadata[1][key],key
maps=[{(p['model'],p['matrix'],p['batch'],p['policy']):p for p in a['profiles']} for a in audits]
assert maps[0].keys()==maps[1].keys() and len(maps[0])==120
groups={}
for key in sorted(maps[0]):
    pair=[m[key] for m in maps]
    assert pair[0]['output_rows']==pair[1]['output_rows'] and pair[0]['inner']==pair[1]['inner']
    shape=(key[2],pair[0]['output_rows'],pair[0]['inner'],key[3])
    for p in pair:
        assert len(p['checks'])==4 and all(c['passed'] and c['bitwise_equal'] and c['comparison_worst']==0 for c in p['checks'])
    groups.setdefault(shape,[]).append(dict(model=key[0],matrix=key[1],
        wall_fractions=[p['wall_fraction'] for p in pair],gpu_fractions=[p['gpu_fraction'] for p in pair]))
profiles=[]
for (batch,n,k,policy),items in sorted(groups.items()):
    gpu=[x for item in items for x in item['gpu_fractions']]
    wall=[x for item in items for x in item['wall_fractions']]
    profiles.append(dict(batch=batch,output_rows=n,inner=k,policy=policy,threads=policy and (32<<policy),
        observations=items,worst_gpu_fraction=max(gpu),mean_gpu_fraction=statistics.mean(gpu),
        worst_wall_fraction=max(wall),eligible=max(gpu)<=.98 and max(wall)<1))
selected={}
for p in profiles:
    if p['eligible']:
        key=(p['batch'],p['output_rows'],p['inner'])
        if key not in selected or p['worst_gpu_fraction']<selected[key]['worst_gpu_fraction']:selected[key]=p
result=dict(component_cohorts=2,selected_original_matrices_per_cohort=20,numeric_checks=960,
    safety_profiles=378,timed_executions=15360,identical_runtime_source_flags_assets_versions_binary=True,
    policy='Require GPU fraction <=0.98 and wall fraction <1 in every observed matrix/cohort for geometry; choose minimum worst GPU fraction; original four-output otherwise.',
    profiles=profiles,selected=list(selected.values()),
    scope='Data-derived dispatch candidates only; repeated resident one-matrix timing differs from inference; selection uses both cohorts and requires fresh full-model confirmation. No model gain or vLLM ratio.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
