"""Apply the predeclared five-percent GPU/null rule without model extrapolation."""
from pathlib import Path
import hashlib,json,statistics,sys
base=Path(sys.argv[1]).resolve();audits=[];metadata=[]
for name in ('initial','confirmation'):
    folder=base/name;a=json.loads((folder/'verification.json').read_text())
    assert a['selected_original_matrices']==20 and a['numeric_checks']==480
    assert a['safety_profiles']==189 and a['timed_executions']==7680 and not a['rejected_profiles']
    assert len(a['profiles'])==120
    trial=a['profiles'][0]['trial'];path=folder/'evidence/logs'/trial/'jobs/metadata.json'
    proof=json.loads((folder/'evidence/manifest.json').read_text());raw=path.read_bytes()
    info=proof['files'][path.relative_to(folder).as_posix()]
    assert len(raw)==info['bytes'] and hashlib.sha256(raw).hexdigest()==info['sha256']
    metadata.append(json.loads(raw));audits.append(a)
for key in ('source_sha256','context_generated_sha256','private_header_sha256','harness_generated_sha256',
            'context_macro_defines','assets','hardware','versions','reference_static_library_sha256','binary_sha256','new_shader_compiled'):
    assert metadata[0][key]==metadata[1][key],key
assert metadata[0]['new_shader_compiled'] is False
maps=[{(p['model'],p['matrix'],p['batch'],p['policy']):p for p in a['profiles']} for a in audits]
assert maps[0].keys()==maps[1].keys() and len(maps[0])==120
groups={};null_records=[]
for key in sorted(maps[0]):
    pair=[m[key] for m in maps]
    assert (pair[0]['output_rows'],pair[0]['inner'])==(pair[1]['output_rows'],pair[1]['inner'])
    for p in pair:assert len(p['checks'])==4 and all(v['passed'] and v['bitwise_equal'] and v['comparison_worst']==0 for v in p['checks'])
    shape=(key[2],pair[0]['output_rows'],pair[0]['inner'],key[3])
    record=dict(model=key[0],matrix=key[1],batch=key[2],policy=key[3],
        gpu_fractions=[p['gpu_fraction'] for p in pair],wall_fractions=[p['wall_fraction'] for p in pair])
    groups.setdefault(shape,[]).append(record)
    if key[3]==1:null_records.append(record)
assert len(null_records)==40
profiles=[];selected={}
for (batch,n,k,policy),items in sorted(groups.items()):
    if policy==1:continue
    null=groups[(batch,n,k,1)]
    null_ok=all(.95<=v<=1.05 for r in null for key in ('gpu_fractions','wall_fractions') for v in r[key])
    gpu=[v for r in items for v in r['gpu_fractions']];wall=[v for r in items for v in r['wall_fractions']]
    profile=dict(batch=batch,output_rows=n,inner=k,threads=32<<policy,observations=items,null_observations=null,
        null_within_five_percent=null_ok,worst_gpu_fraction=max(gpu),worst_wall_fraction=max(wall),
        eligible=null_ok and max(gpu)<=.95 and max(wall)<1)
    profiles.append(profile)
    shape=(batch,n,k)
    if profile['eligible'] and (shape not in selected or profile['worst_gpu_fraction']<selected[shape]['worst_gpu_fraction']):selected[shape]=profile
null_summary=[]
for i,name in enumerate(('initial','confirmation')):
    values={k:[r[k][i] for r in null_records] for k in ('gpu_fractions','wall_fractions')}
    null_summary.append(dict(cohort=name,profiles=40,**{key:dict(minimum=min(v),maximum=max(v),mean=statistics.mean(v),
        outside_five_percent=sum(not .95<=x<=1.05 for x in v)) for key,v in values.items()}))
result=dict(cohorts=2,numeric_checks=960,safety_profiles=378,timed_executions=15360,
    identical_runtime_source_flags_assets_versions_binary=True,same_original_shader_and_pipeline=True,
    null_comparisons_per_cohort=40,null_summary=null_summary,profiles=profiles,selected=list(selected.values()),
    selection_rule='GPU fraction<=0.95 and wall<1 for every duplicate geometry/matrix observation in both cohorts; null GPU/wall within5% in both. Minimum worst GPU fraction wins. Original64 otherwise.',
    scope='Repeated resident one-projection diagnostic/calibration. Null GPU work identical but CPU hook overhead is included; observed variability is not causally attributed. No confidence interval, full-model gain or vLLM ratio.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
