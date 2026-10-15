"""Join two independently audited identical-source component cohorts."""
from pathlib import Path
import hashlib,json,sys

base=Path(sys.argv[1]).resolve();audits=[];metadata=[]
for name in ('initial','confirmation'):
    folder=base/name;audit=json.loads((folder/'verification.json').read_text())
    assert audit['numeric_checks']==72 and audit['control_profiles']==189 and audit['timed_executions']==1152
    assert not audit['rejected_profiles'] and len(audit['profiles'])==18
    trial=audit['profiles'][0]['trial'];path=folder/'evidence/logs'/trial/'jobs/metadata.json'
    proof=json.loads((folder/'evidence/manifest.json').read_text())
    info=proof['files'][path.relative_to(folder).as_posix()]
    raw=path.read_bytes();assert len(raw)==info['bytes'] and hashlib.sha256(raw).hexdigest()==info['sha256']
    metadata.append(json.loads(raw));audits.append(audit)
keys=('source_sha256','context_generated_sha256','private_header_sha256','harness_generated_sha256',
      'context_macro_defines','assets','hardware','versions','reference_static_library_sha256')
for key in keys:assert metadata[0][key]==metadata[1][key],key
maps=[{(p['model'],p['batch'],p['policy']):p for p in a['profiles']} for a in audits]
assert maps[0].keys()==maps[1].keys() and len(maps[0])==18
profiles=[]
for key in sorted(maps[0]):
    pair=[m[key] for m in maps]
    for profile in pair:
        assert len(profile['checks'])==4 and all(v['passed'] and v['bitwise_equal'] and v['comparison_worst']==0 for v in profile['checks'])
    wall=[p['wall_fraction'] for p in pair];gpu=[p['gpu_fraction'] for p in pair]
    profiles.append(dict(model=key[0],batch=key[1],policy=key[2],wall_fractions=wall,gpu_fractions=gpu,
        wall_gain_in_both=all(v<1 for v in wall),gpu_gain_in_both=all(v<1 for v in gpu),
        wall_and_gpu_gain_in_both=all(v<1 for v in wall+gpu)))
result=dict(component_cohorts=2,numeric_records=144,safety_profiles=378,timed_executions=2304,
    identical_runtime_source_flags_assets_and_versions=True,
    binary_sha256=[m['binary_sha256'] for m in metadata],profiles=profiles,
    wall_gain_profiles_in_both=sum(p['wall_gain_in_both'] for p in profiles),
    gpu_gain_profiles_in_both=sum(p['gpu_gain_in_both'] for p in profiles),
    wall_and_gpu_gain_profiles_in_both=sum(p['wall_and_gpu_gain_in_both'] for p in profiles),
    scope='Repeated component directions only; no statistical confidence, complete-model performance or vLLM ratio. Original CPU packing proof reused, not replicated.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
