"""Independently audit archived sources, validation and complete ABBA outputs."""
import hashlib,json,sys
from pathlib import Path
base=Path(sys.argv[1]).resolve();manifest=json.loads((base/'manifest.json').read_text());logs=base/'logs'
for entry in manifest['files']:
    path=base/entry['path'];assert path.is_relative_to(base)
    assert path.stat().st_size==entry['bytes'] and hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==entry['sha256'],entry['path']
assert all(s['exit_code']==0 for s in manifest['jobs'].values())
stages=json.loads((logs/'native-bfloat-body-checks-20261013/jobs/statuses.json').read_text())
assert len(stages)==16 and all(s['exit_code']==0 and s['interruption'] is None for s in stages)
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    assert (logs/'native-bfloat-body-checks-20261013/jobs'/(model+'-private-kv.log')).read_text().count('private_kv_reference PASS')==8
total=0;profiles=[]
for name,count in [('native-bfloat-body-native-20261013',6),('native-bfloat-body-native-confirm-20261013',4)]:
    summary=json.loads((logs/name/'jobs/summary.json').read_text());assert len(summary['profiles'])==count
    for profile in summary['profiles']:
        folder=logs/name/'jobs'/Path(profile['evidence']).name
        records=json.loads((folder/'summary.json').read_text())['records'];assert len(records)==12
        baseline=records[0]
        for value in records:
            assert all(value[key]==baseline[key] for key in ('token_ids','prompt_token_ids','finished','case'))
            needed=([24] if profile['batch']==1 else [22]) if profile['model']=='Qwen3-0.6B' else [26] if profile['model']=='Qwen2.5-Coder-0.5B-Instruct' else [80] if profile['batch']==1 else [32,32,16]
            assert value['prefill_chunk_plan']==needed
            if profile['model']=='Qwen3.5-0.8B':assert value['vision_reencoded'] and value['patch_sha256']==value['case']['image_metadata']['patch_sha256']
        for policy in ('control','candidate'):
            assert len([v for v in records if v['build_policy']==policy and v['request']>0])==4
            for repeat in (0,1):
                worker=folder/(profile['model']+'-b'+str(profile['batch'])+'-'+policy+'-'+str(repeat))
                assert json.loads((worker/'exit.json').read_text())['exit_code']==0
                assert json.loads((worker/'monitor.json').read_text())
        total+=len(records);profiles.append(profile)
assert total==120
for name,count in [('native-bfloat-decode-20261013',272),('native-bfloat-confirm-20261013',136)]:
    checks=json.loads((logs/name/'checks/summary.json').read_text())['checks']
    assert len(checks)==count and all(c['fp64_reference_passed'] and c['padding_preserved'] and c['worst_forward_error_ratio']<=1 for c in checks)
    assert all(c['bits_exact'] for c in checks if 'serial' in c['variant'])
cast=json.loads((logs/'native-bfloat-cast-20261013/jobs/summary.json').read_text())['checks']
assert len(cast)==2 and {v['fast_math'] for v in cast}=={False,True} and all(v['tested']==65280 and v['bit_mismatches']==0 for v in cast)
proof=dict(validation_stages=16,private_cache_profiles=24,component_checks=408,cast_checks=130560,complete_timed_requests=120,source_and_evidence_files_verified=len(manifest['files']),candidate_sha256=manifest['candidate_sha256'],control_sha256=manifest['control_sha256'],full_goal_met=False,primary_build_changed=False,profiles=profiles)
Path(sys.argv[2]).write_bytes((json.dumps(proof,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in proof.items() if k!='profiles'}))
