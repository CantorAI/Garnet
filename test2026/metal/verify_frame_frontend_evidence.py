"""Verify exact frontend output and complete matched native/vLLM executions."""
import hashlib,json,sys
from pathlib import Path
base=Path(sys.argv[1]).resolve();manifest=json.loads((base/'manifest.json').read_text());logs=base/'logs'
for e in manifest['files']:
    p=base/e['path'];assert p.is_relative_to(base)
    assert p.stat().st_size==e['bytes'] and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==e['sha256']
for name,status in manifest['jobs'].items():assert status['exit_code']==(1 if name=='frame-frontend-profile-20261013' else 0)
profile=json.loads((logs/'frame-frontend-profile-retry1-20261013/checks/summary.json').read_text());assert len(profile['records'])==24 and all(v['c_contiguous'] for v in profile['records'])
component_records=0
for name in ('frame-frontend-abba-20261013','frame-frontend-webp-abba-20261013'):
    result=json.loads((logs/name/'checks/summary.json').read_text());assert len(result['records'])==384 and len({v['rgb_sha256'] for v in result['records']})==1
    component_records+=len(result['records'])
compatibility_checks=0
for name in ('frame-frontend-exact-20261013','frame-frontend-exact-fresh-20261013','frame-frontend-selected-exact-20261013'):
    result=json.loads((logs/name/'checks/summary.json').read_text());assert len(result['checks'])==448 and result['exact'] and result['late_exif_cv_fallback_checks']==14
    compatibility_checks+=448+14
inference=logs/'frame-frontend-inference-20261013/jobs';summary=json.loads((inference/'summary.json').read_text());assert len(summary['profiles'])==2
requests=0
for batch in (1,2):
    folder=inference/('b'+str(batch));result=json.loads((folder/'summary.json').read_text());records=result['records'];assert len(records)==24
    reference=records[0]
    for value in records:
        assert all(value[k]==reference[k] for k in ('normalized_token_ids','prompt_token_ids','finished','case','patch_sha256','rgb_sha256'))
        if value['backend']=='garnet':assert value['vision_reencoded'] and value['prefill_chunk_plan']==([80] if batch==1 else [32,32,16])
        else:assert not value['prefix_caching'] and not value['speculative_decoding'] and value['configured_model_dtype'] in ('torch.float32','float32')
    for backend in ('garnet','vllm-metal'):
        for policy in ('control','selected'):
            selected=[v for v in records if v['backend']==backend and v['frontend_policy']==policy];assert len(selected)==6 and len([v for v in selected if v['request']>0])==4
            for repeat in (0,1):
                worker=folder/(backend+'-'+policy+'-'+str(repeat));assert json.loads((worker/'exit.json').read_text())['exit_code']==0 and json.loads((worker/'monitor.json').read_text())
                if backend=='vllm-metal':assert len({x for v in selected if v['repeat']==repeat for x in v['encoder_cache_identities']})==batch*3
    requests+=len(records)
result=dict(verified_files=len(manifest['files']),timed_component_records=component_records,compatibility_comparisons=compatibility_checks,complete_inference_requests=requests,selected_library_sha256=manifest['selected_library_sha256'],counts_as_default_bf16_speedup=False,full_goal_met=False,profiles=summary['profiles'])
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
