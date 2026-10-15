"""Summarize existing warm phase clocks without adding overlapping intervals."""
from pathlib import Path
import hashlib,json,statistics,sys

root=Path(sys.argv[1]).resolve();records=[]
joined=json.loads((root/'combined-verification.json').read_text())
assert joined['full_output_requests']==144 and joined['identical_assets_libraries_sources_and_cases']
for trial,folder,audit_name in (
    ('initial','logs/coalesced-body-native-20261015/native','verification.json'),
    ('repeat','logs/coalesced-body-native-confirm-20261015/jobs','confirmation-verification.json')):
    audit=json.loads((root/audit_name).read_text());assert audit['full_output_requests']==72 and audit['actual_coalesced']
    for profile in audit['profiles']:
        model,batch=profile['model'],profile['batch'];p=root/folder/(model+'-b'+str(batch)+'-attempt0')/'summary.json'
        raw=p.read_bytes();data=json.loads(raw);rows=data['records'];assert len(rows)==12
        for policy in ('control','candidate'):
            warm=[r for r in rows if r['build_policy']==policy and r['request']>0];assert len(warm)==4
            fields={}
            for key in ('ttft_seconds','worker_ttft_seconds','vision_seconds','assembly_seconds','prefill_wall_seconds','reset_seconds','common_preprocessing_seconds'):
                if all(key in r for r in warm):fields[key.removesuffix('_seconds')+'_ms']=1000*statistics.mean(r[key] for r in warm)
            for key in ('prefill_inference_ms',):
                if all(key in r for r in warm):fields[key]=statistics.mean(r[key] for r in warm)
            for key in ('prepare_ms','encode_ms','gpu_ms','wait_ms'):
                if all(key in r.get('prefill_profile',{}) for r in warm):fields['native_prefill_'+key]=statistics.mean(r['prefill_profile'][key] for r in warm)
            assert abs(fields['ttft_ms']-profile['means'][policy]['ttft_ms'])<1e-9
            records.append(dict(trial=trial,model=model,batch=batch,policy=policy,warm_requests=4,means_ms=fields,source=p.relative_to(root).as_posix(),source_sha256=hashlib.sha256(raw).hexdigest()))
assert len(records)==24
result=dict(scope='Existing independently audited native trials only; descriptive phase means. These clocks may overlap and are not additive. No current vLLM ratio or predicted optimization gain.',init_and_first_after_init_excluded=True,phase_records=records,full_goal_met=False)
(root/'phase-means.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(dict(phase_records=len(records),scope=result['scope'])))
