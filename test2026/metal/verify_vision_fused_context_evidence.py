"""Verify archived hashes, terminal jobs, component references and real outputs."""
import hashlib,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();e=root/'evidence'
manifest=json.loads((e/'files.json').read_text())
for name,record in manifest.items():
    path=root/name
    assert path.is_file() and path.stat().st_size==record['bytes'],name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],name
def read(name):return json.loads((e/'logs'/name).read_text())
for name,code in [('vision-fused-context-20261013',1),('vision-fused-context-retry1-20261013',1),('vision-fused-context-retry2-20261013',0),('vision-fused-context-confirm-20261013',0),('vision-fused-context-checks-20261013',0),('vision-fused-context-inference-20261013',0),('vision-fused-context-inference-confirm-20261013',0)]:
    assert read(name+'/exit.json')['exit_code']==code,name
for job,source,path in [
    ('vision-fused-context-20261013','test2026/metal/vision_fused_context_gpu.mm','source-history/initial-compile-failure.mm'),
    ('vision-fused-context-retry1-20261013','test2026/metal/make_vision_fused_context_shader.py','source-history/initial-struct-extraction.py')]:
    expected=read(job+'/jobs/metadata.json')['source_sha256'][source]
    assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected,path
for name in ('vision-fused-context-retry2-20261013','vision-fused-context-confirm-20261013'):
    summary=read(name+'/jobs/summary.json');assert len(summary['checks'])==24
    assert all(c['bits_exact'] and c['max_abs_error']==0 and c['fp64_worst_ratio']<=1 and c['generations']==4 for c in summary['checks'])
    for source,sha in read(name+'/jobs/metadata.json')['source_sha256'].items():
        path=e/'original/metal_context.mm' if source=='src/backends/metal/metal_context.mm' else e/'source'/Path(source).name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,source
statuses=read('vision-fused-context-checks-20261013/jobs/statuses.json')
assert len(statuses)==18 and all(s['exit_code']==0 for s in statuses)
metadata=read('vision-fused-context-checks-20261013/jobs/metadata.json')
assert 'METAL_FUSED_VISION_CONTEXT' in metadata['options']
assert 'METAL_JOINED_PREFILL' not in metadata['options'] and 'METAL_NATIVE_BFLOAT_BODY' not in metadata['options']
for target in ('garnet','garnet_metal'):
    flags=(e/'build'/(target+'-flags.make')).read_text()
    assert '-DGARNET_METAL_FUSED_VISION_CONTEXT=1' in flags
    assert '-DGARNET_METAL_MPS_VISION_SCORES=1' in flags
for name,sha in metadata['source_sha256'].items():
    assert hashlib.sha256((e/'check-source'/name).read_bytes()).hexdigest()==sha,name
def verify_inference(job):
    summary=read(job+'/jobs/summary.json')
    assert len(summary['records'])==24 and len(summary['profiles'])==2
    inference_metadata=read(job+'/jobs/metadata.json')
    assert inference_metadata['library_sha256']['candidate']==metadata['candidate_sha256']
    assert inference_metadata['library_sha256']['control']==metadata['control_sha256']=='73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2'
    assert len(read(job+'/jobs/statuses.json'))==8 and all(v['exit_code']==0 for v in read(job+'/jobs/statuses.json'))
    for profile in summary['profiles']:
        batch=profile['batch'];case=read(job+'/jobs/Qwen3.5-0.8B-b'+str(batch)+'-case.json')
        records=[v for v in summary['records'] if v['case']['batch']==batch];assert len(records)==12
        ids=records[0]['token_ids'];prompt=records[0]['prompt_token_ids'];finished=records[0]['finished']
        for record in records:
            assert record['case']==case and record['vision_reencoded']
            assert record['patch_sha256']==case['image_metadata']['patch_sha256']
            assert record['token_ids']==ids and record['prompt_token_ids']==prompt and record['finished']==finished
            assert record['prefill_chunk_plan']==([80] if batch==1 else [32,32,16])
        for policy in ('control','candidate'):
            warm=[r for r in records if r['build_policy']==policy and r['request']>0];assert len(warm)==4
            means=profile['means'][policy]
            assert abs(means['ttft_ms']-statistics.mean(v['ttft_seconds'] for v in warm)*1000)<1e-9
            assert abs(means['decode_tps']-statistics.mean(v['decode_wall_steps_per_second'] for v in warm))<1e-9
        assert abs(profile['ttft_fraction']-profile['means']['candidate']['ttft_ms']/profile['means']['control']['ttft_ms'])<1e-12
    return summary
initial=verify_inference('vision-fused-context-inference-20261013')
confirm=verify_inference('vision-fused-context-inference-confirm-20261013')
assert [v['token_ids'] for v in initial['records']]==[v['token_ids'] for v in confirm['records']]
print(json.dumps(dict(verified_files=len(manifest),component_shape_checks=48,component_generation_comparisons=192,handoff_stages=18,complete_frame_outputs=48)))
