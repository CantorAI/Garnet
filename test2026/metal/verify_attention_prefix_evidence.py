"""Verify sources, compiled feature, references, complete outputs and warm means."""
import hashlib,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();e=root/'evidence';manifest=json.loads((e/'files.json').read_text())
for name,item in manifest.items():
    data=(root/name).read_bytes();assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],name
def read(job,name):return json.loads((e/'logs'/job/name).read_text())
assert read('attention-prefix-probe-20261013','exit.json')['exit_code']==1
assert read('attention-prefix-guard-checks-20261013','exit.json')['exit_code']==0
guard=json.loads((e/'logs/attention-prefix-guard-checks-20261013/output.log').read_text())['process_exit_checks']
assert [(v['pid_live'],v['verified_service'],v['rejected']) for v in guard]==[(False,False,False),(True,False,True),(True,True,False)]
for job in ('attention-prefix-probe-retry1-20261013','attention-prefix-probe-confirm-20261013'):
    assert read(job,'exit.json')['exit_code']==0
    summary=read(job,'jobs/summary.json');assert len(summary['checks'])==18
    assert all(c['cases']==28 and c['bits_exact'] and c['max_abs_error']==0 and c['fp64_worst_ratio']<=1 for c in summary['checks'])
    rows=[json.loads(line) for line in (e/'logs'/job/'jobs/gpu.log').read_text().splitlines() if line.startswith('{')]
    assert len(rows)==162 and sum(r.get('samples',0) for r in rows)==1728
    for name,sha in read(job,'jobs/metadata.json')['source_sha256'].items():
        if name=='src/backends/metal/metal_context.mm':path=e/'original/metal_context.mm'
        elif name=='test2026/metal/compare_vllm_metal.py':path=e/'original/compare_vllm_metal.py'
        elif name=='test2026/metal/make_attention_prefix_shader.py':path=root/'source-history/make_attention_prefix_initial.py'
        else:path=e/'source'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,name
shader=(e/'logs/attention-prefix-probe-confirm-20261013/jobs/combined.metal').read_text()
native=(e/'source/src/backends/metal/metal_context.mm').read_text()
fragment=native[native.index('struct ChunkAttentionParams {'):native.index('struct ScoresParams {')]
def resolve(text,enabled):
    lines=[];inside=False;keep=True
    for line in text.splitlines(keepends=True):
        if line.strip()=='#ifdef GARNET_ATTENTION_PREFIX':assert not inside;inside=True;keep=enabled
        elif inside and line.strip()=='#else':keep=not enabled
        elif inside and line.strip()=='#endif':inside=False;keep=True
        elif keep:lines.append(line)
    assert not inside;return ''.join(lines)
production=resolve(fragment,True);production=production[production.index('kernel void attention_chunk('):]
tested=shader[shader.index('kernel void attention_chunk_prefix('):].replace('kernel void attention_chunk_prefix(','kernel void attention_chunk(',1)
assert production.strip()==tested.strip(),'Production shortcut differs from tested kernel'
control=resolve(fragment,False)
assert shader.startswith('#include <metal_stdlib>\nusing namespace metal;\n'+control)
assert 'source=[@"#define GARNET_ATTENTION_PREFIX 1\\n" stringByAppendingString:source]' in native
job='attention-prefix-checks-20261013';assert read(job,'exit.json')['exit_code']==0
statuses=read(job,'jobs/statuses.json');assert len(statuses)==16 and all(v['exit_code']==0 for v in statuses)
metadata=read(job,'jobs/metadata.json')
for name,sha in metadata['source_sha256'].items():assert hashlib.sha256((e/'check-source'/name).read_bytes()).hexdigest()==sha,name
for target in ('garnet','garnet_metal'):
    flags=(e/'build'/(target+'-flags.make')).read_text();assert '-DGARNET_METAL_ATTENTION_PREFIX=1' in flags
    for option in ('FUSED_VISION_CONTEXT','JOINED_PREFILL','NATIVE_BFLOAT_BODY'):assert '-DGARNET_METAL_'+option+'=1' not in flags
inference_counts={}
cleanup=json.loads((root/'source-history/cleanup-removed.json').read_text(encoding='utf-8-sig'))
plan=json.loads((root/'source-history/cleanup-plan.json').read_text(encoding='utf-8-sig'))
assert len(cleanup['removed'])==len(plan['items'])==705
assert cleanup['total_bytes']==plan['total_bytes']==sum(v['bytes'] for v in cleanup['removed'])
assert {v['path']:v['sha256'] for v in cleanup['removed']}=={v['path']:v['sha256'] for v in plan['items']}
assert all(v['path'].startswith('logs/') and v['path'].endswith('.bin') for v in cleanup['removed'])
assert len(cleanup['checkpoint_sha256'])==3
for directory in sorted((e/'logs').glob('attention-prefix-inference*')):
    job=directory.name
    if job in ('attention-prefix-inference-20261013','attention-prefix-inference-confirm-20261013','attention-prefix-inference-retry1-20261013','attention-prefix-inference-confirm-retry1-20261013'):
        assert read(job,'exit.json')['exit_code']==1;continue
    assert read(job,'exit.json')['exit_code']==0
    summary=read(job,'jobs/summary.json');profiles=summary['profiles'];records=summary['records']
    assert len(records)==12*len(profiles)
    assert len(profiles)==6 and len(records)==72
    info=read(job,'jobs/metadata.json');identity=info['library_sha256']
    for model,assets in info['assets'].items():assert assets['model.safetensors']==cleanup['checkpoint_sha256']['models/'+model+'/model.safetensors']
    assert identity['candidate']==metadata['candidate_sha256'] and identity['control']==metadata['control_sha256']=='73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2'
    for profile in profiles:
        values=[v for v in records if v['model_name']==profile['model'] and v['case']['batch']==profile['batch']];assert len(values)==12
        for v in values:
            assert v['token_ids']==values[0]['token_ids'] and v['prompt_token_ids']==values[0]['prompt_token_ids'] and v['finished']==values[0]['finished']
            assert v['case']==values[0]['case'] and v['attention_prefix_enabled']==(v['build_policy']=='candidate')
            if profile['model']=='Qwen3.5-0.8B':assert v['vision_reencoded'] and v['patch_sha256']==v['case']['image_metadata']['patch_sha256']
            expected=([24] if profile['batch']==1 else [22]) if profile['model']=='Qwen3-0.6B' else [26] if profile['model']=='Qwen2.5-Coder-0.5B-Instruct' else [80] if profile['batch']==1 else [32,32,16]
            assert v['prefill_chunk_plan']==expected
        for policy in ('control','candidate'):
            warm=[v for v in values if v['build_policy']==policy and v['request']>0];assert len(warm)==4
            means=profile['means'][policy];assert abs(means['ttft_ms']-statistics.mean(v['ttft_seconds'] for v in warm)*1000)<1e-9
            assert abs(means['decode_tps']-statistics.mean(v['decode_wall_steps_per_second'] for v in warm))<1e-9
        assert abs(profile['ttft_fraction']-profile['means']['candidate']['ttft_ms']/profile['means']['control']['ttft_ms'])<1e-12
    inference_counts[job]=len(records)
    disks=list((directory/'jobs').glob('*-disk.json'));assert disks
    assert all(json.loads(p.read_text())['free_bytes']>=2*2**30 for p in disks)
assert len(inference_counts)==2 and sum(inference_counts.values())==144
print(json.dumps(dict(verified_files=len(manifest),component_profiles=36,changed_input_comparisons=1008,component_timed_submissions=3456,validation_stages=16,complete_native_outputs=inference_counts,production_matches_tested_shader=True)))
