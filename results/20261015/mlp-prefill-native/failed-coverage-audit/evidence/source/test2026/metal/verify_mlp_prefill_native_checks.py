"""Frozen expanded audit; preserves original assertions without wrapper dependencies."""
# Generator SHA256: 371fa6f79540780641e5462deeb3a5356a56bd4d89aaeb5ab9deea0377c21557
import hashlib,json,sys,re,math
from pathlib import Path
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=base/name;assert p.resolve().is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
trial='mlp-prefill-native-checks-resume-20261015';assert proof['jobs'][trial]['exit_code']==0
checks=root/'logs'/trial/'jobs';meta=json.loads((checks/'metadata.json').read_text())
assert meta['candidate_sha256']==proof['omitted']['build-mlp-prefill-graph-guardfix/libgarnet.dylib']['sha256']
assert proof['omitted']['build-coalesced-bf16-body/libgarnet.dylib']['sha256']=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'
manifest=json.loads((root/'mlp-prefill-native-staging/manifest.json').read_text())
assert len(manifest)==18
for name,sha in manifest.items():assert digest(root/'source'/name)==sha
repair=root/'logs/mlp-prefill-guard-repair-20261015'
report=json.loads((repair/'report.json').read_text())
assert report['original_trial_archive_sha256']=='eada52786b8f2072c83c810d542823ff58ad9519f07c18b6fa526b5f989465fc'
previous=json.loads((repair/'previous-manifest.json').read_text());assert len(previous)==17
changed=report['changes'];assert len(changed)==3
for name,sha in previous.items():assert manifest[name]==changed.get(name,dict(after=sha))['after']
for name,hashes in changed.items():
    assert digest(repair/'original'/name)==hashes['before'] and digest(root/'source'/name)==hashes['after']
name='test2026/metal/compare_vllm_metal.py'
before=(repair/'original'/name).read_text()
old="'2568b1d1cefdf20d5610393c463169a3b4b2a490e2680dc3b0ca73e0ae1a84f7'):"
new="'2568b1d1cefdf20d5610393c463169a3b4b2a490e2680dc3b0ca73e0ae1a84f7','115a1d9dd0ae7181af4876e57bda55b9948d31c1febf1a12d40b150bce0c622e'):"
assert before.count(old)==1 and before.replace(old,new)==(root/'source'/name).read_text()
assert 'METAL_MLP_PREFILL_GRAPH' in meta['options'] and 'METAL_COALESCED_BF16_BODY' in meta['options']
resume=json.loads((checks/'resume-provenance.json').read_text())
assert resume['prior_archive_sha256']=='6d9fda15edcd9bf2e101918ed7615925b8f3d5f98cfc016ec34cd1da806b2147'
assert resume['candidate_sha256']==meta['candidate_sha256'] and resume['source_manifest']==manifest
assert resume['reused_stages']==['configure','build','ctest']
assert 'METAL_INTERLEAVED_SCREEN_INDEX' in meta['options'] and 'METAL_SMALL_SCREEN_SEED' in meta['options'] and 'METAL_SERIAL_VISION_PREFETCH' in meta['options'] and 'METAL_SCREENED_HEAD' in meta['options'] and 'METAL_GLOBAL_HALF_ALPHA' in meta['options']
assert all(name not in meta['options'] for name in ('METAL_JOINED_GLOBAL_MLP','METAL_VISION_GLOBAL_ALPHA'))
for name,sha in meta['source_sha256'].items():assert digest(root/'source'/name)==sha
stages=json.loads((checks/'statuses.json').read_text());assert len(stages)==28
names={'wide-batch-trained-reference','vision-full-grid','screened-real-heads','configure','build','ctest','qwen35-extended-prefill','serving','mixed-image-pipeline','qwen35-layers','vision-small','vision-pretrained'}
names|={m+suffix for m in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct') for suffix in ('-exact-tails','-layers')}
names|={m+suffix for m in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B') for suffix in ('-private-kv','-head')}
names|={m+'-'+label+'-recovery' for m in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B') for label in ('prefix','screened')}
assert all(v in meta['options'] for v in ('METAL_PREFILL_STREAM','METAL_DIRTY_PREFIX_RESET'))
assert {v['name'] for v in stages}==names
for stage in stages:
    assert stage['exit_code']==0 and stage['interruption'] is None
    accepted=stage['accepted_cases'];assert len(accepted)==(8 if stage['name'].endswith('-exact-tails') else 1)
    assert {c['case_index'] for c in accepted}==set(range(len(accepted)))
    for case in accepted:
        record=case['status'];assert record['exit_code']==0 and record['interruption'] is None
        folder=checks/case['evidence'];assert folder.resolve().is_relative_to(checks)
        assert json.loads((folder/(stage['name']+'-monitor.json')).read_text())
        assert (folder/(stage['name']+'.log')).stat().st_size>0
ctest=next(v for v in stages if v['name']=='ctest')['accepted_cases'][0]
ctext=(checks/ctest['evidence']/'ctest.log').read_text()
assert '100% tests passed' in ctext and 'metal_screened_head_reference' in ctext and 'metal_mps_matmul_reference' in ctext
trained=json.loads((checks/'trained-heads/summary.json').read_text())
assert trained['query_checks']==51 and trained['row_checks']==78 and len(trained['profiles'])==3
assert {p['model'] for p in trained['profiles']}=={'Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'}
for p in trained['profiles']:
    asset_path='logs/screened-head-after-vision-20261015/jobs/'+p['model']+'-assets/'
    assert json.loads((root/asset_path/'manifest.json').read_text())==p['manifest']
    assert proof['omitted'][asset_path+'raw.bin']['sha256']==p['manifest']['raw_sha256']
    for q in p['manifest']['queries']:
        for suffix,key in (('.input.bin','input_sha256'),('.reference.bin','reference_sha256')):
            assert proof['omitted'][asset_path+q['name']+suffix]['sha256']==q[key]
    values=p['checks'];assert len(values)==17 and all(v['seed_rows']==1024 and v['interleaved_dispatches']==1 for v in values)
    assert {v['query'] for v in values}=={q['name'] for q in p['manifest']['queries']}
    assert all(v['raw_bits_exact'] and v['retained_bits_exact'] and v['every_rejection_checked'] and v['greedy_exact'] and v['dispatches']==1 for v in values)
import numpy as np
grid=checks/'full-grid';full=json.loads((grid/'summary.json').read_text())
assert len(full['profiles'])==4 and {(p['batch'],p['profile']) for p in full['profiles']}=={(b,s) for b in (1,2) for s in ('small','full')}
for p in full['profiles']:
    b=p['batch'];label=p['profile'];arrays=[]
    assert p['bitwise_equal'] and p['fresh_encodes_per_policy']==2
    assert p['expected_candidate_dispatches']==(12 if label=='full' else 0)
    for policy in ('control','candidate'):
        stem=policy+'-'+label+'-b'+str(b);raw=json.loads((grid/(stem+'.json')).read_text())
        fixture=json.loads((grid/(stem+'-fixture.json')).read_text())
        assert raw['enabled']==(policy=='candidate') and raw['counts']==[p['expected_candidate_dispatches'] if policy=='candidate' else 0]*2
        assert fixture['batch']==b and fixture['expected_prefetch_dispatches']==p['expected_candidate_dispatches']
        a=np.asarray(raw['values'],dtype='<f4');assert a.size==p['values'] and np.all(np.isfinite(a))
        assert hashlib.sha256(a.tobytes()).hexdigest()==p['embedding_sha256'];arrays.append(a)
        from pathlib import PurePosixPath
        patch_name=str(PurePosixPath(fixture['patch_file']).relative_to('/Users/zsdx/garnet-metal-investigation'))
        assert proof['omitted'][patch_name]['sha256']==p['patch_sha256']
    assert np.array_equal(arrays[0].view('<u4'),arrays[1].view('<u4'))
recovered=0;prefix_recovered=0
for stage in stages:
    if not stage['name'].endswith('-recovery'):continue
    assert len(stage['accepted_cases'])==1
    text=(checks/stage['accepted_cases'][0]['evidence']/(stage['name']+'.log')).read_text()
    screened=stage['name'].endswith('-screened-recovery');prefix='screened_head_session_reference PASS ' if screened else 'prefix_reset_recovery PASS '
    rows=[json.loads(line.split(' ',2)[2]) for line in text.splitlines() if line.startswith(prefix)]
    assert len(rows)==2 and {r['batch'] for r in rows}=={1,2}
    keys=('actual_screened_head','generic_full_logits_preserved','float_head_fallback','nonfinite_poison_recovery','private_sessions_isolated','rebuild_and_close') if screened else ('partial_clear_verified','unknown_writer_full_fallback','nonfinite_request_poisoned','full_recovery_verified','private_sessions_isolated')
    assert all(all(r[k] for k in keys) for r in rows)
    if screened:assert all(r['actual_seed_rows']==1024 and r['actual_interleaved'] for r in rows)
    if screened:recovered+=2
    else:prefix_recovered+=2
assert recovered==prefix_recovered==6
assert all(name in ctext for name in ('metal_prefill_stream_reference','metal_dirty_prefix_reset_reference'))
stage=next(v for v in json.loads((checks/'statuses.json').read_text()) if v['name']=='wide-batch-trained-reference')
text=(checks/stage['accepted_cases'][0]['evidence']/'wide-batch-trained-reference.log').read_text()
ratios=[float(v) for v in re.findall(r'worst_ratio ([0-9.eE+-]+)',text)]
import math
assert len(ratios)==12 and all(math.isfinite(v) and 0<=v<=1 for v in ratios) and text.count('pretrained chunk PASS ')==4
fixtures=list((checks/'wide-reference/fixtures').glob('*/fixture.json'));assert len(fixtures)==4;coverage=set()
from pathlib import PurePosixPath
for path in fixtures:
    f=json.loads(path.read_text());coverage.add((f['kind'],f['layer']))
    assert f['batch']==2 and f['chunk']==80 and f['capacity']==256 and len(f['steps'])==3
    assert f['config']==json.loads((root/'models/Qwen3.5-0.8B/config.json').read_text())['text_config']
    for step in f['steps']:
        for name,shape in zip(step['inputs'],f['shapes']):
            rel=str(PurePosixPath(name).relative_to('/Users/zsdx/garnet-metal-investigation'));assert proof['omitted'][rel]['bytes']==math.prod(shape)*4
        rel=str(PurePosixPath(step['expected']).relative_to('/Users/zsdx/garnet-metal-investigation'));assert proof['omitted'][rel]['bytes']==step['count']*4
assert coverage=={('linear_attention',0),('linear_attention',22),('full_attention',3),('full_attention',23)}

assert 'metal_mlp_prefill_graph_reference' in ctext and '0 tests failed out of 8' in ctext
last=(checks/'CTest-LastTest.log').read_text()
assert 'MLP graph pending producer, ownership, abort/recovery and native bound PASS: 64 cases' in last
for name in ('metal_wide_batch_prefill_reference','metal_interleaved_bf16_body_reference','metal_coalesced_bf16_body_reference'):
    assert name in ctext
result=dict(verified_files=len(proof['files']),validation_stages=28,ctests=8,graph_primitive_cases=64,
    reused_validation_stages=['configure','build','ctest'],
    trained_head_queries=51,trained_head_rows=78,vision_grid_profiles=4,
    screened_recovery_cases=recovered,prefix_recovery_cases=prefix_recovered,
    wide_reference_profiles=4,wide_carried_steps=12,worst_wide_original_bound_ratio=max(ratios),
    candidate_library_sha256=meta['candidate_sha256'],source_manifest=manifest,
    scope='Native correctness/ownership checks only; no complete inference comparison, speed or vLLM claim',full_goal_met=False)
if len(sys.argv)>3:
    with Path(sys.argv[3]).open('rb') as stream:result['archive_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in result.items() if k!='source_manifest'}))
