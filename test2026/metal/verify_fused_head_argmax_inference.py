"""Independent full-gate, trained-recovery and whole-output ABBA audit."""
import hashlib,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=root/name;assert p.resolve().is_relative_to(root) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
for name,sha in proof['candidate'].items():assert digest(root/'source'/name)==sha
for name,sha in proof['original'].items():assert digest(root/'logs/fused-head-argmax-original-source-20261015'/name)==sha
for trial in ('fused-head-argmax-checks-20261015','fused-head-argmax-native-20261015'):assert proof['jobs'][trial]['exit_code']==0
checks=root/'logs/fused-head-argmax-checks-20261015/jobs';meta=json.loads((checks/'metadata.json').read_text())
assert meta['candidate_sha256']==proof['library_sha256']['build-fused-head-argmax']
assert proof['library_sha256']['build-global-half-alpha']=='887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f'
assert 'METAL_FUSED_HEAD_ARGMAX' in meta['options'] and 'METAL_GLOBAL_HALF_ALPHA' in meta['options']
assert all(name not in meta['options'] for name in ('METAL_PREFILL_STREAM','METAL_DIRTY_PREFIX_RESET','METAL_JOINED_GLOBAL_MLP','METAL_VISION_GLOBAL_ALPHA'))
for name,sha in meta['source_sha256'].items():assert digest(root/'source'/name)==sha
stages=json.loads((checks/'statuses.json').read_text());assert len(stages)==19
names={'configure','build','ctest','qwen35-extended-prefill','serving','mixed-image-pipeline','qwen35-layers','vision-small','vision-pretrained'}
names|={m+suffix for m in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct') for suffix in ('-exact-tails','-layers')}
names|={m+suffix for m in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B') for suffix in ('-private-kv','-head')}
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
assert '100% tests passed' in ctext and 'metal_fused_head_argmax_reference' in ctext and 'metal_mps_matmul_reference' in ctext
trial=root/'logs/fused-head-argmax-native-20261015'
for phase in ('recovery','native'):assert json.loads((trial/(phase+'-exit.json')).read_text())['exit_code']==0
dependencies=json.loads((trial/'dependency.json').read_text());assert dependencies['pid']==68935 and dependencies['completed_original_stages']==19 and dependencies['source_manifest']==proof['candidate']
for name,sha in json.loads((trial/'helper-identities.json').read_text()).items():assert digest(trial/'helper-snapshot'/name)==sha
recovery=trial/'recovery';r_stages=json.loads((recovery/'statuses.json').read_text());assert len(r_stages)==3
recovered=0
for stage in r_stages:
    assert stage['exit_code']==0 and stage['interruption'] is None and json.loads((recovery/(stage['name']+'-monitor.json')).read_text())
    rows=[json.loads(line.split(' ',2)[2]) for line in (recovery/(stage['name']+'.log')).read_text().splitlines() if line.startswith('fused_head_session_reference PASS ')]
    assert len(rows)==2 and {v['batch'] for v in rows}=={1,2}
    assert all(all(v[k] for k in ('actual_fused_head','generic_full_logits_preserved','float_head_fallback','nonfinite_poison_recovery','private_sessions_isolated')) for v in rows);recovered+=2
native=trial/'native';profiles=json.loads((native/'summary.json').read_text())['profiles']
assert len(profiles)==6 and {(p['model'],p['batch']) for p in profiles}=={(m,b) for m in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B') for b in (1,2)}
observed=[];requests=0
for profile in profiles:
    folder=native/Path(profile['evidence']).name;model=profile['model'];batch=profile['batch'];label=model+'-b'+str(batch)
    expected=json.loads((folder/(label+'-expected.json')).read_text());case=json.loads((folder/(label+'-case.json')).read_text())
    meta=json.loads((folder/'metadata.json').read_text());assert meta['library_sha256']=={'control':proof['library_sha256']['build-global-half-alpha'],'candidate':proof['library_sha256']['build-fused-head-argmax']}
    stages=json.loads((folder/'statuses.json').read_text());assert len(stages)==4 and all(v['exit_code']==0 for v in stages)
    records=[]
    for stage in stages:
        directory=folder/stage['name'];assert json.loads((directory/'monitor.json').read_text())
        prefix='matched_frame_result ' if model=='Qwen3.5-0.8B' else 'matched_result '
        rows=[json.loads(line.split(' ',1)[1]) for line in (directory/'output.log').read_text().splitlines() if line.startswith(prefix)]
        assert len(rows)==3;policy='candidate' if '-candidate-' in stage['name'] else 'control';repeat=int(stage['name'].rsplit('-',1)[-1])
        for row in rows:
            assert row['case']==case and all(row[k]==expected[k] for k in ('token_ids','prompt_token_ids','finished'))
            assert row['fused_head_argmax_enabled']==(policy=='candidate') and row['global_half_alpha_enabled']
            counts=row['fused_head_argmax_counts'];assert len(counts)==row['decode_steps_after_first']+1 and all(v==(1 if policy=='candidate' else 0) for v in counts)
            assert not row['vision_global_alpha_enabled'] and not row['prefill_stream_enabled'] and not row['dirty_prefix_reset_enabled']
            if model=='Qwen3.5-0.8B':assert row['vision_reencoded'] and row['patch_sha256']==case['image_metadata']['patch_sha256']
            row.update(build_policy=policy,repeat=repeat);records.append(row);requests+=1
    means={policy:dict(decode_tps=statistics.mean(v['decode_wall_steps_per_second'] for v in records if v['build_policy']==policy and v['request']>0),ttft_ms=statistics.mean(v['ttft_seconds'] for v in records if v['build_policy']==policy and v['request']>0)*1000) for policy in ('control','candidate')}
    for policy in means:
        assert sum(v['build_policy']==policy and v['request']>0 for v in records)==4
        for key,value in means[policy].items():assert abs(profile['means'][policy][key]-value)<1e-9
    ratio=means['candidate']['decode_tps']/means['control']['decode_tps'];fraction=means['candidate']['ttft_ms']/means['control']['ttft_ms']
    assert abs(profile['decode_ratio']-ratio)<1e-12 and abs(profile['ttft_fraction']-fraction)<1e-12
    first={policy:dict(ttft_ms=statistics.mean(v['ttft_seconds'] for v in records if v['build_policy']==policy and v['request']==0)*1000,initialization_s=statistics.mean(v['initialization_seconds'] for v in records if v['build_policy']==policy and v['request']==0)) for policy in ('control','candidate')}
    observed.append(dict(model=model,batch=batch,means=means,decode_ratio=ratio,ttft_fraction=fraction,first_request_after_initialization=first))
assert requests==72 and recovered==6
result=dict(verified_files=len(proof['files']),original_validation_stages=19,trained_recovery_cases=6,full_output_requests=requests,profiles=observed,scope='Native comparison only; init and first request after init separate, caches unflushed; no fresh vLLM ratio',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
