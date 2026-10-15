"""Independent original-bound / complete-output audit for internal profile changes."""
import hashlib,json,math,re,statistics,sys
from pathlib import Path,PurePosixPath
root=Path(sys.argv[1]).resolve();proof=json.loads((root/'evidence/manifest.json').read_text());observed=[]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=root/name;assert p.resolve().is_relative_to(root) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
assert 'replanned-vl-prefill-20261015' in proof['jobs'] and set(proof['jobs'])<={'replanned-vl-prefill-20261015','replanned-vl-prefill-confirm-20261015'} and all(v['exit_code']==0 for v in proof['jobs'].values())
base='logs/replanned-vl-prefill-20261015/jobs';jobs=root/'evidence'/base;meta=json.loads((jobs/'metadata.json').read_text())
assert 'Apple M4' in meta['hardware']['machdep.cpu.brand_string'] and int(meta['hardware']['hw.memsize'])==16*2**30
assert meta['library_sha256']=='837d6f3a55598b3169be7569a00a2ab5935fd60e4643546131790a2e654e4471'
assert proof['omitted']['build-interleaved-screen-index/libgarnet.dylib']['sha256']==meta['library_sha256']
assert proof['omitted']['models/Qwen3.5-0.8B/model.safetensors']['sha256']==meta['checkpoint_sha256']
assert proof['omitted']['external/xlang3']['sha256']==meta['runtime_sha256']
for name,sha in meta['source_sha256'].items():assert digest(jobs/'source-snapshot'/name)==sha
for name,sha in meta['source_manifest'].items():assert digest(jobs/'source-snapshot'/name)==sha
configfile=root/'evidence/models/Qwen3.5-0.8B/config.json';assert digest(configfile)==meta['config_sha256'];config=json.loads(configfile.read_text());text_config=config['text_config']
source=(jobs/'source-snapshot/test2026/metal/run_chunk_graph.py').read_text();assert 'bound=5e-6+5e-5*abs(y)' in source and "assert resident==explicit" in source and "'Resident reset retained state'" in source
stages=json.loads((jobs/'statuses.json').read_text());assert len(stages)==3 and [s['name'] for s in stages]==['trained-chunk-references','chunk16','chunk64'] and all(s['exit_code']==0 for s in stages)
assert stages[0]['interruption'] is None and json.loads((jobs/'trained-chunk-references-monitor.json').read_text())
reference=(jobs/'trained-chunk-references.log').read_text();ratios=[float(v) for v in re.findall(r'worst_ratio ([0-9.eE+-]+)',reference)]
assert len(ratios)==36 and all(math.isfinite(v) and 0<=v<=1 for v in ratios) and reference.count('pretrained chunk PASS ')==12
fixtures=list((jobs/'reference/fixtures').glob('*/fixture.json'));assert len(fixtures)==12
coverage=set()
for path in fixtures:
    f=json.loads(path.read_text());kind=f['kind'];layer=f['layer'];chunk=f['chunk'];coverage.add((kind,layer,chunk))
    assert f['config']==text_config and f['batch']==2 and chunk in (16,32,64) and f['capacity']==256 and len(f['steps'])==3
    assert len(f['invalid_inputs'])==(1 if kind=='linear_attention' else 3)
    for step in f['steps']:
        assert len(step['inputs'])==len(f['shapes'])==6
        for name,shape in zip(step['inputs'],f['shapes']):
            rel=str(PurePosixPath(name).relative_to('/Users/zsdx/garnet-metal-investigation'));assert proof['omitted'][rel]['bytes']==math.prod(shape)*4
        rel=str(PurePosixPath(step['expected']).relative_to('/Users/zsdx/garnet-metal-investigation'));assert proof['omitted'][rel]['bytes']==step['count']*4
layers=text_config['num_hidden_layers'];assert coverage=={(kind,layer,chunk) for kind,indices in [('linear_attention',(0,layers-2)),('full_attention',(3,layers-1))] for layer in indices for chunk in (16,32,64)}
outer=json.loads((jobs/'summary.json').read_text())['profiles'];assert len(outer)==2 and {p['maximum'] for p in outer}=={16,64};requests=0
episodes=[(jobs,maximum,'initial') for maximum in (16,64)];common_assets=None
if 'replanned-vl-prefill-confirm-20261015' in proof['jobs']:
    repeat=root/'evidence/logs/replanned-vl-prefill-confirm-20261015/jobs';dep=json.loads((repeat/'dependency.json').read_text());assert dep==dict(metadata=meta,reused_independent_profiles=12,reused_carried_steps=36)
    episodes.append((repeat,64,'repeat'))
for trial,maximum,trial_name in episodes:
    directory=trial/('chunk'+str(maximum));m=json.loads((directory/'metadata.json').read_text());assert m['library_sha256']=={'control':meta['library_sha256'],'candidate':meta['library_sha256']}
    assets=m['assets']['Qwen3.5-0.8B'];assert assets['model.safetensors']==meta['checkpoint_sha256'] and assets['config.json']==meta['config_sha256']
    if common_assets is None:common_assets=assets
    else:assert assets==common_assets
    assert m['internal_prefill_replan']==dict(control_maximum=32,candidate_maximum=maximum,same_library=True,logical_inputs_unchanged=True,physical_total_slots=80,no_fresh_vllm_ratio=True)
    profile=json.loads((directory/'summary.json').read_text())['profiles'];assert len(profile)==1;profile=profile[0]
    trial_outer=json.loads((trial/'summary.json').read_text())['profiles'];assert next(p for p in trial_outer if p['maximum']==maximum)==dict(maximum=maximum,**profile)
    label='Qwen3.5-0.8B-b2';expected=json.loads((directory/(label+'-expected.json')).read_text());original=json.loads((directory/(label+'-case.json')).read_text())
    assert original['batch']==2 and original['chunk']==32 and original['max_tokens']==32 and original['capacity']==256
    steps=json.loads((directory/'statuses.json').read_text());assert len(steps)==4 and all(v['exit_code']==0 for v in steps);records=[]
    assert ['candidate' if '-candidate-' in s['name'] else 'control' for s in steps]==['control','candidate','candidate','control']
    for step in steps:
        d=directory/step['name'];assert json.loads((d/'monitor.json').read_text());policy='candidate' if '-candidate-' in step['name'] else 'control'
        case=json.loads((directory/(label+'-'+policy+'-case.json')).read_text());assert case['chunk']==(maximum if policy=='candidate' else 32)
        assert {k:v for k,v in case.items() if k!='chunk'}=={k:v for k,v in original.items() if k!='chunk'}
        raw=(d/'output.log').read_text();rows=[json.loads(line.split(' ',1)[1]) for line in raw.splitlines() if line.startswith('matched_frame_result ')]
        assert len(rows)==3 and {v['request'] for v in rows}=={0,1,2}
        rss=re.findall(r'^\s*(\d+)\s+maximum resident set size\s*$',raw,re.M);assert len(rss)==1 and int(rss[0])>0
        for row in rows:
            assert row['case']==case and all(row[k]==expected[k] for k in ('token_ids','prompt_token_ids','finished'))
            plan=([16]*5 if maximum==16 else [64,16]) if policy=='candidate' else [32,32,16];assert row['prefill_chunk_plan']==plan and sum(plan)==80
            assert row['vision_reencoded'] and row['patch_sha256']==case['image_metadata']['patch_sha256'] and row['frontend_policy']=='combined' and row['frontend_route']=='threaded-striped'
            assert all(row['rgb_sha256']==v['rgb_sha256'] for v in records)
            assert row['serial_vision_prefetch_enabled'] and row['vision_prefetch_dispatches']==config['vision_config']['depth']
            assert row['small_screen_seed_enabled'] and row['interleaved_screen_enabled'] and row['screened_head_enabled'] and row['global_half_alpha_enabled']
            count=row['decode_steps_after_first']+1;assert row['screened_head_counts']==row['interleaved_screen_counts']==[1]*count and row['screened_head_seed_rows']==[1024]*count
            assert row['prefill_stream_enabled'] and row['dirty_prefix_reset_enabled'] and row['decode_partial_submissions']==0
            assert row['prefill_profile']['prefill_partial_submissions']==((layers-1)//4)*len(plan)
            row.update(policy=policy,peak_individual_rss_bytes=int(rss[0]));records.append(row);requests+=1
    means={policy:dict(decode_tps=statistics.mean(v['decode_wall_steps_per_second'] for v in records if v['policy']==policy and v['request']>0),ttft_ms=1000*statistics.mean(v['ttft_seconds'] for v in records if v['policy']==policy and v['request']>0)) for policy in ('control','candidate')}
    for policy in means:
        assert sum(v['policy']==policy and v['request']>0 for v in records)==4
        for key,value in means[policy].items():assert abs(profile['means'][policy][key]-value)<1e-9
    ratio=means['candidate']['decode_tps']/means['control']['decode_tps'];fraction=means['candidate']['ttft_ms']/means['control']['ttft_ms']
    assert abs(profile['decode_ratio']-ratio)<1e-12 and abs(profile['ttft_fraction']-fraction)<1e-12
    observed.append(dict(trial=trial_name,maximum=maximum,means=means,decode_ratio=ratio,ttft_fraction=fraction))
assert requests==(36 if len(episodes)==3 else 24)
result=dict(verified_files=len(proof['files']),independent_reference_profiles=12,carried_reference_steps=36,worst_original_bound_ratio=max(ratios),full_output_requests=requests,profiles=observed,scope='Same-library native internal VL B2 prefill plans only; original bounds and logical inputs fixed; no fresh vLLM ratio',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
