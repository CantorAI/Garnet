"""Frozen expanded audit; preserves original assertions without wrapper dependencies."""
# Generator SHA256: 36c622272e7e59f49c3bfc2ac23ab518135826ff998fdeb03e4692bc3ae6ad91
import hashlib,json,statistics,sys,re
from pathlib import Path
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
native_trial=sys.argv[3] if len(sys.argv)>3 else 'mlp-prefill-native-20261015'
assert proof['jobs'][native_trial]['exit_code']==0
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=base/name;assert p.resolve().is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
assert proof['library_sha256']['build-coalesced-bf16-body']=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'
saved=sys.argv[:]
try:
    sys.argv=[saved[0],str(base),str(base/'checks-reverification.json')]
    helper=Path(__file__).with_name('verify_mlp_prefill_native_checks.py')
    exec(compile(helper.read_text(),str(helper),'exec'),{'__file__':str(helper)})
finally:sys.argv=saved
checks_audit=json.loads((base/'checks-reverification.json').read_text())
assert checks_audit['candidate_library_sha256']==proof['library_sha256']['build-mlp-prefill-graph-guardfix']
audit=json.loads((root/'logs'/native_trial/'checks-verification.json').read_text())
assert audit['archive_sha256']==proof['checks_archive_sha256']
for key,value in checks_audit.items():assert audit[key]==value,key
identities=json.loads((root/'logs'/native_trial/'source-identities.json').read_text())
for name,sha in identities.items():
    assert digest(root/'source'/name)==sha and digest(root/'logs'/native_trial/'source-snapshot'/name)==sha
native=root/'logs'/native_trial/'native'
profiles=json.loads((native/'summary.json').read_text())['profiles']
assert len(profiles)==6 and {(p['model'],p['batch']) for p in profiles}=={(m,b) for m in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B') for b in (1,2)}
observed=[];requests=0
for profile in profiles:
    folder=native/Path(profile['evidence']).name;model=profile['model'];batch=profile['batch'];label=model+'-b'+str(batch)
    expected=json.loads((folder/(label+'-expected.json')).read_text());case=json.loads((folder/(label+'-case.json')).read_text())
    meta=json.loads((folder/'metadata.json').read_text());assert meta['library_sha256']=={'control':proof['library_sha256']['build-coalesced-bf16-body'],'candidate':proof['library_sha256']['build-mlp-prefill-graph-guardfix']}
    stages=json.loads((folder/'statuses.json').read_text());assert len(stages)==4 and all(v['exit_code']==0 for v in stages)
    records=[]
    for stage in stages:
        directory=folder/stage['name'];assert json.loads((directory/'monitor.json').read_text())
        prefix='matched_frame_result ' if model=='Qwen3.5-0.8B' else 'matched_result '
        rows=[json.loads(line.split(' ',1)[1]) for line in (directory/'output.log').read_text().splitlines() if line.startswith(prefix)]
        assert len(rows)==3;policy='candidate' if '-candidate-' in stage['name'] else 'control';repeat=int(stage['name'].rsplit('-',1)[-1])
        rss=re.findall(r'^\s*(\d+)\s+maximum resident set size\s*$',(directory/'output.log').read_text(),re.M);assert len(rss)==1 and int(rss[0])>0
        original_case=dict(case);case=json.loads((folder/(label+'-'+policy+'-case.json')).read_text())
        assert {k:v for k,v in case.items() if k!='chunk'}=={k:v for k,v in original_case.items() if k!='chunk'}
        for row in rows:
            assert row['case']==case and all(row[k]==expected[k] for k in ('token_ids','prompt_token_ids','finished'))
            assert row['screened_head_enabled'] and row['global_half_alpha_enabled']
            wide=model=='Qwen3.5-0.8B' and batch==2
            assert (row['prefill_profile'].get('wide_batch_prefill_dispatches',0)>0)==wide
            assert row['interleaved_body_enabled']
            assert row['coalesced_block_enabled']
            compressed=row['coalesced_block_counts'];assert len(compressed)==row['decode_steps_after_first']+1 and len(compressed)>1
            assert compressed[0]==0 and all(v>0 for v in compressed[1:]) and len(set(compressed[1:]))==1

            body=row['interleaved_body_counts'];assert len(body)==row['decode_steps_after_first']+1 and len(body)>1 and body[0]==0
            assert all(v>0 for v in body[1:]) and len(set(body[1:]))==1

            if model=='Qwen3.5-0.8B':assert row['prefill_chunk_plan']==[80]
            counts=row['screened_head_counts'];assert len(counts)==row['decode_steps_after_first']+1 and all(v==1 for v in counts)
            assert row['interleaved_screen_enabled']
            assert row['interleaved_screen_counts']==[1]*len(counts)
            assert row['small_screen_seed_enabled']
            assert row['screened_head_seed_rows']==[1024]*len(counts)
            assert not row['vision_global_alpha_enabled']
            assert row['prefill_stream_enabled'] and row['dirty_prefix_reset_enabled']
            if model=='Qwen3.5-0.8B':assert row['vision_reencoded'] and row['patch_sha256']==case['image_metadata']['patch_sha256']
            row['peak_individual_rss_bytes']=int(rss[0])
            assert not row['fused_head_argmax_enabled'] and not row['body_shape_enabled'] and not row['deferred_state_zero_enabled'] and not row['reused_zero_source_enabled']
            if model=='Qwen3.5-0.8B':
                assert row['frontend_policy']=='combined' and row['frontend_route']=='threaded-striped'
                assert row['serial_vision_prefetch_enabled']
                vision_config=json.loads((root/'models'/model/'config.json').read_text())['vision_config']
                assert row['vision_prefetch_dispatches']==vision_config['depth']
                assert not row['parallel_vision_context_enabled'] and row['vision_parallel_dispatches']==0
                assert all(row['rgb_sha256']==v['rgb_sha256'] for v in records)
            configfile=root/'models'/model/'config.json';assert digest(configfile)==meta['assets'][model]['config.json']
            config=json.loads(configfile.read_text());config=config.get('text_config',config)
            submissions=((config['num_hidden_layers']-1)//4)*sum(chunk>1 for chunk in row['prefill_chunk_plan'])
            physical_rows=[batch*chunk for chunk in row['prefill_chunk_plan']]
            eligible=sum((3<=rows<=128) or rows==160 for rows in physical_rows)
            expected_graphs=config['num_hidden_layers']*eligible if policy=='candidate' else 0
            assert row['mlp_prefill_graph_enabled']==(policy=='candidate')
            assert row['mlp_prefill_graph_counts']==[expected_graphs]+[0]*row['decode_steps_after_first']
            assert row['prefill_profile'].get('mlp_prefill_graph_dispatches',0)==expected_graphs
            assert row['decode_partial_submissions']==0
            assert row['prefill_profile'].get('prefill_partial_submissions',0)==submissions+expected_graphs
            assert row['vision_global_alpha_dispatches']==0
            if row['dirty_prefix_reset_enabled']:
                kinds=config.get('layer_types',['full_attention']*config['num_hidden_layers'])
                full=kinds.count('full_attention');linear=kinds.count('linear_attention')
                assert row['dirty_prefix_reset_states']==2*full
                assert row['dirty_prefix_reset_fallback_states']==2*linear
                extent=0 if row['request']==0 else len(row['prompt_token_ids'])+row['decode_steps_after_first']
                dim=config.get('head_dim',config['hidden_size']//config['num_attention_heads'])
                columns=2*full*config['num_key_value_heads']*extent*dim
                if linear:
                    channels=2*config['linear_num_key_heads']*config['linear_key_head_dim']+config['linear_num_value_heads']*config['linear_value_head_dim']
                    columns+=linear*(channels*config['linear_conv_kernel_dim']+config['linear_num_value_heads']*config['linear_key_head_dim']*config['linear_value_head_dim'])
                assert row['dirty_prefix_reset_bytes']==batch*columns*4
            else:assert row['dirty_prefix_reset_bytes']==row['dirty_prefix_reset_states']==row['dirty_prefix_reset_fallback_states']==0
            row.update(build_policy=policy,repeat=repeat);records.append(row);requests+=1
    means={policy:dict(decode_tps=statistics.mean(v['decode_wall_steps_per_second'] for v in records if v['build_policy']==policy and v['request']>0),ttft_ms=statistics.mean(v['ttft_seconds'] for v in records if v['build_policy']==policy and v['request']>0)*1000) for policy in ('control','candidate')}
    for policy in means:
        assert sum(v['build_policy']==policy and v['request']>0 for v in records)==4
        for key,value in means[policy].items():assert abs(profile['means'][policy][key]-value)<1e-9
    peak_rss={p:statistics.mean(v['peak_individual_rss_bytes'] for v in records if v['build_policy']==p and v['request']>0) for p in means}
    assert all(peak_rss[p]==profile['means'][p]['peak_individual_rss_bytes'] for p in peak_rss)
    ratio=means['candidate']['decode_tps']/means['control']['decode_tps'];fraction=means['candidate']['ttft_ms']/means['control']['ttft_ms']
    assert abs(profile['decode_ratio']-ratio)<1e-12 and abs(profile['ttft_fraction']-fraction)<1e-12
    first={policy:dict(ttft_ms=statistics.mean(v['ttft_seconds'] for v in records if v['build_policy']==policy and v['request']==0)*1000,initialization_s=statistics.mean(v['initialization_seconds'] for v in records if v['build_policy']==policy and v['request']==0)) for policy in ('control','candidate')}
    observed.append(dict(model=model,batch=batch,means=means,decode_ratio=ratio,ttft_fraction=fraction,first_request_after_initialization=first,peak_individual_rss_bytes=peak_rss))

assert requests==72
assert 'Apple M4' in proof['hardware']['machdep.cpu.brand_string'] and int(proof['hardware']['hw.memsize'])==16*2**30
for p in profiles:
    directory=native/Path(p['evidence']).name;meta=json.loads((directory/'metadata.json').read_text());model=p['model']
    assert meta['mlp_graph']==dict(candidate=True,control=False,fp32_arithmetic=True,original_bf16_values=True,queue='same-native-queue',synchronous=True,lazy_graph_compile=True,encode_ms_includes_graph_gpu_work=True)
    for name in ('model.safetensors','tokenizer.json'):
        assert proof['omitted']['models/'+model+'/'+name]['sha256']==meta['assets'][model][name]
result=dict(verified_files=len(proof['files']),full_output_requests=requests,profiles=observed,
    source_closure_files=len(identities),actual_graph_routes=True,checks=checks_audit,
    scope='Native complete-model comparison only; lazy graph compilation, first-after-init, warm, initialization and RSS distinct; caches unflushed; no fresh vLLM ratio',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','checks')}))
