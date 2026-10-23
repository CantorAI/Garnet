"""Compare saved candidate/control on the same four-way equal-output groups."""
import argparse,json,statistics
from pathlib import Path
from qwen35_comparator_speculation import saved_comparator_speculation
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--model',required=True);p.add_argument('--batch',required=True,type=int)
p.add_argument('--candidate-suffix',default='-fused-attention');p.add_argument('--control-suffix',default='')
p.add_argument('--vllm-suffix',default='')
a=p.parse_args();root=Path(a.root);prefix=f'matched512-';key=lambda r:(r['trial'],r['workload_group'])
names={'control':f'garnet-{a.model}-b{a.batch}{a.control_suffix}','candidate':f'garnet-{a.model}-b{a.batch}{a.candidate_suffix}',
    'vllm':f'vllm-{a.model}-b{a.batch}{a.vllm_suffix}','lmstudio':f'lmstudio-{a.model}-b{a.batch}-partitioned'}
assert names['control']!=names['candidate'],'Candidate must not be its own control'
data={engine:{key(r):r for r in json.loads((root/f'{prefix}{name}.json').read_text())} for engine,name in names.items()}
assert all(len(rows)==28 for rows in data.values())
for k,base in data['control'].items():
    assert all(rows[k]['input_sha256']==base['input_sha256'] for rows in data.values())
    assert base['prompt_token_ids']==data['candidate'][k]['prompt_token_ids']
def counts(e,r):
    if e in ('control','candidate'):return r['sampled_tokens_including_stop']
    if e=='vllm':return [len(q['token_ids']) for q in r['records']]
    return [q['result']['stats']['total_output_tokens'] for q in r['records']]
equal=[k for k,r in data['control'].items() if not r['warmup'] and all(counts(e,rows[k])==[32]*a.batch for e,rows in data.items())]
median=lambda e:statistics.median(data[e][k]['wall_seconds'] for k in equal)*1000 if equal else None
walls={e:median(e) for e in data}
changed=[{'trial':k[0],'group':k[1],'control_sampled':counts('control',r),'candidate_sampled':counts('candidate',data['candidate'][k])}
    for k,r in data['control'].items() if r['generated_token_ids']!=data['candidate'][k]['generated_token_ids']]
owned={e:sorted({r['owned_memory']['owned_model_bytes'] for r in data[e].values() if not r['warmup']}) for e in ('control','candidate')}
result={'model':a.model,'batch':a.batch,'candidate_suffix':a.candidate_suffix,'control_suffix':a.control_suffix,
    'equal_32_four_way_groups':len(equal),'group_keys':equal,
    'median_group_ms':walls,'candidate_control_latency_fraction':walls['candidate']/walls['control'] if equal else None,
    'candidate_control_speed_ratio':walls['control']/walls['candidate'] if equal else None,
    'same_sequence_cases':28-len(changed),'total_cases':28,'changed_cases':changed,'retained_owned_bytes':owned,
    'goal_met':False,'quality_accepted':False,
    'limitation':'Historical comparator cohorts retained; API/arithmetic differ. Changed outputs require independent analysis; retained ownership is not process peak. No default promotion or goal acceptance.'}
candidate_depth=next(iter(data['candidate'].values()))['profile'].get('speculativeMTP',0)
control_depth=next(iter(data['control'].values()))['profile'].get('speculativeMTP',0)
result['candidate_speculative_mtp_depth']=candidate_depth;result['control_speculative_mtp_depth']=control_depth
candidate_profile=next(iter(data['candidate'].values()))['profile']
result['candidate_speculative_head_candidates']=candidate_profile.get('speculativeHeadCandidates',0)
result['candidate_speculative_prefix_replay']=candidate_profile.get('speculativePrefixReplay',False)
result['candidate_packed_decoder_program']=candidate_profile.get('packedDecoderProgram',False)
result['candidate_resident_controls']=candidate_profile.get('residentControls',False)
result['candidate_serial_graph_reuse']=candidate_profile.get('serialGraphReuse',False)
result['candidate_register_delta_128']=candidate_profile.get('registerDelta128',False)
result['candidate_flat_packed_concat']=candidate_profile.get('flatPackedConcat',False)
result['candidate_serial_target_graph_reuse']=candidate_profile.get('serialTargetGraphReuse',False)
result['candidate_deferred_state_commit']=candidate_profile.get('deferPackedStateCommit',False)
if result['candidate_deferred_state_commit']:
    assert all(r['owned_memory']['speculative']['target_verifier']['deferred_state_commit'] and r['owned_memory']['speculative']['target_verifier']['canonical_state_backup_bytes']==0 for r in data['candidate'].values())
result['candidate_native_prefix_commit']=candidate_profile.get('nativePackedPrefixCommit',False)
if result['candidate_native_prefix_commit']:
    assert all(r['owned_memory']['speculative']['target_verifier']['native_prefix_commit']['ready'] and r['owned_memory']['speculative']['target_verifier']['packed_replay_graph'] is None for r in data['candidate'].values())
result['candidate_shared_output_storage']=candidate_profile.get('sharePackedOutputStorage',False)
result['candidate_pack_prompt']=candidate_profile.get('packPrompt',False)
result['candidate_reuse_verified_head']=candidate_profile.get('reuseVerifiedHead',False)
result['candidate_join_decode_input_projections']=candidate_profile.get('joinDecodeInputProjections',False)
if result['candidate_join_decode_input_projections']:
    counts={r['io_memory']['joined_projection_group_count'] for r in data['candidate'].values()}
    native_counts={sum(g['joined_projection_inputs'] for g in r['owned_memory']['decoder']['graphs']) for r in data['candidate'].values()}
    assert counts==native_counts=={18 if a.model=='2b' else 24},'Joined projection labels differ from actual native graphs'
    result['candidate_actual_joined_projection_groups']=sorted(counts)
build_modes={g['qwen35_whole_program_cuda'] for r in data['candidate'].values() for g in r['owned_memory']['decoder']['graphs'] if 'qwen35_whole_program_cuda' in g}
assert len(build_modes)<=1
result['candidate_whole_program_cuda']=next(iter(build_modes)) if build_modes else None
if any(r.get('native_build_policy',{}).get('whole_program_cuda_requested',False) for r in data['candidate'].values()):
    assert result['candidate_whole_program_cuda'] is True
if result['candidate_shared_output_storage']:
    for r in data['candidate'].values():
        memory=r['owned_memory']['decoder'];assert memory['shared_output_pool_bytes']>0
        assert all(g['external_output_storage'] and g['owned_device_output_bytes']==0 for g in memory['graphs'][:-1])
        assert memory['engine_output_bytes']==memory['shared_output_pool_bytes']+sum(g['owned_device_output_bytes'] for g in memory['graphs'])
result['candidate_zero_aux_streams']=all(r.get('native_build_policy',{}).get('zero_aux_streams_requested',False) for r in data['candidate'].values())
if result['candidate_zero_aux_streams']:
    for r in data['candidate'].values():
        memory=r['owned_memory'];graphs=memory['decoder']['graphs']+[memory['vision']]
        if candidate_depth:
            session=memory['speculative'];graphs=graphs+[session['mtp_graph'],session['target_verifier']['norm_graph'],session['target_verifier']['packed_replay_graph']]
        graphs=[g for g in graphs if g is not None]
        assert all(g['aux_streams']==0 for g in graphs),graphs
if result['candidate_serial_target_graph_reuse']:
    for r in data['candidate'].values():
        assert all(g['contexts']==1 for g in r['owned_memory']['decoder']['graphs'])
        if not r['warmup']:
            graphs=r['owned_memory']['decoder']['graphs']
            if candidate_depth:
                verifier=r['owned_memory']['speculative']['target_verifier']
                graphs=graphs+[verifier['norm_graph'],verifier['packed_replay_graph']]
            graphs=[g for g in graphs if g is not None]
            assert all(g['contexts']==1 and g['serial_graph_reuse_requested'] and g['serial_graph_reuses']>0 for g in graphs),graphs
result['vllm_baseline_suffix']=a.vllm_suffix
result['saved_comparator_speculation']=saved_comparator_speculation(root,a.model,a.batch,'-partitioned',a.vllm_suffix)
result['algorithm_configuration_status']='candidate/control Garnet speculation '+('differs' if candidate_depth!=control_depth else 'has same depth')+'; LM Studio drafting is runtime-audited; vLLM configuration is recorded separately; exact cross-engine proposal settings differ or are unresolved'
result['saved_comparator_speculation_equivalent']=False
if candidate_depth:
    if result['candidate_serial_graph_reuse']:
        assert result['candidate_resident_controls']
        assert all(r['owned_memory']['speculative']['mtp_graph']['contexts']==1 and r['owned_memory']['speculative']['mtp_graph']['serial_graph_reuses']>0 for r in data['candidate'].values())
    assert all(r['speculative']['enabled'] and r['speculative']['target_verification'] and r['speculative']['proposal_source']=='native checkpoint MTP' for r in data['candidate'].values())
    assert all(r['speculative'].get('head_candidates',0)==result['candidate_speculative_head_candidates'] for r in data['candidate'].values())
    if result['candidate_speculative_prefix_replay']:
        expected_commit='packed_delta_replay_kv_selection' if result['candidate_packed_decoder_program'] else 'replay_delta_restore_kv'
        assert all(r['speculative']['partial_commit']==expected_commit and r['speculative_metrics']['target_prefix_recomputes']==0 for r in data['candidate'].values())
out=root/f'matched512-{a.model}-b{a.batch}{a.candidate_suffix}{a.vllm_suffix}-candidate.json';out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
