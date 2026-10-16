"""Verify matched values, actual dtypes, fresh frames, identities and metrics."""
import hashlib,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();e=root/'evidence';files=json.loads((e/'files.json').read_text())
for name,item in files.items():
    data=(root/name).read_bytes();assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],name
selection=json.loads((e/'selection.json').read_text());outputs=0;audits=0;profiles=[]
for name,sha in selection['source_manifest'].items():assert hashlib.sha256((e/'source'/name).read_bytes()).hexdigest()==sha
for job in sorted((e/'logs').glob('vector-vision-vllm-*')):
    assert json.loads((job/'exit.json').read_text())['exit_code']==0
    summary=json.loads((job/'jobs/summary.json').read_text());assert len(summary['profiles'])==6 and len(summary['dtype_audits'])==3
    for audit in summary['dtype_audits']:
        worker=audit['report']['workers'][0]
        assert 'gpu' in worker['mlx_default_device'].lower()
        assert set(worker['actual_parameter_tensors'])=={'mlx.core.float32'}
        assert worker['runner_kv_dtype']=='mlx.core.float32' and set(worker['activation_dtype_probes'].values())=={'mlx.core.float32'}
        audits+=1
    for profile in summary['profiles']:
        model=profile['model'];batch=profile['batch'];folder=job/'jobs'/Path(profile['evidence']).name
        result=json.loads((folder/'summary.json').read_text());values=result['records'];assert len(values)==12
        metadata=json.loads((folder/'metadata.json').read_text());policy=selection['models'][model][str(batch)];chosen=selection['builds'][policy]
        assert metadata['native_library_sha256']==chosen['sha256'] and metadata['native_build_policy']==policy
        assert metadata['runtime_sha256']==selection['runtime_sha256'] and metadata['guard_source_sha256']==selection['guard_source_sha256']
        proof=json.loads((folder/'exact-value-equivalence.json').read_text());assert proof['all_values_bit_exact'] and all(v['all_values_bit_exact'] for v in proof['tensors'])
        assert metadata['source_checkpoint_sha256']==proof['source_sha256'] and metadata['comparator_checkpoint_sha256']==proof['destination_sha256']
        assert hashlib.sha256((folder/'exact-value-equivalence.json').read_bytes()).hexdigest()==metadata['equivalence_proof_sha256']
        assert hashlib.sha256((folder/'runtime-selection.json').read_bytes()).hexdigest()==metadata['selection_sha256']
        assert metadata['threaded_decoder_sha256']=='3c86ac196227b3729477cd24704ea4cea4e796fe5f9933c7a5a20361a32804a3'
        for name,sha in metadata['source_sha256'].items():assert hashlib.sha256((e/'source/test2026/metal'/name).read_bytes()).hexdigest()==sha
        for backend in ('garnet','vllm-metal'):
            for repeat in (0,1):
                worker=folder/(backend+'-'+str(repeat));lines=(worker/'output.log').read_text().splitlines()
                prefix='matched_frame_result ' if model=='Qwen3.5-0.8B' else 'matched_result '
                raw=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith(prefix)];assert len(raw)==3
                processes=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('matched_process ')];assert processes and processes[-1]['exit_code']==0
                stored=[v for v in values if v['backend']==backend and v['repeat']==repeat];assert len(stored)==3
                for observed,record in zip(raw,stored):
                    assert all(record[k]==v for k,v in observed.items()) and record['process']==processes[-1]
                monitor=json.loads((worker/'monitor.json').read_text());assert monitor and all(v['guard_passed'] for v in monitor)
        for value in values:
            assert all(value[k]==values[0][k] for k in ('normalized_token_ids','prompt_token_ids','finished','case'))
            assert value['decode_steps_after_first']==max(map(len,value['normalized_token_ids']))-1
            if value['backend']=='garnet':
                assert value['attention_prefix_enabled']==chosen['attention_prefix_enabled']
                assert value['screened_head_enabled'] and value['screened_head_counts']==[1]*(value['decode_steps_after_first']+1)
                assert value['small_screen_seed_enabled'] and value['interleaved_screen_enabled']
                assert value['interleaved_body_enabled'] and value['coalesced_block_enabled']
                for name in ('interleaved_body_counts','coalesced_block_counts'):
                    counts=value[name]
                    assert len(counts)==value['decode_steps_after_first']+1 and len(counts)>1
                    assert counts[0]==0 and all(v>0 for v in counts[1:]) and len(set(counts[1:]))==1

                assert value['screened_head_seed_rows']==[1024]*(value['decode_steps_after_first']+1)
                assert value['interleaved_screen_counts']==[1]*(value['decode_steps_after_first']+1)
                assert (value['prefill_profile'].get('wide_batch_prefill_dispatches',0)>0)==(model=='Qwen3.5-0.8B' and batch==2)
                if model=='Qwen3.5-0.8B':assert value['serial_vision_prefetch_enabled'] and value['vision_prefetch_dispatches']==12 and value['vector_vision_context_enabled'] and value['vision_vector_dispatches']==12

                assert not value['fused_head_argmax_enabled'] and not value['body_shape_enabled']
                assert value['prefill_stream_enabled'] and value['dirty_prefix_reset_enabled']
                assert not value['deferred_state_zero_enabled'] and not value['reused_zero_source_enabled']

                assert value['global_half_alpha_enabled'] and value['prefill_profile'].get('global_half_alpha_dispatches',0)>0
                assert not value['packed_prefill_norm_enabled'] and not value['residual_norm_enabled'] and not value['constant_head_enabled'] and not value['direct_state_zero_enabled']
                expected=([24] if batch==1 else [22]) if model=='Qwen3-0.6B' else [26] if model=='Qwen2.5-Coder-0.5B-Instruct' else [80]
                assert value['prefill_chunk_plan']==expected
            else:assert not value['prefix_caching'] and not value['speculative_decoding'] and value['configured_model_dtype'] in ('torch.float32','float32')
            if model=='Qwen3.5-0.8B':
                assert value['patch_sha256']==value['case']['image_metadata']['patch_sha256']
                assert value['frontend_policy']=='combined' and value['frontend_route']=='threaded-striped'
                assert all(value['rgb_sha256']==v['rgb_sha256'] for v in values)

                if value['backend']=='garnet':assert value['vision_reencoded']
        for backend in ('garnet','vllm-metal'):
            warm=[v for v in values if v['backend']==backend and v['request']>0];assert len(warm)==4
            means=result['means'][backend]
            assert abs(means['ttft_ms']-statistics.mean(v['ttft_seconds'] for v in warm)*1000)<1e-9
            assert abs(means['decode_tps']-statistics.mean(v['decode_wall_steps_per_second'] for v in warm))<1e-9
            for repeat in (0,1):
                worker=folder/(backend+'-'+str(repeat));assert json.loads((worker/'exit.json').read_text())['exit_code']==0
                assert json.loads((worker/'monitor.json').read_text())
                if model=='Qwen3.5-0.8B' and backend=='vllm-metal':assert len({identity for v in values if v['backend']==backend and v['repeat']==repeat for identity in v['encoder_cache_identities']})==batch*3
        assert abs(result['decode_ratio']-result['means']['garnet']['decode_tps']/result['means']['vllm-metal']['decode_tps'])<1e-12
        assert abs(result['ttft_fraction']-result['means']['garnet']['ttft_ms']/result['means']['vllm-metal']['ttft_ms'])<1e-12
        assert result['decode_target_met']==(result['decode_ratio']>=2) and result['ttft_target_met']==(result['ttft_fraction']<=.5)
        assert all(json.loads(p.read_text())['free_bytes']>=2*2**30 for p in folder.glob('*-disk.json'))
        outputs+=12;profiles.append({k:v for k,v in result.items() if k!='records'})
assert outputs==72 and audits==3 and len(profiles)==6
verification=dict(actual_vector_vision_requests=12,scope='One complete matched FP32 cohort; exact outputs/dtypes/route/source/clock audits, no two-cohort completion claim',full_goal_met=False,verified_files=len(files),actual_dtype_audits=audits,complete_matched_outputs=outputs,profiles=len(profiles),all_measured_targets_met=all(p['decode_target_met'] and p['ttft_target_met'] for p in profiles))
if len(sys.argv)>2:Path(sys.argv[2]).write_bytes((json.dumps(verification,indent=2)+'\n').encode())
print(json.dumps(verification))
