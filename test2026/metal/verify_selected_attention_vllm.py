"""Verify matched values, actual dtypes, fresh frames, identities and metrics."""
import hashlib,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();e=root/'evidence';files=json.loads((e/'files.json').read_text())
for name,item in files.items():
    data=(root/name).read_bytes();assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],name
selection=json.loads((e/'selection.json').read_text());outputs=0;audits=0;profiles=[]
for job in sorted((e/'logs').glob('selected-attention-vllm*')):
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
        proof=json.loads((folder/'exact-value-equivalence.json').read_text());assert proof['all_values_bit_exact'] and all(v['all_values_bit_exact'] for v in proof['tensors'])
        assert metadata['source_checkpoint_sha256']==proof['source_sha256'] and metadata['comparator_checkpoint_sha256']==proof['destination_sha256']
        assert hashlib.sha256((folder/'exact-value-equivalence.json').read_bytes()).hexdigest()==metadata['equivalence_proof_sha256']
        assert hashlib.sha256((folder/'runtime-selection.json').read_bytes()).hexdigest()==metadata['selection_sha256']
        for value in values:
            assert all(value[k]==values[0][k] for k in ('normalized_token_ids','prompt_token_ids','finished','case'))
            assert value['decode_steps_after_first']==max(map(len,value['normalized_token_ids']))-1
            if value['backend']=='garnet':
                assert value['attention_prefix_enabled']==chosen['attention_prefix_enabled']
                expected=([24] if batch==1 else [22]) if model=='Qwen3-0.6B' else [26] if model=='Qwen2.5-Coder-0.5B-Instruct' else [80] if batch==1 else [32,32,16]
                assert value['prefill_chunk_plan']==expected
            else:assert not value['prefix_caching'] and not value['speculative_decoding'] and value['configured_model_dtype'] in ('torch.float32','float32')
            if model=='Qwen3.5-0.8B':
                assert value['patch_sha256']==value['case']['image_metadata']['patch_sha256']
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
print(json.dumps(dict(verified_files=len(files),actual_dtype_audits=audits,complete_matched_outputs=outputs,profiles=len(profiles),all_measured_targets_met=all(p['decode_target_met'] and p['ttft_target_met'] for p in profiles))))
