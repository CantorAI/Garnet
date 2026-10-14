"""Verify exact fresh CPU inputs; optionally verify matched inference evidence."""
import hashlib,json,statistics,sys
from pathlib import Path
r=Path(sys.argv[1]).resolve();e=r/'evidence';files=json.loads((e/'files.json').read_text());omitted=json.loads((e/'omitted-proof.json').read_text())
for name,item in files.items():
    raw=(r/name).read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],name
cpu_records=0
for folder in sorted((e/'logs').glob('webp-thread-probe*')):
    assert json.loads((folder/'exit.json').read_text())['exit_code']==0
    summary=json.loads((folder/'jobs/summary.json').read_text());metadata=summary['metadata'];records=summary['records'];assert len(records)==384
    assert metadata['webp']=='1.6.0' and metadata['native_version']==[1,6,0]
    for name,expected in metadata['source_sha256'].items():assert hashlib.sha256((folder/'jobs/source-snapshot'/name).read_bytes()).hexdigest()==expected
    assert len(summary['checks'])==2 and all(c['full_rgb_exact'] for c in summary['checks'])
    assert len({v['rgb_sha256'] for v in records})==1
    for v in records:assert v['patch_sha256']==metadata['cases'][str(v['batch'])]['image_metadata']['patch_sha256']
    for result in summary['results']:
        values=[v for v in records if v['batch']==result['batch'] and v['candidate']==result['candidate']]
        assert result['pillow_ms']==statistics.median(v['elapsed_ms'] for v in values if v['policy']=='pillow')
        assert result['candidate_ms']==statistics.median(v['elapsed_ms'] for v in values if v['policy']==result['candidate'])
    cpu_records+=len(records)
failed=e/'logs/webp-thread-exact-20261013'
assert json.loads((failed/'exit.json').read_text())['exit_code']==1
assert '/DLC/PIL/.dylibs/libwebp.7.dylib' in (failed/'output.log').read_text()
exact=e/'logs/webp-thread-exact-retry1-20261013';assert json.loads((exact/'exit.json').read_text())['exit_code']==0
test=json.loads((exact/'jobs/summary.json').read_text());assert test['metadata']['checks']==956 and len(test['records'])==239 and test['metadata']['native_fixtures']==16
assert test['metadata']['library_sha256']==omitted['logs/webp-thread-probe-confirm-20261013/jobs/libwebp_thread.dylib']['sha256']
native_outputs=0
for folder in sorted((e/'logs').glob('webp-thread-inference*')):
    assert json.loads((folder/'exit.json').read_text())['exit_code']==0
    top=json.loads((folder/'jobs/summary.json').read_text());assert len(top['profiles'])==2 and len(top['dtype_audits'])==1
    worker=top['dtype_audits'][0]['report']['workers'][0]
    assert 'gpu' in worker['mlx_default_device'].lower() and set(worker['actual_parameter_tensors'])=={'mlx.core.float32'}
    assert worker['runner_kv_dtype']=='mlx.core.float32' and set(worker['activation_dtype_probes'].values())=={'mlx.core.float32'}
    for profile in top['profiles']:
        job=folder/'jobs'/Path(profile['evidence']).name;data=json.loads((job/'summary.json').read_text());rows=data['records'];assert len(rows)==24
        metadata=json.loads((job/'metadata.json').read_text());assert metadata['frontend_bridge_sha256']==test['metadata']['library_sha256']
        proof=json.loads((job/'exact-value-equivalence.json').read_text())
        assert proof['all_values_bit_exact'] and all(v['all_values_bit_exact'] for v in proof['tensors'])
        assert metadata['source_checkpoint_sha256']==proof['source_sha256'] and metadata['comparator_checkpoint_sha256']==proof['destination_sha256']
        assert hashlib.sha256((job/'exact-value-equivalence.json').read_bytes()).hexdigest()==metadata['equivalence_proof_sha256']
        expected_library='f2c5c448ec4f3c8431be37b48a2634d8fa4cc31de4aa163ec367a319c9263088' if profile['batch']==1 else '73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2'
        assert metadata['native_library_sha256']==expected_library
        for row in rows:
            assert all(row[k]==rows[0][k] for k in ('normalized_token_ids','prompt_token_ids','finished','patch_sha256','rgb_sha256','case'))
            assert row['frontend_setup_seconds']>=0
            assert row['decode_steps_after_first']==max(map(len,row['normalized_token_ids']))-1
            assert row['patch_sha256']==row['case']['image_metadata']['patch_sha256']
            if row['backend']=='garnet':
                assert row['vision_reencoded']
                assert row['prefill_chunk_plan']==([80] if profile['batch']==1 else [32,32,16])
            else:assert not row['prefix_caching'] and not row['speculative_decoding'] and row['configured_model_dtype'] in ('torch.float32','float32')
        for backend in ('garnet','vllm-metal'):
            for policy in ('selected','threaded'):
                warm=[v for v in rows if v['backend']==backend and v['frontend_policy']==policy and v['request']>0];assert len(warm)==4
                means=data['policies'][policy]['means'][backend]
                assert abs(means['ttft_ms']-statistics.mean(v['ttft_seconds'] for v in warm)*1000)<1e-9
                assert abs(means['decode_tps']-statistics.mean(v['decode_wall_steps_per_second'] for v in warm))<1e-9
                for repeat in (0,1):
                    values=[v for v in rows if v['backend']==backend and v['frontend_policy']==policy and v['repeat']==repeat]
                    child=job/(backend+'-'+policy+'-'+str(repeat));assert json.loads((child/'exit.json').read_text())['exit_code']==0
                    assert json.loads((child/'monitor.json').read_text())
                    if backend=='vllm-metal':assert len({identity for v in values for identity in v['encoder_cache_identities']})==profile['batch']*3
        for policy in ('selected','threaded'):
            value=data['policies'][policy];means=value['means']
            assert abs(value['decode_ratio']-means['garnet']['decode_tps']/means['vllm-metal']['decode_tps'])<1e-12
            assert abs(value['ttft_fraction']-means['garnet']['ttft_ms']/means['vllm-metal']['ttft_ms'])<1e-12
            assert value['decode_target_met']==(value['decode_ratio']>=2) and value['ttft_target_met']==(value['ttft_fraction']<=.5)
        native_outputs+=len(rows)
print(json.dumps(dict(files=len(files),cpu_timed_records=cpu_records,compatibility_checks=956,complete_matched_inference_outputs=native_outputs)))
