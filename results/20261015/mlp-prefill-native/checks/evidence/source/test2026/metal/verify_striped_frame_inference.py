"""Independently verify actual FP32 GPU audit, pixel equivalence and full outputs."""
import hashlib,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();proof=json.loads((root/'evidence/manifest.json').read_text());outputs=0;observed=[]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,p in proof['files'].items():
    f=root/name;assert f.resolve().is_relative_to(root) and f.stat().st_size==p['bytes'] and digest(f)==p['sha256']
assert proof['native_library_sha256']=='887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f'
for trial,status in proof['jobs'].items():
    assert status['exit_code']==0;out=root/'evidence/logs'/trial/'jobs';summary=json.loads((out/'summary.json').read_text());assert len(summary['profiles'])==2 and len(summary['dtype_audits'])==1
    dtype=summary['dtype_audits'][0];audit=out/Path(dtype['evidence']).name
    assert json.loads((audit/'exit.json').read_text())['exit_code']==0 and json.loads((audit/'monitor.json').read_text())
    worker=dtype['report']['workers'][0];assert 'gpu' in worker['mlx_default_device'].lower()
    assert set(worker['actual_parameter_tensors'])=={'mlx.core.float32'} and worker['runner_kv_dtype']=='mlx.core.float32' and set(worker['activation_dtype_probes'].values())=={'mlx.core.float32'}
    for profile in summary['profiles']:
        j=out/Path(profile['evidence']).name;batch=profile['batch'];data=json.loads((j/'summary.json').read_text());meta=json.loads((j/'metadata.json').read_text());rows=data['records'];assert len(rows)==24
        assert meta['native_library_sha256']==proof['native_library_sha256'] and meta['learned_values_bit_exact']
        for name,sha in meta['source_sha256'].items():assert digest(root/'evidence/source'/name)==sha
        equivalent=json.loads((j/'exact-value-equivalence.json').read_text());assert equivalent['all_values_bit_exact'] and all(v['all_values_bit_exact'] for v in equivalent['tensors'])
        assert digest(j/'exact-value-equivalence.json')==meta['equivalence_proof_sha256'] and meta['source_checkpoint_sha256']==equivalent['source_sha256'] and meta['comparator_checkpoint_sha256']==equivalent['destination_sha256']
        statuses=json.loads((j/'statuses.json').read_text());assert len(statuses)==8 and all(v['exit_code']==0 for v in statuses)
        for row in rows:
            assert all(row[k]==rows[0][k] for k in ('normalized_token_ids','prompt_token_ids','finished','patch_sha256','rgb_sha256','case'))
            assert row['frontend_route']==('parallel' if row['frontend_policy']=='striped' else 'selected') and row['frontend_setup_seconds']>=0
            assert row['patch_sha256']==row['case']['image_metadata']['patch_sha256'] and row['decode_steps_after_first']==max(map(len,row['normalized_token_ids']))-1
            if row['backend']=='garnet':
                assert row['vision_reencoded'] and row['global_half_alpha_enabled'] and not row['body_shape_enabled']
                assert row['prefill_profile'].get('global_half_alpha_dispatches',0)>0 and row['prefill_chunk_plan']==([80] if batch==1 else [32,32,16])
            else:assert not row['prefix_caching'] and not row['speculative_decoding'] and row['configured_model_dtype'] in ('torch.float32','float32')
        for backend in ('garnet','vllm-metal'):
            for policy in ('selected','striped'):
                warm=[r for r in rows if r['backend']==backend and r['frontend_policy']==policy and r['request']>0];assert len(warm)==4
                means=data['policies'][policy]['means'][backend]
                for key,field,scale in (('ttft_ms','ttft_seconds',1000),('decode_tps','decode_wall_steps_per_second',1),('preprocessing_ms','common_preprocessing_seconds',1000),('initialization_s','initialization_seconds',1)):
                    assert abs(means[key]-statistics.mean(r[field] for r in warm)*scale)<1e-9
                for repeat in (0,1):
                    child=j/(backend+'-'+policy+'-'+str(repeat));assert json.loads((child/'exit.json').read_text())['exit_code']==0 and json.loads((child/'monitor.json').read_text())
                    values=[r for r in rows if r['backend']==backend and r['frontend_policy']==policy and r['repeat']==repeat];assert {r['request'] for r in values}=={0,1,2}
                    if backend=='vllm-metal':assert len({v for r in values for v in r['encoder_cache_identities']})==batch*3
        for policy in ('selected','striped'):
            p=data['policies'][policy];m=p['means'];assert abs(p['decode_ratio']-m['garnet']['decode_tps']/m['vllm-metal']['decode_tps'])<1e-12 and abs(p['ttft_fraction']-m['garnet']['ttft_ms']/m['vllm-metal']['ttft_ms'])<1e-12
            assert p['decode_target_met']==(p['decode_ratio']>=2) and p['ttft_target_met']==(p['ttft_fraction']<=.5)
        for backend,p in data['frontend_effects'].items():
            assert abs(p['ttft_fraction']-data['policies']['striped']['means'][backend]['ttft_ms']/data['policies']['selected']['means'][backend]['ttft_ms'])<1e-12
            assert abs(p['decode_ratio']-data['policies']['striped']['means'][backend]['decode_tps']/data['policies']['selected']['means'][backend]['decode_tps'])<1e-12
        observed.append(dict(trial=trial,batch=batch,policies=data['policies'],frontend_effects=data['frontend_effects'],first_after_init={backend:{policy:dict(ttft_ms=statistics.mean(r['ttft_seconds'] for r in rows if r['backend']==backend and r['frontend_policy']==policy and r['request']==0)*1000,frontend_setup_s=statistics.mean(r['frontend_setup_seconds'] for r in rows if r['backend']==backend and r['frontend_policy']==policy and r['request']==0)) for policy in ('selected','striped')} for backend in ('garnet','vllm-metal')}));outputs+=len(rows)
result=dict(verified_files=len(proof['files']),actual_fp32_gpu_audits=len(proof['jobs']),complete_matched_requests=outputs,profiles=observed,scope='VL-only frontend comparison on selected GPU kernels. Same learned values/FP32 arithmetic; compact native weight storage versus F32 comparator storage differ. Fresh pixels and encoder identities, no prefix/speculation. Init/setup/first use separate; caches unflushed.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
