"""Exercise actual controller subprocess ordering with no GPU/model execution."""
import json,os,runpy,sys,tempfile
from pathlib import Path
from unittest.mock import patch

repo=Path(__file__).resolve().parents[2]
class FirstGarnet(Exception): pass
with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary); work=root/'work'; ids=[1]*256
    for name in ('arithmetic','code-tracing'):
        saved=work/'prompt-benchmarks'/name;saved.mkdir(parents=True)
        (saved/'request.json').write_text(json.dumps({'input_ids':ids}))
        (saved/'expected.json').write_text('{}')
    reference=root/'reference.json'
    reference.write_text(json.dumps(dict(batch=256,output_tokens_per_request=512,
        max_context_tokens_per_request=1024,optimization_environment={})))
    profile=root/'resident.json'
    profile.write_text(json.dumps(dict(resident_profile_schema=1,kernel_environment={},
        plan=dict(batch=256,capacity=1024,max_tokens=256))))
    def execute(label,flags):
        commands=[]; validations=[]
        def fake_run(command,**kwargs):
            parts=list(map(str,command));commands.append(parts)
            if any(x.endswith('run_resident_batch_tp2.py') for x in parts):
                assert kwargs['env']['GARNET_BATCH_PLAN_ONLY']=='1'
            if parts[0]=='bash':
                assert 'benchmark_batch_tp2.sh' in parts[1]
                raise FirstGarnet
            if any(x.endswith('run_vllm_batch_throughput.py') for x in parts):
                output=Path(parts[4]);batch=int(parts[5]);length=int(parts[6])
                output.write_text(json.dumps(dict(input_token_ids=ids,batch=batch,
                    output_tokens_per_request=length,max_context_tokens_per_request=1024,
                    vllm_version='0.31.0',full_request_output_tokens_per_second=100,
                    sampled_peak_gpu_memory_mib=[1,1],decode_trials=[dict(
                        decode_aggregate_output_tokens_per_second=110,
                        full_request_output_tokens_per_second=100,
                        request_first_token_seconds=[1]*batch)]*3,
                    kv_profile_after_trials=[dict(layers=[{'dtype':'torch.bfloat16'}],
                        allocated_unique_backing_bytes=1,logical_full_history_bytes=1,
                        logical_retained_history_bytes=1)]*2)))
            if any(x.endswith('validate_batch_results.py') for x in parts):
                validations.append(parts)
        target=root/label
        with patch.dict(os.environ,{'CANTORAI_ROOT':str(root),'GARNET_BENCH_WORK_DIR':str(work)}), \
             patch.object(sys,'argv',['paired_batch_suite.py',str(reference),str(target),
                 'arithmetic','code-tracing','--resident-profile',str(profile)]+flags), \
             patch('subprocess.check_output',return_value=''), \
             patch('subprocess.run',side_effect=fake_run), \
             patch('importlib.metadata.version',return_value='0.31.0'):
            try: runpy.run_path(str(repo/'tools/gpt_oss/paired_batch_suite.py'),run_name='__main__')
            except FirstGarnet: assert not flags
        manifest=json.loads((target/'manifest.json').read_text())
        inference=[c for c in commands if c[0]=='bash' or any(x.endswith('run_vllm_batch_throughput.py') for x in c)]
        assert len(validations)==2
        assert all(c[1].endswith('run_resident_batch_tp2.py') for c in commands[:2])
        assert all('vllm' in c and 'garnet' not in c for c in manifest['cases'])
        assert len(inference)==(2 if flags else 3)
        assert all(any(x.endswith('run_vllm_batch_throughput.py') for x in c) for c in inference[:2])
        if flags:
            assert manifest['phase_order']==['admission','vllm']
            assert 'Garnet inference has not run' in manifest['suite_scope']
        else:
            assert inference[2][0]=='bash' and 'suite_scope' not in manifest
    execute('references',['--vllm-only'])
    execute('paired',[])
    reject=root/'reject'
    with patch.object(sys,'argv',['paired_batch_suite.py',str(reference),str(reject),
            '--vllm-only','--reuse-vllm-manifest','missing.json']):
        try: runpy.run_path(str(repo/'tools/gpt_oss/paired_batch_suite.py'),run_name='__main__')
        except ValueError as error: assert 'cannot reuse' in str(error)
        else: raise AssertionError('Incompatible reference phase flags accepted')
    assert not reject.exists()
print('Actual controller boundaries: fresh V-only never invokes Garnet; default all V precedes G; reuse conflict rejects PASS')
