"""Host-controller boundary: saved admission must not leak into new GPU jobs.

Stops at the real subprocess boundary. No GPU/model process is executed.
"""
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from unittest.mock import patch

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / 'tools/gpt_oss'))
from resident_budget import normalize_kernel_environment
from audit_resident_session import (kernel_environment as audit_kernel_environment,
    normalize_kernel_environment as normalize_audit_environment)

router_policy = 'GARNET_GPT_OSS_TENSOR_ROUTER_EXPERT_WARPS'
peer_cta_policy = 'GARNET_GPT_OSS_BF16_PEER_GROUP_CTAS'
peer_thread_policy = 'GARNET_GPT_OSS_BF16_PEER_GROUP_THREADS'
grid_signal_policy = 'GARNET_GPT_OSS_BF16_PEER_GRID_SIGNALS'
defaults = {router_policy:'4', peer_cta_policy:'64', peer_thread_policy:'256', grid_signal_policy:'0'}
assert normalize_kernel_environment({}) == defaults
assert normalize_kernel_environment({router_policy:'4'}) == defaults
assert normalize_kernel_environment({}) != normalize_kernel_environment({router_policy:'2'})
assert normalize_kernel_environment({router_policy:'1'}) == {
    router_policy:'1', peer_cta_policy:'64', peer_thread_policy:'256', grid_signal_policy:'0'}
assert normalize_audit_environment({}) == defaults
assert audit_kernel_environment({'optimization_environment':{'GARNET_TRT_SYNC_ALLOCATOR':'0'}}) == {
    'GARNET_TRT_SYNC_ALLOCATOR':'0',**defaults}
assert normalize_audit_environment({}) != normalize_audit_environment({router_policy:'2'})
assert normalize_audit_environment({router_policy:'malformed'}) == {
    router_policy:'malformed', peer_cta_policy:'64', peer_thread_policy:'256', grid_signal_policy:'0'}
assert normalize_audit_environment({grid_signal_policy:'1'}) != defaults

class ObservedCommand(Exception):
    pass

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    work = root / 'work'
    saved = work / 'prompt-benchmarks/arithmetic'
    saved.mkdir(parents=True)
    (saved / 'request.json').write_text(json.dumps({'input_ids': [1]*256}))
    (saved / 'expected.json').write_text('{}')
    reference = root / 'reference.json'
    reference.write_text(json.dumps(dict(batch=256, input_token_ids=[1]*256,
        input_tokens_per_request=256, prefill_chunk_tokens=16, output_tokens_per_request=512,
        max_context_tokens_per_request=1024, optimization_environment={
            'GARNET_RESIDENT_PROFILE': '/old/binary/admission.json',
            'GARNET_RESIDENT_WARMUPS': '99',
            'GARNET_RESIDENT_REUSE_OUTPUT': '1',
            'GARNET_RESIDENT_NATIVE_GREEDY_MERGE': '1',
            'GARNET_RESIDENT_FINAL_PREFILL_SAMPLE_ONLY': '1',
            'GARNET_RESIDENT_PATTERN_UPDATES': '1',
            'GARNET_RESIDENT_NATIVE_GREEDY_STATE': '1',
            'GARNET_BATCH_PREFILL_CHUNK': '16',
            'GARNET_GPT_OSS_MARLIN_PREPACKED': '1',
            'GARNET_GPT_OSS_MARLIN_CTAS_PER_SM': '1',
            'GARNET_GPT_OSS_TENSOR_ROUTER_EXPERT_WARPS': '4',
            'GARNET_GPT_OSS_PREFILL_TILED_64': '1',
            'GARNET_GPT_OSS_PREFILL_GQA_QUERY_TILE': '2',
            'GARNET_GPT_OSS_DIRECT_ALLREDUCE': '1',
            'GARNET_GPT_OSS_DIRECT_BATCH_ALLREDUCE': '1',
            'GARNET_GPT_OSS_DIRECT_LARGE_BATCH_ALLREDUCE': '1',
            'GARNET_GPT_OSS_BF16_PEER_GROUP': '1',
            peer_cta_policy: '64'})))
    fresh = root / 'new-admission.json'
    fresh.write_text(json.dumps(dict(resident_profile_schema=1,
        kernel_environment={'GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK':'64'},
        plan=dict(batch=256, capacity=1024, max_tokens=16))))
    def execute(name, args, expected, shape=None):
        observed=[]
        def stop(command, **kwargs):
            environment=kwargs['env']
            if name=='paired_batch_suite.py':
                runner=('run_resident_batch_tp2.py' if '--resident-profile' in args
                        else 'run_tp_batch_throughput.py')
                assert str(command[1]).endswith(runner), command
                assert environment['GARNET_BATCH_PLAN_ONLY']=='1'
            for key,value in expected.items():
                assert environment.get(key)==value, (name,key,environment.get(key),value)
            assert '/old/binary/admission.json' not in environment.values()
            if shape is not None:
                specification=json.loads(Path(command[-1]).read_text())
                for key,value in shape.items():
                    assert specification[key]==value,(key,specification[key],value)
                assert specification['scope'].startswith('Unvalidated')
                assert json.loads(reference.read_text())['batch']==256
            observed.append(command)
            raise ObservedCommand
        with patch.dict(os.environ, {
                'CANTORAI_ROOT': str(root), 'GARNET_BENCH_WORK_DIR': str(work),
                'GARNET_RESIDENT_PROFILE': '/inherited/stale-admission.json',
                'GARNET_RESIDENT_WARMUPS': '42',
                'GARNET_RESIDENT_REUSE_OUTPUT': '1',
                'GARNET_RESIDENT_NATIVE_GREEDY_MERGE': '1',
                'GARNET_RESIDENT_FINAL_PREFILL_SAMPLE_ONLY': '1',
                'GARNET_RESIDENT_PATTERN_UPDATES': '1',
                'GARNET_RESIDENT_NATIVE_GREEDY_STATE': '1',
                'GARNET_RESIDENT_RUNTIME_SWITCH_US': '1000',
                'GARNET_GPT_OSS_BF16_DECODE_ALLREDUCE': '1',
                'GARNET_GPT_OSS_DECODE_ROUTER_TENSORCORE': '1',
                'GARNET_GPT_OSS_HYBRID_KV': '1',
                'GARNET_GPT_OSS_MARLIN_FAST_HOST_PACK': '1',
                'GARNET_GPT_OSS_MARLIN_CTAS_PER_SM': '2',
                'GARNET_GPT_OSS_TENSOR_ROUTER_EXPERT_WARPS': '2',
                'GARNET_GPT_OSS_MARLIN_BOUNDED_PREFILL': '1'}), \
             patch.object(sys,'argv',[name]+list(map(str,args))), \
             patch('subprocess.check_output',return_value=''), \
             patch('subprocess.run',side_effect=stop), \
             patch('importlib.metadata.version',return_value='0.31.0'):
            try:
                runpy.run_path(str(repo/'tools/gpt_oss'/name),run_name='__main__')
            except ObservedCommand:
                pass
        assert len(observed)==1, (name,observed)
    execute('profile_tp_engine_memory.py',[
        reference,root/'profile','--padded-prefill','--large-prefill-block','64',
        '--prefill-ctas','2','--prefill-down-k','64','--prefill-down-ctas','1',
        '--prefill-up-stages','2',
        '--direct-max-batch','256','--direct-ctas','16'],{
            'GARNET_RESIDENT_PROFILE':None, 'GARNET_RESIDENT_WARMUPS':None,
            'GARNET_RESIDENT_PADDED_PREFILL':'1',
            'GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK':'64',
            'GARNET_GPT_OSS_MARLIN_CTAS_PER_SM':'1',
            'GARNET_GPT_OSS_MARLIN_PREFILL_CTAS_PER_SM':'2',
            'GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_K':'64',
            'GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_CTAS_PER_SM':'1',
            'GARNET_GPT_OSS_MARLIN_PREFILL_UP_STAGES':'2',
            'GARNET_GPT_OSS_DIRECT_MAX_BATCH':'256',
            'GARNET_GPT_OSS_DIRECT_BATCH_CTAS':'16'})
    with patch.object(sys,'argv',['profile_tp_engine_memory.py',str(reference),
            str(root/'invalid-prefill-up-stages-without-large-tile'),
            '--prefill-up-stages','2']), \
            patch('subprocess.run') as gpu_command, patch('subprocess.check_output') as gpu_observation:
        try:
            runpy.run_path(str(repo/'tools/gpt_oss/profile_tp_engine_memory.py'),run_name='__main__')
        except ValueError as error:
            assert str(error)=='Prefill up-stage override requires the large-prefill64 path'
        else:
            raise AssertionError('Up-stage override without large-prefill64 must fail before GPU work')
        gpu_command.assert_not_called();gpu_observation.assert_not_called()
    execute('profile_tp_engine_memory.py',[reference,root/'peer-group-cta128',
        '--bf16-peer-group-ctas','128'],{
            'GARNET_GPT_OSS_BF16_PEER_GROUP':'1',
            peer_cta_policy:'128'})
    execute('profile_tp_engine_memory.py',[reference,root/'peer-group-cta188',
        '--bf16-peer-group-ctas','188'],{
            'GARNET_GPT_OSS_BF16_PEER_GROUP':'1',
            peer_cta_policy:'188'})
    execute('profile_tp_engine_memory.py',[reference,root/'peer-grid-signals',
        '--bf16-peer-grid-signals','1'],{
            'GARNET_GPT_OSS_BF16_PEER_GROUP':'1',
            'GARNET_GPT_OSS_BF16_PEER_GRID_SIGNALS':'1'})
    with patch.object(sys,'argv',['profile_tp_engine_memory.py',str(reference),
            str(root/'invalid-grid-signals-without-group'),
            '--bf16-peer-group','0','--bf16-peer-grid-signals','1']), \
            patch('subprocess.run') as gpu_command, patch('subprocess.check_output') as gpu_observation:
        try:
            runpy.run_path(str(repo/'tools/gpt_oss/profile_tp_engine_memory.py'),run_name='__main__')
        except ValueError as error:
            assert str(error)=='Grid signaling requires the recorded owned peer group'
        else:
            raise AssertionError('Grid signaling without group must fail before GPU work')
        gpu_command.assert_not_called();gpu_observation.assert_not_called()
    with patch.object(sys,'argv',['profile_tp_engine_memory.py',str(reference),
            str(root/'invalid-peer-group-cta-without-group'),
            '--bf16-peer-group','0','--bf16-peer-group-ctas','188']), \
            patch('subprocess.run') as gpu_command, patch('subprocess.check_output') as gpu_observation:
        try:
            runpy.run_path(str(repo/'tools/gpt_oss/profile_tp_engine_memory.py'),run_name='__main__')
        except ValueError as error:
            assert str(error)=='Peer-group CTA policy requires the owned peer group'
        else:
            raise AssertionError('CTA188 without peer-group owner must fail before GPU subprocess')
        gpu_command.assert_not_called();gpu_observation.assert_not_called()
    execute('profile_tp_engine_memory.py',[reference,root/'default-router-warps'],{
        'GARNET_GPT_OSS_TENSOR_ROUTER_EXPERT_WARPS':'4'})
    saved_reference=reference.read_bytes()
    default_reference=json.loads(saved_reference)
    del default_reference['optimization_environment']['GARNET_GPT_OSS_TENSOR_ROUTER_EXPERT_WARPS']
    reference.write_text(json.dumps(default_reference))
    execute('profile_tp_engine_memory.py',[reference,root/'unset-router-warps'],{
        'GARNET_GPT_OSS_TENSOR_ROUTER_EXPERT_WARPS':None})
    reference.write_bytes(saved_reference)
    for value in (2,4):
        execute('profile_tp_engine_memory.py',[
            reference,root/f'router-warps{value}','--tensor-router-warps',str(value)],{
                'GARNET_GPT_OSS_TENSOR_ROUTER_EXPERT_WARPS':str(value)})
    for value in (2,4):
        execute('profile_tp_engine_memory.py',[
            reference,root/f'prefill-gqa-tile{value}','--prefill-gqa-query-tile',str(value)],{
                'GARNET_GPT_OSS_PREFILL_TILED_64':'1',
                'GARNET_GPT_OSS_PREFILL_GQA_QUERY_TILE':str(value)})
    with patch.object(sys,'argv',['profile_tp_engine_memory.py',str(reference),
            str(root/'invalid-gqa-prefill-tile'),'--prefill-gqa-query-tile','4']), \
            patch('subprocess.run') as gpu_command, patch('subprocess.check_output') as gpu_observation:
        unpadded=json.loads(reference.read_text())
        unpadded['optimization_environment']['GARNET_GPT_OSS_PREFILL_TILED_64']='0'
        reference.write_text(json.dumps(unpadded))
        try:
            runpy.run_path(str(repo/'tools/gpt_oss/profile_tp_engine_memory.py'),run_name='__main__')
        except ValueError as error:
            assert str(error)=='GQA query tile override requires recorded tiled head64 prefill'
        else:
            raise AssertionError('GQA query override without tiled prefill must fail before GPU subprocess')
        gpu_command.assert_not_called();gpu_observation.assert_not_called()
        reference.write_bytes(saved_reference)
    with patch.object(sys,'argv',['profile_tp_engine_memory.py',str(reference),
            str(root/'invalid-saved-gqa-prefill-tile')]), \
            patch('subprocess.run') as gpu_command, patch('subprocess.check_output') as gpu_observation:
        malformed=json.loads(reference.read_text())
        malformed['optimization_environment']['GARNET_GPT_OSS_PREFILL_GQA_QUERY_TILE']='3'
        reference.write_text(json.dumps(malformed))
        try:
            runpy.run_path(str(repo/'tools/gpt_oss/profile_tp_engine_memory.py'),run_name='__main__')
        except ValueError as error:
            assert str(error)=='Invalid shared-query GQA prefill tile'
        else:
            raise AssertionError('Malformed saved GQA query tile must fail before GPU subprocess')
        gpu_command.assert_not_called();gpu_observation.assert_not_called()
        reference.write_bytes(saved_reference)
    with patch.object(sys,'argv',['profile_tp_engine_memory.py',str(reference),str(root/'invalid-router-warps'),
            '--tensor-router-warps','1']), patch('subprocess.run') as gpu_command:
        try:
            runpy.run_path(str(repo/'tools/gpt_oss/profile_tp_engine_memory.py'),run_name='__main__')
        except SystemExit as error:
            assert error.code==2
        else:
            raise AssertionError('Unsupported router geometry must fail before subprocess')
        gpu_command.assert_not_called()
    original_reference=reference.read_bytes()
    for malformed in ('1','2x','',' 2','04'):
        bad=json.loads(original_reference);bad['optimization_environment']['GARNET_GPT_OSS_TENSOR_ROUTER_EXPERT_WARPS']=malformed
        reference.write_text(json.dumps(bad))
        with patch.object(sys,'argv',['profile_tp_engine_memory.py',str(reference),str(root/'malformed-router-policy')]), \
                patch('subprocess.run') as gpu_command, patch('subprocess.check_output') as gpu_observation:
            try:
                runpy.run_path(str(repo/'tools/gpt_oss/profile_tp_engine_memory.py'),run_name='__main__')
            except ValueError as error:
                assert str(error)=='Invalid tensor-core router warp policy'
            else:
                raise AssertionError('Malformed recorded router policy must fail before GPU subprocess')
            gpu_command.assert_not_called();gpu_observation.assert_not_called()
    reference.write_bytes(original_reference)
    for value in (1,2):
        execute('profile_tp_engine_memory.py',[
            reference,root/f'decode-cta-profile{value}','--padded-prefill',
            '--decode-ctas',str(value),'--prefill-ctas','1'],{
                'GARNET_GPT_OSS_MARLIN_CTAS_PER_SM':str(value),
                'GARNET_GPT_OSS_MARLIN_PREFILL_CTAS_PER_SM':'1'})
    execute('profile_tp_engine_memory.py',[
        reference,root/'projection-cta-profile','--padded-prefill',
        '--prefill-up-ctas','2','--prefill-down-ctas','1'],{
            'GARNET_GPT_OSS_MARLIN_PREFILL_UP_CTAS_PER_SM':'2',
            'GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_CTAS_PER_SM':'1'})
    for value in (2,4):
        with patch.object(sys,'argv',['profile_tp_engine_memory.py',str(reference),str(root/f'invalid-decode-cta{value}'),
                '--decode-ctas',str(value)]), patch('subprocess.run') as gpu_command:
            try:
                runpy.run_path(str(repo/'tools/gpt_oss/profile_tp_engine_memory.py'),run_name='__main__')
            except ValueError as error:
                assert value==2 and str(error)=='Decode CTA override requires an explicit prefill CTA count'
            except SystemExit as error:
                assert value==4 and error.code==2
            else:
                raise AssertionError('Unsafe decode override must fail before subprocess work')
            gpu_command.assert_not_called()
    for value in (0,1):
        execute('profile_tp_engine_memory.py',[
            reference,root/f'bf16-profile{value}','--padded-prefill',
            '--bf16-decode-allreduce',str(value),'--bounded-prefill',str(value),
            '--decode-router-tensorcore',str(value)],{
                'GARNET_GPT_OSS_BF16_DECODE_ALLREDUCE':str(value),
                'GARNET_GPT_OSS_MARLIN_BOUNDED_PREFILL':str(value),
                'GARNET_GPT_OSS_DECODE_ROUTER_TENSORCORE':str(value),
                'GARNET_RESIDENT_PROFILE':None,'GARNET_RESIDENT_WARMUPS':None})
        execute('profile_tp_engine_memory.py',[
            reference,root/f'hybrid-profile{value}','--padded-prefill','--hybrid-kv',str(value)],{
                'GARNET_GPT_OSS_HYBRID_KV':str(value),
                'GARNET_RESIDENT_PROFILE':None,'GARNET_RESIDENT_WARMUPS':None})
        execute('profile_tp_engine_memory.py',[
            reference,root/f'host-pack-profile{value}','--padded-prefill',
            '--fast-host-pack',str(value)],{
                'GARNET_GPT_OSS_MARLIN_FAST_HOST_PACK':str(value),
                'GARNET_RESIDENT_PROFILE':None,'GARNET_RESIDENT_WARMUPS':None})
    execute('profile_tp_engine_memory.py',[
        reference,root/'larger-profile','--padded-prefill','--batch','448',
        '--context','768','--prefill-chunk','8','--output','512'],{
            'GARNET_RESIDENT_PROFILE':None,'GARNET_RESIDENT_WARMUPS':None,
            'GARNET_BATCH_PREFILL_CHUNK':'8','GARNET_RESIDENT_PADDED_PREFILL':'1',
            'GARNET_GPT_OSS_MARLIN_BOUNDED_PREFILL':None,
            'GARNET_GPT_OSS_DECODE_ROUTER_TENSORCORE':None,
            'GARNET_GPT_OSS_HYBRID_KV':None,
            'GARNET_GPT_OSS_MARLIN_FAST_HOST_PACK':None,
            'GARNET_GPT_OSS_BF16_DECODE_ALLREDUCE':None},
        shape=dict(batch=448,input_tokens_per_request=256,output_tokens_per_request=512,
                   max_context_tokens_per_request=768,prefill_chunk_tokens=8))
    saved_host=json.loads(reference.read_text())
    saved_host['optimization_environment']['GARNET_RESIDENT_RUNTIME_SWITCH_US']='1000'
    reference.write_text(json.dumps(saved_host))
    execute('paired_batch_suite.py',[reference,root/'paired','arithmetic',
        '--resident-profile',fresh],{'GARNET_RESIDENT_PROFILE':str(fresh.resolve()),
            'GARNET_RESIDENT_WARMUPS':'1',
            'GARNET_RESIDENT_REUSE_OUTPUT':'0',
            'GARNET_RESIDENT_NATIVE_GREEDY_MERGE':'0',
            'GARNET_RESIDENT_FINAL_PREFILL_SAMPLE_ONLY':'0',
            'GARNET_RESIDENT_PATTERN_UPDATES':'0',
            'GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK':'64'})
    for flags in range(32):
        values=[str((flags >> index) & 1) for index in range(5)]
        execute('paired_batch_suite.py',[reference,root/f'host-candidates{flags}','arithmetic',
            '--resident-profile',fresh,'--reuse-output',values[0],
            '--native-greedy-merge',values[1],'--final-prefill-sample-only',values[2],
            '--pattern-updates',values[3],'--native-greedy-state',values[4]],
            dict(zip(('GARNET_RESIDENT_REUSE_OUTPUT','GARNET_RESIDENT_NATIVE_GREEDY_MERGE',
                'GARNET_RESIDENT_FINAL_PREFILL_SAMPLE_ONLY','GARNET_RESIDENT_PATTERN_UPDATES',
                'GARNET_RESIDENT_NATIVE_GREEDY_STATE'),values)))
    with patch.object(sys,'argv',['paired_batch_suite.py',str(reference),str(root/'invalid-nonresident'),
            'arithmetic','--reuse-output','1']), patch('subprocess.run') as gpu_command:
        try:
            runpy.run_path(str(repo/'tools/gpt_oss/paired_batch_suite.py'),run_name='__main__')
        except ValueError as error:
            assert str(error)=='Host lifecycle candidates require a resident Garnet comparison'
        else:
            raise AssertionError('Nonresident host candidate must fail before subprocess work')
        gpu_command.assert_not_called()
    execute('paired_batch_suite.py',[reference,root/'nonresident','arithmetic'],{
        'GARNET_RESIDENT_PROFILE':None,'GARNET_RESIDENT_WARMUPS':None})
    for value in (0,100,200,1000):
        execute('paired_batch_suite.py',[reference,root/f'runtime-switch{value}','arithmetic',
            '--resident-profile',fresh,'--runtime-switch-us',str(value)],
            {'GARNET_RESIDENT_RUNTIME_SWITCH_US':str(value)})
    execute('paired_batch_suite.py',[reference,root/'runtime-switch-default','arithmetic',
        '--resident-profile',fresh],{'GARNET_RESIDENT_RUNTIME_SWITCH_US':'0'})
    for extra in ([],['--resident-profile',str(fresh),'--vllm-only']):
        with patch.object(sys,'argv',['paired_batch_suite.py',str(reference),
                str(root/('invalid-runtime-switch'+str(len(extra)))),'arithmetic',
                '--runtime-switch-us','200',*extra]),patch('subprocess.run') as gpu_command:
            try:
                runpy.run_path(str(repo/'tools/gpt_oss/paired_batch_suite.py'),run_name='__main__')
            except ValueError as error:
                assert str(error)=='Runtime scheduling requires a resident Garnet comparison'
            else:
                raise AssertionError('Nonresident/vLLM scheduling enablement must fail before subprocess')
            gpu_command.assert_not_called()
print('Saved/inherited admission excluded; explicit fresh profile and candidate overrides reach only the intended subprocess PASS')
