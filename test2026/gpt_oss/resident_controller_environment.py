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
            'GARNET_BATCH_PREFILL_CHUNK': '16',
            'GARNET_GPT_OSS_MARLIN_PREPACKED': '1',
            'GARNET_GPT_OSS_DIRECT_ALLREDUCE': '1',
            'GARNET_GPT_OSS_DIRECT_BATCH_ALLREDUCE': '1',
            'GARNET_GPT_OSS_DIRECT_LARGE_BATCH_ALLREDUCE': '1'})))
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
                'GARNET_RESIDENT_WARMUPS': '42'}), \
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
        '--direct-max-batch','256','--direct-ctas','16'],{
            'GARNET_RESIDENT_PROFILE':None, 'GARNET_RESIDENT_WARMUPS':None,
            'GARNET_RESIDENT_PADDED_PREFILL':'1',
            'GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK':'64',
            'GARNET_GPT_OSS_MARLIN_PREFILL_CTAS_PER_SM':'2',
            'GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_K':'64',
            'GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_CTAS_PER_SM':'1',
            'GARNET_GPT_OSS_DIRECT_MAX_BATCH':'256',
            'GARNET_GPT_OSS_DIRECT_BATCH_CTAS':'16'})
    execute('profile_tp_engine_memory.py',[
        reference,root/'larger-profile','--padded-prefill','--batch','448',
        '--context','768','--prefill-chunk','8','--output','512'],{
            'GARNET_RESIDENT_PROFILE':None,'GARNET_RESIDENT_WARMUPS':None,
            'GARNET_BATCH_PREFILL_CHUNK':'8','GARNET_RESIDENT_PADDED_PREFILL':'1'},
        shape=dict(batch=448,input_tokens_per_request=256,output_tokens_per_request=512,
                   max_context_tokens_per_request=768,prefill_chunk_tokens=8))
    execute('paired_batch_suite.py',[reference,root/'paired','arithmetic',
        '--resident-profile',fresh],{'GARNET_RESIDENT_PROFILE':str(fresh.resolve()),
            'GARNET_RESIDENT_WARMUPS':'1',
            'GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK':'64'})
    execute('paired_batch_suite.py',[reference,root/'nonresident','arithmetic'],{
        'GARNET_RESIDENT_PROFILE':None,'GARNET_RESIDENT_WARMUPS':None})
print('Saved/inherited admission excluded; explicit fresh profile and candidate overrides reach only the intended subprocess PASS')
