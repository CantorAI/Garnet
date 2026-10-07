"""Actual paired-controller boundaries: all V validation precedes one shared G process."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from unittest.mock import patch

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / 'tools/gpt_oss'))
names = ['arithmetic', 'code-tracing', 'instruction-following']

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary); work = root / 'work'
    model = root / 'models/gpt-oss-120b-hf'; model.mkdir(parents=True)
    (model / 'tokenizer.json').write_text('{}')
    for index, name in enumerate(names):
        saved = work / 'prompt-benchmarks' / name; saved.mkdir(parents=True)
        (saved / 'request.json').write_text(json.dumps({'input_ids': [index + 1] * 3}))
        (saved / 'expected.json').write_text('{}')
    reference = root / 'reference.json'
    reference.write_text(json.dumps(dict(batch=2, output_tokens_per_request=16,
        max_context_tokens_per_request=64, optimization_environment={
            'GARNET_RESIDENT_SESSION': '1', 'GARNET_BATCH_PREFILL_CHUNK': '2'})))
    profile = root / 'profile.json'
    profile.write_text(json.dumps(dict(resident_profile_schema=1, kernel_environment={},
        plan=dict(batch=2, capacity=64, max_tokens=2))))

    def execute(label, mixed=False):
        directory = root / label
        events = []
        def run(command, **kwargs):
            parts = list(map(str, command))
            if any(part.endswith('run_resident_batch_tp2.py') for part in parts):
                assert kwargs['env']['GARNET_BATCH_PLAN_ONLY'] == '1'
                assert 'GARNET_RESIDENT_SESSION' not in kwargs['env']
                events.append('admission')
            elif any(part.endswith('run_vllm_batch_throughput.py') for part in parts):
                name = Path(parts[4]).name.split('.')[0]
                events.append('V:' + name)
                ids = json.loads(Path(parts[3]).read_text())['input_ids']
                Path(parts[4]).write_text(json.dumps(dict(input_token_ids=ids, batch=2,
                    output_tokens_per_request=16, max_context_tokens_per_request=64,
                    vllm_version='0.31.0', full_request_output_tokens_per_second=100,
                    sampled_peak_gpu_memory_mib=[1, 1], decode_trials=[dict(
                        decode_aggregate_output_tokens_per_second=110,
                        full_request_output_tokens_per_second=100,
                        request_first_token_seconds=[1, 1])] * 3,
                    kv_profile_after_trials=[dict(layers=[{'dtype': 'torch.bfloat16'}],
                        allocated_unique_backing_bytes=1, logical_full_history_bytes=1,
                        logical_retained_history_bytes=1)] * 2)))
            elif any(part.endswith('validate_batch_results.py') for part in parts):
                name = Path(parts[3]).name.split('.')[0]
                events.append('validateV:' + name)
                Path(parts[5]).write_text(json.dumps(dict(all_pass=True,
                    slots=[dict(trial=t, slot=s, **{'pass': True}) for t in range(3) for s in range(2)])))
            elif parts[0] == 'bash':
                assert events == ['admission'] + [event for name in names for event in ('V:' + name, 'validateV:' + name)]
                assert kwargs['env']['GARNET_RESIDENT_SESSION'] == '1'
                assert kwargs['env']['GARNET_RESIDENT_WARMUPS'] == '1'
                events.append('G:session')
                specification = json.loads(Path(parts[2]).read_text())
                completed = []
                for index, case in enumerate(specification['cases']):
                    ids = json.loads(Path(case['request']).read_text())['input_ids']
                    result, validation = Path(case['result']), Path(case['validation'])
                    result.write_text(json.dumps(dict(input_token_ids=ids, batch=2,
                        output_tokens_per_request=16, max_context_tokens_per_request=64,
                        full_request_output_tokens_per_second=100, sampled_peak_gpu_memory_mib=[1, 1],
                        kv_cache_dtype='bfloat16', prefill_seconds=1,
                        kv_cache_allocated_bytes_per_gpu=1, logical_kv_bytes_per_gpu_at_completion=1,
                        complete_warmup_count=1, engine_pair_reused=index > 0, case_preparation_seconds=.1,
                        decode_trials=[dict(decode_aggregate_output_tokens_per_second=110,
                            full_request_output_tokens_per_second=100, full_request_wall_seconds=2,
                            request_first_token_seconds=[1, 1], prefill_kv_reused_for_decode_trial=False)] * 3)))
                    validation.write_text(json.dumps(dict(all_pass=True,
                        slots=[dict(trial=t, slot=s, **{'pass': True}) for t in range(3) for s in range(2)])))
                    completed.append(dict(name=case['name'], result=str(result), validation=str(validation),
                        result_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),
                        validation_sha256=hashlib.sha256(validation.read_bytes()).hexdigest()))
                Path(parts[3]).write_text(json.dumps(dict(session_complete=True,
                    session_manifest_sha256=hashlib.sha256(Path(parts[2]).read_bytes()).hexdigest(),
                    cold_startup_seconds_excluding_module_imports=210, prefill_build_seconds=75,
                    decode_build_seconds=75, cases=completed)))
            else:
                raise AssertionError(parts)
        if mixed:
            request = work / 'prompt-benchmarks/code-tracing/request.json'
            request.write_text(json.dumps({'input_ids': [2] * 4}))
        with patch.dict(os.environ, {'CANTORAI_ROOT': str(root), 'GARNET_BENCH_WORK_DIR': str(work),
                'GARNET_RESIDENT_SESSION': '1'}, clear=True), \
             patch.object(sys, 'argv', ['paired_batch_suite.py', str(reference), str(directory)] + names +
                ['--resident-profile', str(profile), '--resident-session', '--context', '64']), \
             patch('subprocess.check_output', return_value=''), patch('subprocess.run', side_effect=run), \
             patch('importlib.metadata.version', return_value='0.31.0'):
            try: runpy.run_path(str(repo / 'tools/gpt_oss/paired_batch_suite.py'), run_name='__main__')
            except ValueError as error:
                assert mixed and 'must match' in str(error)
            else: assert not mixed
        if mixed:
            assert not events, 'Mixed input lengths started an admission/reference process'
            return
        assert events[-1] == 'G:session' and events.count('admission') == 1
        manifest = json.loads((directory / 'manifest.json').read_text())
        assert manifest['resident_session']['cold_startup_seconds_excluding_module_imports'] == 210
        assert manifest['phase_order'] == ['admission', 'vllm', 'garnet']
        for index, case in enumerate(manifest['cases']):
            assert case['garnet']['engine_pair_reused'] == (index > 0)
            assert 'prefill_build_load_seconds' not in case['garnet']
            assert 'cold_startup_seconds' not in case['garnet']
            assert case['garnet']['median_full_request_tok_s'] == 100
            if index:
                assert case['admission_shared_with'] == names[0]
    execute('complete')
    execute('mixed', mixed=True)

print('Paired resident session: one admission, all optimized V validated first, one G process, shared startup only once; mixed shapes reject before subprocess PASS; CPU only')
