"""Independent artifact fixtures and adversarial mutations for the CPU raw auditor.

No runtime, planner, admission or session implementation imports. Green flags
are deliberately retained when quality/accounting evidence is corrupted.
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / 'tools/gpt_oss'))
from audit_resident_session import ENGINE_SOURCE_PATHS, audit

NAMES = ['arithmetic', 'code-tracing', 'instruction-following']
IDENTITY = ('schema', 'mode', 'cache_key', 'hardware', 'batch', 'capacity', 'max_tokens',
    'kv_pages', 'config', 'local_kv_heads', 'expert_weight_shards', 'moe_intermediate_shards',
    'marlin_prepacked', 'compact_vocab_greedy', 'marlin_workspace_layout', 'collective_workspace_layout')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Evidence:
    def __init__(self, root, hybrid=False, archived=False):
        self.root, self.archived = root, archived
        self.paths, self.data = {}, {}
        for name in ['session', 'manifest', 'controls', 'profile', 'tokenizer']:
            self.paths[name] = root / (name + '.json')
        model = root / 'model'; model.mkdir()
        self.paths['tokenizer'] = model / 'tokenizer.json'
        self.data['tokenizer'] = {'test': 'independent decoded answer callback'}
        plan = dict(schema=4 if hybrid else 3, mode='gpt-oss-tensor-parallel-tp2', cache_key='test-cache',
            batch=2, capacity=64, max_tokens=4, kv_pages=8, local_kv_heads=2,
            config=dict(num_hidden_layers=4, head_dim=16, num_key_value_heads=4, sliding_window=17, vocab_size=128),
            hardware=[dict(id=i, total_bytes=8 << 30) for i in range(2)],
            expert_weight_shards=False, moe_intermediate_shards=True, marlin_prepacked=True,
            compact_vocab_greedy=True, marlin_workspace_layout={'max_tokens': 4096},
            collective_workspace_layout='independent-test',
            weight_storage_estimate={'prepacked_marlin_constant_bytes': 64 << 20})
        # Independent exact dimensions/bytes, not derived by calling the auditor.
        kv, auxiliary = (49_152, 32) if hybrid else (65_536, 0)
        if hybrid:
            plan['kv_layout'] = dict(schema=1, mode='gpt-oss-alternating-hybrid-kv-v1', dtype='bfloat16',
                global_layers=[1, 3], window_layers=[0, 2], global_shape=[2, 8, 16, 2, 16],
                window_shape=[2, 4, 16, 2, 16], window_table_shape=[2, 4],
                window_pages_per_request=2, logical_pages_per_request=4, max_prefill_tokens=4,
                sliding_window=17, page_size=16, physical_layer_rule='logical_layer//2',
                extra_tensor_arguments=['window_keys', 'window_values', 'window_table'],
                shared_kv_bytes=kv, shared_auxiliary_bytes=auxiliary)
        native = {'libgarnet.so': 'b' * 64, 'libgarnet_gpt_oss.so': 'c' * 64}
        profile = dict(resident_profile_schema=1, source_commit='a' * 40,
            plan=plan, plan_identity={key: copy.deepcopy(plan[key]) for key in IDENTITY},
            native_binaries=native, hardware_csv='index,uuid,driver\n0,GPU-zero,595\n1,GPU-one,595\n',
            kernel_environment={'GARNET_GPT_OSS_MARLIN_PREPACKED': '1', 'GARNET_TRT_SYNC_ALLOCATOR': '0'}, padded_prefill=True,
            engine_statistics=[dict(device=i, phase=phase, total_weights_bytes=64 << 20,
                context_device_memory_upper_bound_bytes=1 << 20) for i in range(2) for phase in ['prefill', 'decode']])
        if hybrid:
            profile['plan_identity']['kv_layout'] = copy.deepcopy(plan['kv_layout'])
        self.data['profile'] = profile
        # Ceil(128MiB*1.05) + two 1MiB contexts + independently counted banks + 2GiB reserve.
        required = 2_290_509_415 + kv + auxiliary
        admission = dict(ranks=[dict(device=i, required_bytes=required,
            budget_bytes=int((8 << 30) * .9), shared_kv_bytes=kv,
            runtime_graph_reserve_bytes=2 << 30, contexts_per_engine=1, weight_margin_fraction=.05)
            for i in range(2)], concurrency='serial complete batches; one context per engine')
        if hybrid:
            for rank in admission['ranks']: rank['shared_auxiliary_bytes'] = auxiliary
        self.data['manifest'] = dict(resident_session_schema=1, batch=2, output=16, context=64,
            prefill_chunk=4, tokenizer=self.record(model), cases=[])
        self.data['controls'] = dict(phase_order=['admission', 'revalidate_saved_vllm', 'garnet'], cases=[])
        self.data['session'] = dict(resident_session_schema=1, session_complete=True, active_case=None,
            phase='complete', case_order=NAMES, cases=[], source_commit='a' * 40,
            native_binaries=native, hardware_csv=profile['hardware_csv'],
            session_manifest=self.ref('manifest'), session_result=self.ref('session'),
            tokenizer=self.record(model), resident_profile=self.ref('profile'), resident_admission=admission,
            cold_startup_seconds_excluding_module_imports=8., prefill_build_seconds=2.,
            decode_build_seconds=2., admission_and_engine_checksum_seconds=1., sampled_peak_gpu_memory_mib=[3000, 3001])
        for index, name in enumerate(NAMES):
            for suffix in ['request', 'expected', 'result', 'control', 'validation']:
                key = name + '.' + suffix
                self.paths[key] = root / (key + '.json')
            self.data[name + '.request'] = dict(input_ids=[index + 1] * 3, device_ids=[0, 1], reserve_mb=1024, memory_fraction=.9)
            self.data[name + '.expected'] = dict(answer=index)
            matrix = [[index + 10] * 16 for _ in range(2)]
            trials = [dict(trial=t, token_ids_by_request=copy.deepcopy(matrix),
                prefill_kv_reused_for_decode_trial=False, prefill_seconds=1., decode_wall_seconds=3.,
                full_request_wall_seconds=4., full_request_output_tokens_per_second=8.,
                decode_aggregate_output_tokens_per_second=10., request_first_token_seconds=[1., 1.],
                request_completion_seconds=[4., 4.], prefill_step_seconds=[.8], decode_step_seconds=[.1] * 15)
                for t in range(3)]
            result = dict(source_commit='a' * 40, native_binaries=native, hardware_csv=profile['hardware_csv'],
                hardware=copy.deepcopy(plan['hardware']), resident_profile=self.ref('profile'),
                resident_admission=copy.deepcopy(admission), optimization_environment=profile['kernel_environment'],
                batch=2, input_token_ids=[index + 1] * 3, input_tokens_per_request=3,
                output_tokens_per_request=16, max_context_tokens_per_request=64, prefill_chunk_tokens=4,
                padded_prefill=True, kv_pages_per_gpu=8, kv_cache_dtype='bfloat16', kv_cache_allocated_bytes_per_gpu=kv,
                padded_tail_tokens=1, logical_kv_bytes_per_gpu_after_prefill=3072,
                logical_kv_bytes_per_gpu_at_completion=18432, gpu_memory_samples_mib=[[2000, 2001], [3000, 3001]],
                sampled_peak_gpu_memory_mib=[3000, 3001], complete_warmup_count=1,
                complete_warmups=[dict(seconds=4., token_ids_by_request=copy.deepcopy(matrix))],
                decode_trials=trials, median_full_request_output_tokens_per_second=8.,
                token_ids_by_request=copy.deepcopy(matrix), prefill_seconds=1., full_request_output_tokens_per_second=8.)
            if hybrid:
                result.update(kv_layout=copy.deepcopy(plan['kv_layout']), shared_kv_auxiliary_bytes_per_gpu=32,
                    logical_retained_kv_bytes_per_gpu_after_prefill=3072,
                    logical_retained_kv_bytes_per_gpu_at_completion=17920)
            self.data[name + '.control'] = copy.deepcopy(result)
            result.update(case_index=index, case_name=name, engine_pair_reused=index > 0,
                session_manifest=self.ref('manifest'), session_result=self.ref('session'), request=self.ref(name + '.request'))
            self.data[name + '.result'] = result
            self.data[name + '.validation'] = dict(all_pass=True, slots=[dict(trial=t, slot=s,
                first_final_json={'answer': index}, **{'pass': True}) for t in range(3) for s in range(2)])
            self.data['manifest']['cases'].append(dict(name=name, request=self.ref(name + '.request'),
                expected=self.ref(name + '.expected'), result=self.ref(name + '.result'), validation=self.ref(name + '.validation')))
            self.data['controls']['cases'].append(dict(name=name, request=self.ref(name + '.request'),
                expected=self.ref(name + '.expected'), batch=2, output=16, context=64, input=3,
                garnet={'result': self.ref(name + '.control')}))
            self.data['session']['cases'].append(dict(name=name, result=self.ref(name + '.result'),
                validation=self.ref(name + '.validation'), answer_checks=6, sampled_peak_gpu_memory_mib=[3000, 3001],
                median_full_request_output_tokens_per_second=8.))

    def record(self, path):
        if self.archived: return '/workspace/CantorAI/' + path.relative_to(self.root).as_posix()
        return str(path)

    def ref(self, key): return self.record(self.paths[key])

    def refresh(self):
        for key in ['profile', 'tokenizer', 'manifest']:
            dump(self.paths[key], self.data[key])
        session = self.data['session']
        session.update(session_manifest_sha256=digest(self.paths['manifest']),
            resident_profile_sha256=digest(self.paths['profile']), tokenizer_sha256=digest(self.paths['tokenizer']))
        for index, name in enumerate(NAMES):
            for suffix in ['request', 'expected', 'validation']:
                dump(self.paths[name + '.' + suffix], self.data[name + '.' + suffix])
            for suffix in ['result', 'control']:
                value = self.data[name + '.' + suffix]
                value['resident_profile_sha256'] = session['resident_profile_sha256']
                if suffix == 'result':
                    value.update(request_sha256=digest(self.paths[name + '.request']),
                        session_manifest_sha256=session['session_manifest_sha256'])
                dump(self.paths[name + '.' + suffix], value)
            if index < len(session['cases']):
                session['cases'][index].update(result_sha256=digest(self.paths[name + '.result']),
                    validation_sha256=digest(self.paths[name + '.validation']), expected_sha256=digest(self.paths[name + '.expected']))
        dump(self.paths['session'], session); dump(self.paths['controls'], self.data['controls'])

    def audit(self, source_repository=None):
        def decode(ids):
            return '<|channel|>final<|message|>' + json.dumps({'answer': ids[0] - 10})
        return audit(self.paths['session'], self.paths['controls'], decode,
            self.root if self.archived else None, source_repository)


class RawAudit(unittest.TestCase):
    def test_explicit_profile_origin_proof_with_complete_archived_session(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); repository = root / 'git'; repository.mkdir()
            def git(*args):
                return subprocess.check_output(['git', '-C', str(repository), *args], text=True).strip()
            git('init', '-q'); git('config', 'user.name', 'CPU archive proof')
            git('config', 'user.email', 'cpu-proof@example.invalid')
            names = []
            for selected in ENGINE_SOURCE_PATHS:
                name = selected + '/fixture.py' if '.' not in Path(selected).name else selected
                path = repository / name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('CPU immutable engine source\n', encoding='utf-8'); names.append(name)
            git('add', '.'); git('commit', '-qm', 'CPU measured engine program')
            measured = git('rev-parse', 'HEAD')
            (repository / 'docs.txt').write_text('CPU reporting-only change\n', encoding='utf-8')
            git('add', '.'); git('commit', '-qm', 'CPU execution reporting')
            executed = git('rev-parse', 'HEAD')
            archive = root / 'archive'; archive.mkdir()
            evidence = Evidence(archive, archived=True)
            evidence.data['profile']['source_commit'] = measured
            evidence.data['session']['source_commit'] = executed
            for name in NAMES:
                for suffix in ('result', 'control'):
                    evidence.data[name + '.' + suffix]['source_commit'] = executed
            for name in names:
                target = archive / 'Garnet' / name; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((repository / name).read_bytes())
            evidence.refresh()
            with self.assertRaisesRegex(ValueError, 'explicit engine-source Git proof'):
                evidence.audit()
            result = evidence.audit(repository)
            proof = result['profile_source_identity']
            self.assertEqual(proof['measured_source_commit'], measured)
            self.assertEqual(proof['execution_source_commit'], executed)
            self.assertTrue(proof['archived_execution_bytes_verified'])
            self.assertEqual(sum(case['answer_checks'] for case in result['cases']), 18)
            target.write_bytes(target.read_bytes() + b'changed archived program')
            with self.assertRaisesRegex(ValueError, 'execution Git object'):
                evidence.audit(repository)

    def test_positive_full_hybrid_and_archive_paths(self):
        for hybrid in [False, True]:
            for archived in [False, True]:
                with self.subTest(hybrid=hybrid, archived=archived), tempfile.TemporaryDirectory() as temp:
                    evidence = Evidence(Path(temp), hybrid, archived); evidence.refresh()
                    result = evidence.audit()
                    self.assertEqual(sum(case['answer_checks'] for case in result['cases']), 18)
                    self.assertEqual(result['shared_cold_preparation_seconds_excluding_imports'], 8.)
                    self.assertIn('unassessed', result['scope'])

    def test_semantic_corruption_with_valid_updated_hashes(self):
        def change(key, field, value):
            return lambda e: e.data[key].__setitem__(field, value)
        def trial(field, value, control=False):
            return lambda e: e.data['arithmetic.' + ('control' if control else 'result')]['decode_trials'][0].__setitem__(field, value)
        mutations = {
            'partial session': change('session', 'session_complete', False),
            'active session': change('session', 'active_case', 'arithmetic'),
            'failed session': change('session', 'failure', {'type': 'RuntimeError'}),
            'wrong order': change('session', 'case_order', list(reversed(NAMES))),
            'missing case': lambda e: e.data['session']['cases'].pop(),
            'duplicate case': lambda e: e.data['session']['cases'][1].update(name='arithmetic'),
            'no V first': change('controls', 'phase_order', ['garnet', 'vllm']),
            'session control': change('controls', 'resident_session', {'result': 'same'}),
            'source': change('arithmetic.control', 'source_commit', 'd' * 40),
            'hardware': change('arithmetic.control', 'hardware_csv', 'different GPU'),
            'native': change('arithmetic.control', 'native_binaries', {'libgarnet.so': 'd' * 64}),
            'kernel': change('arithmetic.control', 'optimization_environment', {}),
            'allocator policy': change('arithmetic.control', 'optimization_environment',
                {'GARNET_GPT_OSS_MARLIN_PREPACKED': '1', 'GARNET_TRT_SYNC_ALLOCATOR': '1'}),
            'profile binding': change('arithmetic.control', 'resident_profile', 'missing-file'),
            'request binding': change('arithmetic.result', 'request', 'missing-file'),
            'source profile': change('profile', 'source_commit', 'd' * 40),
            'plan identity': lambda e: e.data['profile']['plan_identity'].update(capacity=128),
            'batch boolean': change('manifest', 'batch', True),
            'input shortened': change('arithmetic.result', 'input_token_ids', [1]),
            'input bool': change('arithmetic.request', 'input_ids', [True, 1, 1]),
            'input devices': change('code-tracing.request', 'device_ids', [1, 0]),
            'output cap': change('arithmetic.result', 'output_tokens_per_request', 8),
            'context cap': change('arithmetic.result', 'max_context_tokens_per_request', 32),
            'KV dtype': change('arithmetic.result', 'kv_cache_dtype', 'float16'),
            'KV allocation': change('arithmetic.result', 'kv_cache_allocated_bytes_per_gpu', 1),
            'logical history': change('arithmetic.result', 'logical_kv_bytes_per_gpu_at_completion', 1),
            'padding': change('arithmetic.result', 'padded_tail_tokens', 0),
            'no warmup': change('arithmetic.result', 'complete_warmup_count', 0),
            'warmup differs': change('arithmetic.result', 'complete_warmups', [{'seconds': 4., 'token_ids_by_request': [[11] * 16] * 2}]),
            'diagnostic': change('arithmetic.result', 'profiled_diagnostic', True),
            'missing trial': lambda e: e.data['arithmetic.result']['decode_trials'].pop(),
            'duplicate trial': trial('trial', 1),
            'KV reused': trial('prefill_kv_reused_for_decode_trial', True),
            'control KV reused': trial('prefill_kv_reused_for_decode_trial', True, True),
            'raw answer false green': trial('token_ids_by_request', [[11] * 16] * 2),
            'short trajectory': trial('token_ids_by_request', [[10] * 15] * 2),
            'timing': trial('full_request_wall_seconds', 3.),
            'control timing': trial('full_request_output_tokens_per_second', 100., True),
            'TTFT': trial('request_first_token_seconds', [.1, .1]),
            'completion': trial('request_completion_seconds', [1., 1.]),
            'missing steps': trial('decode_step_seconds', [.1]),
            'invalid steps': trial('decode_step_seconds', [1.] * 15),
            'nan timing': trial('decode_wall_seconds', float('nan')),
            'median': change('arithmetic.result', 'median_full_request_output_tokens_per_second', 100.),
            'control median': change('arithmetic.control', 'median_full_request_output_tokens_per_second', 100.),
            'summary throughput': change('arithmetic.result', 'full_request_output_tokens_per_second', 100.),
            'summary matrix': change('arithmetic.result', 'token_ids_by_request', [[11] * 16] * 2),
            'duplicate answer slots': lambda e: e.data['arithmetic.validation']['slots'].__setitem__(1, copy.deepcopy(e.data['arithmetic.validation']['slots'][0])),
            'wrong answer flag': lambda e: e.data['arithmetic.validation']['slots'][0].update(first_final_json={'answer': 9}),
            'case reuse': change('arithmetic.result', 'engine_pair_reused', True),
            'repeated startup': change('arithmetic.result', 'cold_startup_seconds_excluding_module_imports', 8.),
            'cold decomposition': change('session', 'cold_startup_seconds_excluding_module_imports', 1.),
            'underreserve': lambda e: e.data['arithmetic.result']['resident_admission']['ranks'][0].update(runtime_graph_reserve_bytes=1),
            'control underreserve': lambda e: e.data['arithmetic.control']['resident_admission']['ranks'][0].update(runtime_graph_reserve_bytes=1),
            'weight margin': lambda e: e.data['arithmetic.result']['resident_admission']['ranks'][0].update(weight_margin_fraction=0.),
            'oversized budget': lambda e: e.data['arithmetic.result']['resident_admission']['ranks'][0].update(budget_bytes=8 << 30),
            'false peak': change('arithmetic.result', 'sampled_peak_gpu_memory_mib', [1, 1]),
            'peak over budget': change('arithmetic.result', 'gpu_memory_samples_mib', [[9000, 9000]]),
            'session max': change('session', 'sampled_peak_gpu_memory_mib', [1, 1]),
            'negative bounds': lambda e: e.data['profile']['engine_statistics'][0].update(total_weights_bytes=-1),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name), tempfile.TemporaryDirectory() as temp:
                evidence = Evidence(Path(temp)); mutate(evidence); evidence.refresh()
                with self.assertRaises((ValueError, KeyError, FileNotFoundError, IndexError)):
                    evidence.audit()

    def test_false_green_answers_even_matching_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            evidence = Evidence(Path(temp))
            for suffix in ['result', 'control']:
                value = evidence.data['arithmetic.' + suffix]
                for trial in value['decode_trials']: trial['token_ids_by_request'] = [[11] * 16] * 2
                value['complete_warmups'][0]['token_ids_by_request'] = [[11] * 16] * 2
            evidence.refresh()
            with self.assertRaisesRegex(ValueError, 'Raw output answer'):
                evidence.audit()

    def test_hybrid_ring_and_byte_mutations(self):
        for field, value in [('window_pages_per_request', 1), ('window_layers', [1, 3]),
                ('shared_kv_bytes', 1), ('shared_auxiliary_bytes', 0), ('max_prefill_tokens', 1)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                evidence = Evidence(Path(temp), hybrid=True)
                evidence.data['profile']['plan']['kv_layout'][field] = value
                evidence.data['profile']['plan_identity']['kv_layout'][field] = value
                evidence.refresh()
                with self.assertRaises(ValueError): evidence.audit()

    def test_checksum_and_archive_escape(self):
        for kind in ['result', 'tokenizer', 'escape']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                evidence = Evidence(Path(temp), archived=True); evidence.refresh()
                if kind == 'result': evidence.paths['arithmetic.result'].write_text('{}')
                elif kind == 'tokenizer': evidence.paths['tokenizer'].write_text('{"mutated": true}')
                else:
                    evidence.data['session']['resident_profile'] = '/workspace/CantorAI/../outside.json'
                    dump(evidence.paths['session'], evidence.data['session'])
                with self.assertRaises((ValueError, KeyError)): evidence.audit()


if __name__ == '__main__':
    unittest.main()
