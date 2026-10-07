"""Execute the production resident runner against poisoned CPU KV and injected failures.

No CUDA, model weights or speed claims. The stateful runtime independently checks
full input rewrites, ragged future KV, per-case warmups, resource reuse and timing.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types
from unittest.mock import patch

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / 'tools/gpt_oss'))
import resident_budget as budget
from resident_session import read_session


class InjectedFailure(RuntimeError):
    pass


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    weights, cache = root / 'weights', root / 'cache'
    weights.mkdir(); cache.mkdir()
    tokenizer = root / 'tokenizer'; tokenizer.mkdir()
    (tokenizer / 'tokenizer.json').write_text('{}')
    devices = [dict(id=rank, total_bytes=16 << 30, free_bytes=15 << 30) for rank in range(2)]
    plan = dict(schema=3, mode='tensor-parallel', cache_key='session-test',
        hardware=['rank0', 'rank1'], batch=2, capacity=64, max_tokens=2,
        kv_pages=8, local_kv_heads=1,
        config=dict(num_hidden_layers=2, head_dim=32, vocab_size=64),
        expert_weight_shards=False, moe_intermediate_shards=True,
        marlin_prepacked=True, compact_vocab_greedy=True,
        marlin_workspace_layout='fixture', collective_workspace_layout='fixture-v11',
        weight_storage_estimate=dict(prepacked_marlin_constant_bytes=96 << 20))
    binaries = {'libgarnet.so': 'core', 'libgarnet_gpt_oss.so': 'plugin'}
    checkpoint = {'immutable_fixture': 'header'}
    statistics = []
    for rank in range(2):
        for phase in ('prefill', 'decode'):
            path = cache / f'{rank}-{phase}.engine'; path.write_bytes(b'engine-fixture')
            statistics.append(dict(device=rank, phase=phase, total_weights_bytes=100 << 20,
                context_device_memory_upper_bound_bytes=1 << 20, engine_path=str(path),
                engine_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    profile = root / 'profile.json'
    profile.write_text(json.dumps(dict(resident_profile_schema=1, padded_prefill=True,
        plan_identity=budget.plan_identity(plan), native_binaries=binaries,
        hardware_csv='fixture-hardware', checkpoint=checkpoint, cache_root=str(cache),
        kernel_environment={}, engine_statistics=statistics)))
    prompt_ids = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def specification(directory):
        directory.mkdir()
        cases = []
        for index, ids in enumerate(prompt_ids):
            request, expected = directory / f'request{index}.json', directory / f'expected{index}.json'
            request.write_text(json.dumps({'input_ids': ids})); expected.write_text('{}')
            cases.append(dict(name=f'case{index}', request=str(request), expected=str(expected),
                result=str(directory / f'case{index}.json'), validation=str(directory / f'validation{index}.json')))
        manifest = directory / 'input.json'
        payload = dict(resident_session_schema=1, batch=2, output=16, context=64,
            prefill_chunk=2, validation_python=str(Path(sys.executable).resolve()),
            tokenizer=str(tokenizer), cases=cases)
        manifest.write_text(json.dumps(payload))
        return manifest, directory / 'session.json', payload

    class Tensor:
        def __init__(self, values, dtype, shape, device):
            self.data, self.dtype, self.shape, self.device = list(values), dtype, list(shape), device

    class State:
        def __init__(self, failure=None):
            self.clock = 0.
            self.device = 0
            self.allocations = []; self.loads = []; self.releases = 0
            self.model_releases = []
            self.starts = []; self.steps = []; self.validation_calls = []
            self.prepared = {}; self.in_request = False
            self.kv = [[-999] * 64 for _ in range(2)]
            self.prefix = None; self.request_index = -1; self.failure = failure

        def now(self):
            self.clock += .001
            return self.clock

        def set_device(self, device):
            old = self.device; self.device = device
            return old

        def tensor(self, values, dtype, shape, device):
            item = Tensor(values, dtype, shape, self.device)
            self.allocations.append(item)
            return item

        def update(self, tensor, values):
            assert tensor.device == self.device and len(values) == len(tensor.data)
            self.clock += .01
            tensor.data[:] = values

        def load(self, *args, **kwargs):
            self.loads.append(args[4])
            self.clock += 2
            state, phase = self, args[4]
            class Model:
                def release(self):
                    assert phase not in state.model_releases
                    state.model_releases.append(phase)
            return Model()

        def check_inputs(self, phase, prepared):
            identity = tuple(id(tensor) for activation, controls in prepared for tensor in [activation] + controls)
            if phase in self.prepared:
                assert identity == self.prepared[phase], 'Rank tensors reallocated between requests/cases'
            self.prepared[phase] = identity
            for rank, (activation, controls) in enumerate(prepared):
                assert activation.device == rank and all(t.device == rank for t in controls)
                assert activation.data == prepared[0][0].data
                assert all(t.data == prepared[0][1][index].data for index, t in enumerate(controls))
            return prepared[0]

        def prefill(self, prepared, **kwargs):
            activation, controls = self.check_inputs('prefill', prepared)
            offset, length = controls[3].data[0], controls[2].data[0]
            positions = controls[0].data
            real = length - offset
            if offset == 0:
                assert not self.in_request
                self.request_index += 1
                if self.failure == 'inference' and self.request_index == 5:
                    raise InjectedFailure('Second-case measured request failed')
                self.in_request = True
                self.prefix = []
                self.starts.append(self.request_index)
            for slot in range(2):
                assert positions[slot * 2:slot * 2 + 2] == [offset, offset + 1]
                self.kv[slot][offset:offset + 2] = activation.data[slot * 2:slot * 2 + 2]
            self.prefix.extend(activation.data[:real])
            self.steps.append(('prefill', self.request_index, offset, real))
            self.clock += .02
            return {'token_ids': [sum(self.prefix) % 31 + 10] * 2}

        def decode(self, prepared, **kwargs):
            activation, controls = self.check_inputs('decode', prepared)
            position = controls[0].data[0]
            assert self.in_request and len(self.prefix) == 3
            assert self.prefix == prompt_ids[self.request_index // 4]
            # A reused KV allocation is poisoned after every complete request.
            # All three true input positions must have been freshly rewritten;
            # the first decode must replace the ragged padded future position.
            for slot in range(2):
                assert self.kv[slot][:3] == self.prefix
                assert controls[3].data[slot] == position
                self.kv[slot][position] = activation.data[slot]
            self.steps.append(('decode', self.request_index, position))
            self.clock += .02
            result = {'token_ids': [(value + 1) % 63 for value in activation.data]}
            if position == 17:
                self.in_request = False
                self.kv = [[-999] * 64 for _ in range(2)]
            return result

        def check_output(self, command, **kwargs):
            if command[0] == 'git':
                assert type(kwargs['cwd']) is str
                return 'fixture-source'
            assert not self.in_request, 'Memory subprocess entered the timed request'
            self.clock += .5
            if self.failure == 'load-sample' and self.loads == [True]:
                raise InjectedFailure('Memory observation failed after prefill load')
            if self.failure == 'memory' and self.request_index == 7:
                return '0, 16000\n1, 16000\n'
            return '0, 101\n1, 102\n'

        def validate(self, command, **kwargs):
            assert not self.in_request and kwargs['check'] is True
            self.clock += .7
            result, output = Path(command[3]), Path(command[5])
            data = json.loads(result.read_text())
            self.validation_calls.append(data['case_name'])
            rows = [dict(trial=trial, slot=slot, **{'pass': True}) for trial in range(3) for slot in range(2)]
            if self.failure == 'answer' and data['case_index'] == 1:
                rows[0]['pass'] = False
            output.write_text(json.dumps(dict(all_pass=all(row['pass'] for row in rows), slots=rows)))
            if not all(row['pass'] for row in rows):
                raise subprocess.CalledProcessError(1, command)
            return types.SimpleNamespace(returncode=0)

    def execute(label, failure=None, single=False):
        manifest, summary, payload = specification(root / label)
        state = State(failure)
        class Pair:
            prefill_stages = [dict(device_id=rank) for rank in range(2)]
            decode_stages = [dict(device_id=rank) for rank in range(2)]
            forward_prefill = lambda self, *a, **k: state.prefill(*a, **k)
            forward_decode = lambda self, *a, **k: state.decode(*a, **k)
            def release(self):
                state.releases += 1
                assert state.releases == 1
                for model in self.models:
                    model.release()
        def build_pair(prefill, decode):
            first = prefill()
            try:
                second = decode(('same-shared-kv',))
            except BaseException:
                first.release()
                raise
            result = Pair(); result.models = [first, second]
            return result
        garnet = types.SimpleNamespace(cuda_devices_json=lambda: json.dumps(devices),
            cuda_set_device=state.set_device, tensor_from_host=state.tensor, tensor_update_from_host=state.update)
        pipeline = types.SimpleNamespace(make_tensor_parallel_plan=lambda *a, **k: copy.deepcopy(plan),
            build_tensor_parallel=state.load)
        owner = types.SimpleNamespace(ResidentTensorParallel=types.SimpleNamespace(build=build_pair))
        env = dict(GARNET_BATCH_CONTEXT_CAPACITY='64', GARNET_BATCH_PREFILL_CHUNK='2',
            GARNET_RESIDENT_PROFILE=str(profile), GARNET_RESIDENT_WARMUPS='1')
        arguments = (['--session', str(weights), str(cache), str(manifest), str(summary), '2', '16'] if not single
            else [str(weights), str(cache), payload['cases'][0]['request'], payload['cases'][0]['result'], '2', '16'])
        original_write = Path.write_text
        def write_result(path, *args, **kwargs):
            if failure == 'reporting' and path == Path(payload['cases'][1]['result']):
                raise InjectedFailure('Result reporting failed after completed trials')
            return original_write(path, *args, **kwargs)
        with patch.dict(sys.modules, {'garnet': garnet, 'pipeline': pipeline, 'garnet_pipeline': owner}), \
             patch.dict(os.environ, env, clear=True), patch.object(sys, 'argv', ['runner'] + arguments), \
             patch.object(budget, 'native_identity', return_value=binaries), \
             patch.object(budget, 'hardware_identity', return_value='fixture-hardware'), \
             patch.object(budget, 'checkpoint_identity', return_value=checkpoint), \
             patch('time.perf_counter', side_effect=state.now), \
             patch('subprocess.check_output', side_effect=state.check_output), \
             patch('subprocess.run', side_effect=state.validate), \
             patch.object(Path, 'write_text', new=write_result):
            try:
                runpy.run_path(str(repo / 'tools/gpt_oss/run_resident_batch_tp2.py'), run_name='__main__')
            except (InjectedFailure, subprocess.CalledProcessError, RuntimeError):
                assert failure
            else:
                assert not failure
        if failure == 'load-sample':
            assert state.loads == [True] and state.releases == 0 and state.model_releases == [True]
            partial = json.loads(summary.with_suffix('.session.partial.json').read_text())
            assert partial['phase'] == 'engine-startup' and not partial['session_complete']
            assert not partial['cases'] and 'failure' in partial and not summary.exists()
            assert not state.allocations and not state.starts
            return
        assert state.loads == [True, False] and state.releases == 1 and state.model_releases == [True, False]
        assert len(state.allocations) == 24
        first = json.loads(Path(payload['cases'][0]['result']).read_text())
        assert first['input_token_ids'] == prompt_ids[0]
        assert len(first['complete_warmups']) == 1 and len(first['decode_trials']) == 3
        for trial in first['decode_trials']:
            assert len(trial['prefill_step_seconds']) == 2 and len(trial['decode_step_seconds']) == 15
            assert abs(trial['full_request_wall_seconds'] - trial['prefill_seconds'] - trial['decode_wall_seconds']) < 1e-10
            # Independent injected work accounting:17 GPU calls and8 host
            # updates/call plus timer ticks. A .5s memory observation, .7s
            # answer check, or2s engine load inside the request must fail.
            assert abs(trial['full_request_wall_seconds'] - 1.736) < 1e-10
            assert not trial['prefill_kv_reused_for_decode_trial']
        if single:
            assert 'session_result' not in first and 'prefill_build_seconds' in first
            assert state.validation_calls == [] and len(state.starts) == 4
            return first
        assert 'cold_startup_seconds_excluding_module_imports' not in first
        partial = json.loads(summary.with_suffix('.session.partial.json').read_text())
        if failure:
            assert not summary.exists() and not partial['session_complete']
            assert partial['active_case'] == 'case1' and len(partial['cases']) == 1 and 'failure' in partial
            assert not Path(payload['cases'][2]['result']).exists()
            if failure == 'inference':
                assert len(state.starts) == 5 and not Path(payload['cases'][1]['result']).exists()
            elif failure in ('reporting', 'memory'):
                assert len(state.starts) == 8 and state.validation_calls == ['case0']
                assert not Path(payload['cases'][1]['result']).exists()
                saved = json.loads(Path(payload['cases'][1]['result']).with_suffix('.trials.partial.json').read_text())
                assert len(saved['decode_trials']) == 3
                assert Path(payload['cases'][1]['result']).with_suffix('.memory-failure.json').exists() == (failure == 'memory')
            else:
                assert len(state.starts) == 8 and state.validation_calls == ['case0', 'case1']
        else:
            report = json.loads(summary.read_text())
            assert report['session_complete'] and len(report['cases']) == 3
            assert len(state.starts) == 12 and state.validation_calls == ['case0', 'case1', 'case2']
            for index, case in enumerate(payload['cases']):
                data = json.loads(Path(case['result']).read_text())
                assert data['engine_pair_reused'] == (index > 0)
                assert data['input_token_ids'] == prompt_ids[index]
                assert len(data['decode_trials']) == 3 and len(data['complete_warmups']) == 1
                assert data['sampled_peak_gpu_memory_mib'] == [101, 102]
                for request in range(index * 4, index * 4 + 4):
                    assert [step[2:] for step in state.steps if step[:2] == ('prefill', request)] == [(0, 2), (2, 1)]
                assert report['cases'][index]['answer_checks'] == 6
            return first

    control = execute('single', single=True)
    shared = execute('complete')
    assert [trial['token_ids_by_request'] for trial in control['decode_trials']] == [
        trial['token_ids_by_request'] for trial in shared['decode_trials']]
    execute('failed-inference', 'inference')
    execute('failed-answer', 'answer')
    execute('failed-reporting', 'reporting')
    execute('failed-memory', 'memory')
    execute('failed-load-sample', 'load-sample')

    # Fail closed before model construction for mixed shape/device/safety,
    # invalid IDs, aliased paths, and retained partial/decoded artifacts.
    for mode in ('length', 'device', 'reserve', 'fraction', 'token', 'alias', 'partial', 'decoded'):
        manifest, summary, payload = specification(root / ('reject-' + mode))
        second = Path(payload['cases'][1]['request'])
        request = json.loads(second.read_text())
        if mode == 'length': request['input_ids'].append(10)
        if mode == 'device': request['device_ids'] = [1, 0]
        if mode == 'reserve': request['reserve_mb'] = 2048
        if mode == 'fraction': request['memory_fraction'] = .8
        if mode == 'token': request['input_ids'][0] = True
        second.write_text(json.dumps(request))
        if mode == 'alias': payload['cases'][1]['result'] = payload['cases'][0]['request']
        if mode == 'partial': Path(payload['cases'][1]['result']).with_suffix('.trials.partial.json').write_text('old evidence')
        if mode == 'decoded': (manifest.parent / 'validation1.trial0.slot0.txt').write_text('old decoded answer')
        manifest.write_text(json.dumps(payload))
        try: read_session(manifest, summary, 2, 16, 64, 2, protected_paths=[profile])
        except (ValueError, FileExistsError): pass
        else: raise AssertionError('Unsafe session accepted: ' + mode)

print('Actual resident single/session runner: poisoned KV rewrite, ragged tail, tensor lifetime, timing, answers, failure cleanup and preflight hazards PASS; CPU only')
