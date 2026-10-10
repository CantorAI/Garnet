"""Actual compiled BF16 transport, retained resources and phase-stream parity.

Fresh original/group processes are required because selectors are immutable.
All matrices survive before quality checks. No serving-throughput claim.
"""
import concurrent.futures
import gc
import importlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import garnet as G
import garnet_gpt_oss as extension

folder = Path(sys.argv[1]).resolve()
if folder.exists():
    raise FileExistsError(folder)
folder.mkdir(parents=True)
group_flag = os.environ.get('GARNET_GPT_OSS_BF16_PEER_GROUP', '0')
if group_flag not in ('0', '1'):
    raise ValueError('GARNET_GPT_OSS_BF16_PEER_GROUP must be 0 or 1')
group_mode = group_flag == '1'
group_ctas_text = os.environ.get('GARNET_GPT_OSS_BF16_PEER_GROUP_CTAS', '64')
if group_ctas_text not in ('64', '128', '188') or (not group_mode and group_ctas_text != '64'):
    raise ValueError('Peer-group CTA policy must be 64/128/188 and enabled')
group_ctas = int(group_ctas_text)
group_threads_text = os.environ.get('GARNET_GPT_OSS_BF16_PEER_GROUP_THREADS', '256')
if group_threads_text not in ('256', '512') or (not group_mode and group_threads_text != '256'):
    raise ValueError('Peer-group threads-per-CTA policy must be 256/512 and enabled')
group_threads = int(group_threads_text)
grid_flag = os.environ.get('GARNET_GPT_OSS_BF16_PEER_GRID_SIGNALS', '0')
if grid_flag not in ('0','1') or (not group_mode and grid_flag != '0'):
    raise ValueError('Grid signaling must be0/1 and requires owned peer group')
grid_mode = grid_flag == '1'
group_description = None
G.bind_operator_module(extension)
repo = Path(__file__).resolve().parents[2]
template = (repo / 'xModel/gpt_oss/120b/tp_all_reduce_test.py').read_text()
batch, hidden = 512, 2880
tokens = [8, 1]
elements = [batch * t * hidden for t in tokens]
options = json.dumps(dict(ctas=group_ctas, phase_elements=elements))
models = []
prepared = []
records = []


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def value(integer):
    return struct.unpack('<f', struct.pack('<I', integer))[0]


def finite(n):
    return n ^ 0x80 if n & 0x7f80 == 0x7f80 else n


def period(family, offset):
    inputs = [[], []]
    expected = []
    for i in range(65536):
        a = finite((i + offset) & 65535)
        b = [a, a ^ 0x8000, finite((a + 1) & 65535), a & 0x8000,
             (a & 0x8000) | 1, (a & 0x8000) | 0x3f80][family]
        x, y = value(a << 16), value(b << 16)
        exact = x + y
        try:
            summed = bits(exact)
        except OverflowError:
            summed = 0xff800000 if exact < 0 else 0x7f800000
        expected.append(((summed + 0x7fff + ((summed >> 16) & 1)) & 0xffff0000) & 0xffffffff)
        inputs[0].append(x)
        inputs[1].append(y)
    return inputs, struct.pack('<65536I', *expected)


def initialize():
    sources = []
    for phase, sequence in enumerate(tokens):
        for rank in range(2):
            root = folder / f'phase{phase}-rank{rank}'
            root.mkdir(exist_ok=True)
            source = template.replace('TP_RANK = 0', f'TP_RANK = {rank}')
            source = source.replace("boundary='required')", "boundary='required', atomic=True, cuda_graph=True)")
            source = source.replace('hidden_size=8, tp_rank=tp_rank',
                f'hidden_size={hidden}, tp_rank=tp_rank, bf16_communication=1, prefill={int(phase == 0)}')
            path = root / 'tp_all_reduce_test.py'
            path.write_text(source)
            (root / '__init__.py').write_text('')
            shutil.copy2(repo / 'xModel/gpt_oss/120b/tensor_compat.py', root / 'tensor_compat.py')
            sources.append(path)
    # Refresh once after the complete package tree exists, before any import.
    importlib.invalidate_caches()
    for phase, sequence in enumerate(tokens):
        stages, tensors = [], []
        for rank in range(2):
            path = sources[phase * 2 + rank]
            root = path.parent
            G.cuda_set_device(rank)
            model = G.load_model(str(path), runtime_mode='compiled_xmodel', backend='tensorrt',
                precision='bf16', entry_function='GptOssTpCollective',
                input_shapes=[[batch, sequence, hidden]], input_dtypes=['float32'],
                cache_dir=str(root / 'engine'))
            assert model.runtime_status()['ready'], model.runtime_status()
            status = model.runtime_status()
            assert status['engine_partition_count'] == 1
            assert any(r['cuda_graph'] for r in status['execution_plan']['regions'])
            stages.append(model)
            tensors.append(G.tensor_from_host([0.] * elements[phase], dtype='float32',
                shape=[batch, sequence, hidden], device='cuda'))
        models.append(stages)
        prepared.append(tensors)


def run(pool, group, phase, family, offset, tag):
    if group is not None:
        extension.peer_group_bind_phase(group, phase)
    inputs, expected = period(family, offset)
    count = elements[phase]
    for rank in range(2):
        G.cuda_set_device(rank)
        values = (inputs[rank] * ((count + 65535) // 65536))[:count]
        G.tensor_update_from_host(prepared[phase][rank], values)

    def forward(rank):
        G.cuda_set_device(rank)
        request = dict(inputs=[prepared[phase][rank]], reuse_output=True)
        if group is not None:
            request.update(operator_execution_group=group,
                operator_execution_phase=phase, operator_execution_rank=rank)
        result = models[phase][rank].forward(request)
        assert result['status'] == 'ok', result
        G.cuda_synchronize()
        output = G.tensor_to_cpu(result['output']).tolist()
        raw = struct.pack('<' + str(count) + 'f', *output)
        path = folder / (tag + f'-rank{rank}.bin')
        with path.open('xb') as f:
            f.write(raw)
        wanted = (expected * ((count + 65535) // 65536))[:count * 4]
        assert raw == wanted, ('COMPLETE_COMPILED_MATRIX_MISMATCH', tag, rank)
        return dict(path=path.name, phase=phase, rank=rank, family=family,
            offset=offset, values=count, every_bit_exact=True)

    results = [pool.submit(forward, rank) for rank in range(2)]
    records.extend(future.result() for future in results)
    print('COMPILED_PEER_PHASE_COMPLETE', tag, phase, family, offset, flush=True)


group = None
try:
    initialize()
    if group_mode:
        for invalid in [dict(ctas=64.0, phase_elements=elements),
                dict(ctas=2**80, phase_elements=elements),
                dict(ctas=32, phase_elements=elements),
                dict(ctas=64, phase_elements=[0, elements[1]]),
                dict(ctas=64, phase_elements=[elements[0] + 8, elements[1]]),
                dict(ctas=64, phase_elements=[elements[0], 7]),
                dict(ctas=64, phase_elements=elements, extra=True)]:
            try:
                extension.peer_group(json.dumps(invalid))
            except Exception as e:
                assert 'invalid' in str(e).lower(), str(e)
            else:
                raise AssertionError('Malformed native storage contract accepted')
        print('COMPILED_PEER_OPTIONS_NEGATIVES_COMPLETE', 5, flush=True)
    group = extension.peer_group(options) if group_mode else None
    if group_mode:
        group_description = json.loads(extension.peer_group_status_json(group))
        assert group_description['ctas'] == group_ctas
        assert group_description.get('threads_per_cta', 256) == group_threads
        expected_schema = 4 if group_threads == 512 else (3 if grid_mode else 2)
        expected_protocol = ('owned-mapped-bf16-peer-grid-threads-v1' if grid_mode else 'owned-mapped-bf16-peer-threads-v1') if group_threads == 512 else ('owned-mapped-bf16-peer-grid-v3' if grid_mode else 'owned-mapped-bf16-peer-v2')
        assert group_description['schema'] == expected_schema
        assert group_description['protocol'] == expected_protocol
        assert group_description.get('grid_signals',0) == int(grid_mode)
        print('NATIVE_GROUP', json.dumps(group_description), flush=True)
        G.cuda_set_device(0)
        for rank, phase in [(-1, 0), (2, 0), (0, -1), (0, 2), (0, 1)]:
            try:
                models[0][0].forward(dict(inputs=[prepared[0][0]],
                    operator_execution_group=group,
                    operator_execution_rank=rank, operator_execution_phase=phase))
            except Exception:
                pass
            else:
                raise AssertionError(('Invalid group rank or inactive phase accepted', rank, phase))
        print('COMPILED_PEER_GROUP_NEGATIVES_COMPLETE', 5, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        # First call warms, second captures; later calls replay both phase pairs with
        # updated input storage. Both cached engines remain alive together.
        for family in range(6):
            for phase in range(2):
                run(pool, group, phase, family, 131 + family * 31 + phase * 17,
                    f'family{family}-phase{phase}')
    statuses = [[model.runtime_status() for model in phase] for phase in models]
    if group_mode:
        closed_group = group
        assert extension.peer_group_release(group)
        assert extension.peer_group_release(group)
        group = None
        gc.collect()
        for action in [lambda: extension.peer_group_status_json(closed_group),
                lambda: extension.peer_group_bind_phase(closed_group, 0),
                lambda: models[0][0].forward(dict(inputs=[prepared[0][0]],
                    operator_execution_group=closed_group,
                    operator_execution_rank=0, operator_execution_phase=0))]:
            try:
                action()
            except Exception:
                pass
            else:
                raise AssertionError('Released native handle accepted an operation')
        print('COMPILED_PEER_RELEASE_NEGATIVES_COMPLETE', 3, flush=True)
        try:
            unexpected = extension.peer_group(options)
        except Exception as e:
            assert 'not ready' in str(e).lower(), str(e)
        else:
            raise AssertionError('Cached graphs failed to retain exclusive native owner')
    for phase in models:
        for model in phase:
            model.release_runtime()
    models.clear()
    # Cached resource references are gone; a new owner can now be acquired.
    if group_mode:
        group = extension.peer_group(options)
        extension.peer_group_bind_phase(group, 1)
        extension.peer_group_bind_phase(group, 0)
        assert extension.peer_group_release(group)
        assert extension.peer_group_release(group)
        group = extension.peer_group(options)
        assert extension.peer_group_release(group)
        group = None
        gc.collect()
    (folder / 'result.json').write_text(json.dumps(dict(protocol='compiled-bf16-peer-group-v1',
        group_mode=group_mode, group_ctas=group_ctas if group_mode else None,
        group_threads_per_cta=group_threads if group_mode else None,
        grid_signals=grid_mode, native_group_description=group_description, batch=batch, tokens=tokens, hidden=hidden,
        matrices=records, all_complete=True,
        compiled_statuses=statuses,
        scope='Compiled transport and retained graph ownership only; no full pretrained/refit/serving qualification'), indent=2))
    print('COMPILED_PEER_GROUP_ALL_GATES_COMPLETE', len(records), flush=True)
finally:
    for phase in models:
        for model in phase:
            model.release_runtime()
    if group is not None:
        extension.peer_group_release(group)
    group = None
