"""Compiled provider authorization across fresh/cache loads and release.

Run exclusively on TP2 with the peer group enabled. This is a lifecycle gate,
not a throughput benchmark. Complete output rows are retained without tolerances.
"""
import concurrent.futures
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import garnet as G
import garnet_gpt_oss as extension

pattern_flag = os.environ.get('GARNET_RESIDENT_PATTERN_UPDATES', '0')
if pattern_flag not in ('0', '1'):
    raise ValueError('GARNET_RESIDENT_PATTERN_UPDATES must be 0 or 1')
pattern_updates = pattern_flag == '1'
group_flag = os.environ.get('GARNET_GPT_OSS_BF16_PEER_GROUP', '1')
if group_flag != '1':
    raise ValueError('Operator-group lifecycle gate requires GARNET_GPT_OSS_BF16_PEER_GROUP=1')
group_ctas_text = os.environ.get('GARNET_GPT_OSS_BF16_PEER_GROUP_CTAS', '64')
if group_ctas_text not in ('64', '128'):
    raise ValueError('Peer-group CTA policy must be 64 or 128')
group_ctas = int(group_ctas_text)
folder = Path(sys.argv[1]).resolve()
folder.mkdir(exist_ok=False, parents=True)
repo = Path(__file__).resolve().parents[2]
G.bind_operator_module(extension)
template = (repo / 'xModel/gpt_oss/120b/tp_all_reduce_test.py').read_text()
if pattern_updates:
    # The compiled group must consume the new count on its own execution stream.
    # Zero is deliberately invalid: a missing dependency cannot pass by luck.
    template = template.replace("{'name': 'x', 'kind': 'tensor'},",
        "{'name': 'x', 'kind': 'tensor'}, {'name': 'valid_count', 'kind': 'tensor'},")
    template = template.replace('def GptOssTpCollective(x, tp_rank):',
        'def GptOssTpCollective(x, valid_count, tp_rank):')
    template = template.replace("    return x * T.unary_op('gpt_oss_tp_all_reduce', hidden_size=8, tp_rank=tp_rank)",
        "    reduced = x * T.unary_op('gpt_oss_tp_all_reduce', hidden_size=8, tp_rank=tp_rank)\n"
        "    return reduced * T.binary_op('select_last_valid_sequence') * valid_count")
hidden, tokens = 2880, [2, 1]
paths = []
for phase, sequence in enumerate(tokens):
    for rank in range(2):
        root = folder / f'phase{phase}-rank{rank}'; root.mkdir()
        source = template.replace('TP_RANK = 0', f'TP_RANK = {rank}')
        source = source.replace('hidden_size=8, tp_rank=tp_rank',
            f'hidden_size={hidden}, tp_rank=tp_rank, bf16_communication=1, prefill={int(phase == 0)}')
        path = root / 'tp_all_reduce_test.py'; path.write_text(source)
        (root / '__init__.py').write_text('')
        shutil.copy2(repo / 'xModel/gpt_oss/120b/tensor_compat.py', root / 'tensor_compat.py')
        paths.append(path)
foreign_path = paths[0].parent / 'plain_relu.py'
foreign_path.write_text("from .tensor_compat import tensor\nT = tensor()\n"
    "GARNET_MODEL_SPEC = {'arguments': [{'name': 'x', 'kind': 'tensor'}]}\n"
    "@T.fusion(name='plain_relu', role='decoder_layer', boundary='required')\n"
    "def Plain(x):\n    return x * T.unary_op('relu')\n")
importlib.invalidate_caches()
models, tensors, counts, records, statuses = [], [], [], [], []
released_responses = []
group = foreign = None

def load():
    for phase, sequence in enumerate(tokens):
        ranks = []
        for rank in range(2):
            G.cuda_set_device(rank); path = paths[phase * 2 + rank]
            model = G.load_model(str(path), runtime_mode='compiled_xmodel', backend='tensorrt',
                precision='bf16', entry_function='GptOssTpCollective',
                input_shapes=([[1, sequence, hidden], [1]] if pattern_updates else [[1, sequence, hidden]]),
                input_dtypes=(['float32', 'int32'] if pattern_updates else ['float32']),
                cache_dir=str(path.parent / 'engine'))
            status = model.runtime_status(); assert status['ready'], status
            ranks.append(model)
        models.append(ranks)

def rejected(model, tensor):
    try:
        model.forward(dict(inputs=[tensor], operator_execution_group=group,
            operator_execution_rank=0, operator_execution_phase=0))
    except Exception as e:
        assert 'not required' in str(e).lower(), str(e)
    else:
        raise AssertionError('Missing/released model provider membership authorized a native group')

try:
    load()
    for phase, sequence in enumerate(tokens):
        ranks = []
        for rank in range(2):
            G.cuda_set_device(rank)
            ranks.append(G.tensor_from_host([float((i % 7) + rank) for i in range(sequence * hidden)],
                shape=[1, sequence, hidden], dtype='float32', device='cuda'))
        tensors.append(ranks)
        if pattern_updates:
            count_ranks = []
            for rank in range(2):
                G.cuda_set_device(rank)
                count_ranks.append(G.tensor_from_host([0], shape=[1], dtype='int32', device='cuda'))
            counts.append(count_ranks)
    G.cuda_set_device(0)
    foreign = G.load_model(str(foreign_path), runtime_mode='compiled_xmodel', backend='tensorrt',
        precision='bf16', entry_function='Plain', input_shapes=[[1, 2, hidden]],
        input_dtypes=['float32'], cache_dir=str(folder / 'plain-engine'))
    assert foreign.runtime_status()['ready'], foreign.runtime_status()
    control = foreign.forward(dict(inputs=[tensors[0][0]])); assert control['status'] == 'ok', control
    control_actual = G.tensor_to_cpu(control['output']).tolist()
    assert control_actual == [float(i % 7) for i in range(2 * hidden)]
    group = extension.peer_group(json.dumps(dict(ctas=group_ctas, phase_elements=[s * hidden for s in tokens])))
    rejected(foreign, tensors[0][0])
    for reload in range(2):
        if reload:
            load()
        status = [[m.runtime_status() for m in ranks] for ranks in models]
        assert all(s['forbidden_path_counters']['graph_cache_hits'] == reload for ranks in status for s in ranks)
        statuses.append(status)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for phase, sequence in enumerate(tokens):
                extension.peer_group_bind_phase(group, phase)
                def forward(rank):
                    G.cuda_set_device(rank)
                    inputs = [tensors[phase][rank]]
                    valid = sequence if reload == 0 else 1
                    if pattern_updates:
                        assert G.tensor_update_int_patterns_async([counts[phase][rank]], [[valid]])
                        inputs.append(counts[phase][rank])
                    result = models[phase][rank].forward(dict(inputs=inputs,
                        operator_execution_group=group, operator_execution_phase=phase,
                        operator_execution_rank=rank))
                    assert result['status'] == 'ok', result
                    G.cuda_synchronize(); actual = G.tensor_to_cpu(result['output']).tolist()
                    expected = ([float(2 * (((valid - 1) * hidden + i) % 7) + 1) for i in range(hidden)]
                        if pattern_updates else [float(2 * (i % 7) + 1) for i in range(sequence * hidden)])
                    assert actual == expected, (reload, phase, rank)
                    row = dict(reload=reload, phase=phase, rank=rank, actual=actual, expected=expected)
                    if pattern_updates:
                        count_actual = G.tensor_to_cpu(counts[phase][rank]).tolist()
                        assert count_actual == [valid], count_actual
                        row.update(valid_count=valid, count_actual=count_actual)
                    return row
                records.extend(f.result() for f in [pool.submit(forward, rank) for rank in range(2)])
                (folder / 'result.partial.json').write_text(json.dumps(dict(
                    matrices=records, statuses=statuses, released_responses=released_responses,
                    default_control_actual=control_actual, complete=False), indent=2))
        extension.peer_group_bind_phase(group, 0)
        for ranks in models:
            for model in ranks: model.release_runtime()
        # Model::ReleaseRuntime destroys the compiled runtime. Its established
        # public contract returns this exact error before any group entry.
        released = models[0][0].forward(dict(inputs=[tensors[0][0]],
            operator_execution_group=group, operator_execution_rank=0,
            operator_execution_phase=0))
        assert (released['status'] == 'error' and
            released['error_code'] == 'compiled_graph_not_ready' and 'output' not in released), released
        released_responses.append(released)
        (folder / 'result.partial.json').write_text(json.dumps(dict(
            matrices=records, statuses=statuses, released_responses=released_responses,
            default_control_actual=control_actual, complete=False), indent=2))
        models.clear()
    assert extension.peer_group_release(group)
    assert extension.peer_group_release(group)
    group = None
    # The script release and all cached engine leases have actually retired.
    group = extension.peer_group(json.dumps(dict(ctas=group_ctas, phase_elements=[s * hidden for s in tokens])))
    assert extension.peer_group_release(group); group = None
    (folder / 'result.json').write_text(json.dumps(dict(protocol='operator-requirements-lifecycle-v1',
        pattern_updates=pattern_updates, group_ctas=group_ctas,
        statuses=statuses, matrices=records, negative_foreign=1, negative_released=2,
        default_control_actual=control_actual, released_responses=released_responses,
        default_control_exact=True, passed=True), indent=2))
    print('OPERATOR_REQUIREMENTS_FRESH_CACHE_RELEASE_COMPLETE 8', flush=True)
finally:
    for ranks in models:
        for model in ranks: model.release_runtime()
    if foreign is not None: foreign.release_runtime()
    if group is not None: extension.peer_group_release(group)
