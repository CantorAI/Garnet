"""Actual two-GPU stage execution, local KV ownership and cached-engine parity."""
import json
import sys
from pathlib import Path
import garnet as G

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / 'tools/gpt_oss'))
from pipeline import make_plan, build_pipeline
weights, cache = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
devices = json.loads(G.cuda_devices_json())
assert len(devices) >= 2, 'test requires two visible CUDA GPUs'
devices = devices[:2]
print('devices', devices, flush=True)
expected = json.loads((weights / 'expected.json').read_text())
plan = make_plan(weights, devices, batch=2, capacity=16, tokens=3, reserve_bytes=256 << 20)
assert [s['start'] for s in plan['stages']] == [0, 1]
print('placement', plan['stages'], flush=True)


def tensor(data, dtype, shape):
    return G.tensor_from_host(data, dtype=dtype, shape=shape, device='cuda')


def compare(result, target, label):
    actual = G.tensor_to_cpu(result['output']).tolist()
    reference = [v for row in target for v in row]
    assert len(actual) == len(reference), (len(actual), len(reference))
    maximum = 0.
    for value, wanted in zip(actual, reference):
        maximum = max(maximum, abs(value - wanted))
        assert abs(value - wanted) <= .025 * (1 + abs(wanted)), (label, value, wanted)
    print(label, maximum, flush=True)


def forward(model, ids, positions, lengths, slots, active):
    G.cuda_set_device(0)
    batch, tokens = len(ids), len(ids[0])
    controls = [tensor([v for row in positions for v in row], 'int64', [batch, tokens]),
                tensor([0, 1], 'int32', [2, 1]), tensor(lengths, 'int32', [2]),
                tensor(slots, 'int32', [2]), tensor(active, 'int32', [2])]
    return model.forward(tensor([v for row in ids for v in row], 'int64', [batch, tokens]), controls)


prefill = build_pipeline(weights, cache, plan, 3, True)
compare(forward(prefill, [row[:3] for row in expected['ids']], [[0, 1, 2]] * 2,
                [3, 3], [0, 0], [1, 1]),
        [row for batch in expected['logits'] for row in batch[:3]], 'two GPU prefill')
kv = [(s['keys'], s['values']) for s in prefill.stages]
prefill.release()
last_only = build_pipeline(weights, cache, plan, 3, True, last_token_logits=True)
compare(forward(last_only, [row[:3] for row in expected['ids']], [[0, 1, 2]] * 2,
                [3, 3], [0, 0], [1, 1]),
        [batch[2] for batch in expected['logits']], 'two GPU last-token prefill logits')
last_only.release()
decode = build_pipeline(weights, cache, plan, 1, False, kv)
ids = [[row[3]] for row in expected['ids']]
compare(forward(decode, ids, [[3], [3]], [4, 4], [3, 3], [1, 1]),
        [batch[3] for batch in expected['logits']], 'two GPU cached decode')
saved = [[G.tensor_to_cpu(t).tolist() for t in pair] for pair in kv]
forward(decode, [[5], [1]], [[3], [3]], [4, 4], [3, 3], [1, 0])
updated = [[G.tensor_to_cpu(t).tolist() for t in pair] for pair in kv]
for before, after in zip(saved, updated):
    for old, new in zip(before, after):
        # Each local KV tensor has one layer, two pages, one page per request.
        assert old[len(old) // 2:] == new[len(new) // 2:]
assert saved[0][0][:len(saved[0][0]) // 2] != updated[0][0][:len(updated[0][0]) // 2]
decode.release()
decode = build_pipeline(weights, cache, plan, 1, False, kv)
compare(forward(decode, ids, [[3], [3]], [4, 4], [3, 3], [1, 1]),
        [batch[3] for batch in expected['logits']], 'two GPU engine reload')
decode.release()
# Native transfers preserve bytes across GPUs, including return to source GPU.
G.cuda_set_device(0)
t = tensor([1, 7, -3], 'int32', [3])
assert G.tensor_to_cpu(G.tensor_to_device(G.tensor_to_device(t, 1), 0)).tolist() == [1, 7, -3]
cpu = G.tensor_from_host([9, -2], dtype='int64', shape=[2], device='cpu')
assert G.tensor_to_cpu(G.tensor_to_device(cpu, 1)).tolist() == [9, -2]
for invalid in (-1, len(devices), .5):
    try:
        G.cuda_set_device(invalid)
        raise AssertionError('invalid device accepted')
    except RuntimeError:
        pass
print('multi-gpu-parity-passed', flush=True)
