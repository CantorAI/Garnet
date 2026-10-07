"""Inspect TP2 numerical error with independent CPU-reference token prefixes.

This does not replace exact free-generation or compact/full equivalence gates.
"""
import json
import os
from pathlib import Path
import sys
import garnet as G
repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / 'tools/gpt_oss'))
from pipeline import make_tensor_parallel_plan, build_tensor_parallel
weights, cache, output = map(Path, sys.argv[1:4])
batch = int(sys.argv[4]) if len(sys.argv)>4 else 1
if not 1<=batch<=512:
    raise ValueError('Teacher-forced batch must be from1 to512')
if output.exists():
    raise FileExistsError(output)
expected = json.loads((weights / 'expected.json').read_text())
ids = json.loads((weights / 'request.json').read_text())['input_ids']
os.environ.pop('GARNET_GPT_OSS_COMPACT_VOCAB_GREEDY', None)
plan = make_tensor_parallel_plan(weights, json.loads(G.cuda_devices_json())[:2],
                                 batch=batch, capacity=16, tokens=len(ids), reserve_bytes=0)
def tensor(values, dtype, shape):
    return G.tensor_from_host(values, dtype=dtype, shape=shape, device='cuda')
G.cuda_set_device(0)
table = tensor(list(range(batch)), 'int32', [batch, 1])
active = tensor([1]*batch, 'int32', [batch])
model = build_tensor_parallel(weights, cache, plan, len(ids), True, last_token_logits=True)
steps, kv = [], None
try:
    for step, wanted in enumerate(expected['generation_logits']):
        if step:
            if step == 1:
                kv = [(s['keys'], s['values']) for s in model.stages]
                model.release()
                model = build_tensor_parallel(weights, cache, plan, 1, False, kv)
            tokens = [expected['generated'][step - 1]]
            positions = [len(ids) + step - 1]
            start = positions[0]
        else:
            tokens, positions, start = ids, list(range(len(ids))), 0
        length = len(ids) + step
        reply = model.forward(tensor(tokens*batch, 'int64', [batch, len(tokens)]),
            [tensor(positions*batch, 'int64', [batch, len(tokens)]), table,
             tensor([length]*batch, 'int32', [batch]), tensor([start]*batch, 'int32', [batch]), active])
        G.cuda_set_device(0)
        actual = G.tensor_to_cpu(reply['output']).tolist()
        assert len(actual) == batch*len(wanted)
        errors = [abs(a - b) for a, b in zip(actual, wanted*batch)]
        within = all(e <= .025 * (1 + abs(b)) for e, b in zip(errors, wanted*batch))
        actual_top = sorted(range(len(wanted)), key=lambda i: (-actual[i], i))[:2]
        cpu_top = sorted(range(len(wanted)), key=lambda i: (-wanted[i], i))[:2]
        row = dict(step=step, batch=batch, logits_per_request=len(wanted), cpu_top=cpu_top, tp_top=actual_top,
                   cpu_margin=wanted[cpu_top[0]] - wanted[cpu_top[1]],
                   tp_margin=actual[actual_top[0]] - actual[actual_top[1]],
                   maximum_absolute_error=max(errors), within_existing_tolerance=within,
                   cpu_logits=wanted, tp_logits=actual)
        steps.append(row)
        print({k: v for k, v in row.items() if not k.endswith('logits')}, flush=True)
finally:
    model.release()
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(dict(measurement='teacher-forced CPU prefixes, diagnostic only', batch=batch, steps=steps), indent=2))
if not all(s['within_existing_tolerance'] for s in steps):
    raise RuntimeError('TP teacher-forced logits exceed the established compiled parity tolerance')
