"""XLang3: run_pipeline_tokens.py original-weights cache request.json result.json.

Request may specify device_ids (default: all visible GPUs), memory_fraction,
reserve_mb and max_new_tokens. Builds an automatic contiguous layer placement.
"""
import json
import sys
import time
from pathlib import Path
import garnet as G
from pipeline import make_plan, build_pipeline

assert len(sys.argv) == 5, 'expected weights, cache, request JSON, result JSON'
weights, cache = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
request = json.loads(Path(sys.argv[3]).read_text())
ids = request['input_ids']
limit = int(request.get('max_new_tokens', 32))
assert ids and 1 <= limit <= 256 and len(ids) + limit <= 4096
devices = json.loads(G.cuda_devices_json())
selected = request.get('device_ids', [d['id'] for d in devices])
assert selected and len(set(selected)) == len(selected)
devices = [next(d for d in devices if d['id'] == id) for id in selected]
plan = make_plan(weights, devices, batch=1, capacity=len(ids) + limit, tokens=len(ids),
                 reserve_bytes=int(request.get('reserve_mb', 1024)) << 20,
                 memory_fraction=float(request.get('memory_fraction', .9)))
assert all(isinstance(t, int) and 0 <= t < plan['config']['vocab_size'] for t in ids)
stops = request.get('stop_token_ids', [])


def tensor(data, dtype, shape):
    return G.tensor_from_host(data, dtype=dtype, shape=shape, device='cuda')


G.cuda_set_device(devices[0]['id'])
pages = (plan['capacity'] + 15) // 16
table = tensor(list(range(pages)), 'int32', [1, pages])
length = tensor([len(ids)], 'int32', [1])
slot = tensor([0], 'int32', [1])
active = tensor([1], 'int32', [1])
started = time.perf_counter()
model = build_pipeline(weights, cache, plan, len(ids), True)
load_seconds = time.perf_counter() - started
started = time.perf_counter()
result = model.forward(tensor(ids, 'int64', [1, len(ids)]),
    [tensor(list(range(len(ids))), 'int64', [1, len(ids)]), table, length, slot, active], True)
generated = [int(result['token_id'])]
prefill_seconds = time.perf_counter() - started
kv = [(s['keys'], s['values']) for s in model.stages]
model.release()
decode_seconds = 0.
decode_wall_seconds = 0.
decode_prepare_seconds = 0.
if limit > 1 and generated[-1] not in stops:
    started = time.perf_counter()
    model = build_pipeline(weights, cache, plan, 1, False, kv)
    token = tensor([generated[-1]], 'int64', [1, 1])
    position = tensor([len(ids)], 'int64', [1, 1])
    decode_prepare_seconds = time.perf_counter() - started
    decode_started = time.perf_counter()
    for offset in range(1, limit):
        index = len(ids) + offset - 1
        G.tensor_update_from_host(token, [generated[-1]])
        G.tensor_update_from_host(position, [index])
        G.tensor_update_from_host(length, [index + 1])
        G.tensor_update_from_host(slot, [index])
        started = time.perf_counter()
        result = model.forward(token, [position, table, length, slot, active], True)
        decode_seconds += time.perf_counter() - started
        generated.append(int(result['token_id']))
        if generated[-1] in stops:
            break
    decode_wall_seconds = time.perf_counter() - decode_started
    model.release()
Path(sys.argv[4]).write_text(json.dumps({'token_ids': generated, 'placement': plan,
    'load_seconds': load_seconds, 'prefill_seconds': prefill_seconds, 'decode_seconds': decode_seconds,
    'decode_prepare_seconds': decode_prepare_seconds, 'decode_wall_seconds': decode_wall_seconds,
    'validation': 'experimental sequential GPU pipeline; full pretrained validation pending'}, indent=2))
print('Generated', len(generated), 'tokens across', len(devices), 'GPUs', flush=True)
