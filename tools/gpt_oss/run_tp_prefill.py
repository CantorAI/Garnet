"""XLang3: repeat one GPT-OSS TP2 prefill request without decoding.

Usage: xlang3 run_tp_prefill.py weights cache request.json result.json
The first pass can include lazy weight packing. Later passes measure warm
prefill with the same model, input, and KV pages.
"""
import json
import sys
import time
from pathlib import Path

import garnet as G
from pipeline import build_tensor_parallel, make_tensor_parallel_plan


if len(sys.argv) != 5:
    raise SystemExit('expected weights, cache, request JSON, result JSON')
weights, cache = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
request = json.loads(Path(sys.argv[3]).read_text())
ids = request['input_ids']
repeats = int(request.get('prefill_repeats', 3))
if not ids or len(ids) > 4096 or not 2 <= repeats <= 10:
    raise ValueError('invalid input length or prefill_repeats')

devices = json.loads(G.cuda_devices_json())
selected = request.get('device_ids', [d['id'] for d in devices[:2]])
if len(selected) != 2 or len(set(selected)) != 2:
    raise ValueError('prefill benchmark requires two distinct GPUs')
devices = [next(d for d in devices if d['id'] == device_id) for device_id in selected]
plan = make_tensor_parallel_plan(weights, devices, batch=1,
    capacity=len(ids) + int(request.get('max_new_tokens', 32)), tokens=len(ids),
    reserve_bytes=int(request.get('reserve_mb', 1024)) << 20,
    memory_fraction=float(request.get('memory_fraction', .9)))
if not all(isinstance(token, int) and 0 <= token < plan['config']['vocab_size'] for token in ids):
    raise ValueError('input contains an invalid token ID')

G.cuda_set_device(devices[0]['id'])
pages = (plan['capacity'] + 15) // 16
def tensor(data, dtype, shape):
    return G.tensor_from_host(data, dtype=dtype, shape=shape, device='cuda')

token = tensor(ids, 'int64', [1, len(ids)])
position = tensor(list(range(len(ids))), 'int64', [1, len(ids)])
table = tensor(list(range(pages)), 'int32', [1, pages])
length = tensor([len(ids)], 'int32', [1])
slot = tensor([0], 'int32', [1])
active = tensor([1], 'int32', [1])
model = build_tensor_parallel(weights, cache, plan, len(ids), True,
                              last_token_logits=True)
seconds, token_ids = [], []
try:
    for i in range(repeats):
        start = time.perf_counter()
        result = model.forward(token, [position, table, length, slot, active], True)
        seconds.append(time.perf_counter() - start)
        token_ids.append(int(result['token_id']))
        print('prefill', i, 'seconds', seconds[-1], 'token', token_ids[-1], flush=True)
finally:
    model.release()
if len(set(token_ids)) != 1:
    raise RuntimeError('repeated prefill produced different first tokens')
Path(sys.argv[4]).write_text(json.dumps({
    'token_ids': token_ids, 'prefill_seconds': seconds,
    'warm_prefill_seconds': seconds[1:], 'placement': plan,
}, indent=2))
