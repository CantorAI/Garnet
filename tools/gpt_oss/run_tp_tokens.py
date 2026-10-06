"""XLang3: run_tp_tokens.py original-weights cache request.json result.json."""
import json
import sys
import time
from pathlib import Path
import garnet as G
from pipeline import make_tensor_parallel_plan, build_tensor_parallel

assert len(sys.argv) == 5, 'expected weights, cache, request JSON, result JSON'
weights, cache = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
request = json.loads(Path(sys.argv[3]).read_text())
ids = request['input_ids']
limit = int(request.get('max_new_tokens', 32))
assert ids and 1 <= limit <= 256 and len(ids) + limit <= 4096
devices = json.loads(G.cuda_devices_json())
selected = request.get('device_ids', [d['id'] for d in devices[:2]])
assert len(selected) == 2 and len(set(selected)) == 2
devices = [next(d for d in devices if d['id'] == device_id) for device_id in selected]
plan = make_tensor_parallel_plan(weights, devices, batch=1,
    capacity=len(ids) + limit, tokens=len(ids),
    reserve_bytes=int(request.get('reserve_mb', 1024)) << 20,
    memory_fraction=float(request.get('memory_fraction', .9)))
print('TP2 plan', json.dumps({k: plan[k] for k in ('cache_key', 'estimated_per_gpu_bytes', 'hardware')}), flush=True)
assert all(isinstance(token, int) and 0 <= token < plan['config']['vocab_size'] for token in ids)
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
print('Building TP2 prefill engines', flush=True)
model = build_tensor_parallel(weights, cache, plan, len(ids), True, last_token_logits=True)
load_seconds = time.perf_counter() - started
started = time.perf_counter()
print('Running paired TP2 prefill', flush=True)
result = model.forward(tensor(ids, 'int64', [1, len(ids)]),
    [tensor(list(range(len(ids))), 'int64', [1, len(ids)]), table, length, slot, active], True)
generated = [int(result['token_id'])]
prefill_seconds = time.perf_counter() - started
print('TP2 prefill complete', prefill_seconds, flush=True)
kv = [(stage['keys'], stage['values']) for stage in model.stages]
model.release()
decode_seconds = decode_wall_seconds = decode_prepare_seconds = 0.
decode_step_seconds = []
if limit > 1 and generated[-1] not in stops:
    started = time.perf_counter()
    print('Building TP2 decode engines', flush=True)
    model = build_tensor_parallel(weights, cache, plan, 1, False, kv)
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
        step_seconds = time.perf_counter() - started
        decode_step_seconds.append(step_seconds)
        decode_seconds += step_seconds
        generated.append(int(result['token_id']))
        if generated[-1] in stops:
            break
    decode_wall_seconds = time.perf_counter() - decode_started
    print('TP2 decode complete', decode_wall_seconds, flush=True)
    model.release()

# The first decode invocation can lazily prepare rank-local Marlin weights.
# Keep its latency visible, but exclude it from steady-state throughput just
# as TTFT is kept separate from decode throughput in the vLLM measurements.
warm_decode_seconds = sum(decode_step_seconds[1:])
warm_decode_tokens = max(0, len(decode_step_seconds) - 1)
warm_decode_tokens_per_second = (warm_decode_tokens / warm_decode_seconds
    if warm_decode_seconds > 0 else None)

Path(sys.argv[4]).write_text(json.dumps({
    'token_ids': generated, 'placement': plan, 'load_seconds': load_seconds,
    'prefill_seconds': prefill_seconds, 'decode_seconds': decode_seconds,
    'decode_prepare_seconds': decode_prepare_seconds,
    'decode_wall_seconds': decode_wall_seconds,
    'decode_step_seconds': decode_step_seconds,
    'warm_decode_seconds': warm_decode_seconds,
    'warm_decode_tokens': warm_decode_tokens,
    'warm_decode_tokens_per_second': warm_decode_tokens_per_second,
    'validation': 'experimental replicated-weight, expert-parallel TP2; full pretrained validation pending'
}, indent=2))
print('Generated', len(generated), 'tokens with two-rank GPT-OSS expert parallelism', flush=True)
