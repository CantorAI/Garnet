"""XLang3: run_tp_batch_throughput.py WEIGHTS CACHE REQUEST RESULT BATCH OUTPUT_TOKENS.

Repeat one saved prompt across a fixed decode batch. This measures aggregate
output throughput with identical work in every slot and no early-stop bias.
"""
import json
import sys
import time
from pathlib import Path

import garnet as G
from pipeline import build_tensor_parallel, make_tensor_parallel_plan


assert len(sys.argv) == 7, 'expected weights, cache, request, result, batch, output tokens'
weights, cache = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
request = json.loads(Path(sys.argv[3]).read_text())
result_path = Path(sys.argv[4])
batch, output_tokens = int(sys.argv[5]), int(sys.argv[6])
ids = request['input_ids']
assert 1 <= batch <= 16 and 16 <= output_tokens <= 512
assert ids and len(ids) + output_tokens <= 4096

available = json.loads(G.cuda_devices_json())
selected = request.get('device_ids', [d['id'] for d in available[:2]])
assert len(selected) == 2 and len(set(selected)) == 2
devices = [next(d for d in available if d['id'] == device_id) for device_id in selected]
plan = make_tensor_parallel_plan(weights, devices, batch=batch,
    capacity=len(ids) + output_tokens, tokens=len(ids),
    reserve_bytes=int(request.get('reserve_mb', 1024)) << 20,
    memory_fraction=float(request.get('memory_fraction', .9)))
print('TP2 batch plan', json.dumps({k: plan[k] for k in
    ('cache_key', 'estimated_per_gpu_bytes', 'batch')}), flush=True)
assert all(isinstance(token, int) and 0 <= token < plan['config']['vocab_size']
           for token in ids)


def tensor(data, dtype, shape):
    return G.tensor_from_host(data, dtype=dtype, shape=shape, device='cuda')


G.cuda_set_device(devices[0]['id'])
pages_per_request = (plan['capacity'] + 15) // 16
table_data = [batch_index * pages_per_request + page
              for batch_index in range(batch)
              for page in range(pages_per_request)]
table = tensor(table_data, 'int32', [batch, pages_per_request])
length = tensor([len(ids)] * batch, 'int32', [batch])
slot = tensor([0] * batch, 'int32', [batch])
active = tensor([1] * batch, 'int32', [batch])
input_ids = tensor(ids * batch, 'int64', [batch, len(ids)])
positions = tensor(list(range(len(ids))) * batch, 'int64', [batch, len(ids)])

started = time.perf_counter()
print('Building TP2 batch prefill engines', flush=True)
model = build_tensor_parallel(weights, cache, plan, len(ids), True,
                              last_token_logits=True)
prefill_build_seconds = time.perf_counter() - started
started = time.perf_counter()
prefill = model.forward(input_ids, [positions, table, length, slot, active],
                        True, sample_batch=True)
prefill_seconds = time.perf_counter() - started
generated = [[int(token)] for token in prefill['token_ids']]
assert len(generated) == batch, (len(generated), batch)
print('TP2 batch prefill complete', prefill_seconds, flush=True)
kv = [(stage['keys'], stage['values']) for stage in model.stages]
model.release()

started = time.perf_counter()
print('Building TP2 batch decode engines', flush=True)
model = build_tensor_parallel(weights, cache, plan, 1, False, kv)
decode_build_seconds = time.perf_counter() - started
token = tensor([row[-1] for row in generated], 'int64', [batch, 1])
position = tensor([len(ids)] * batch, 'int64', [batch, 1])
rank_inputs = []
for stage in model.stages:
    previous = G.cuda_set_device(stage['device_id'])
    try:
        rank_inputs.append((G.tensor_to_device(token, stage['device_id']),
            [G.tensor_to_device(control, stage['device_id']) for control in
             (position, table, length, slot, active)]))
    finally:
        G.cuda_set_device(previous)

step_seconds = []
for offset in range(1, output_tokens):
    tokens = [row[-1] for row in generated]
    index = len(ids) + offset - 1
    started = time.perf_counter()
    for stage, (local_token, controls) in zip(model.stages, rank_inputs):
        previous = G.cuda_set_device(stage['device_id'])
        try:
            G.tensor_update_from_host(local_token, tokens)
            G.tensor_update_from_host(controls[0], [index] * batch)
            G.tensor_update_from_host(controls[2], [index + 1] * batch)
            G.tensor_update_from_host(controls[3], [index] * batch)
        finally:
            G.cuda_set_device(previous)
    reply = model.forward_rank_local(rank_inputs, True, sample_batch=True)
    sampled = [int(value) for value in reply['token_ids']]
    assert len(sampled) == batch, (len(sampled), batch)
    for row, value in zip(generated, sampled):
        row.append(value)
    step_seconds.append(time.perf_counter() - started)
    if offset % 32 == 0:
        print('Decoded', offset, 'of', output_tokens - 1, 'batch steps', flush=True)
model.release()

warm_seconds = sum(step_seconds[1:])
warm_output_tokens = batch * max(0, len(step_seconds) - 1)
decode_seconds = sum(step_seconds)
decode_output_tokens = batch * len(step_seconds)
result_path.write_text(json.dumps({
    'token_ids_by_request': generated,
    'input_tokens_per_request': len(ids), 'output_tokens_per_request': output_tokens,
    'batch': batch, 'prefill_seconds': prefill_seconds,
    'prefill_build_seconds': prefill_build_seconds,
    'decode_build_seconds': decode_build_seconds,
    'decode_step_seconds': step_seconds,
    'decode_output_tokens': decode_output_tokens,
    'decode_aggregate_output_tokens_per_second':
        decode_output_tokens / decode_seconds if decode_seconds > 0 else None,
    'warm_decode_seconds': warm_seconds,
    'warm_output_tokens': warm_output_tokens,
    'warm_aggregate_output_tokens_per_second':
        warm_output_tokens / warm_seconds if warm_seconds > 0 else None,
    'total_output_tokens': batch * output_tokens,
    'decode_wall_seconds': decode_seconds,
    'full_request_output_tokens_per_second':
        batch * output_tokens / (prefill_seconds + decode_seconds),
    'measurement': 'homogeneous fixed-size TP2 batch, greedy, no early stop',
}, indent=2))
print('Decode aggregate output tok/s', decode_output_tokens / decode_seconds,
      'warm after first decode tok/s', warm_output_tokens / warm_seconds,
      flush=True)
