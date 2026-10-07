"""XLang3: run_tp_batch_throughput.py WEIGHTS CACHE REQUEST RESULT BATCH OUTPUT_TOKENS.

Repeat one saved prompt across a fixed decode batch. This measures aggregate
output throughput with identical work in every slot and no early-stop bias.
"""
import json
import os
import subprocess
import sys
import time
import ctypes
from pathlib import Path

import garnet as G
from pipeline import build_tensor_parallel, make_tensor_parallel_plan


assert len(sys.argv) == 7, 'expected weights, cache, request, result, batch, output tokens'
weights, cache = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
repo = Path(__file__).resolve().parents[2]
source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo),
                                        text=True).strip()
optimization_environment = {name: value for name, value in os.environ.items()
    if name.startswith(('GARNET_GPT_OSS_', 'GARNET_TP_', 'GARNET_BATCH_'))
    and not name.endswith(('_TOKEN', '_KEY', '_SECRET', '_PASSWORD'))}
request = json.loads(Path(sys.argv[3]).read_text())
result_path = Path(sys.argv[4])
batch, output_tokens = int(sys.argv[5]), int(sys.argv[6])
ids = request['input_ids']
assert 1 <= batch <= 512 and 16 <= output_tokens <= 2048
capacity = int(os.environ.get('GARNET_BATCH_CONTEXT_CAPACITY', '4096'))
assert ids and len(ids) + output_tokens <= capacity <= 4096
configured_chunk = int(os.environ.get('GARNET_BATCH_PREFILL_CHUNK', '0'))
assert 0 <= configured_chunk <= 4096
chunk_limit = configured_chunk or len(ids)
prefill_chunks = [(offset, min(chunk_limit, len(ids) - offset))
                  for offset in range(0, len(ids), chunk_limit)]

available = json.loads(G.cuda_devices_json())
selected = request.get('device_ids', [d['id'] for d in available[:2]])
assert len(selected) == 2 and len(set(selected)) == 2
devices = [next(d for d in available if d['id'] == device_id) for device_id in selected]
plan = make_tensor_parallel_plan(weights, devices, batch=batch,
    capacity=capacity, tokens=min(chunk_limit, len(ids)),
    reserve_bytes=int(request.get('reserve_mb', 1024)) << 20,
    memory_fraction=float(request.get('memory_fraction', .9)))
print('TP2 batch plan', json.dumps({k: plan[k] for k in
    ('cache_key', 'estimated_per_gpu_bytes', 'batch')}), flush=True)
assert all(isinstance(token, int) and 0 <= token < plan['config']['vocab_size']
           for token in ids)
if os.environ.get('GARNET_BATCH_PLAN_ONLY') == '1':
    if result_path.exists():
        raise FileExistsError('Refusing to overwrite admission evidence: ' + str(result_path))
    result_path.write_text(json.dumps({'measurement': 'memory admission only; no engines loaded',
        'source_commit': source_commit, 'plan': plan, 'batch': batch,
        'input_tokens_per_request': len(ids), 'output_tokens_per_request': output_tokens,
        'max_context_tokens_per_request': capacity, 'prefill_chunks': prefill_chunks}, indent=2))
    print('TP2 admission passed without loading engines', flush=True)
    sys.exit(0)


def tensor(data, dtype, shape):
    return G.tensor_from_host(data, dtype=dtype, shape=shape, device='cuda')


def gpu_memory_mib():
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        text=True)
    return [int(value.strip()) for value in output.splitlines()]


G.cuda_set_device(devices[0]['id'])
pages_per_request = (plan['capacity'] + 15) // 16
table_data = [batch_index * pages_per_request + page
              for batch_index in range(batch)
              for page in range(pages_per_request)]
table = tensor(table_data, 'int32', [batch, pages_per_request])
active = tensor([1] * batch, 'int32', [batch])
prefill_build_seconds = prefill_seconds = 0.
prefill_step_seconds = []
warm_prefill = os.environ.get('GARNET_BATCH_PREFILL_WARMUP') == '1'
prefill_warmup_step_seconds = []
prefill_engine_memory_samples_mib = []
prefill_completed_memory_samples_mib = []
profile_prefill = os.environ.get('GARNET_GPT_OSS_PROFILE_PREFILL') == '1'
profile_prefill_chunk = int(os.environ.get('GARNET_GPT_OSS_PROFILE_PREFILL_CHUNK', '0'))
assert 0 <= profile_prefill_chunk < len(prefill_chunks)
profile_steps = int(os.environ.get('GARNET_GPT_OSS_PROFILE_DECODE_STEPS', '0'))
profile_start = int(os.environ.get('GARNET_GPT_OSS_PROFILE_DECODE_START', '10'))
assert profile_steps >= 0
if profile_steps:
    assert 1 <= profile_start and profile_start + profile_steps <= output_tokens
profiler = ctypes.CDLL('libcudart.so') if profile_prefill or profile_steps else None
model = kv = prefill = None
loaded_tokens = None
for chunk_index, (offset, chunk_tokens) in enumerate(prefill_chunks):
    if chunk_tokens != loaded_tokens:
        if model is not None:
            kv = [(stage['keys'], stage['values']) for stage in model.stages]
            model.release()
        started = time.perf_counter()
        print('Building TP2 batch prefill engines for', chunk_tokens,
              'tokens', flush=True)
        model = build_tensor_parallel(weights, cache, plan, chunk_tokens, True,
                                      kv, last_token_logits=True)
        prefill_build_seconds += time.perf_counter() - started
        loaded_tokens = chunk_tokens
    prefill_engine_memory_samples_mib.append(gpu_memory_mib())
    print('TP2 prefill memory before execution MiB',
          prefill_engine_memory_samples_mib[-1], flush=True)
    chunk_ids = ids[offset:offset + chunk_tokens]
    input_ids = tensor(chunk_ids * batch, 'int64', [batch, chunk_tokens])
    positions = tensor(list(range(offset, offset + chunk_tokens)) * batch,
                       'int64', [batch, chunk_tokens])
    length = tensor([offset + chunk_tokens] * batch, 'int32', [batch])
    slot = tensor([offset] * batch, 'int32', [batch])
    if warm_prefill:
        # vLLM warms the same request before measurement. Rewriting this
        # chunk's identical KV slots makes its graph/packing work warm too;
        # earlier chunks retain their measured causal context.
        warm_started = time.perf_counter()
        model.forward(input_ids, [positions, table, length, slot, active],
                      True, sample_batch=True)
        prefill_warmup_step_seconds.append(time.perf_counter() - warm_started)
    capture_prefill = profile_prefill and chunk_index == profile_prefill_chunk
    if capture_prefill and profiler.cudaProfilerStart() != 0:
        raise RuntimeError('cudaProfilerStart failed for prefill')
    started = time.perf_counter()
    try:
        prefill = model.forward(input_ids, [positions, table, length, slot, active],
                                True, sample_batch=True)
    finally:
        step = time.perf_counter() - started
        if capture_prefill:
            status = profiler.cudaProfilerStop()
            if status:
                print('cudaProfilerStop failed for prefill:', status,
                      file=sys.stderr, flush=True)
    prefill_seconds += step
    prefill_step_seconds.append(step)
    prefill_completed_memory_samples_mib.append(gpu_memory_mib())
    print('TP2 batch prefill chunk', offset, chunk_tokens, 'complete',
          step, flush=True)
prefill_engine_memory_mib = prefill_engine_memory_samples_mib[-1]
prefill_completed_memory_mib = prefill_completed_memory_samples_mib[-1]
generated = [[int(token)] for token in prefill['token_ids']]
assert len(generated) == batch, (len(generated), batch)
print('TP2 batch prefill complete', prefill_seconds, flush=True)
kv = [(stage['keys'], stage['values']) for stage in model.stages]
model.release()

started = time.perf_counter()
print('Building TP2 batch decode engines', flush=True)
model = build_tensor_parallel(weights, cache, plan, 1, False, kv)
decode_build_seconds = time.perf_counter() - started
decode_engine_memory_mib = gpu_memory_mib()
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

inline_batch = batch <= 64 and os.environ.get('GARNET_GPT_OSS_INLINE_BATCH_CONTROLS') == '1'
inline_single = not inline_batch and batch == 1 and os.environ.get('GARNET_GPT_OSS_INLINE_CONTROL_KERNEL') == '1'


def forward_decode_step(tokens, index):
    if not inline_single and not inline_batch:
        for stage, (local_token, controls) in zip(model.stages, rank_inputs):
            previous = G.cuda_set_device(stage['device_id'])
            try:
                G.tensor_update_from_host(local_token, tokens)
                G.tensor_update_from_host(controls[0], [index] * batch)
                G.tensor_update_from_host(controls[2], [index + 1] * batch)
                G.tensor_update_from_host(controls[3], [index] * batch)
            finally:
                G.cuda_set_device(previous)
    return model.forward_rank_local(rank_inputs, True,
        scalar_values=[tokens[0], index, index + 1, index] if inline_single and not inline_batch else None,
        vector_values=[tokens, [index] * batch, [index + 1] * batch, [index] * batch]
            if inline_batch else None,
        sample_batch=True)


# Warm the decode engine and its rank-local expert preparation once, just as
# the vLLM runner executes a warmup batch before its timed batch. This writes
# the first decode KV slot; the timed first step overwrites that same slot.
started = time.perf_counter()
forward_decode_step([row[-1] for row in generated], len(ids))
decode_warmup_seconds = time.perf_counter() - started
memory_samples_mib = [gpu_memory_mib()]
trial_count = int(os.environ.get('GARNET_BATCH_DECODE_TRIALS', '1'))
assert 1 <= trial_count <= 5
initial_generated = [row[:] for row in generated]
decode_trials = []


def run_decode_trial(trial):
    rows = [row[:] for row in initial_generated]
    steps = []
    profiling = False
    trial_started = warm_started = time.perf_counter()
    for offset in range(1, output_tokens):
        if profile_steps and trial == 0 and offset == profile_start:
            if profiler.cudaProfilerStart() != 0:
                raise RuntimeError('cudaProfilerStart failed')
            profiling = True
        tokens = [row[-1] for row in rows]
        index = len(ids) + offset - 1
        started = time.perf_counter()
        reply = forward_decode_step(tokens, index)
        sampled = [int(value) for value in reply['token_ids']]
        assert len(sampled) == batch, (len(sampled), batch)
        for row, value in zip(rows, sampled):
            row.append(value)
        steps.append(time.perf_counter() - started)
        if offset == 1:
            warm_started = time.perf_counter()
        if profiling and offset == profile_start + profile_steps - 1:
            if profiler.cudaProfilerStop() != 0:
                raise RuntimeError('cudaProfilerStop failed')
            profiling = False
    finished = time.perf_counter()
    if profiling and profiler.cudaProfilerStop() != 0:
        raise RuntimeError('cudaProfilerStop failed')
    return {'trial': trial, 'token_ids_by_request': rows,
        'decode_step_seconds': steps, 'decode_call_seconds_sum': sum(steps),
        'decode_wall_seconds': finished - trial_started,
        'warm_decode_seconds': finished - warm_started,
        'prefill_kv_reused_for_decode_trial': trial > 0,
        'decode_aggregate_output_tokens_per_second':
            batch * len(steps) / (finished - trial_started)}


for trial in range(trial_count):
    # Later trials reuse only the original input KV. Each decode writes the
    # same positions afresh; context lengths hide any leftover future slots.
    # These are repeated decode trials, not additional full-request trials.
    measured = run_decode_trial(trial)
    decode_trials.append(measured)
    memory_samples_mib.append(gpu_memory_mib())
    print('Decode trial', trial, 'aggregate output tok/s',
          measured['decode_aggregate_output_tokens_per_second'], flush=True)
model.release()

generated = decode_trials[0]['token_ids_by_request']
step_seconds = decode_trials[0]['decode_step_seconds']
warm_seconds = decode_trials[0]['warm_decode_seconds']
warm_output_tokens = batch * max(0, len(step_seconds) - 1)
decode_seconds = decode_trials[0]['decode_wall_seconds']
decode_output_tokens = batch * len(step_seconds)
kv_bytes_per_token_per_gpu = (plan['config']['num_hidden_layers'] * 2 *
    plan['local_kv_heads'] * plan['config']['head_dim'] * 2)
all_memory_samples = (prefill_engine_memory_samples_mib +
    prefill_completed_memory_samples_mib + [decode_engine_memory_mib] + memory_samples_mib)
result_path.write_text(json.dumps({
    'source_commit': source_commit,
    'optimization_environment': optimization_environment,
    'hardware': plan['hardware'],
    'estimated_per_gpu_bytes': plan['estimated_per_gpu_bytes'],
    'expert_weight_shards': plan.get('expert_weight_shards', False),
    'compact_vocab_greedy': plan.get('compact_vocab_greedy', False),
    'weight_storage_estimate': plan.get('weight_storage_estimate'),
    'marlin_workspace_layout': plan.get('marlin_workspace_layout'),
    'collective_workspace_layout': plan.get('collective_workspace_layout'),
    'memory_budget_per_gpu_bytes': [stage['budget_bytes'] for stage in plan['stages']],
    'token_ids_by_request': generated,
    'identical_output_across_duplicate_requests':
        all(row == generated[0] for row in generated),
    'input_tokens_per_request': len(ids), 'output_tokens_per_request': output_tokens,
    'input_token_ids': ids,
    'batch': batch, 'prefill_seconds': prefill_seconds,
    'prefill_chunk_tokens': configured_chunk,
    'prefill_chunks': prefill_chunks,
    'prefill_step_seconds': prefill_step_seconds,
    'prefill_warmup_step_seconds_excluded': prefill_warmup_step_seconds,
    'prefill_measurement': ('one identical-KV warmup per chunk before timed execution'
                            if warm_prefill else 'first execution after engine loading'),
    'profile_prefill_chunk': profile_prefill_chunk if profile_prefill else None,
    'max_context_tokens_per_request': capacity,
    'kv_cache_allocated_bytes_per_gpu': (
        plan['config']['num_hidden_layers'] * 2 * plan['kv_pages'] * 16 *
        plan['local_kv_heads'] * plan['config']['head_dim'] * 2),
    'kv_pages_per_gpu': plan['kv_pages'],
    'kv_cache_dtype': 'bfloat16',
    'kv_bytes_per_token_per_gpu': kv_bytes_per_token_per_gpu,
    'logical_kv_tokens_per_request_at_completion': len(ids) + output_tokens - 1,
    'logical_kv_bytes_per_gpu_after_prefill': batch * len(ids) * kv_bytes_per_token_per_gpu,
    'logical_kv_bytes_per_gpu_at_completion':
        batch * (len(ids) + output_tokens - 1) * kv_bytes_per_token_per_gpu,
    'sampled_peak_gpu_memory_mib': [max(row[rank] for row in all_memory_samples)
                                   for rank in range(len(devices))],
    'gpu_memory_mib_after_prefill_engine': prefill_engine_memory_mib,
    'gpu_memory_mib_after_prefill': prefill_completed_memory_mib,
    'gpu_memory_mib_after_prefill_engine_by_chunk':
        prefill_engine_memory_samples_mib,
    'gpu_memory_mib_after_prefill_by_chunk':
        prefill_completed_memory_samples_mib,
    'gpu_memory_mib_after_decode_engine': decode_engine_memory_mib,
    'gpu_memory_mib_during_decode': memory_samples_mib,
    'prefill_build_seconds': prefill_build_seconds,
    'decode_build_seconds': decode_build_seconds,
    'decode_warmup_seconds_excluded': decode_warmup_seconds,
    'decode_step_seconds': step_seconds,
    'decode_call_seconds_sum': sum(step_seconds),
    'decode_trials': decode_trials,
    'inline_single_request_controls': inline_single,
    'inline_batch_controls': inline_batch,
    'profile_decode_steps': profile_steps,
    'profile_decode_start': profile_start if profile_steps else None,
    'profiler_output': os.environ.get('GARNET_BENCH_NSYS_OUTPUT'),
    'profile_reference_result': os.environ.get('GARNET_BENCH_REFERENCE_JSON'),
    'profile_prefill': profile_prefill,
    'decode_output_tokens': decode_output_tokens,
    'decode_aggregate_output_tokens_per_second':
        decode_output_tokens / decode_seconds if decode_seconds > 0 else None,
    'warm_decode_seconds': warm_seconds,
    'warm_output_tokens': warm_output_tokens,
    'warm_aggregate_output_tokens_per_second':
        warm_output_tokens / warm_seconds if warm_seconds > 0 else None,
    'total_output_tokens': batch * output_tokens,
    'decode_wall_seconds': decode_seconds,
    'request_completion_seconds': [prefill_seconds + decode_seconds] * batch,
    'full_request_output_tokens_per_second':
        batch * output_tokens / (prefill_seconds + decode_seconds),
    'measurement': 'homogeneous fixed-size TP2 batch, greedy, no early stop; actual decode wall window without progress logging or GPU-memory subprocesses; one decode warmup excluded',
    'measurement_limits': 'Execution throughput excludes prefill/decode engine loading and handoff; later decode trials reuse original input KV and do not measure another prefill. Sampled GPU memory is not an exhaustive allocation peak.',
}, indent=2))
print('Decode aggregate output tok/s', decode_output_tokens / decode_seconds,
      'warm after first decode tok/s', warm_output_tokens / warm_seconds,
      flush=True)
