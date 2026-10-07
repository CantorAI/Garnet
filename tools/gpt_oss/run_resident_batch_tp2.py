"""XLang3: WEIGHTS CACHE REQUEST RESULT BATCH OUTPUT; measured resident profile via env.

Every trial reruns the entire input and generation. Timers include controls,
input updates, sampling, and phase transition; no logging/memory subprocesses
inside the timed window. Engine loading is separately reported cold startup.
"""
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import garnet as G
from pipeline import build_tensor_parallel, make_tensor_parallel_plan
from garnet_pipeline import ResidentTensorParallel
from resident_budget import (admit_resident, native_identity, hardware_identity,
    kernel_environment, checkpoint_identity, validate_engine_files, file_sha256)

if len(sys.argv) != 7:
    raise ValueError('Expected weights, cache, request, result, batch, output')
weights, cache, request_path, result_path = map(Path, sys.argv[1:5])
batch, output_tokens = map(int, sys.argv[5:])
if result_path.exists():
    raise FileExistsError(result_path)
request = json.loads(request_path.read_text())
ids = request['input_ids']
capacity = int(os.environ['GARNET_BATCH_CONTEXT_CAPACITY'])
chunk = int(os.environ['GARNET_BATCH_PREFILL_CHUNK'])
if not (1 <= batch <= 512 and 16 <= output_tokens <= 2048 and
        0 < chunk <= len(ids) and len(ids) + output_tokens <= capacity <= 4096):
    raise ValueError('Invalid fixed batch/input/output/context profile')
if math.ceil(len(ids) / chunk) * chunk > capacity:
    raise ValueError('Padded final prefill chunk exceeds allocated KV context')
profile_path = Path(os.environ['GARNET_RESIDENT_PROFILE'])
profile = json.loads(profile_path.read_text())
padded = bool(profile['padded_prefill'])
if len(ids) % chunk and not padded:
    raise ValueError('Ragged tail requires validated padded prefill engine')
if any(os.environ.get(key, '0') != '0' for key in
       ('GARNET_GPT_OSS_PROFILE_PREFILL', 'GARNET_GPT_OSS_PROFILE_DECODE_STEPS')):
    raise ValueError('Resident throughput runner requires unprofiled execution')
available = json.loads(G.cuda_devices_json())
selected = request.get('device_ids', [d['id'] for d in available[:2]])
if len(selected) != 2 or len(set(selected)) != 2:
    raise ValueError('Exactly two distinct rank devices required')
devices = [next(d for d in available if d['id'] == index) for index in selected]
plan = make_tensor_parallel_plan(weights, devices, batch=batch, capacity=capacity,
    tokens=chunk, reserve_bytes=int(request.get('reserve_mb', 1024)) << 20,
    memory_fraction=float(request.get('memory_fraction', .9)))
if not ids or any(type(token) is not int or not 0 <= token < plan['config']['vocab_size'] for token in ids):
    raise ValueError('Invalid input token ID')
repo = Path(__file__).resolve().parents[2]
build = Path(os.environ.get('GARNET_BUILD_DIR', repo.parent / 'out/build/gpt-oss'))
admission = admit_resident(profile, plan, devices, binaries=native_identity(build),
    hardware_csv=hardware_identity(), environment=kernel_environment(),
    checkpoint=checkpoint_identity(weights), cache=cache, padded_prefill=padded,
    memory_fraction=float(request.get('memory_fraction', .9)))
validate_engine_files(profile)
print('Resident admission', json.dumps(admission), flush=True)

def memory():
    return [int(x.strip()) for x in subprocess.check_output(['nvidia-smi',
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()]

samples = [memory()]
load_seconds = {}
def load(prefill, kv=None):
    start = time.perf_counter()
    model = build_tensor_parallel(weights, cache, plan, chunk if prefill else 1,
        prefill, kv, last_token_logits=prefill, padded_prefill=prefill and padded)
    load_seconds['prefill' if prefill else 'decode'] = time.perf_counter() - start
    samples.append(memory())
    return model

pair = ResidentTensorParallel.build(lambda: load(True), lambda kv: load(False, kv))
pages = (capacity + 15) // 16
table_data = [row * pages + page for row in range(batch) for page in range(pages)]

def tensors(stages, tokens):
    inputs = []
    for stage in stages:
        previous = G.cuda_set_device(stage['device_id'])
        try:
            def tensor(data, dtype, shape):
                return G.tensor_from_host(data, dtype=dtype, shape=shape, device='cuda')
            inputs.append((tensor([0] * (batch * tokens), 'int64', [batch, tokens]),
                [tensor([0] * (batch * tokens), 'int64', [batch, tokens]),
                 tensor(table_data, 'int32', [batch, pages]),
                 tensor([1] * batch, 'int32', [batch]), tensor([0] * batch, 'int32', [batch]),
                 tensor([1] * batch, 'int32', [batch])]))
        finally:
            G.cuda_set_device(previous)
    return inputs

def update(stages, prepared, tokens, positions, length, slot):
    for stage, (activation, controls) in zip(stages, prepared):
        previous = G.cuda_set_device(stage['device_id'])
        try:
            G.tensor_update_from_host(activation, tokens)
            G.tensor_update_from_host(controls[0], positions)
            G.tensor_update_from_host(controls[2], [length] * batch)
            G.tensor_update_from_host(controls[3], [slot] * batch)
        finally:
            G.cuda_set_device(previous)

prefill_inputs = decode_inputs = None
trials, warmups = [], []
warmup_count = int(os.environ.get('GARNET_RESIDENT_WARMUPS', '1'))
if not 1 <= warmup_count <= 3:
    pair.release()
    raise ValueError('Expected one to three complete warmups; match baseline setting')

def run_request(trial):
    start = time.perf_counter()
    prefill_steps = []
    reply = None
    for offset in range(0, len(ids), chunk):
        step = time.perf_counter()
        real = min(chunk, len(ids) - offset)
        if not 1 <= real <= chunk:
            raise ValueError('Last-valid count must be within prefill sequence')
        # Causal real rows never consume padded future KV, and subsequent
        # decode rewrites that future range. Selection uses true count.
        tokens = (ids[offset:offset + real] + [0] * (chunk - real)) * batch
        positions = list(range(offset, offset + chunk)) * batch
        update(pair.prefill_stages, prefill_inputs, tokens, positions, offset + real, offset)
        reply = pair.forward_prefill(prefill_inputs, sample=True, sample_batch=True)
        prefill_steps.append(time.perf_counter() - step)
    rows = [[int(value)] for value in reply['token_ids']]
    if len(rows) != batch:
        raise ValueError('Missing prefill output requests')
    first_token = time.perf_counter()
    decode_steps = []
    for offset in range(1, output_tokens):
        step = time.perf_counter()
        index = len(ids) + offset - 1
        update(pair.decode_stages, decode_inputs, [row[-1] for row in rows],
            [index] * batch, index + 1, index)
        reply = pair.forward_decode(decode_inputs, sample=True, sample_batch=True)
        tokens = reply['token_ids']
        if len(tokens) != batch:
            raise ValueError('Missing decode output requests')
        for row, token in zip(rows, tokens):
            row.append(int(token))
        decode_steps.append(time.perf_counter() - step)
    finish = time.perf_counter()
    return dict(trial=trial, token_ids_by_request=rows, prefill_kv_reused_for_decode_trial=False,
        prefill_step_seconds=prefill_steps, prefill_seconds=first_token - start,
        decode_step_seconds=decode_steps, decode_wall_seconds=finish - first_token,
        decode_aggregate_output_tokens_per_second=batch * (output_tokens - 1) / (finish - first_token),
        full_request_wall_seconds=finish - start,
        full_request_output_tokens_per_second=batch * output_tokens / (finish - start),
        request_first_token_seconds=[first_token - start] * batch,
        request_completion_seconds=[finish - start] * batch)

try:
    prefill_inputs = tensors(pair.prefill_stages, chunk)
    decode_inputs = tensors(pair.decode_stages, 1)
    for warmup in range(warmup_count):
        measured = run_request(-1 - warmup)
        warmups.append(dict(seconds=measured['full_request_wall_seconds'],
            token_ids_by_request=measured['token_ids_by_request']))
        samples.append(memory())
        print('Complete warmup', warmup, 'seconds', warmups[-1]['seconds'], flush=True)
    for trial in range(3):
        trials.append(run_request(trial))
        samples.append(memory())
        print('Complete trial', trial, 'output tok/s', trials[-1]['full_request_output_tokens_per_second'], flush=True)
finally:
    pair.release()

kv_per_token = plan['config']['num_hidden_layers'] * 2 * plan['local_kv_heads'] * plan['config']['head_dim'] * 2
result_path.write_text(json.dumps(dict(source_commit=subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
    resident_profile=str(profile_path.resolve()), resident_profile_sha256=file_sha256(profile_path),
    resident_admission=admission, native_binaries=profile['native_binaries'],
    optimization_environment={key:value for key,value in os.environ.items()
        if key.startswith(('GARNET_GPT_OSS_', 'GARNET_TP_', 'GARNET_BATCH_', 'GARNET_RESIDENT_'))
        and not key.endswith(('_TOKEN', '_KEY', '_SECRET', '_PASSWORD'))},
    hardware=plan['hardware'], hardware_csv=profile['hardware_csv'], batch=batch,
    input_token_ids=ids, input_tokens_per_request=len(ids), output_tokens_per_request=output_tokens,
    max_context_tokens_per_request=capacity, prefill_chunk_tokens=chunk,
    padded_prefill=padded, padded_tail_tokens=(-len(ids)) % chunk,
    kv_cache_dtype='bfloat16', kv_pages_per_gpu=plan['kv_pages'],
    kv_cache_allocated_bytes_per_gpu=plan['kv_pages'] * 16 * kv_per_token,
    logical_kv_bytes_per_gpu_after_prefill=batch * len(ids) * kv_per_token,
    logical_kv_bytes_per_gpu_at_completion=batch * (len(ids) + output_tokens - 1) * kv_per_token,
    sampled_peak_gpu_memory_mib=[max(row[rank] for row in samples) for rank in range(2)],
    gpu_memory_samples_mib=samples, prefill_build_seconds=load_seconds['prefill'],
    decode_build_seconds=load_seconds['decode'], complete_warmup_count=warmup_count,
    complete_warmups=warmups, decode_trials=trials, token_ids_by_request=trials[0]['token_ids_by_request'],
    prefill_seconds=trials[0]['prefill_seconds'],
    full_request_output_tokens_per_second=trials[0]['full_request_output_tokens_per_second'],
    median_full_request_output_tokens_per_second=statistics.median(t['full_request_output_tokens_per_second'] for t in trials),
    measurement='homogeneous fixed-size TP2 resident batch, greedy, no early stop; three complete warmed requests',
    measurement_limits='Cold engine startup and warmup excluded, as for vLLM; every trial rewrites full input KV and includes controls/input preparation, sampling and phase transition. Sampled memory is not exhaustive peak; serial batches only, no continuous scheduler.'), indent=2))
