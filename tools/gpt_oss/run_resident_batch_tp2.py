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
from kv_layout import kv_memory, logical_retained_kv_bytes
from resident_capture import ResidentCapture
from resident_session import read_session, session_partial
from resident_budget import (admit_resident, native_identity, hardware_identity,
    kernel_environment, checkpoint_identity, validate_engine_files, file_sha256)

startup_started = time.perf_counter()

session_mode = len(sys.argv) > 1 and sys.argv[1] == '--session'
arguments = sys.argv[2:] if session_mode else sys.argv[1:]
if len(arguments) != 6:
    raise ValueError('Expected weights, cache, request, result, batch, output')
weights, cache, request_path, result_path = map(Path, arguments[:4])
batch, output_tokens = map(int, arguments[4:])
if result_path.exists():
    raise FileExistsError(result_path)
capacity = int(os.environ['GARNET_BATCH_CONTEXT_CAPACITY'])
chunk = int(os.environ['GARNET_BATCH_PREFILL_CHUNK'])
profile_path = Path(os.environ['GARNET_RESIDENT_PROFILE'])
session_result_path, session_metadata = result_path, None
if session_mode:
    if os.environ.get('GARNET_BATCH_PLAN_ONLY') == '1':
        raise ValueError('Session inference cannot be selected for admission-only mode')
    cases, session_metadata = read_session(request_path, session_result_path,
        batch, output_tokens, capacity, chunk, protected_paths=[profile_path])
    request = cases[0]['request']
else:
    request = json.loads(request_path.read_text())
    cases = [dict(name='single-request', request_path=request_path,
        result_path=result_path, request=request, input_ids=request['input_ids'])]
ids = cases[0]['input_ids']
if not (1 <= batch <= 512 and 16 <= output_tokens <= 2048 and
        0 < chunk <= len(ids) and len(ids) + output_tokens <= capacity <= 4096):
    raise ValueError('Invalid fixed batch/input/output/context profile')
if math.ceil(len(ids) / chunk) * chunk > capacity:
    raise ValueError('Padded final prefill chunk exceeds allocated KV context')
profile = json.loads(profile_path.read_text())
padded = bool(profile['padded_prefill'])
if len(ids) % chunk and not padded:
    raise ValueError('Ragged tail requires validated padded prefill engine')
capture = ResidentCapture(os.environ, len(ids), chunk, output_tokens)
if session_mode and (capture.enabled or os.environ.get('GARNET_BENCH_NSYS_OUTPUT')):
    raise ValueError('Session throughput requires uninstrumented requests')
warmup_count = int(os.environ.get('GARNET_RESIDENT_WARMUPS', '1'))
if not 1 <= warmup_count <= 3 or (session_mode and warmup_count != 1):
    raise ValueError('Expected one complete warmup for a session, or one to three for a single case')
available = json.loads(G.cuda_devices_json())
selected = request.get('device_ids', [d['id'] for d in available[:2]])
if len(selected) != 2 or len(set(selected)) != 2:
    raise ValueError('Exactly two distinct rank devices required')
devices = [next(d for d in available if d['id'] == index) for index in selected]
plan = make_tensor_parallel_plan(weights, devices, batch=batch, capacity=capacity,
    tokens=chunk, reserve_bytes=int(request.get('reserve_mb', 1024)) << 20,
    memory_fraction=float(request.get('memory_fraction', .9)))
if any(not case['input_ids'] or any(type(token) is not int or
       not 0 <= token < plan['config']['vocab_size'] for token in case['input_ids']) for case in cases):
    raise ValueError('Invalid input token ID')
repo = Path(__file__).resolve().parents[2]
# The embedded Python subprocess bridge requires a string cwd. Resolve this
# metadata before GPU work, so a reporting failure cannot discard a trial.
source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
    cwd=str(repo), text=True).strip()
if not session_mode and result_path.with_suffix('.trials.partial.json').exists():
    raise FileExistsError(result_path.with_suffix('.trials.partial.json'))
build = Path(os.environ.get('GARNET_BUILD_DIR', repo.parent / 'out/build/gpt-oss'))
admission = admit_resident(profile, plan, devices, binaries=native_identity(build),
    hardware_csv=hardware_identity(), environment=kernel_environment(),
    checkpoint=checkpoint_identity(weights), cache=cache, padded_prefill=padded,
    memory_fraction=float(request.get('memory_fraction', .9)))
validate_engine_files(profile)
admission_seconds = time.perf_counter() - startup_started
print('Resident admission', json.dumps(admission), flush=True)
if os.environ.get('GARNET_BATCH_PLAN_ONLY') == '1':
    result_path.write_text(json.dumps(dict(
        measurement='Strict resident admission only; no engines loaded or inference',
        admission_only=True, source_commit=source_commit, plan=plan, admission=admission,
        batch=batch, input_token_ids=ids, input_tokens_per_request=len(ids),
        output_tokens_per_request=output_tokens, max_context_tokens_per_request=capacity,
        prefill_chunk_tokens=chunk, admission_seconds=admission_seconds),indent=2))
    print('Strict measured resident admission passed; paired residency and quality unproven',flush=True)
    sys.exit(0)

def memory():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    by_device = {int(index.strip()): int(value.strip()) for index,value in (row.split(',') for row in rows)}
    return [by_device[device['id']] for device in devices]

samples = [memory()]
load_seconds = {}
def load(prefill, kv=None):
    print('Resident phase load start', 'prefill' if prefill else 'decode', flush=True)
    start = time.perf_counter()
    model = build_tensor_parallel(weights, cache, plan, chunk if prefill else 1,
        prefill, kv, last_token_logits=prefill, padded_prefill=prefill and padded)
    try:
        load_seconds['prefill' if prefill else 'decode'] = time.perf_counter() - start
        samples.append(memory())
        print('Resident phase load complete', 'prefill' if prefill else 'decode',
              'seconds', load_seconds['prefill' if prefill else 'decode'], flush=True)
    except BaseException:
        model.release()
        raise
    return model

pages = (capacity + 15) // 16
table_data = [row * pages + page for row in range(batch) for page in range(pages)]
kv_per_token = plan['config']['num_hidden_layers'] * 2 * plan['local_kv_heads'] * plan['config']['head_dim'] * 2
kv_allocated, auxiliary_allocated = kv_memory(plan)
session_report = (dict(**session_metadata, source_commit=source_commit,
    session_complete=False, cases=[], active_case=None, phase='engine-startup',
    resident_profile=str(profile_path.resolve()),
    resident_profile_sha256=file_sha256(profile_path), resident_admission=admission,
    native_binaries=profile['native_binaries'], hardware_csv=profile['hardware_csv'])
    if session_mode else None)
if session_report is not None:
    session_partial(session_result_path).write_text(json.dumps(session_report, indent=2))
try:
    pair = ResidentTensorParallel.build(lambda: load(True), lambda kv: load(False, kv))
except BaseException as error:
    if session_report is not None:
        session_report['failure'] = dict(type=type(error).__name__, message=str(error))
        session_partial(session_result_path).write_text(json.dumps(session_report, indent=2))
    raise

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

def run_request(trial, ids):
    start = time.perf_counter()
    prefill_steps = []
    reply = None
    for chunk_index, offset in enumerate(range(0, len(ids), chunk)):
        capture.prefill_begin(trial, chunk_index)
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
        capture.prefill_end(trial, chunk_index)
        prefill_steps.append(time.perf_counter() - step)
    rows = [[int(value)] for value in reply['token_ids']]
    if len(rows) != batch:
        raise ValueError('Missing prefill output requests')
    first_token = time.perf_counter()
    decode_steps = []
    for offset in range(1, output_tokens):
        capture.decode_begin(trial, offset)
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
        capture.decode_end(trial, offset)
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

def measure_case(case, index, cold_startup_seconds):
    case_started = time.perf_counter()
    ids, result_path = case['input_ids'], case['result_path']
    partial_path = result_path.with_suffix('.trials.partial.json')
    # Include the shared startup samples in each admission/peak check, but
    # keep this case's execution samples separate from earlier prompts.
    case_samples = list(samples)
    if session_mode:
        case_samples.append(memory())
        print('Session case start', index, case['name'], flush=True)
    warmups, trials = [], []
    case_preparation_seconds = time.perf_counter() - case_started
    session_case = (dict(**session_metadata, case_index=index, case_name=case['name'],
        request=str(case['request_path'].resolve()), request_sha256=case['request_sha256'],
        engine_pair_reused=index > 0, case_preparation_seconds=case_preparation_seconds,
        startup_scope='Session shared cold preparation is reported once in session_result')
        if session_mode else {})
    for warmup in range(warmup_count):
        print('Complete warmup start', warmup, flush=True)
        measured = run_request(-1 - warmup, ids)
        warmups.append(dict(seconds=measured['full_request_wall_seconds'],
            token_ids_by_request=measured['token_ids_by_request']))
        case_samples.append(memory())
        print('Complete warmup', warmup, 'seconds', warmups[-1]['seconds'], flush=True)
    for trial in range(3):
        trials.append(run_request(trial, ids))
        case_samples.append(memory())
        partial_path.write_text(json.dumps(dict(source_commit=source_commit,
            batch=batch, input_token_ids=ids, input_tokens_per_request=len(ids),
            output_tokens_per_request=output_tokens, max_context_tokens_per_request=capacity,
            measurement='Incomplete resident run; preserve completed trial matrices, not a success claim',
            complete_warmups=warmups, decode_trials=trials, gpu_memory_samples_mib=case_samples,
            **session_case, **capture.metadata()), indent=2))
        print('Complete trial', trial, 'output tok/s', trials[-1]['full_request_output_tokens_per_second'], flush=True)
    capture.assert_complete()
    peaks = [max(row[rank] for row in case_samples) for rank in range(2)]
    if any(peak * (1 << 20) > admitted['budget_bytes'] for peak, admitted in zip(peaks, admission['ranks'])):
        result_path.with_suffix('.memory-failure.json').write_text(json.dumps(
            dict(admission=admission, samples_mib=case_samples, peaks_mib=peaks), indent=2))
        raise RuntimeError('Observed resident memory exceeds admitted budget')
    layout_report = (dict(kv_layout=plan['kv_layout'],
        shared_kv_auxiliary_bytes_per_gpu=auxiliary_allocated,
        kv_page_counts_per_bank=dict(global_bank=plan['kv_pages'],
            window_bank=plan['kv_layout']['window_shape'][1]),
        logical_retained_kv_bytes_per_gpu_after_prefill=logical_retained_kv_bytes(plan, len(ids)),
        logical_retained_kv_bytes_per_gpu_at_completion=logical_retained_kv_bytes(plan, len(ids)+output_tokens-1),
        logical_kv_description='Full-history equivalent token volume; window bank recycles expired history')
        if 'kv_layout' in plan else {})
    # Session cases deliberately omit repeated cold-load fields. Their full
    # request timings remain directly comparable to the single-case runner.
    startup_report = (dict(prefill_build_seconds=load_seconds['prefill'],
        decode_build_seconds=load_seconds['decode'],
        admission_and_engine_checksum_seconds=admission_seconds,
        cold_startup_seconds_excluding_module_imports=cold_startup_seconds)
        if not session_mode else session_case)
    result_path.write_text(json.dumps(dict(source_commit=source_commit,
        resident_profile=str(profile_path.resolve()), resident_profile_sha256=file_sha256(profile_path),
        resident_admission=admission, native_binaries=profile['native_binaries'],
        optimization_environment={'GARNET_TRT_SYNC_ALLOCATOR': os.environ.get('GARNET_TRT_SYNC_ALLOCATOR', '0'),
            **{key:value for key,value in os.environ.items()
            if key.startswith(('GARNET_GPT_OSS_', 'GARNET_TP_', 'GARNET_BATCH_', 'GARNET_RESIDENT_'))
            and not key.endswith(('_TOKEN', '_KEY', '_SECRET', '_PASSWORD'))}},
        hardware=plan['hardware'], hardware_csv=profile['hardware_csv'], batch=batch,
        input_token_ids=ids, input_tokens_per_request=len(ids), output_tokens_per_request=output_tokens,
        max_context_tokens_per_request=capacity, prefill_chunk_tokens=chunk,
        padded_prefill=padded, padded_tail_tokens=(-len(ids)) % chunk,
        kv_cache_dtype='bfloat16', kv_pages_per_gpu=plan['kv_pages'],
        kv_cache_allocated_bytes_per_gpu=kv_allocated,
        logical_kv_bytes_per_gpu_after_prefill=batch * len(ids) * kv_per_token,
        logical_kv_bytes_per_gpu_at_completion=batch * (len(ids) + output_tokens - 1) * kv_per_token,
        sampled_peak_gpu_memory_mib=peaks,
        gpu_memory_samples_mib=case_samples, complete_warmup_count=warmup_count,
        complete_warmups=warmups, decode_trials=trials, token_ids_by_request=trials[0]['token_ids_by_request'],
        prefill_seconds=trials[0]['prefill_seconds'],
        full_request_output_tokens_per_second=trials[0]['full_request_output_tokens_per_second'],
        median_full_request_output_tokens_per_second=statistics.median(t['full_request_output_tokens_per_second'] for t in trials),
        measurement=('Instrumented resident diagnostic; timings are NOT benchmark evidence' if capture.enabled else
            'homogeneous fixed-size TP2 resident batch, greedy, no early stop; three complete warmed requests'),
        measurement_limits=('Nsight captures perturb timings; do not compare these rates to vLLM or accept them as an unprofiled reference. ' if capture.enabled else '') +
            'Cold engine startup and warmup excluded, as for vLLM; every trial rewrites full input KV and includes controls/input preparation, sampling and phase transition. Sampled memory is not exhaustive peak; serial batches only, no continuous scheduler.',
        **startup_report, **layout_report, **capture.metadata()), indent=2))
    completed = dict(name=case['name'], result=str(result_path.resolve()),
        result_sha256=file_sha256(result_path), sampled_peak_gpu_memory_mib=peaks,
        median_full_request_output_tokens_per_second=statistics.median(
            trial['full_request_output_tokens_per_second'] for trial in trials))
    if session_mode:
        if (file_sha256(case['expected_path']) != case['expected_sha256'] or
                file_sha256(Path(session_metadata['tokenizer']) / 'tokenizer.json') != session_metadata['tokenizer_sha256']):
            raise RuntimeError('Session answer reference or tokenizer changed after preflight')
        subprocess.run([session_metadata['validation_python'],
            str(repo / 'tools/gpt_oss/validate_batch_results.py'), session_metadata['tokenizer'],
            str(result_path), str(case['expected_path']), str(case['validation_path'])], check=True)
        validation = json.loads(case['validation_path'].read_text())
        outcomes = validation.get('slots', [])
        if (validation.get('all_pass') is not True or len(outcomes) != 3 * batch or
                {(item['trial'], item['slot']) for item in outcomes} !=
                {(trial, slot) for trial in range(3) for slot in range(batch)} or
                any(item.get('pass') is not True for item in outcomes)):
            raise RuntimeError('Session case did not pass every answer before next case')
        completed.update(validation=str(case['validation_path'].resolve()),
            validation_sha256=file_sha256(case['validation_path']),
            expected_sha256=case['expected_sha256'], answer_checks=len(outcomes))
    return completed


try:
    prefill_inputs = tensors(pair.prefill_stages, chunk)
    decode_inputs = tensors(pair.decode_stages, 1)
    cold_startup_seconds = time.perf_counter() - startup_started
    if session_mode:
        session_report.update(phase='requests',
            prefill_build_seconds=load_seconds['prefill'], decode_build_seconds=load_seconds['decode'],
            admission_and_engine_checksum_seconds=admission_seconds,
            cold_startup_seconds_excluding_module_imports=cold_startup_seconds,
            measurement='One admitted pair reused for serial same-shape cases; no continuous scheduling',
            measurement_limits='Cold startup shared once; each case has one complete warmup and three full requests. No input KV reuse; sampled memory is not exhaustive. Every case answer is validated before advancing; performance requires separate matched comparison.')
    for index, case in enumerate(cases):
        if session_mode:
            session_report['active_case'] = case['name']
            session_partial(session_result_path).write_text(json.dumps(session_report, indent=2))
        completed = measure_case(case, index, cold_startup_seconds)
        if session_mode:
            session_report['cases'].append(completed)
            session_report['active_case'] = None
            session_partial(session_result_path).write_text(json.dumps(session_report, indent=2))
except BaseException as error:
    if session_report is not None:
        session_report['failure'] = dict(type=type(error).__name__, message=str(error))
        session_partial(session_result_path).write_text(json.dumps(session_report, indent=2))
    raise
finally:
    try:
        capture.abort()
    finally:
        pair.release()
if session_mode:
    session_report['session_complete'] = True
    session_report['phase'] = 'complete'
    session_report['sampled_peak_gpu_memory_mib'] = [
        max(case['sampled_peak_gpu_memory_mib'][rank] for case in session_report['cases']) for rank in range(2)]
    session_result_path.write_text(json.dumps(session_report, indent=2))
    print('Resident session complete; all case answers passed; performance requires matched audit', flush=True)
