"""XLang3: WEIGHTS CACHE REQUEST RESULT BATCH OUTPUT; measured resident profile via env.

Every trial reruns the entire input and generation. Timers include controls,
input updates, sampling, and phase transition; no logging/memory subprocesses
inside the timed window. Engine loading is separately reported cold startup.
"""
import json
import hashlib
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
from resident_runtime_scheduling import POLICY as runtime_switch_policy, parse_switch_us, RuntimeSwitchScope
from token_budget_scheduler import QueuedPrefill, plan_token_budget_iteration
from resident_budget import (admit_resident, native_identity, hardware_identity,
    kernel_environment, checkpoint_identity, validate_engine_files, file_sha256,
    validate_runtime_candidate_shape)

startup_started = time.perf_counter()

def optional_flag(name):
    value = os.environ.get(name, '0')
    if value not in ('0', '1'):
        raise ValueError(name + ' must be 0 or 1')
    return value == '1'

# Host-lifecycle choices do not change graph math or the KV profile. A new
# native binary still requires an exact matching measured resident profile.
reuse_output = optional_flag('GARNET_RESIDENT_REUSE_OUTPUT')
native_candidate_merge = optional_flag('GARNET_RESIDENT_NATIVE_GREEDY_MERGE')
final_prefill_sample_only = optional_flag('GARNET_RESIDENT_FINAL_PREFILL_SAMPLE_ONLY')
pattern_updates = optional_flag('GARNET_RESIDENT_PATTERN_UPDATES')
native_greedy_state = optional_flag('GARNET_RESIDENT_NATIVE_GREEDY_STATE')
token_budget_scheduler = optional_flag('GARNET_RESIDENT_TOKEN_BUDGET_SCHEDULER')
if native_greedy_state and native_candidate_merge:
    raise ValueError('Native greedy state replaces the independent native merge path')
resident_host_candidates = dict(reuse_output=reuse_output,
    native_candidate_merge=native_candidate_merge,
    final_prefill_sample_only=final_prefill_sample_only,
    pattern_updates=pattern_updates, native_greedy_state=native_greedy_state,
    token_budget_scheduler=token_budget_scheduler)
runtime_switch_us = parse_switch_us(os.environ.get(runtime_switch_policy, '0'))

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
    # Fail on host-package/tokenizer problems before plan/checksum/engine load.
    # The actual validator uses the caller's preserved venv path.
    subprocess.run([session_metadata['validation_python'],
        str(Path(__file__).resolve().parent/'validate_batch_results.py'),
        '--preflight', session_metadata['tokenizer']], check=True)
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
if pattern_updates and (2 * chunk + 2 > 768 or batch + 3 > 768):
    raise ValueError('Resident integer patterns exceed the bounded native payload')
profile = json.loads(profile_path.read_text())
padded = bool(profile['padded_prefill'])
phase_batches = profile.get('phase_engine_batches',
    dict(prefill=batch, decode=batch))
if (not isinstance(phase_batches, dict) or set(phase_batches) != {'prefill','decode'} or
        any(type(value) is not int or not 1 <= value <= batch for value in phase_batches.values())):
    raise ValueError('Resident profile has invalid phase engine batches')
validate_runtime_candidate_shape(profile, batch=batch, input_tokens=len(ids),
    output_tokens=output_tokens, context=capacity, prefill_chunk=chunk,
    phase_engine_batches=phase_batches)
if token_budget_scheduler:
    if (session_mode or batch != 224 or len(ids) != 2005 or output_tokens != 1024 or
            capacity != 3072 or
            phase_batches != dict(prefill=224, decode=224) or
            not profile.get('candidate_shape') or
            profile['candidate_shape'].get('prefill_chunk_tokens') != chunk or
            profile['candidate_shape'].get('input_ids_sha256') != hashlib.sha256(
                json.dumps(ids, separators=(',', ':')).encode()).hexdigest() or
            (len(ids) % chunk and not padded) or
            min(112, batch) * chunk > 4096 or
            reuse_output or native_candidate_merge or final_prefill_sample_only or
            pattern_updates or native_greedy_state or runtime_switch_us):
        raise ValueError('Token-budget scheduler requires the exact qualified-shape candidate gates')
elif phase_batches != dict(prefill=batch, decode=batch):
    raise ValueError('Split phase profiles require explicit token-budget scheduler opt-in')
if len(ids) % chunk and not padded:
    raise ValueError('Ragged tail requires validated padded prefill engine')
capture = ResidentCapture(os.environ, len(ids), chunk, output_tokens)
if token_budget_scheduler and capture.enabled:
    raise ValueError('Token-budget scheduler cannot be combined with instrumented captures')
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
    memory_fraction=float(request.get('memory_fraction', .9)),
    prefill_engine_batch=phase_batches['prefill'],
    decode_engine_batch=phase_batches['decode'])
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
        prefill, kv, last_token_logits=prefill, padded_prefill=prefill and padded,
        engine_batch=phase_batches['prefill' if prefill else 'decode'])
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

peer_group_status = None
group = None
try:
    if 'operator_execution_layout' in plan:
        import garnet_gpt_oss as extension
        layout = plan['operator_execution_layout']
        group = extension.peer_group(json.dumps(dict(phase_elements=layout['phase_elements'], ctas=layout['ctas'])))
        peer_group_status = json.loads(extension.peer_group_status_json(group))
        if peer_group_status != layout:
            raise ValueError('Native peer resources differ from admitted contract')
        pair.attach_operator_execution_group(group, extension.peer_group_bind_phase,
            extension.peer_group_release)
        print('Resident native execution group', json.dumps(peer_group_status), flush=True)
        samples.append(memory())
except BaseException:
    try:
        pair.release()
    finally:
        if group is not None:
            extension.peer_group_release(group)
        group = None
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

def phase_tensors(stages, tokens, slots):
    """Create fixed-shape cohort inputs with page tables for global KV slots."""
    inputs = []
    page_table = [row * pages + page for row in slots for page in range(pages)]
    for stage in stages:
        previous = G.cuda_set_device(stage['device_id'])
        try:
            count = len(slots)
            def tensor(data, dtype, shape):
                return G.tensor_from_host(data, dtype=dtype, shape=shape, device='cuda')
            inputs.append((tensor([0] * (count * tokens), 'int64', [count, tokens]),
                [tensor([0] * (count * tokens), 'int64', [count, tokens]),
                 tensor(page_table, 'int32', [count, pages]),
                 tensor([1] * count, 'int32', [count]),
                 tensor(list(slots), 'int32', [count]),
                 tensor([1] * count, 'int32', [count])]))
        finally:
            G.cuda_set_device(previous)
    return inputs

prefill_inputs = decode_inputs = None
prefill_cohort_inputs = None

def run_token_budget_request(trial, ids, start):
    """Decode active requests first, then FIFO-prefill up to one bounded cohort."""
    request_ids = [f'q{slot:03d}' for slot in range(batch)]
    slots = {request_id: slot for slot, request_id in enumerate(request_ids)}
    offsets = {request_id: 0 for request_id in request_ids}
    remaining = {request_id: len(ids) for request_id in request_ids}
    active = []
    rows = {}
    first_token_times = {}
    completion_times = {}
    prefill_steps = []
    decode_steps = []
    decode_batch_sizes = []
    scheduler_steps = []
    while any(remaining.values()) or active:
        queued = [QueuedPrefill(request_id, remaining[request_id], offsets[request_id])
                  for request_id in request_ids if remaining[request_id] > 0]
        decision = plan_token_budget_iteration(active, queued, token_budget=8192,
            max_sequences=batch, prefill_chunk=chunk, prefill_cohort=112,
            max_peer_rows=4096, max_new_prefills=112)
        if decision.priority_request_ids != tuple(active):
            raise RuntimeError('Token-budget scheduler failed to retain every active request')
        if decision.scheduled_tokens > 8192:
            raise RuntimeError('Token-budget scheduler exceeded the aggregate token budget')
        scheduler_steps.append(dict(active_decode=len(decision.priority_request_ids),
            prefill_tokens=sum(item.real_tokens for cohort in decision.prefill_cohorts
                               for item in cohort), scheduled_tokens=decision.scheduled_tokens,
            prefill_cohorts=[len(cohort) for cohort in decision.prefill_cohorts]))

        # Garnet has separate phase engines. Preserve vLLM's priority ordering,
        # while timing both serial forwards inside the same full-request window.
        if active:
            active_slots = [slots[request_id] for request_id in active]
            slot_key = tuple(active_slots)
            if slot_key not in decode_inputs_by_slot_group:
                raise RuntimeError('Active sequence slots do not match an admitted decode cohort')
            lengths = [len(rows[request_id]) for request_id in active]
            if any(length != lengths[0] for length in lengths):
                raise RuntimeError('Fixed cohort decode histories diverged')
            tokens = [rows[request_id][-1] for request_id in active]
            positions = [len(ids) + length - 1 for length in lengths]
            step = time.perf_counter()
            reply = pair.forward_decode(decode_inputs_by_slot_group[slot_key], sample=True,
                sample_batch=True, vector_values=[tokens, positions,
                [len(ids) + length for length in lengths], active_slots])
            decode_steps.append(time.perf_counter() - step)
            decode_batch_sizes.append(len(active))
            tokens = reply.get('token_ids', [])
            if len(tokens) != len(active):
                raise ValueError('Decode phase returned the wrong active sequence count')
            for request_id, token in zip(active, tokens):
                rows[request_id].append(int(token))
                if len(rows[request_id]) == output_tokens:
                    completion_times[request_id] = time.perf_counter() - start
            active = [request_id for request_id in active
                      if len(rows[request_id]) < output_tokens]

        completed_prefills = []
        for cohort in decision.prefill_cohorts:
            cohort_ids = [item.request_id for item in cohort]
            cohort_slots = [slots[item] for item in cohort_ids]
            offset = cohort[0].input_offset
            real = cohort[0].real_tokens
            if any(item.input_offset != offset or item.real_tokens != real
                   for item in cohort):
                raise ValueError('Fixed-batch harness received a heterogeneous prefill cohort')
            token_row = ids[offset:offset + real] + [0] * (chunk - real)
            position_row = list(range(offset, offset + chunk))
            cohort_index = 0 if cohort_slots == list(range(112)) else 1
            if cohort_slots != list(range(cohort_index * 112, (cohort_index + 1) * 112)):
                raise ValueError('Fixed scheduler cohort order/slot ownership changed')
            inputs = prefill_cohort_inputs[cohort_index]
            values = [token_row * len(cohort_ids), position_row * len(cohort_ids),
                [offset + real] * len(cohort_ids), cohort_slots]
            final_slice = offset + real == len(ids)
            step = time.perf_counter()
            reply = pair.forward_prefill(inputs, sample=final_slice, sample_batch=True,
                vector_values=values)
            prefill_steps.append(time.perf_counter() - step)
            if final_slice:
                if len(reply.get('token_ids', [])) != len(cohort_ids):
                    raise ValueError('Prefill cohort returned the wrong sampled token count')
                for request_id, token in zip(cohort_ids, reply['token_ids']):
                    rows[request_id] = [int(token)]
                    first_token_times[request_id] = time.perf_counter() - start
            for item in cohort:
                offsets[item.request_id] += item.real_tokens
                remaining[item.request_id] -= item.real_tokens
                if remaining[item.request_id] == 0:
                    completed_prefills.append(item.request_id)
        if any(value < 0 for value in remaining.values()):
            raise RuntimeError('Prefill planner advanced beyond an input prompt')
        for request_id in completed_prefills:
            if len(rows.get(request_id, ())) != 1 or request_id in active:
                raise RuntimeError('Completed prompt was duplicated or missed its first token')
            active.append(request_id)
        if not active and not any(remaining.values()):
            break

    finish = time.perf_counter() - start
    token_rows = [rows[request_id] for request_id in request_ids]
    if (set(rows) != set(request_ids) or any(len(row) != output_tokens for row in token_rows) or
            set(first_token_times) != set(request_ids) or set(completion_times) != set(request_ids)):
        raise ValueError('Token-budget scheduler returned an incomplete token trajectory')
    earliest_first = min(first_token_times.values())
    latest_last = max(completion_times.values())
    decode_wall = latest_last - earliest_first
    prefill_wall = max(first_token_times.values())
    return dict(trial=trial, token_ids_by_request=token_rows,
        prefill_kv_reused_for_decode_trial=False, prefill_step_seconds=prefill_steps,
        prefill_seconds=prefill_wall, decode_step_seconds=decode_steps,
        decode_batch_sizes=decode_batch_sizes, decode_wall_seconds=decode_wall,
        decode_aggregate_output_tokens_per_second=batch * (output_tokens - 1) / decode_wall,
        full_request_wall_seconds=finish,
        full_request_output_tokens_per_second=batch * output_tokens / finish,
        request_first_token_seconds=[first_token_times[item] for item in request_ids],
        request_completion_seconds=[completion_times[item] for item in request_ids],
        benchmark_started_ts=start,
        request_timings=[dict(first_token_ts=start + first_token_times[item],
                              last_token_ts=start + completion_times[item])
                         for item in request_ids],
        token_budget_scheduler=dict(token_budget=8192, max_peer_rows=4096,
            chunk_tokens=chunk, scheduling_steps=scheduler_steps,
            execution='decode forward before FIFO prefill forward; sequential engine calls'))

def run_request(trial, ids):
    start = time.perf_counter()
    state_handle = G.greedy_batch_state_create(batch, output_tokens) if native_greedy_state else None
    execute = (lambda: run_token_budget_request(trial, ids, start)) if token_budget_scheduler else (
        lambda: run_request_body(trial, ids, start, state_handle))
    # Keep the established default path's timing and call sequence unchanged.
    if not runtime_switch_us and state_handle is None:
        result = execute()
        result.update(runtime_scheduling=dict(requested_microseconds=0,
            default_seconds=None, effective_seconds=None, restored_seconds=None, applied=False),
            runtime_scheduling_finish_seconds=0., native_greedy_state=False,
            native_greedy_state_finish_seconds=0., request_wrapper_finish_seconds=0.)
        return result
    scheduling = None
    try:
        if runtime_switch_us:
            with RuntimeSwitchScope(runtime_switch_us) as scheduling:
                result = execute()
        else:
            result = execute()
    finally:
        if state_handle is not None:
            release_started = time.perf_counter()
            if not G.greedy_batch_state_release(state_handle):
                raise RuntimeError('Native greedy batch state release failed')
            release_seconds = time.perf_counter() - release_started
        else:
            release_seconds = 0.
    wall = time.perf_counter() - start
    body_wall = result['full_request_wall_seconds']
    result['native_greedy_state'] = bool(native_greedy_state)
    result['native_greedy_state_finish_seconds'] = release_seconds
    result['full_request_wall_seconds'] = wall
    result['full_request_output_tokens_per_second'] = batch * output_tokens / wall
    result['request_completion_seconds'] = [wall] * batch
    result['runtime_scheduling'] = dict(scheduling.metadata) if scheduling else dict(
        requested_microseconds=0, default_seconds=None, effective_seconds=None,
        restored_seconds=None, applied=False)
    # The full window includes interval set/verification/restoration as well
    # as preparation, sampling and output assembly. Decode retains its own
    # first-token-to-last-token window and is not the acceptance metric.
    wrapper_residual = max(0., wall - body_wall - release_seconds)
    result['runtime_scheduling_finish_seconds'] = wrapper_residual if scheduling else 0.
    result['request_wrapper_finish_seconds'] = wrapper_residual if not scheduling else 0.
    return result


def run_request_body(trial, ids, start, state_handle=None):
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
        token_pattern = ids[offset:offset + real] + [0] * (chunk - real)
        position_pattern = list(range(offset, offset + chunk))
        patterns = None
        if pattern_updates:
            patterns = [token_pattern, position_pattern, [offset + real], [offset]]
        else:
            update(pair.prefill_stages, prefill_inputs, token_pattern * batch,
                position_pattern * batch, offset + real, offset)
        need_sample = not final_prefill_sample_only or offset + real == len(ids)
        options = dict(sample=need_sample, sample_batch=True,
            reuse_output=reuse_output, native_candidate_merge=native_candidate_merge,
            native_greedy_state=state_handle)
        if patterns is not None:
            options['pattern_values'] = patterns
        reply = pair.forward_prefill(prefill_inputs, **options)
        capture.prefill_end(trial, chunk_index)
        prefill_steps.append(time.perf_counter() - step)
    rows = None if state_handle is not None else [[int(value)] for value in reply['token_ids']]
    if rows is not None and len(rows) != batch:
        raise ValueError('Missing prefill output requests')
    first_token = time.perf_counter()
    decode_steps = []
    for offset in range(1, output_tokens):
        capture.decode_begin(trial, offset)
        step = time.perf_counter()
        index = len(ids) + offset - 1
        # Native state keeps its token history outside Python; feed the last
        # selected batch directly from the preceding forward result.
        tokens = (reply['token_ids'] if state_handle is not None else
                  [row[-1] for row in rows])
        options = dict(sample=True, sample_batch=True,
            reuse_output=reuse_output, native_candidate_merge=native_candidate_merge,
            native_greedy_state=state_handle)
        if pattern_updates:
            options['pattern_values'] = [tokens, [index], [index + 1], [index]]
        else:
            update(pair.decode_stages, decode_inputs, tokens, [index] * batch, index + 1, index)
        reply = pair.forward_decode(decode_inputs, **options)
        tokens = reply['token_ids']
        if len(tokens) != batch:
            raise ValueError('Missing decode output requests')
        if state_handle is None:
            for row, token in zip(rows, tokens):
                row.append(int(token))
        capture.decode_end(trial, offset)
        decode_steps.append(time.perf_counter() - step)
    if state_handle is not None:
        rows = G.greedy_batch_state_history(state_handle)
        if len(rows) != batch or any(len(row) != output_tokens for row in rows):
            raise ValueError('Native greedy history has an incomplete request')
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
        # Preserve full request evidence, including set/restore verification and
        # finalization costs. Keep the existing seconds alias for older readers.
        warmups.append(dict(measured, seconds=measured['full_request_wall_seconds']))
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
        operator_execution_resources=peer_group_status,
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
        resident_host_candidates=resident_host_candidates,
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
            ('vLLM-style token-budget fixed 224-slot TP2 scheduler candidate, greedy, no early stop; three complete warmed requests'
             if token_budget_scheduler else
             'homogeneous fixed-size TP2 resident batch, greedy, no early stop; three complete warmed requests')),
        measurement_limits=('Nsight captures perturb timings; do not compare these rates to vLLM or accept them as an unprofiled reference. ' if capture.enabled else '') +
            ('Cold engine startup and warmup excluded, as for vLLM; every trial rewrites full input KV and includes controls/input preparation, FIFO chunk scheduling, sampling and phase transitions. The opt-in scheduler admits only the fixed 224/112/36 profile; it is not a general continuous batch server. Sampled memory is not exhaustive peak.'
             if token_budget_scheduler else
             'Cold engine startup and warmup excluded, as for vLLM; every trial rewrites full input KV and includes controls/input preparation, sampling and phase transition. Sampled memory is not exhaustive peak; serial batches only, no continuous scheduler.'),
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
    if token_budget_scheduler:
        prefill_cohort_inputs = [phase_tensors(pair.prefill_stages, chunk,
            list(range(start, start + 112))) for start in (0, 112)]
        decode_inputs_by_slot_group = {
            tuple(range(112)): phase_tensors(pair.decode_stages, 1, list(range(112))),
            tuple(range(112, 224)): phase_tensors(pair.decode_stages, 1, list(range(112, 224))),
            tuple(range(224)): phase_tensors(pair.decode_stages, 1, list(range(224))),
        }
    else:
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
        try:
            pair.release()
        finally:
            if group is not None:
                extension.peer_group_release(group)
            group = None
if session_mode:
    session_report['session_complete'] = True
    session_report['phase'] = 'complete'
    session_report['sampled_peak_gpu_memory_mib'] = [
        max(case['sampled_peak_gpu_memory_mib'][rank] for case in session_report['cases']) for rank in range(2)]
    session_result_path.write_text(json.dumps(session_report, indent=2))
    print('Resident session complete; all case answers passed; performance requires matched audit', flush=True)
