"""Independent CPU raw audit of a resident session against fresh single-case controls.

No Garnet, placement, KV-layout or admission implementation imports. This proves
same-shape session evidence, not the all-four maximum-throughput/vLLM objective.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

ENGINE_SOURCE_PATHS=('src','python','xModel','tools/gpt_oss/pipeline.py',
    'tools/gpt_oss/kv_layout.py','tools/gpt_oss/engine_profile_shape.py',
    'tools/gpt_oss/resident_budget.py','tools/gpt_oss/peer_group_layout.py',
    'tools/gpt_oss/profile_tp_engine_memory.py','plugins/gpt_oss/include/gpt_oss_peer_group_options.h',
    'plugins/gpt_oss/cuda/tp_peer_group_options.cpp','plugins/gpt_oss/cuda/tp_peer_group.cu',
    'plugins/gpt_oss/cuda/tp_peer_owner.cuh')
ROUTER_WARP_POLICY='GARNET_GPT_OSS_TENSOR_ROUTER_EXPERT_WARPS'
RESIDENT_HOST_FLAGS={
    'GARNET_RESIDENT_REUSE_OUTPUT':'reuse_output',
    'GARNET_RESIDENT_NATIVE_GREEDY_MERGE':'native_candidate_merge',
    'GARNET_RESIDENT_FINAL_PREFILL_SAMPLE_ONLY':'final_prefill_sample_only',
    'GARNET_RESIDENT_PATTERN_UPDATES':'pattern_updates',
    'GARNET_RESIDENT_NATIVE_GREEDY_STATE':'native_greedy_state'}


def normalize_kernel_environment(environment):
    """Map only absent recorded router/peer geometry to its legacy default."""
    normalized=dict(environment)
    normalized.setdefault(ROUTER_WARP_POLICY,'4')
    normalized.setdefault('GARNET_GPT_OSS_BF16_PEER_GROUP_CTAS','64')
    normalized.setdefault('GARNET_GPT_OSS_BF16_PEER_GROUP_THREADS','256')
    normalized.setdefault('GARNET_GPT_OSS_BF16_PEER_GRID_SIGNALS','0')
    normalized.setdefault('GARNET_GPT_OSS_FUSED_MOE_TP_REDUCE','0')
    return normalized


def profile_source_identity(profile, observed_commit, source_repository=None, archive_root=None):
    """Explicit Git-object proof for reused measurements, never relabel a profile."""
    measured_commit=profile['source_commit']
    if not all(re.fullmatch('[0-9a-f]{40}',value) for value in (measured_commit,observed_commit)):
        raise ValueError('Invalid immutable profile/execution source commit')
    identity=dict(measured_source_commit=measured_commit,execution_source_commit=observed_commit)
    if measured_commit==observed_commit:
        return dict(identity,compatibility='same immutable commit')
    if source_repository is None:
        raise ValueError('Different profile origin requires explicit engine-source Git proof')
    def blobs(commit):
        raw=subprocess.check_output(['git','-C',str(source_repository),'ls-tree','-rz','--full-tree',commit,
            '--',*ENGINE_SOURCE_PATHS])
        entries={}
        for item in raw.split(b'\0'):
            if not item:continue
            prefix,name=item.split(b'\t',1);mode,kind,object_id=prefix.decode().split()
            path=name.decode('utf-8')
            if (kind!='blob' or mode not in ('100644','100755') or
                    not re.fullmatch('[0-9a-f]{40}',object_id)):
                raise ValueError('Engine source requires regular immutable Git blobs')
            entries[path]=dict(mode=mode,git_blob_sha1=object_id)
        for selected in ENGINE_SOURCE_PATHS:
            if not any(path==selected or path.startswith(selected+'/') for path in entries):
                raise ValueError('Incomplete engine-program source closure')
        return entries
    measured,executed=blobs(measured_commit),blobs(observed_commit)
    if measured!=executed:
        raise ValueError('Measured engine-program source differs from execution commit')
    if archive_root is not None:
        root=Path(archive_root).resolve()
        for name,entry in executed.items():
            path=(root/'Garnet'/name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('Missing actual archived execution-program source')
            payload=path.read_bytes()
            object_id=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
            if object_id!=entry['git_blob_sha1']:
                raise ValueError('Actual archived source does not match execution Git object')
    digest=hashlib.sha256(json.dumps(executed,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(identity,compatibility='identical immutable engine-program Git blobs',
        engine_program_blobs=executed,engine_program_manifest_sha256=digest,
        archived_execution_bytes_verified=archive_root is not None,
        limits='Profile origin retained; runner/control identity and native/cache/hardware/math/timing guards remain separate')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def close(actual, expected):
    if (type(actual) not in (int, float) or type(expected) not in (int, float) or
            not math.isfinite(actual) or not math.isfinite(expected) or
            abs(actual - expected) > 1e-8 * max(1., abs(expected))):
        raise ValueError('Inconsistent finite timing/accounting value')


def kernel_environment(result):
    environment = {key: value for key, value in result['optimization_environment'].items()
        if key.startswith(('GARNET_GPT_OSS_', 'GARNET_TP_')) and
        not key.startswith('GARNET_GPT_OSS_PROFILE_') and
        key not in ('GARNET_GPT_OSS_WEIGHTS', 'GARNET_GPT_OSS_CACHE', 'GARNET_GPT_OSS_TOKENIZER')}
    policy = result['optimization_environment'].get('GARNET_TRT_SYNC_ALLOCATOR', '0')
    if policy not in ('0', '1'):
        raise ValueError('Invalid allocator policy in result')
    environment['GARNET_TRT_SYNC_ALLOCATOR'] = policy
    return normalize_kernel_environment(environment)


def resident_host_candidates(result):
    environment=result.get('optimization_environment', {})
    actual={}
    for name,field in RESIDENT_HOST_FLAGS.items():
        value=environment.get(name,'0')
        if value not in ('0','1'):
            raise ValueError('Invalid resident host candidate environment: '+name)
        actual[field]=value=='1'
    if actual['native_greedy_state'] and actual['native_candidate_merge']:
        raise ValueError('Mutually exclusive native greedy paths were both enabled')
    recorded=result.get('resident_host_candidates')
    if recorded is None:
        if any(actual.values()):
            raise ValueError('Enabled host candidate lacks explicit result provenance')
    elif recorded!=actual:
        raise ValueError('Recorded resident host candidates differ from effective environment')
    return actual


def allocated_kv(plan):
    """Derive bytes independently from dimensions, never trust declared totals."""
    batch, capacity, chunk = plan['batch'], plan['capacity'], plan['max_tokens']
    config, heads = plan['config'], plan['local_kv_heads']
    layers, dimension = config['num_hidden_layers'], config['head_dim']
    if (any(type(value) is not int or value <= 0 for value in
            (batch, capacity, chunk, layers, dimension, heads)) or chunk > capacity or
            config['num_key_value_heads'] != heads * 2 or not plan['marlin_prepacked']):
        raise ValueError('Invalid KV allocation dimensions')
    pages = (capacity + 15) // 16
    if plan['kv_pages'] != batch * pages:
        raise ValueError('Physical full-history page reservation differs')
    full = layers * 2 * batch * pages * 16 * heads * dimension * 2
    if 'kv_layout' not in plan:
        if plan['schema'] != 3:
            raise ValueError('Unknown full-history KV schema')
        return full, 0
    window = config['sliding_window']
    ring = min(pages, (window + chunk - 1 + 15) // 16)
    if layers % 2 or config['num_key_value_heads'] != heads * 2 or window <= 0:
        raise ValueError('Invalid complete alternating TP2 KV configuration')
    layout = plan['kv_layout']
    if (plan['schema'] != 4 or layout['schema'] != 1 or
            layout['mode'] != 'gpt-oss-alternating-hybrid-kv-v1' or layout['dtype'] != 'bfloat16' or
            layout['global_layers'] != list(range(1, layers, 2)) or
            layout['window_layers'] != list(range(0, layers, 2)) or
            layout['global_shape'] != [layers // 2, batch * pages, 16, heads, dimension] or
            layout['window_shape'] != [layers // 2, batch * ring, 16, heads, dimension] or
            layout['window_table_shape'] != [batch, pages] or
            layout['window_pages_per_request'] != ring or layout['logical_pages_per_request'] != pages or
            layout['max_prefill_tokens'] != chunk or layout['sliding_window'] != window or
            layout['page_size'] != 16 or layout['physical_layer_rule'] != 'logical_layer//2' or
            layout['extra_tensor_arguments'] != ['window_keys', 'window_values', 'window_table']):
        raise ValueError('Hybrid KV layout does not match independently derived banks')
    actual = (layers // 2) * 2 * batch * (pages + ring) * 16 * heads * dimension * 2
    auxiliary = batch * pages * 4
    if layout['shared_kv_bytes'] != actual or layout['shared_auxiliary_bytes'] != auxiliary:
        raise ValueError('Hybrid KV declared byte totals differ')
    return actual, auxiliary


def memory_evidence(value, profile, kv, auxiliary):
    ranks, plan = value['resident_admission']['ranks'], profile['plan']
    if ([rank['device'] for rank in ranks] != [device['id'] for device in plan['hardware']] or
            len(ranks) != 2 or len({rank['device'] for rank in ranks}) != 2 or
            value['hardware'] != plan['hardware'] or len(profile['engine_statistics']) != 4):
        raise ValueError('Missing or reordered rank/hardware admission')
    samples = value['gpu_memory_samples_mib']
    if not samples or any(len(row) != 2 or any(type(x) is not int or x < 0 for x in row) for row in samples):
        raise ValueError('Invalid sampled rank memory')
    peaks = [max(row[rank] for row in samples) for rank in range(2)]
    if peaks != value['sampled_peak_gpu_memory_mib']:
        raise ValueError('Sampled peak not derived from raw samples')
    for peak, rank, device in zip(peaks, ranks, plan['hardware']):
        rows = [row for row in profile['engine_statistics'] if row['device'] == rank['device']]
        if len(rows) != 2 or {row['phase'] for row in rows} != {'prefill', 'decode'}:
            raise ValueError('Invalid two-phase weight/context records')
        if any(type(row[key]) is not int or row[key] <= 0 for row in rows for key in
                ('total_weights_bytes', 'context_device_memory_upper_bound_bytes')):
            raise ValueError('Invalid measured engine bounds')
        lower = plan['weight_storage_estimate']['prepacked_marlin_constant_bytes']
        if any(row['total_weights_bytes'] < lower for row in rows):
            raise ValueError('Engine weights below recorded packed-weight lower bound')
        weights = sum(row['total_weights_bytes'] for row in rows)
        contexts = sum(row['context_device_memory_upper_bound_bytes'] for row in rows)
        reserve = rank['runtime_graph_reserve_bytes']
        required = (weights * 105 + 99) // 100 + contexts + kv + auxiliary + reserve
        if (type(reserve) is not int or rank['weight_margin_fraction'] != .05 or
                rank['contexts_per_engine'] != 1 or reserve < 2 << 30 or
                rank['shared_kv_bytes'] != kv or rank.get('shared_auxiliary_bytes', 0) != auxiliary or
                rank['required_bytes'] != required or not required <= rank['budget_bytes'] <= int(device['total_bytes'] * .9) or
                peak * (1 << 20) > rank['budget_bytes']):
            raise ValueError('Weakened or inconsistent memory admission')
    return peaks


def timed_trial(trial, batch, output, length, chunk):
    full, decode_time, prefill = (trial[key] for key in
        ('full_request_wall_seconds', 'decode_wall_seconds', 'prefill_seconds'))
    if not full > decode_time > 0 or not prefill > 0:
        raise ValueError('Invalid complete request timing')
    scheduling = trial.get('runtime_scheduling')
    overhead = trial.get('runtime_scheduling_finish_seconds', 0.)
    greedy_finish = trial.get('native_greedy_state_finish_seconds', 0.)
    wrapper_finish = trial.get('request_wrapper_finish_seconds', 0.)
    native_state = trial.get('native_greedy_state', False)
    if (not math.isfinite(overhead) or overhead < 0 or
            not math.isfinite(wrapper_finish) or wrapper_finish < 0):
        raise ValueError('Invalid request finalization duration')
    if (type(native_state) is not bool or not math.isfinite(greedy_finish) or greedy_finish < 0 or
            ('native_greedy_state_finish_seconds' not in trial and native_state) or
            (not native_state and greedy_finish != 0)):
        raise ValueError('Invalid native greedy state finalization evidence')
    if scheduling is not None:
        requested = scheduling.get('requested_microseconds')
        if type(requested) is not int or requested not in (0,100,200,1000):
            raise ValueError('Invalid recorded runtime scheduling policy')
        if requested:
            if scheduling.get('applied') is not True or 'runtime_scheduling_finish_seconds' not in trial:
                raise ValueError('Missing applied scheduling timing')
            original = scheduling.get('default_seconds')
            effective = scheduling.get('effective_seconds')
            restored = scheduling.get('restored_seconds')
            if any(type(x) not in (int,float) or not math.isfinite(x) or not 0 < x < 1
                   for x in (original,effective,restored)):
                raise ValueError('Invalid recorded scheduling intervals')
            close(effective, requested/1000000.)
            close(restored, original)
        elif (overhead != 0 or scheduling.get('applied') is not False or
                any(scheduling.get(k) is not None for k in ('default_seconds','effective_seconds','restored_seconds'))):
            raise ValueError('Disabled scheduling has applied state or overhead')
    elif overhead != 0:
        raise ValueError('Scheduling overhead lacks its policy')
    close(full, prefill + decode_time + overhead + greedy_finish + wrapper_finish)
    close(trial['full_request_output_tokens_per_second'], batch * output / full)
    close(trial['decode_aggregate_output_tokens_per_second'], batch * (output - 1) / decode_time)
    if (len(trial['request_first_token_seconds']) != batch or len(trial['request_completion_seconds']) != batch or
            len(trial['prefill_step_seconds']) != math.ceil(length / chunk) or
            len(trial['decode_step_seconds']) != output - 1):
        raise ValueError('Missing per-request or per-step timings')
    for first, finish in zip(trial['request_first_token_seconds'], trial['request_completion_seconds']):
        close(first, prefill); close(finish, full)
    for steps, limit in ((trial['prefill_step_seconds'], prefill), (trial['decode_step_seconds'], decode_time)):
        if any(not math.isfinite(step) or step <= 0 for step in steps) or sum(steps) > limit + 1e-8:
            raise ValueError('Invalid step timings outside full request')
    return batch * output / full


def audit(session_path, controls_path, decode, archive_root=None, source_repository=None):
    def locate(recorded):
        if archive_root is not None:
            prefix = '/workspace/CantorAI/'
            recorded = str(recorded).replace('\\', '/')
            if not recorded.startswith(prefix):
                raise ValueError('Archive reference outside expected target workspace')
            root = Path(archive_root).resolve()
            path = (root / str(recorded)[len(prefix):]).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Archive path escapes evidence root')
        else:
            path = Path(recorded)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    session, controls = read(session_path), read(controls_path)
    names = ['arithmetic', 'code-tracing', 'instruction-following']
    if (session.get('resident_session_schema') != 1 or session.get('session_complete') is not True or
            session.get('active_case') is not None or session.get('failure') or
            session.get('phase') != 'complete' or session.get('case_order') != names or
            [case['name'] for case in session['cases']] != names or
            [case['name'] for case in controls['cases']] != names or controls.get('resident_session') or
            controls['phase_order'] not in (['admission', 'vllm', 'garnet'],
                ['admission', 'revalidate_saved_vllm', 'garnet'])):
        raise ValueError('Require all three completed session cases and independent single-case controls')
    manifest_path = locate(session['session_manifest'])
    manifest = read(manifest_path)
    if (sha(manifest_path) != session['session_manifest_sha256'] or
            [case['name'] for case in manifest['cases']] != names or
            not re.fullmatch('[0-9a-f]{40}', session['source_commit'])):
        raise ValueError('Session manifest/source identity changed')
    tokenizer_path = locate(session['tokenizer'].rstrip('/\\') + '/tokenizer.json')
    if sha(tokenizer_path) != session['tokenizer_sha256']:
        raise ValueError('Saved tokenizer changed')
    profile_path = locate(session['resident_profile'])
    profile = read(profile_path)
    if (sha(profile_path) != session['resident_profile_sha256'] or
            profile.get('resident_profile_schema') != 1 or
            profile['native_binaries'] != session['native_binaries'] or
            profile['hardware_csv'] != session['hardware_csv']):
        raise ValueError('Session profile/native/hardware identity changed')
    cold = session['cold_startup_seconds_excluding_module_imports']
    if not math.isfinite(cold) or cold <= 0:
        raise ValueError('Missing real session cold preparation')
    if any(not math.isfinite(session[key]) or session[key] < 0 for key in
           ('prefill_build_seconds', 'decode_build_seconds', 'admission_and_engine_checksum_seconds')):
        raise ValueError('Invalid shared phase/checksum timing')
    if sum(session[key] for key in ('prefill_build_seconds', 'decode_build_seconds',
                                   'admission_and_engine_checksum_seconds')) > cold:
        raise ValueError('Shared startup decomposition exceeds actual cold duration')
    plan = profile['plan']
    identity_keys = ('schema', 'mode', 'cache_key', 'hardware', 'batch', 'capacity', 'max_tokens',
        'kv_pages', 'config', 'local_kv_heads', 'expert_weight_shards', 'moe_intermediate_shards',
        'marlin_prepacked', 'compact_vocab_greedy', 'marlin_workspace_layout', 'collective_workspace_layout')
    identity = {key: plan[key] for key in identity_keys}
    if 'kv_layout' in plan:
        identity['kv_layout'] = plan['kv_layout']
    if (profile['plan_identity'] != identity or
            set(session['native_binaries']) != {'libgarnet.so', 'libgarnet_gpt_oss.so'} or
            any(not re.fullmatch('[0-9a-f]{64}', value) for value in session['native_binaries'].values())):
        raise ValueError('Measured plan/source/native identity is inconsistent')
    source_identity=profile_source_identity(profile,session['source_commit'],source_repository,archive_root)
    kv, auxiliary = allocated_kv(plan)
    batch, output, capacity, chunk = (manifest[key] for key in ('batch', 'output', 'context', 'prefill_chunk'))
    if (any(type(value) is not int for value in (batch, output, capacity, chunk)) or
            plan['batch'] != batch or plan['capacity'] != capacity or plan['max_tokens'] != chunk or
            not 1 <= batch <= 512 or not 16 <= output <= 2048 or not 1 <= chunk <= capacity <= 4096):
        raise ValueError('Session shape differs from strict engine profile')
    report = dict(scope='Independent session equivalence and complete-request evidence; all-four speed goal unassessed',
        session_sha256=sha(session_path), controls_manifest_sha256=sha(controls_path),
        source_commit=session['source_commit'], native_binaries=session['native_binaries'],
        profile_source_identity=source_identity,
        shared_cold_preparation_seconds_excluding_imports=cold, cases=[],
        limits=['CPU artifact verification, not runtime GPU instrumentation.',
                'Sampled memory is not exhaustive peak; serial same-shape batches only.',
                'Recorded binary/cache/checkpoint identities require separate runtime admission and provenance gates.',
                'Requires separate all-four latest optimized vLLM/max-throughput assessment.'])
    requests, case_paths = [], set()
    for index, (record, specification, control_case) in enumerate(zip(
            session['cases'], manifest['cases'], controls['cases'])):
        result_path, validation_path = locate(record['result']), locate(record['validation'])
        result, validation = read(result_path), read(validation_path)
        request_path, expected_path = locate(specification['request']), locate(specification['expected'])
        request, expected = read(request_path), read(expected_path)
        control_path = locate(control_case['garnet']['result'])
        control = read(control_path)
        paths = {result_path.resolve(), validation_path.resolve(), control_path.resolve()}
        if len(paths) != 3 or case_paths.intersection(paths):
            raise ValueError('Case/control evidence paths alias')
        case_paths.update(paths)
        requests.append((len(request['input_ids']), request.get('device_ids'),
            request.get('reserve_mb', 1024), request.get('memory_fraction', .9)))
        if (sha(locate(control_case['request'])) != sha(request_path) or
                sha(locate(control_case['expected'])) != sha(expected_path) or
                any(control_case[key] != value for key, value in
                    [('batch', batch), ('output', output), ('context', capacity), ('input', len(request['input_ids']))]) or
                locate(result['request']).resolve() != request_path.resolve() or
                locate(result['session_manifest']).resolve() != manifest_path.resolve() or
                locate(result['session_result']).resolve() != Path(session_path).resolve() or
                locate(session['session_result']).resolve() != Path(session_path).resolve()):
            raise ValueError('Case/control request or session evidence binding differs')
        if (str(record['result']) != specification['result'] or
                str(record['validation']) != specification['validation'] or
                sha(result_path) != record['result_sha256'] or
                sha(validation_path) != record['validation_sha256'] or
                sha(expected_path) != record['expected_sha256'] or
                sha(request_path) != result['request_sha256'] or
                result['case_index'] != index or result['case_name'] != names[index] or
                result['engine_pair_reused'] != (index > 0) or
                result['session_manifest_sha256'] != session['session_manifest_sha256'] or
                result['session_result'] != session['session_result'] or
                'cold_startup_seconds_excluding_module_imports' in result or
                any(key in result for key in ('prefill_build_seconds', 'decode_build_seconds',
                    'admission_and_engine_checksum_seconds'))):
            raise ValueError('Case hash/order/reuse/startup accounting mismatch')
        for field in ('source_commit', 'native_binaries', 'hardware_csv', 'resident_profile_sha256'):
            if result[field] != session[field] or control[field] != result[field]:
                raise ValueError('Fresh single-case control identity differs: ' + field)
        profile_environment=normalize_kernel_environment(profile['kernel_environment'])
        if kernel_environment(result) != profile_environment or kernel_environment(control) != profile_environment:
            raise ValueError('Session/control kernel settings differ from measured profile')
        for value in (result, control):
            host_candidates=resident_host_candidates(value)
            if (sha(locate(value['resident_profile'])) != session['resident_profile_sha256'] or
                    value['padded_prefill'] != profile['padded_prefill'] or
                    value['kv_pages_per_gpu'] != plan['kv_pages']):
                raise ValueError('Actual profile/padding/page identity differs')
            for item in value['complete_warmups']+value['decode_trials']:
                if item.get('native_greedy_state',False) is not host_candidates['native_greedy_state']:
                    raise ValueError('Request native-state flag differs from effective host environment')
            if (value['batch'] != batch or value['output_tokens_per_request'] != output or
                    value['input_token_ids'] != request['input_ids'] or
                    value['input_tokens_per_request'] != len(request['input_ids']) or
                    value['max_context_tokens_per_request'] != capacity or
                    value['prefill_chunk_tokens'] != chunk or value.get('profiled_diagnostic') or
                    value.get('profile_decode_steps') or value.get('profile_prefill') or
                    value['complete_warmup_count'] != 1 or len(value['complete_warmups']) != 1 or
                    len(value['decode_trials']) != 3 or value['kv_cache_dtype'] != 'bfloat16' or
                    value['kv_cache_allocated_bytes_per_gpu'] != kv or
                    value['padded_tail_tokens'] != (-len(request['input_ids'])) % chunk):
                raise ValueError('Incomplete/profiled/mismatched workload or KV evidence')
            per_layer = batch * 2 * plan['local_kv_heads'] * plan['config']['head_dim'] * 2
            layers = plan['config']['num_hidden_layers']
            length = len(request['input_ids'])
            if (value['logical_kv_bytes_per_gpu_after_prefill'] != layers * length * per_layer or
                    value['logical_kv_bytes_per_gpu_at_completion'] != layers * (length + output - 1) * per_layer):
                raise ValueError('Logical input/output KV history differs')
            if 'kv_layout' in plan:
                window = plan['config']['sliding_window']
                if (value['kv_layout'] != plan['kv_layout'] or value['shared_kv_auxiliary_bytes_per_gpu'] != auxiliary or
                        value['logical_retained_kv_bytes_per_gpu_after_prefill'] != (layers // 2) * (length + min(length, window)) * per_layer or
                        value['logical_retained_kv_bytes_per_gpu_at_completion'] != (layers // 2) *
                        (length + output - 1 + min(length + output - 1, window)) * per_layer):
                    raise ValueError('Hybrid retained-history/bank report differs')
        if (not request['input_ids'] or len(request['input_ids']) + output > capacity or
                any(type(token) is not int or not 0 <= token < plan['config']['vocab_size'] for token in request['input_ids'])):
            raise ValueError('Request was shortened or does not fit reservation')
        peaks = memory_evidence(result, profile, kv, auxiliary)
        memory_evidence(control, profile, kv, auxiliary)
        if peaks != record['sampled_peak_gpu_memory_mib'] or result['resident_admission'] != session['resident_admission']:
            raise ValueError('Case memory differs from shared admission/record')
        outcomes = {(item['trial'], item['slot']): item for item in validation['slots']}
        if (validation.get('all_pass') is not True or len(validation['slots']) != 3 * batch or
                len(outcomes) != 3 * batch or record['answer_checks'] != 3 * batch):
            raise ValueError('Missing unique answer validations')
        rates, control_rates, matrices = [], [], []
        for trial_index, (trial, control_trial) in enumerate(zip(result['decode_trials'], control['decode_trials'])):
            matrix = trial['token_ids_by_request']
            if (trial['trial'] != trial_index or trial['prefill_kv_reused_for_decode_trial'] is not False or
                    matrix != control_trial['token_ids_by_request'] or len(matrix) != batch):
                raise ValueError('Single/session trajectories differ or input KV was reused')
            if control_trial['prefill_kv_reused_for_decode_trial'] is not False or control_trial['trial'] != trial_index:
                raise ValueError('Single-case control reused input KV or missed a trial')
            control_rates.append(timed_trial(control_trial, batch, output, length, chunk))
            rates.append(timed_trial(trial, batch, output, length, chunk))
            for slot, ids in enumerate(matrix):
                if len(ids) != output or any(type(token) is not int or not 0 <= token < plan['config']['vocab_size'] for token in ids):
                    raise ValueError('Missing output token trajectory')
                matches = re.findall(r'<\|channel\|>final<\|message\|>\s*(\{.*?\})', decode(ids), re.DOTALL)
                checked = outcomes[trial_index, slot]
                if (not matches or json.loads(matches[0]) != expected or
                        checked.get('pass') is not True or checked['first_final_json'] != expected):
                    raise ValueError('Raw output answer disagrees with expected task or green flag')
            matrices.append(matrix)
        if (any(matrix != matrices[0] for matrix in matrices[1:]) or
                result['complete_warmups'][0]['token_ids_by_request'] != matrices[0] or
                control['complete_warmups'][0]['token_ids_by_request'] != matrices[0]):
            raise ValueError('Session warmup/repeat matrices are not exact')
        close(statistics.median(rates), result['median_full_request_output_tokens_per_second'])
        close(statistics.median(rates), record['median_full_request_output_tokens_per_second'])
        close(statistics.median(control_rates), control['median_full_request_output_tokens_per_second'])
        for value in (result, control):
            first = value['decode_trials'][0]
            if value['token_ids_by_request'] != first['token_ids_by_request']:
                raise ValueError('Top-level token matrix differs from complete first trial')
            close(value['prefill_seconds'], first['prefill_seconds'])
            close(value['full_request_output_tokens_per_second'], first['full_request_output_tokens_per_second'])
        report['cases'].append(dict(name=names[index], exact_single_case_trajectories=True,
            answer_checks=3 * batch, raw_sha256=sha(result_path), control_sha256=sha(control_path),
            full_request_median_tok_s=statistics.median(rates), sampled_peak_mib=peaks,
            kv_allocated_bytes_per_rank=kv, auxiliary_allocated_bytes_per_rank=auxiliary))
    if any(identity != requests[0] for identity in requests[1:]):
        raise ValueError('Session requests use different input/device/memory shapes')
    if session['sampled_peak_gpu_memory_mib'] != [max(case['sampled_peak_mib'][rank] for case in report['cases']) for rank in range(2)]:
        raise ValueError('Session maximum omits a case peak')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('single_case_manifest', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--archive-root', type=Path)
    parser.add_argument('--source-repository',type=Path,
        help='Explicit trusted Git repository proving identical engine-program blobs for a reused historical profile')
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    from tokenizers import Tokenizer
    saved = read(args.session)
    tokenizer_path = Path(saved['tokenizer']) / 'tokenizer.json'
    if args.archive_root is not None:
        tokenizer_path = args.archive_root / str(tokenizer_path).removeprefix('/workspace/CantorAI/')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    report = audit(args.session, args.single_case_manifest,
        lambda ids: tokenizer.decode(ids, skip_special_tokens=False), args.archive_root,args.source_repository)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('All three resident-session raw matrices and evidence match single-case controls; full speed goal unassessed')
