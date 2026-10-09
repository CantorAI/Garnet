"""Fail-closed admission for serial TP2 resident engines using measured bounds.

This does not admit concurrent requests/contexts. Runtime/graph reserve is an
explicit allowance; sampled execution peaks must still be checked separately.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from kv_layout import kv_memory
from peer_group_layout import validate_peer_group_layout


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def native_identity(build):
    return {name: file_sha256(Path(build) / 'bin' / name)
            for name in ('libgarnet.so', 'libgarnet_gpt_oss.so')}


def hardware_identity():
    return subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version',
        '--format=csv'], text=True)


def kernel_environment():
    environment = {key: value for key, value in os.environ.items()
        if key.startswith(('GARNET_GPT_OSS_', 'GARNET_TP_'))
        and not key.endswith(('_TOKEN', '_KEY', '_SECRET', '_PASSWORD'))
        and not key.startswith('GARNET_GPT_OSS_PROFILE_')
        and key not in ('GARNET_GPT_OSS_WEIGHTS', 'GARNET_GPT_OSS_CACHE',
                        'GARNET_GPT_OSS_TOKENIZER')}
    environment['GARNET_TRT_SYNC_ALLOCATOR'] = os.environ.get('GARNET_TRT_SYNC_ALLOCATOR', '0')
    # Bind the effective router geometry, including its legacy default, into
    # the measured profile/cache identity. Otherwise an unset environment and
    # an explicit four-warp policy would produce indistinguishable profiles.
    environment.setdefault('GARNET_GPT_OSS_TENSOR_ROUTER_EXPERT_WARPS', '4')
    return environment


def normalize_kernel_environment(environment):
    """Interpret only the pre-OPT79 missing router key as legacy default4."""
    normalized = dict(environment)
    normalized.setdefault('GARNET_GPT_OSS_TENSOR_ROUTER_EXPERT_WARPS', '4')
    return normalized


def checkpoint_identity(weights):
    # Engine refit reads the checkpoint. Header + size/mtime identity detects
    # replaced tensors without rescanning sixty GB before every admission.
    root = Path(weights).resolve()
    files = []
    for path in sorted(root.rglob('*.safetensors')):
        stat = path.stat()
        with path.open('rb') as handle:
            count = int.from_bytes(handle.read(8), 'little')
            if not 0 < count <= 256 << 20:
                raise ValueError('Invalid checkpoint header size')
            header = handle.read(count)
            if len(header) != count:
                raise ValueError('Truncated checkpoint header')
        files.append(dict(path=str(path.relative_to(root)), bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns, header_sha256=hashlib.sha256(header).hexdigest()))
    if not files:
        raise ValueError('No checkpoint files')
    return dict(root=str(root), config_sha256=file_sha256(root / 'config.json'), files=files)


def plan_identity(plan):
    identity = {key: plan[key] for key in ('schema', 'mode', 'cache_key', 'hardware',
        'batch', 'capacity', 'max_tokens', 'kv_pages', 'config', 'local_kv_heads',
        'expert_weight_shards', 'moe_intermediate_shards', 'marlin_prepacked',
        'compact_vocab_greedy', 'marlin_workspace_layout', 'collective_workspace_layout')}
    if 'kv_layout' in plan:
        # Recompute/validate the bank contract before trusting a profile.
        kv_memory(plan)
        identity['kv_layout'] = plan['kv_layout']
    if 'operator_execution_layout' in plan:
        validate_peer_group_layout(plan['operator_execution_layout'], plan['batch'],
            plan['max_tokens'], plan['config']['hidden_size'])
        identity['operator_execution_layout'] = plan['operator_execution_layout']
    return identity


def admit_resident(profile, plan, devices, *, binaries, hardware_csv, environment,
                   checkpoint, cache, padded_prefill, reserve_bytes=2 << 30,
                   memory_fraction=.9):
    """Pure calculation after caller collects actual runtime identities."""
    if not plan.get('marlin_prepacked'):
        raise ValueError('Resident admission requires prepacked engine-owned constants')
    if reserve_bytes < 2 << 30 or not 0 < memory_fraction <= .9:
        raise ValueError('Resident safety reserve/fraction cannot be weakened')
    expected = dict(native_binaries=binaries, hardware_csv=hardware_csv,
        kernel_environment=environment, checkpoint=checkpoint,
        cache_root=str(Path(cache).resolve()), padded_prefill=bool(padded_prefill),
        plan_identity=plan_identity(plan))
    if profile.get('resident_profile_schema') != 1:
        raise ValueError('Regenerate resident profile with native/cache/checkpoint identities')
    for key, value in expected.items():
        actual = profile.get(key)
        if key == 'kernel_environment':
            actual = normalize_kernel_environment(actual or {})
            value = normalize_kernel_environment(value)
        if actual != value:
            raise ValueError('Resident profile identity mismatch: ' + key)
    rows = profile['engine_statistics']
    if len(rows) != 4 or len(devices) != 2:
        raise ValueError('Resident profile requires exactly two phases per rank')
    records = {(row['device'], row['phase']): row for row in rows}
    if len(records) != 4:
        raise ValueError('Duplicate resident engine statistic')
    kv_bytes, auxiliary_bytes = kv_memory(plan)
    packed_lower_bound = plan['weight_storage_estimate']['prepacked_marlin_constant_bytes']
    group_bytes = (validate_peer_group_layout(plan['operator_execution_layout'],
        plan['batch'], plan['max_tokens'], plan['config']['hidden_size'])
        if 'operator_execution_layout' in plan else 0)
    result = []
    for device in devices:
        bounds = []
        for phase in ('prefill', 'decode'):
            row = records.get((device['id'], phase))
            if not row:
                raise ValueError('Missing rank/phase engine statistic')
            weight = row['total_weights_bytes']
            context = row['context_device_memory_upper_bound_bytes']
            if (type(weight) is not int or weight < packed_lower_bound or
                    type(context) is not int or context <= 0):
                raise ValueError('Invalid resident weight/context bound')
            bounds.append((weight, context))
        required = math.ceil(sum(w for w, _ in bounds) * 1.05) + sum(c for _, c in bounds) + kv_bytes + auxiliary_bytes + reserve_bytes + group_bytes
        budget = min(device['free_bytes'], int(device['total_bytes'] * memory_fraction))
        if required > budget:
            raise ValueError('Resident engines exceed memory budget on GPU' + str(device['id']))
        result.append(dict(device=device['id'], required_bytes=required, budget_bytes=budget,
            shared_kv_bytes=kv_bytes, runtime_graph_reserve_bytes=reserve_bytes,
            contexts_per_engine=1, weight_margin_fraction=.05))
        if 'kv_layout' in plan:
            result[-1]['shared_auxiliary_bytes'] = auxiliary_bytes
        if group_bytes:
            result[-1]['operator_execution_storage_bytes'] = group_bytes
    return dict(ranks=result, concurrency='serial complete batches; one context per engine',
        limits='Measured engine bounds plus explicit reserve; execution peak validation still required')


def validate_engine_files(profile):
    # Serialized engine identities are checked separately so CPU admission
    # tests do not depend on remote engine paths.
    for row in profile['engine_statistics']:
        if file_sha256(row['engine_path']) != row['engine_sha256']:
            raise ValueError('Resident engine cache changed after profiling')
