"""GPT-OSS adapter for Garnet's generic GPU pipeline.

The original packed checkpoint stays on disk. Each engine loads only weights
referenced by its layer range, and each stage allocates only its local KV cache.
"""
import json
import math
import hashlib
import os
import shutil
import struct
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / 'python'))
from garnet_pipeline import Pipeline, plan_layers


def weight_sizes(weights):
    sizes = {}
    for path in sorted(Path(weights).rglob('*.safetensors')):
        with path.open('rb') as handle:
            header_length = struct.unpack('<Q', handle.read(8))[0]
            if header_length > 256 * 1024 * 1024:
                raise ValueError('safetensors header exceeds 256 MB')
            header = json.loads(handle.read(header_length))
        for name, item in header.items():
            if name == '__metadata__':
                continue
            if name in sizes:
                raise ValueError('duplicate checkpoint tensor: ' + name)
            offsets = item['data_offsets']
            size = offsets[1] - offsets[0]
            if size < 0:
                raise ValueError('invalid checkpoint offsets')
            # Packed expert weights remain packed; dense BF16 constants may be
            # lowered in FP32 by the correctness backend.
            sizes[name] = size * (2 if item['dtype'] == 'BF16' else 1)
    if not sizes:
        raise ValueError('no original-layout safetensors weights found')
    return sizes


def make_plan(weights, devices, batch=1, capacity=4096, tokens=1,
              reserve_bytes=1 << 30, memory_fraction=.9):
    config = json.loads((Path(weights) / 'config.json').read_text())
    if batch <= 0 or capacity <= 0 or tokens <= 0 or tokens > capacity:
        raise ValueError('invalid batch/context/token capacity')
    sizes = weight_sizes(weights)
    layers = config['num_hidden_layers']
    costs = [0] * layers
    first, last = 0, 0
    for name, size in sizes.items():
        if name.startswith('block.'):
            index = int(name.split('.')[1])
            costs[index] += size
        elif name == 'embedding.weight':
            first += size
        elif name in ('norm.scale', 'unembedding.weight'):
            last += size
        else:
            raise ValueError('unexpected original-layout tensor: ' + name)
    if not first or not last or any(n == 0 for n in costs):
        raise ValueError('checkpoint has missing layer/embedding/output weights')
    pages = batch * math.ceil(capacity / 16)
    kv_per_layer = 2 * pages * 16 * config['num_key_value_heads'] * config['head_dim'] * 2
    costs = [math.ceil(n * 1.15) + kv_per_layer for n in costs]
    # Conservative stage activation/logit allowance plus explicit runtime reserve.
    buffers = batch * tokens * (config['hidden_size'] * 64 + config['vocab_size'] * 4 +
                               config['intermediate_size'] * config['experts_per_token'] * 4)
    plan = plan_layers(devices, costs, math.ceil(first * 1.15), math.ceil(last * 1.15),
                       reserve_bytes + buffers, memory_fraction)
    plan['batch'] = batch
    plan['capacity'] = capacity
    plan['max_tokens'] = tokens
    plan['kv_pages'] = pages
    plan['config'] = config
    return plan


def make_tensor_parallel_plan(weights, devices, batch=1, capacity=4096, tokens=1,
                              reserve_bytes=1 << 30, memory_fraction=.9):
    """Plan a two-rank GPT-OSS attention and expert tensor-parallel stage."""
    if len(devices) != 2 or len({d['id'] for d in devices}) != 2:
        raise ValueError('GPT-OSS TP2 requires exactly two distinct GPUs')
    config = json.loads((Path(weights) / 'config.json').read_text())
    if batch <= 0 or capacity <= 0 or tokens <= 0 or tokens > capacity:
        raise ValueError('invalid batch/context/token capacity')
    if config['num_experts'] < 2:
        raise ValueError('expert parallelism requires at least two experts')
    if not 0 < memory_fraction <= 1 or reserve_bytes < 0:
        raise ValueError('invalid memory budget')
    pages = batch * math.ceil(capacity / 16)
    local_kv_heads = config['num_key_value_heads'] // 2
    if config['num_key_value_heads'] % 2:
        raise ValueError('GPT-OSS TP2 requires an even KV head count')
    kv_bytes = (config['num_hidden_layers'] * 2 * pages * 16 *
                local_kv_heads * config['head_dim'] * 2)
    activation_bytes = batch * tokens * (config['hidden_size'] * 64 +
        config['vocab_size'] * 4 + config['intermediate_size'] * config['experts_per_token'] * 4)
    # Dense BF16 constants are budgeted at FP32 size because TensorRT may
    # promote some projections during engine construction.
    full_weights = sum(weight_sizes(weights).values())
    required = math.ceil(full_weights * 1.05) + kv_bytes + activation_bytes + reserve_bytes
    hardware = [{k: d[k] for k in ('id', 'name', 'total_bytes', 'compute_major',
                'compute_minor', 'pci_bus_id', 'peer_access')} for d in devices]
    budgets = [int(min(d['free_bytes'], d['total_bytes'] * memory_fraction)) for d in devices]
    if any(required > budget for budget in budgets):
        raise ValueError('GPT-OSS TP2 checkpoint estimate exceeds per-GPU memory budget')
    identity = {'schema': 3, 'mode': 'gpt-oss-tensor-parallel-tp2', 'hardware': hardware,
        'checkpoint': str(Path(weights).resolve()), 'capacity': capacity, 'batch': batch,
        'memory_fraction': memory_fraction, 'reserve_bytes': reserve_bytes,
        'layer_cuda_graph': True}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    stages = [{'device_id': d['id'], 'rank': rank, 'start': 0,
               'end': config['num_hidden_layers'], 'estimated_bytes': required,
               'budget_bytes': budget}
              for rank, (d, budget) in enumerate(zip(devices, budgets))]
    return {'schema': 3, 'mode': 'gpt-oss-tensor-parallel-tp2', 'cache_key': key,
            'hardware': hardware, 'stages': stages, 'batch': batch,
            'capacity': capacity, 'max_tokens': tokens, 'kv_pages': pages,
            'config': config, 'estimated_per_gpu_bytes': required,
            'local_kv_heads': local_kv_heads}


def build_pipeline(weights, cache, plan, tokens, prefill, kv=None, last_token_logits=False):
    import garnet as G
    if tokens < 1 or tokens > plan['max_tokens']:
        raise ValueError('token shape exceeds placement profile')
    config, batch = plan['config'], plan['batch']
    cache = Path(cache) / plan['cache_key']
    cache.mkdir(parents=True, exist_ok=True)
    (cache / 'placement.json').write_text(json.dumps(plan, indent=2))
    root = repo / 'xModel/gpt_oss/120b'
    stages = []
    previous = G.cuda_set_device(plan['stages'][0]['device_id'])
    try:
        for rank, placement in enumerate(plan['stages']):
            G.cuda_set_device(placement['device_id'])
            start, end = placement['start'], placement['end']
            stage_cache = cache / ('prefill' if prefill else 'decode') / str(tokens) / str(rank)
            if last_token_logits:
                stage_cache = stage_cache / 'last-token-logits'
            model_root = stage_cache / 'xmodel'
            model_root.mkdir(parents=True, exist_ok=True)
            for name in ('__init__.py', 'tensor_compat.py', 'gpt_oss_llm.py', 'model.json'):
                shutil.copy2(root / name, model_root / name)
            source = (root / 'stage.py').read_text()
            source = source.replace('STAGE_START = 0', 'STAGE_START = ' + str(start))
            source = source.replace('STAGE_END = 1', 'STAGE_END = ' + str(end))
            source = source.replace('STAGE_PREFILL = 1', 'STAGE_PREFILL = ' + str(int(prefill)))
            source = source.replace('STAGE_LAST_TOKEN = 0', 'STAGE_LAST_TOKEN = ' + str(int(last_token_logits)))
            (model_root / 'stage.py').write_text(source)
            shape = [end - start, plan['kv_pages'], 16, config['num_key_value_heads'], config['head_dim']]
            if kv is None:
                keys = G.tensor_zeros(shape, 'bfloat16')
                values = G.tensor_zeros(shape, 'bfloat16')
            else:
                keys, values = kv[rank]
            first_shape = [batch, tokens] if start == 0 else [batch, tokens, config['hidden_size']]
            model = G.load_model(str(model_root / 'stage.py'), runtime_mode='compiled_xmodel',
                backend='tensorrt', precision='bf16', entry_function='GptOssStage', weights=str(weights),
                cache_dir=str(stage_cache / 'engine'),
                input_shapes=[first_shape, [batch, tokens], shape, shape,
                              [batch, math.ceil(plan['capacity'] / 16)], [batch], [batch], [batch]],
                input_dtypes=['int64' if start == 0 else 'float32', 'int64', 'bfloat16', 'bfloat16',
                              'int32', 'int32', 'int32', 'int32'],
                compile={'builder_workspace_mb': 256, 'builder_optimization_level': 1,
                         'partition': {'enable_preferred_boundaries': False,
                                       'max_atomic_regions_per_partition': 0}})
            status = model.runtime_status()
            if not status['ready']:
                raise RuntimeError(str(status))
            stages.append(dict(placement, model=model, keys=keys, values=values))
    except Exception:
        if stages:
            Pipeline(stages).release()
        raise
    finally:
        G.cuda_set_device(previous)
    return Pipeline(stages)


def build_tensor_parallel(weights, cache, plan, tokens, prefill, kv=None,
                          last_token_logits=False):
    """Build paired engines with sharded attention heads and rank-local MoE."""
    import garnet as G
    optimization_level = int(os.environ.get('GARNET_GPT_OSS_TRT_OPT_LEVEL', '1'))
    workspace_mb = int(os.environ.get('GARNET_GPT_OSS_TRT_WORKSPACE_MB', '256'))
    if not 0 <= optimization_level <= 5 or not 256 <= workspace_mb <= 4096:
        raise ValueError('unsupported GPT-OSS TensorRT optimization or workspace setting')
    if tokens < 1 or tokens > plan['max_tokens']:
        raise ValueError('token shape exceeds TP2 placement profile')
    config, batch = plan['config'], plan['batch']
    cache = Path(cache) / plan['cache_key']
    cache.mkdir(parents=True, exist_ok=True)
    (cache / 'placement.json').write_text(json.dumps(plan, indent=2))
    root = repo / 'xModel/gpt_oss/120b'
    stages = []
    previous = G.cuda_set_device(plan['stages'][0]['device_id'])
    try:
        for rank, placement in enumerate(plan['stages']):
            device = placement['device_id']
            G.cuda_set_device(device)
            start, end = 0, config['num_hidden_layers']
            stage_cache = cache / ('prefill' if prefill else 'decode') / str(tokens) / str(rank)
            # The GEMV choice changes the compiled TensorRT graph, so do not reuse
            # an engine built with the other projection implementation.
            if not prefill and os.environ.get('GARNET_GPT_OSS_DECODE_GEMV') == '1':
                stage_cache = stage_cache / 'bf16-gemv-v1'
            if last_token_logits:
                stage_cache = stage_cache / 'last-token-logits'
            model_root = stage_cache / 'xmodel'
            model_root.mkdir(parents=True, exist_ok=True)
            for name in ('__init__.py', 'tensor_compat.py', 'gpt_oss_llm.py', 'model.json'):
                shutil.copy2(root / name, model_root / name)
            source = (root / 'stage.py').read_text()
            source = source.replace('STAGE_START = 0', 'STAGE_START = 0')
            source = source.replace('STAGE_END = 1', 'STAGE_END = ' + str(end))
            source = source.replace('STAGE_PREFILL = 1', 'STAGE_PREFILL = ' + str(int(prefill)))
            source = source.replace('STAGE_LAST_TOKEN = 0', 'STAGE_LAST_TOKEN = ' + str(int(last_token_logits)))
            source = source.replace('STAGE_TP_RANK = -1', 'STAGE_TP_RANK = ' + str(rank))
            (model_root / 'stage.py').write_text(source)
            shape = [end, plan['kv_pages'], 16, plan['local_kv_heads'], config['head_dim']]
            if kv is None:
                keys = G.tensor_zeros(shape, 'bfloat16')
                values = G.tensor_zeros(shape, 'bfloat16')
            else:
                keys, values = kv[rank]
            first_shape = [batch, tokens]
            model = G.load_model(str(model_root / 'stage.py'), runtime_mode='compiled_xmodel',
                backend='tensorrt', precision='bf16', entry_function='GptOssStage', weights=str(weights),
                cache_dir=str(stage_cache / 'engine'),
                input_shapes=[first_shape, [batch, tokens], shape, shape,
                              [batch, math.ceil(plan['capacity'] / 16)], [batch], [batch], [batch]],
                input_dtypes=['int64', 'int64', 'bfloat16', 'bfloat16', 'int32', 'int32', 'int32', 'int32'],
                compile={'builder_workspace_mb': workspace_mb,
                         'builder_optimization_level': optimization_level,
                         'partition': {'enable_preferred_boundaries': False,
                                       'max_atomic_regions_per_partition': 0}})
            status = model.runtime_status()
            if not status['ready']:
                raise RuntimeError(str(status))
            stages.append(dict(placement, model=model, keys=keys, values=values))
    except Exception:
        for stage in stages:
            G.cuda_set_device(stage['device_id'])
            stage['model'].release_runtime()
        raise
    finally:
        G.cuda_set_device(previous)
    from garnet_pipeline import TensorParallel
    return TensorParallel(stages)
