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
from kv_layout import hybrid_requested, hybrid_layout, checked_layout, window_page_table


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
    layout = (hybrid_layout(config, batch, capacity, tokens, local_kv_heads)
              if hybrid_requested() else None)
    if layout is not None:
        kv_bytes = layout['shared_kv_bytes'] + layout['shared_auxiliary_bytes']
    activation_bytes = batch * tokens * (config['hidden_size'] * 64 +
        config['vocab_size'] * 4 + config['intermediate_size'] * config['experts_per_token'] * 4)
    # Dense BF16 constants are budgeted at FP32 size because TensorRT may
    # promote some projections during engine construction.
    sizes = weight_sizes(weights)
    full_weights = sum(sizes.values())
    expert_weight_shards = os.environ.get('GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS') == '1'
    intermediate_shards = os.environ.get('GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS') == '1'
    prepacked_flag = os.environ.get('GARNET_GPT_OSS_MARLIN_PREPACKED', '0')
    if prepacked_flag not in ('0', '1'):
        raise ValueError('Marlin prepacking flag must be0/1')
    marlin_prepacked = prepacked_flag == '1'
    if marlin_prepacked and not (intermediate_shards or expert_weight_shards):
        raise ValueError('TP2 prepacking requires explicit expert/intermediate partitioning')
    if intermediate_shards and (expert_weight_shards or config['intermediate_size'] % 64):
        raise ValueError('Intermediate TP2 needs intermediate divisible by64 and expert-axis sharding disabled')
    weight_estimate = full_weights
    weight_storage = {'original_constants_estimated_bytes': full_weights,
        'includes_lazy_marlin_repacking': False}
    if expert_weight_shards:
        experts = config['num_experts']
        local_experts = (experts + 1) // 2  # worst rank for odd expert counts
        suffixes = ('.mlp.mlp1_weight.blocks', '.mlp.mlp1_weight.scales',
                    '.mlp.mlp2_weight.blocks', '.mlp.mlp2_weight.scales',
                    '.mlp.mlp1_bias', '.mlp.mlp2_bias')
        expert_weights = sum(size for name, size in sizes.items() if name.endswith(suffixes))
        if expert_weights % experts:
            raise ValueError('expert weight storage is not divisible by expert count')
        original_constants = full_weights - expert_weights + expert_weights // experts * local_experts
        h, intermediate = config['hidden_size'], config['intermediate_size']
        up_k, up_n = (h + 63) // 64 * 64, (2 * intermediate + 127) // 128 * 128
        down_k, down_n = (intermediate + 127) // 128 * 128, (h + 63) // 64 * 64
        packed_elements = config['num_hidden_layers'] * local_experts * (up_k * up_n + down_k * down_n)
        repacked = packed_elements // 2 + packed_elements // 32
        weight_estimate = original_constants + repacked
        weight_storage = {'original_constants_estimated_bytes': original_constants,
            'lazy_marlin_repacked_bytes': repacked, 'includes_lazy_marlin_repacking': True,
            'worst_rank_local_experts': local_experts}
    if intermediate_shards:
        suffixes = ('.mlp.mlp1_weight.blocks', '.mlp.mlp1_weight.scales',
                    '.mlp.mlp2_weight.blocks', '.mlp.mlp2_weight.scales', '.mlp.mlp1_bias')
        partitioned = sum(size for name, size in sizes.items() if name.endswith(suffixes))
        if partitioned % 2:
            raise ValueError('Intermediate TP2 checkpoint bytes must divide equally')
        original_constants = full_weights - partitioned // 2
        h, local_i = config['hidden_size'], config['intermediate_size'] // 2
        up_k, up_n = (h + 63) // 64 * 64, (2 * local_i + 127) // 128 * 128
        down_k, down_n = (local_i + 127) // 128 * 128, (h + 63) // 64 * 64
        packed_elements = config['num_hidden_layers'] * config['num_experts'] * (up_k * up_n + down_k * down_n)
        repacked = packed_elements // 2 + packed_elements // 32
        weight_estimate = original_constants + repacked
        weight_storage = {'original_constants_estimated_bytes': original_constants,
            'lazy_marlin_repacked_bytes': repacked, 'includes_lazy_marlin_repacking': True,
            'all_experts_per_rank': config['num_experts'], 'intermediate_per_rank': local_i,
            'down_bias_owner_rank': 0}
    if marlin_prepacked:
        # Packed storage can remove only validated original quantized tensors.
        # Refuse malformed/missing tensors instead of underestimating admission.
        h, intermediate = config['hidden_size'], config['intermediate_size']
        for layer in range(config['num_hidden_layers']):
            for projection, n, k in ((1, 2 * intermediate, h), (2, h, intermediate)):
                for suffix, divisor in (('blocks', 2), ('scales', 32)):
                    name = f'block.{layer}.mlp.mlp{projection}_weight.{suffix}'
                    expected = config['num_experts'] * n * k // divisor
                    if k % 32 or sizes.get(name) != expected:
                        raise ValueError('Invalid original Marlin checkpoint storage: ' + name)
        quantized = sum(size for name, size in sizes.items() if name.endswith(
            ('.mlp.mlp1_weight.blocks', '.mlp.mlp1_weight.scales',
             '.mlp.mlp2_weight.blocks', '.mlp.mlp2_weight.scales')))
        removed = (quantized // config['num_experts'] * local_experts
                   if expert_weight_shards else quantized // 2)
        weight_estimate -= removed
        weight_storage['original_constants_estimated_bytes'] -= removed
        weight_storage['prepacked_marlin_constant_bytes'] = weight_storage.pop('lazy_marlin_repacked_bytes')
        weight_storage['includes_lazy_marlin_repacking'] = False
        weight_storage['original_quant_constants_eliminated_bytes'] = removed
        weight_storage['marlin_prepacked_layout_version'] = 1
    from peer_group_layout import peer_group_layout, validate_peer_group_layout
    execution_layout = peer_group_layout(batch, tokens, config['hidden_size'])
    group_bytes = validate_peer_group_layout(execution_layout, batch, tokens, config['hidden_size'])
    required = math.ceil(weight_estimate * 1.05) + kv_bytes + activation_bytes + reserve_bytes + group_bytes
    hardware = [{k: d[k] for k in ('id', 'name', 'total_bytes', 'compute_major',
                'compute_minor', 'pci_bus_id', 'peer_access')} for d in devices]
    budgets = [int(min(d['free_bytes'], d['total_bytes'] * memory_fraction)) for d in devices]
    if any(required > budget for budget in budgets):
        raise ValueError('GPT-OSS TP2 checkpoint estimate exceeds per-GPU memory budget')
    identity = {'schema': 3, 'mode': 'gpt-oss-tensor-parallel-tp2', 'hardware': hardware,
        'checkpoint': str(Path(weights).resolve()), 'capacity': capacity, 'batch': batch,
        'memory_fraction': memory_fraction, 'reserve_bytes': reserve_bytes,
        'layer_cuda_graph': True}
    if layout is not None:
        identity.update(schema=4, kv_layout=layout)
    if expert_weight_shards:
        # Weight storage changes serialized constants and the plugin contract.
        identity['expert_weight_shards'] = True
    if intermediate_shards:
        identity['moe_intermediate_shards'] = True
    if marlin_prepacked:
        identity['marlin_prepacked_layout_version'] = 1
    # These options change getWorkspaceSize()/buffer offsets. A serialized
    # engine built with a smaller layout cannot safely serve the larger one.
    identity['marlin_workspace_layout'] = marlin_workspace_profile()
    identity['collective_workspace_layout'] = collective_workspace_profile()
    if execution_layout is not None:
        identity['operator_execution_layout'] = execution_layout
    compact_greedy = os.environ.get('GARNET_GPT_OSS_COMPACT_VOCAB_GREEDY') == '1'
    if compact_greedy:
        if config['vocab_size'] % 2 or not (0 < config['vocab_size'] <= (1 << 24)):
            raise ValueError('Compact TP2 greedy requires equal vocabulary shards and exact FP32 token IDs')
        identity['compact_vocab_greedy'] = True
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    stages = [{'device_id': d['id'], 'rank': rank, 'start': 0,
               'end': config['num_hidden_layers'], 'estimated_bytes': required,
               'budget_bytes': budget}
              for rank, (d, budget) in enumerate(zip(devices, budgets))]
    result = {'schema': identity['schema'], 'mode': 'gpt-oss-tensor-parallel-tp2', 'cache_key': key,
            'hardware': hardware, 'stages': stages, 'batch': batch,
            'capacity': capacity, 'max_tokens': tokens, 'kv_pages': pages,
            'config': config, 'estimated_per_gpu_bytes': required,
            'local_kv_heads': local_kv_heads,
            'expert_weight_shards': expert_weight_shards,
            'moe_intermediate_shards': intermediate_shards,
            'marlin_prepacked': marlin_prepacked,
            'compact_vocab_greedy': compact_greedy,
            'weight_storage_estimate': weight_storage,
            'marlin_workspace_layout': identity['marlin_workspace_layout'],
            'collective_workspace_layout': identity['collective_workspace_layout']}
    if execution_layout is not None:
        result['operator_execution_layout'] = execution_layout
    if layout is not None:
        result['kv_layout'] = layout
    return result


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


def marlin_workspace_profile():
    try:
        maximum = int(os.environ.get('GARNET_GPT_OSS_MARLIN_MAX_TOKENS', '8'))
        block = int(os.environ.get('GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK', '8'))
        decode = int(os.environ.get('GARNET_GPT_OSS_MARLIN_DECODE_BLOCK', '0'))
        large = int(os.environ.get('GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK', '0'))
    except ValueError as error:
        raise ValueError('Marlin workspace profile must use integer settings') from error
    profile = {'max_tokens': maximum if maximum in (512, 4096) else 8,
            'prefill_block': 32 if block == 32 else 8,
            'decode_block_override': decode if decode in (8, 32) else 0}
    if large == 64:
        profile['large_prefill_block'] = 64
    bounded = os.environ.get('GARNET_GPT_OSS_MARLIN_BOUNDED_PREFILL', '0')
    if bounded not in ('0', '1'):
        raise ValueError('Bounded Marlin prefill flag must be0 or1')
    if bounded == '1':
        profile['bounded_prefill_outer_rows'] = 8192
        profile['bounded_prefill_subcall_rows'] = 4096
    return profile


def estimate_tp2_marlin_workspace_bytes(config, rows, prefill=True):
    """Include every aligned buffer in the plugin's Marlin workspace layout."""
    profile = marlin_workspace_profile()
    if (prefill and profile['max_tokens'] == 4096 and
            4096 < rows <= profile.get('bounded_prefill_outer_rows',4096)):
        rows = profile['bounded_prefill_subcall_rows']
    h, intermediate = config['hidden_size'], config['intermediate_size']
    experts, top_k = config['num_experts'], config['experts_per_token']
    if not (0 < rows <= profile['max_tokens'] and 0 < h <= 16384 and h % 32 == 0
            and 0 < intermediate <= 65536 and intermediate % 32 == 0
            and 0 < experts <= 256 and 0 < top_k <= min(8, experts)):
        return 0
    up_k, up_n = (h + 63) // 64 * 64, (2 * intermediate + 127) // 128 * 128
    down_k, down_n = (intermediate + 127) // 128 * 128, (h + 63) // 64 * 64
    slots, n = rows * top_k, max(up_n, down_n)
    block = 8
    if rows >= 128:
        block = profile['prefill_block']
        if not prefill and profile['decode_block_override']:
            block = profile['decode_block_override']
    if prefill and rows >= 1024 and profile.get('large_prefill_block') == 64:
        block = 64
    padded = slots + experts * block
    buffers = (slots * 4, slots * 4, rows * experts * 4, rows * up_k * 2,
               slots * up_n * 2, slots * down_k * 2, slots * down_n * 2,
               padded * 4, ((padded + block - 1) // block) * 4, 4,
               experts * (n // 64) * 16 * 4,
               2 * min(n * slots * 8, 512 * 4 * 8 * 256) * 4)
    size = 0
    for buffer in buffers:
        size = (size + 15) // 16 * 16 + buffer
    return size


def estimate_tp2_moe_workspace_bytes(config, rows):
    """Mirror the grouped fallback scratch in GptOssMoeWorkspace()."""
    top_k = config['experts_per_token']
    experts = config['num_experts']
    slots = rows * top_k
    scratch = (slots * (4 + 4) + rows * experts * 4 +
               slots * config['intermediate_size'] * 4)
    if rows >= 16:
        tile_rows = 64 if rows > 512 else 32
        tasks = (slots + tile_rows - 1) // tile_rows + experts
        scratch += 4 * (experts + experts * slots + 1 + 2 * tasks)
        scratch += 4 * slots * config['hidden_size']
    return scratch


def collective_workspace_profile():
    # V11 retains the two-BF16-buffer contract and invalidates older engines.
    return 'v11-bf16-pair-prefill128-decode128-512-s1'


def build_tensor_parallel(weights, cache, plan, tokens, prefill, kv=None,
                          last_token_logits=False, padded_prefill=False):
    """Build paired engines with sharded attention heads and rank-local MoE."""
    import garnet as G
    layout = checked_layout(plan)
    if (layout is not None) != hybrid_requested():
        raise ValueError('Hybrid KV flag changed after planning; regenerate the TP2 profile')
    if kv is not None and len(kv) != len(plan['stages']):
        raise ValueError('Exactly one shared KV resource group per rank required')
    if plan.get('collective_workspace_layout') != collective_workspace_profile():
        raise ValueError('Collective workspace layout changed; regenerate the TP2 profile')
    if padded_prefill and not (prefill and last_token_logits):
        raise ValueError('Padded prefill requires last-valid-token logits')
    compact_greedy = plan.get('compact_vocab_greedy', False)
    if compact_greedy and prefill and not last_token_logits:
        raise ValueError('Compact greedy prefill requires one last-token row per request')
    optimization_level = int(os.environ.get('GARNET_GPT_OSS_TRT_OPT_LEVEL', '1'))
    workspace_mb = int(os.environ.get('GARNET_GPT_OSS_TRT_WORKSPACE_MB', '256'))
    if not 0 <= optimization_level <= 5 or not 256 <= workspace_mb <= 4096:
        raise ValueError('unsupported GPT-OSS TensorRT optimization or workspace setting')
    if tokens < 1 or tokens > plan['max_tokens']:
        raise ValueError('token shape exceeds TP2 placement profile')
    config, batch = plan['config'], plan['batch']
    moe_config = dict(config)
    if plan.get('marlin_prepacked', False) != (os.environ.get('GARNET_GPT_OSS_MARLIN_PREPACKED') == '1'):
        raise ValueError('Marlin storage flag changed after planning; regenerate the TP2 profile')
    if plan.get('moe_intermediate_shards', False):
        moe_config['intermediate_size'] //= 2
    if plan.get('marlin_workspace_layout', marlin_workspace_profile()) != marlin_workspace_profile():
        raise ValueError('Marlin workspace settings changed after planning; regenerate the TP2 profile')
    # Match the actual plugin workspace contract in both phases. Batch256
    # decode attention needs260MiB even though the MoE scratch is smaller.
    rows = batch * tokens
    scratch = max(estimate_tp2_moe_workspace_bytes(moe_config, rows),
                  estimate_tp2_marlin_workspace_bytes(moe_config, rows, prefill))
    if (prefill and rows >= 128) or (not prefill and tokens == 1 and 128 <= batch <= 512):
        scratch = max(scratch, rows * config['hidden_size'] * 4)
    if not prefill:
        scratch = max(scratch, batch * (config['num_attention_heads'] // 2) * 64 * 130 * 4)
    needed_mb = (scratch + (1 << 20) - 1) >> 20
    auto_workspace_mb = 1 << max(0, needed_mb - 1).bit_length()
    if auto_workspace_mb > 4096:
        raise ValueError('GPT-OSS TP2 scratch exceeds 4 GiB builder workspace')
    if auto_workspace_mb > workspace_mb:
        workspace_mb = auto_workspace_mb
        print('GPT-OSS TP2', 'prefill' if prefill else 'decode', 'builder workspace', workspace_mb,
              'MiB for', rows, 'rows', flush=True)
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
                stage_cache = stage_cache / 'bf16-gemv-v2'
            if last_token_logits:
                stage_cache = stage_cache / 'last-token-logits'
            if padded_prefill:
                stage_cache = stage_cache / 'padded-prefill-v1'
            model_root = stage_cache / 'xmodel'
            model_root.mkdir(parents=True, exist_ok=True)
            for name in ('__init__.py', 'tensor_compat.py', 'gpt_oss_llm.py', 'model.json'):
                shutil.copy2(root / name, model_root / name)
            if layout is not None:
                shutil.copy2(root / 'gpt_oss_hybrid_llm.py', model_root / 'gpt_oss_hybrid_llm.py')
            if padded_prefill and layout is None:
                llm_path = model_root / 'gpt_oss_llm.py'
                llm_source = llm_path.read_text()
                original = 'x = x * T.unary_op("last_token")'
                if llm_source.count(original) != 1:
                    raise ValueError('Padded prefill source contract changed')
                llm_path.write_text(llm_source.replace(original,
                    'valid_count = context_length * T.binary_op("sub") * slot_position\n'
                    '            x = x * T.binary_op("select_last_valid_sequence") * valid_count'))
            source = (root / ('stage_hybrid.py' if layout is not None else 'stage.py')).read_text()
            source = source.replace('STAGE_START = 0', 'STAGE_START = 0')
            source = source.replace('STAGE_END = 1', 'STAGE_END = ' + str(end))
            source = source.replace('STAGE_PREFILL = 1', 'STAGE_PREFILL = ' + str(int(prefill)))
            source = source.replace('STAGE_LAST_TOKEN = 0', 'STAGE_LAST_TOKEN = ' + str(int(last_token_logits)))
            source = source.replace('STAGE_TP_RANK = -1', 'STAGE_TP_RANK = ' + str(rank))
            source = source.replace('STAGE_EXPERT_WEIGHT_SHARD = 0',
                'STAGE_EXPERT_WEIGHT_SHARD = ' + str(int(plan.get('expert_weight_shards', False))))
            source = source.replace('STAGE_COMPACT_GREEDY = 0',
                'STAGE_COMPACT_GREEDY = ' + str(int(compact_greedy)))
            source = source.replace('STAGE_MOE_INTERMEDIATE_SHARD = 0',
                'STAGE_MOE_INTERMEDIATE_SHARD = ' + str(int(plan.get('moe_intermediate_shards', False))))
            source = source.replace('STAGE_MARLIN_PREPACKED = 0',
                'STAGE_MARLIN_PREPACKED = ' + str(int(plan.get('marlin_prepacked', False))))
            if layout is not None:
                source = source.replace('STAGE_PADDED_PREFILL = 0',
                    'STAGE_PADDED_PREFILL = ' + str(int(padded_prefill)))
            (model_root / 'stage.py').write_text(source)
            shape = (layout['global_shape'] if layout is not None else
                [end, plan['kv_pages'], 16, plan['local_kv_heads'], config['head_dim']])
            auxiliary = {}
            if kv is None:
                keys = G.tensor_zeros(shape, 'bfloat16')
                values = G.tensor_zeros(shape, 'bfloat16')
                if layout is not None:
                    auxiliary = dict(
                        window_keys=G.tensor_zeros(layout['window_shape'], 'bfloat16'),
                        window_values=G.tensor_zeros(layout['window_shape'], 'bfloat16'),
                        window_table=G.tensor_from_host(window_page_table(layout), dtype='int32',
                            shape=layout['window_table_shape'], device='cuda'))
            else:
                shared = kv[rank]
                if layout is not None:
                    if (not isinstance(shared, dict) or
                            shared.get('shared_resource_layout') != layout or
                            shared.get('extra_input_names') != tuple(layout['extra_tensor_arguments']) or
                            set(shared.get('shared_resources', {})) != set(layout['extra_tensor_arguments'])):
                        raise ValueError('Hybrid KV phases need every bank/table with the exact layout')
                    keys, values = shared['keys'], shared['values']
                    auxiliary = dict(shared['shared_resources'])
                else:
                    keys, values = shared
            first_shape = [batch, tokens]
            input_shapes = [first_shape, [batch, tokens], shape, shape,
                [batch, math.ceil(plan['capacity'] / 16)], [batch], [batch], [batch]]
            input_dtypes = ['int64', 'int64', 'bfloat16', 'bfloat16',
                'int32', 'int32', 'int32', 'int32']
            if layout is not None:
                input_shapes += [layout['window_shape'], layout['window_shape'], layout['window_table_shape']]
                input_dtypes += ['bfloat16', 'bfloat16', 'int32']
            model = G.load_model(str(model_root / 'stage.py'), runtime_mode='compiled_xmodel',
                backend='tensorrt', precision='bf16', entry_function='GptOssStage', weights=str(weights),
                cache_dir=str(stage_cache / 'engine'),
                input_shapes=input_shapes, input_dtypes=input_dtypes,
                compile={'builder_workspace_mb': workspace_mb,
                         'builder_optimization_level': optimization_level,
                         'partition': {'enable_preferred_boundaries': False,
                                       'max_atomic_regions_per_partition': 0}})
            try:
                status = model.runtime_status()
                if not status['ready']:
                    raise RuntimeError(str(status))
            except Exception:
                model.release_runtime()
                raise
            stage = dict(placement, model=model, keys=keys, values=values)
            if layout is not None:
                stage.update(shared_resources=auxiliary,
                    shared_resource_layout=json.loads(json.dumps(layout)),
                    extra_input_names=tuple(layout['extra_tensor_arguments']))
            stages.append(stage)
    except Exception:
        for stage in stages:
            G.cuda_set_device(stage['device_id'])
            stage['model'].release_runtime()
        raise
    finally:
        G.cuda_set_device(previous)
    from garnet_pipeline import TensorParallel
    return TensorParallel(stages, greedy_candidate_pairs=compact_greedy)
