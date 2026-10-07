"""Experimental GPT-OSS TP2 full-history/window-bank contract.

The window ring includes every token written by the largest prefill tile
before attention reads. All logical positions and context reservations remain
unchanged. These calculations do not establish GPU correctness or admission.
"""
import os


def hybrid_requested():
    flag = os.environ.get('GARNET_GPT_OSS_HYBRID_KV', '0')
    if flag not in ('0', '1'):
        raise ValueError('Hybrid KV flag must be0/1')
    return flag == '1'


def hybrid_layout(config, batch, capacity, max_prefill_tokens, local_kv_heads):
    fields = (batch, capacity, max_prefill_tokens, local_kv_heads,
              config['num_hidden_layers'], config['head_dim'], config['sliding_window'])
    if any(type(n) is not int or n <= 0 for n in fields):
        raise ValueError('Hybrid KV dimensions must be positive integers')
    layers, head_dim, window = fields[-3:]
    if layers % 2 or max_prefill_tokens > capacity:
        raise ValueError('Hybrid KV requires complete alternating layers and bounded prefill')
    if config['num_key_value_heads'] != 2 * local_kv_heads:
        raise ValueError('Hybrid KV requires complete TP2 KV head partitioning')
    logical_pages = (capacity + 15) // 16
    ring_pages = min(logical_pages, (window + max_prefill_tokens - 1 + 15) // 16)
    global_shape = [layers // 2, batch * logical_pages, 16, local_kv_heads, head_dim]
    window_shape = [layers // 2, batch * ring_pages, 16, local_kv_heads, head_dim]
    # Two BF16 banks (K/V) at two bytes per element.
    element_bytes = 2 * 2 * (layers // 2) * 16 * local_kv_heads * head_dim
    return dict(schema=1, mode='gpt-oss-alternating-hybrid-kv-v1',
        page_size=16, dtype='bfloat16', max_prefill_tokens=max_prefill_tokens,
        sliding_window=window, window_layers=list(range(0, layers, 2)),
        global_layers=list(range(1, layers, 2)), physical_layer_rule='logical_layer//2',
        logical_pages_per_request=logical_pages, window_pages_per_request=ring_pages,
        global_shape=global_shape, window_shape=window_shape,
        window_table_shape=[batch, logical_pages],
        shared_kv_bytes=element_bytes * batch * (logical_pages + ring_pages),
        shared_auxiliary_bytes=batch * logical_pages * 4,
        extra_tensor_arguments=['window_keys', 'window_values', 'window_table'])


def checked_layout(plan):
    layout = plan.get('kv_layout')
    if layout is None:
        return None
    if plan['schema'] != 4 or plan['mode'] != 'gpt-oss-tensor-parallel-tp2':
        raise ValueError('Hybrid KV requires the versioned complete TP2 plan')
    expected = hybrid_layout(plan['config'], plan['batch'], plan['capacity'],
                             plan['max_tokens'], plan['local_kv_heads'])
    if layout != expected or plan['kv_pages'] != plan['batch'] * expected['logical_pages_per_request']:
        raise ValueError('Hybrid KV layout changed; regenerate plan/profile/cache')
    if (len(plan['stages']) != 2 or len({s['device_id'] for s in plan['stages']}) != 2 or
            [s['rank'] for s in plan['stages']] != [0,1] or
            any(s['start'] != 0 or s['end'] != plan['config']['num_hidden_layers']
                for s in plan['stages'])):
        raise ValueError('Hybrid KV supports only complete alternating TP2 stages')
    return layout


def window_page_table(layout):
    batch, logical = layout['window_table_shape']
    ring = layout['window_pages_per_request']
    return [row * ring + page % ring for row in range(batch) for page in range(logical)]


def kv_memory(plan):
    layout = checked_layout(plan)
    if layout is not None:
        return layout['shared_kv_bytes'], layout['shared_auxiliary_bytes']
    return (plan['config']['num_hidden_layers'] * 2 * plan['kv_pages'] * 16 *
            plan['local_kv_heads'] * plan['config']['head_dim'] * 2, 0)


def logical_retained_kv_bytes(plan, length):
    if type(length) is not int or not 0 <= length <= plan['capacity']:
        raise ValueError('Logical retained KV length exceeds context reservation')
    per_layer = plan['batch'] * 2 * plan['local_kv_heads'] * plan['config']['head_dim'] * 2
    layout = checked_layout(plan)
    layers = plan['config']['num_hidden_layers']
    if layout is None:
        return layers * length * per_layer
    return (layers // 2) * (length + min(length, layout['sliding_window'])) * per_layer
