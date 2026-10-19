"""Exact external arena contract, independent of engine workspace/reserve."""
import os


def peer_group_layout(batch, tokens, hidden, *, prefill_batch=None, decode_batch=None):
    flag = os.environ.get('GARNET_GPT_OSS_BF16_PEER_GROUP', '0')
    if flag not in ('0', '1'):
        raise ValueError('Owned peer group flag must be0/1')
    grid_flag = os.environ.get('GARNET_GPT_OSS_BF16_PEER_GRID_SIGNALS', '0')
    if grid_flag not in ('0', '1') or (flag == '0' and grid_flag != '0'):
        raise ValueError('Grid signaling must be0/1 and requires owned peer group')
    ctas_text = os.environ.get('GARNET_GPT_OSS_BF16_PEER_GROUP_CTAS', '64')
    if ctas_text not in ('64', '128', '188') or (flag == '0' and ctas_text != '64'):
        raise ValueError('Owned peer group CTA policy must be64/128/188 and enabled')
    threads_text = os.environ.get('GARNET_GPT_OSS_BF16_PEER_GROUP_THREADS', '256')
    if threads_text not in ('256', '512') or (flag == '0' and threads_text != '256'):
        raise ValueError('Owned peer group threads-per-CTA policy must be256/512 and enabled')
    prefill_batch = batch if prefill_batch is None else prefill_batch
    decode_batch = batch if decode_batch is None else decode_batch
    if (type(batch) is not int or type(tokens) is not int or
            type(prefill_batch) is not int or type(decode_batch) is not int or
            not 0 < prefill_batch <= batch or not 0 < decode_batch <= batch):
        raise ValueError('Peer phase batches must fit the shared KV-pool capacity')
    if flag == '0':
        return None
    ctas = int(ctas_text)
    counts = [prefill_batch * tokens * hidden, decode_batch * hidden]
    if (any(type(v) is not int or v <= 0 for v in (batch, tokens, hidden)) or
            hidden != 2880 or counts[0] > 4096 * 2880):
        raise ValueError('Owned peer group requires hidden2880 and at most4096 rows')
    capacity = max(counts)
    # One BF16 arena,128-byte epochs and U32 faults per CTA; four mapped
    #128-byte signal words per CTA. Keep all amounts outside the2GiB reserve.
    result = dict(schema=2, id='gpt_oss', backend='tensorrt',
        protocol='owned-mapped-bf16-peer-v2', ctas=ctas, phase_elements=counts,
        capacity_elements=capacity, owned_bytes_per_rank=capacity * 2 + ctas * 132,
        mapped_host_bytes=ctas * 512,
        concurrency='serial paired phases, one invocation per rank')
    if grid_flag == '1':
        result.update(schema=3, protocol='owned-mapped-bf16-peer-grid-v3', grid_signals=1)
    if threads_text == '512':
        result.update(schema=4,
            protocol=('owned-mapped-bf16-peer-grid-threads-v1' if grid_flag == '1'
                      else 'owned-mapped-bf16-peer-threads-v1'),
            threads_per_cta=512)
    if prefill_batch != batch or decode_batch != batch:
        result.update(prefill_batch=prefill_batch, decode_batch=decode_batch)
    return result


def validate_peer_group_layout(layout, batch, tokens, hidden, *,
                               prefill_batch=None, decode_batch=None):
    # Validate a stored contract without consulting mutable process flags.
    if layout is None:
        return 0
    prefill_batch = (layout.get('prefill_batch', batch) if prefill_batch is None
                     else prefill_batch)
    decode_batch = (layout.get('decode_batch', batch) if decode_batch is None
                    else decode_batch)
    if (type(prefill_batch) is not int or type(decode_batch) is not int or
            not 0 < prefill_batch <= batch or not 0 < decode_batch <= batch):
        raise ValueError('Invalid stored peer phase batches')
    counts = [prefill_batch * tokens * hidden, decode_batch * hidden]
    ctas = layout.get('ctas') if isinstance(layout, dict) else None
    if hidden != 2880 or not 0 < counts[0] <= 4096 * 2880 or ctas not in (64,128,188):
        raise ValueError('Invalid stored peer capacity')
    expected = dict(schema=2, id='gpt_oss', backend='tensorrt',
        protocol='owned-mapped-bf16-peer-v2', ctas=ctas, phase_elements=counts,
        capacity_elements=max(counts), owned_bytes_per_rank=max(counts) * 2 + ctas * 132,
        mapped_host_bytes=ctas * 512,
        concurrency='serial paired phases, one invocation per rank')
    if layout.get('grid_signals') == 1 and type(layout.get('grid_signals')) is int:
        expected.update(schema=3, protocol='owned-mapped-bf16-peer-grid-v3', grid_signals=1)
    threads = layout.get('threads_per_cta', 256)
    if type(threads) is not int or threads not in (256, 512):
        raise ValueError('Invalid stored peer thread geometry')
    if threads == 512:
        expected.update(schema=4,
            protocol=('owned-mapped-bf16-peer-grid-threads-v1'
                      if expected.get('grid_signals') == 1
                      else 'owned-mapped-bf16-peer-threads-v1'),
            threads_per_cta=512)
    if prefill_batch != batch or decode_batch != batch:
        expected.update(prefill_batch=prefill_batch, decode_batch=decode_batch)
    if layout != expected:
        raise ValueError('Peer group storage/lifetime contract mismatch')
    # Conservatively charge the entire mapped amount against EACH GPU.
    return expected['owned_bytes_per_rank'] + expected['mapped_host_bytes']
