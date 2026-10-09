"""Exact external arena contract, independent of engine workspace/reserve."""
import os


def peer_group_layout(batch, tokens, hidden):
    flag = os.environ.get('GARNET_GPT_OSS_BF16_PEER_GROUP', '0')
    if flag not in ('0', '1'):
        raise ValueError('Owned peer group flag must be0/1')
    grid_flag = os.environ.get('GARNET_GPT_OSS_BF16_PEER_GRID_SIGNALS', '0')
    if grid_flag not in ('0', '1') or (flag == '0' and grid_flag != '0'):
        raise ValueError('Grid signaling must be0/1 and requires owned peer group')
    ctas_text = os.environ.get('GARNET_GPT_OSS_BF16_PEER_GROUP_CTAS', '64')
    if ctas_text not in ('64', '128', '188') or (flag == '0' and ctas_text != '64'):
        raise ValueError('Owned peer group CTA policy must be64/128/188 and enabled')
    if flag == '0':
        return None
    ctas = int(ctas_text)
    counts = [batch * tokens * hidden, batch * hidden]
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
    return result


def validate_peer_group_layout(layout, batch, tokens, hidden):
    # Validate a stored contract without consulting mutable process flags.
    if layout is None:
        return 0
    counts = [batch * tokens * hidden, batch * hidden]
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
    if layout != expected:
        raise ValueError('Peer group storage/lifetime contract mismatch')
    # Conservatively charge the entire mapped amount against EACH GPU.
    return expected['owned_bytes_per_rank'] + expected['mapped_host_bytes']
