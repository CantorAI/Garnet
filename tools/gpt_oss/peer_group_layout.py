"""Exact external arena contract, independent of engine workspace/reserve."""
import os


def peer_group_layout(batch, tokens, hidden):
    flag = os.environ.get('GARNET_GPT_OSS_BF16_PEER_GROUP', '0')
    if flag not in ('0', '1'):
        raise ValueError('Owned peer group flag must be0/1')
    if flag == '0':
        return None
    counts = [batch * tokens * hidden, batch * hidden]
    if (any(type(v) is not int or v <= 0 for v in (batch, tokens, hidden)) or
            hidden != 2880 or counts[0] > 4096 * 2880):
        raise ValueError('Owned peer group requires hidden2880 and at most4096 rows')
    capacity = max(counts)
    # One BF16 arena,128-byte epochs and U32 faults per CTA; four mapped
    #128-byte signal words per CTA. Keep all amounts outside the2GiB reserve.
    return dict(schema=1, id='gpt_oss', backend='tensorrt',
        protocol='owned-mapped-bf16-peer-v1', ctas=64, phase_elements=counts,
        capacity_elements=capacity, owned_bytes_per_rank=capacity * 2 + 64 * 132,
        mapped_host_bytes=64 * 512,
        concurrency='serial paired phases, one invocation per rank')


def validate_peer_group_layout(layout, batch, tokens, hidden):
    # Validate a stored contract without consulting mutable process flags.
    if layout is None:
        return 0
    counts = [batch * tokens * hidden, batch * hidden]
    if hidden != 2880 or not 0 < counts[0] <= 4096 * 2880:
        raise ValueError('Invalid stored peer capacity')
    expected = dict(schema=1, id='gpt_oss', backend='tensorrt',
        protocol='owned-mapped-bf16-peer-v1', ctas=64, phase_elements=counts,
        capacity_elements=max(counts), owned_bytes_per_rank=max(counts) * 2 + 64 * 132,
        mapped_host_bytes=64 * 512,
        concurrency='serial paired phases, one invocation per rank')
    if layout != expected:
        raise ValueError('Peer group storage/lifetime contract mismatch')
    # Conservatively charge the entire mapped amount against EACH GPU.
    return expected['owned_bytes_per_rank'] + expected['mapped_host_bytes']
