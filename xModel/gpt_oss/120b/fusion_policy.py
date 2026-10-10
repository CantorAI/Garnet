# SPDX-License-Identifier: Apache-2.0
"""Pure operator selection policy for opt-in GPT-OSS prefill fusion."""


BASE_MOE_OPERATOR = 'gpt_oss_moe_mxfp4'
FUSED_PREFILL_OPERATOR = 'gpt_oss_moe_tp_reduce_bf16'


def select_moe_operator(prefill, tp_rank, moe_intermediate_shard, enabled,
                        peer_group_enabled, expert_weight_shard=0):
    """Select fusion only for explicit intermediate-sharded TP2 prefill."""
    if enabled not in ('0', '1'):
        raise ValueError('GARNET_GPT_OSS_FUSED_MOE_TP_REDUCE must be 0 or 1')
    if enabled != '1':
        return BASE_MOE_OPERATOR
    if type(prefill) is bool:
        is_prefill = prefill
    elif type(prefill) is int and prefill in (0, 1):
        is_prefill = bool(prefill)
    else:
        raise ValueError('prefill must be bool or integer 0/1')
    if not is_prefill:
        return BASE_MOE_OPERATOR
    if peer_group_enabled != '1':
        raise ValueError(
            'Fused GPT-OSS MoE peer reduction requires the owned peer group')
    if (type(tp_rank) is not int or tp_rank not in (0, 1) or
            type(moe_intermediate_shard) is not int or moe_intermediate_shard != 1):
        raise ValueError(
            'Fused GPT-OSS MoE peer reduction requires intermediate-sharded TP2 prefill')
    if type(expert_weight_shard) is not int or expert_weight_shard != 0:
        raise ValueError(
            'Fused GPT-OSS MoE peer reduction requires unsharded expert weights')
    return FUSED_PREFILL_OPERATOR
