# SPDX-License-Identifier: Apache-2.0
"""Pure operator selection policy for opt-in GPT-OSS prefill fusion."""


BASE_MOE_OPERATOR = 'gpt_oss_moe_mxfp4'
FUSED_PREFILL_OPERATOR = 'gpt_oss_moe_tp_reduce_bf16'


def select_moe_operator(prefill, tp_rank, moe_intermediate_shard, enabled):
    """Select fusion only for explicit intermediate-sharded TP2 prefill."""
    if enabled not in ('0', '1'):
        raise ValueError('GARNET_GPT_OSS_FUSED_MOE_TP_REDUCE must be 0 or 1')
    if enabled != '1' or not bool(prefill):
        return BASE_MOE_OPERATOR
    if tp_rank not in (0, 1) or moe_intermediate_shard != 1:
        raise ValueError(
            'Fused GPT-OSS MoE peer reduction requires intermediate-sharded TP2 prefill')
    return FUSED_PREFILL_OPERATOR
