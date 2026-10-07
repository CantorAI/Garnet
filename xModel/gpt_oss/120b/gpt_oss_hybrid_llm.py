# SPDX-License-Identifier: Apache-2.0
# Experimental full alternating TP2 stage; selected only by an explicit plan.
from . import gpt_oss_llm as llm
from .tensor_compat import tensor
T = tensor()


def forward_stage(x, position_ids, key_pages, value_pages, page_table,
                  context_length, slot_position, active_mask,
                  window_keys, window_values, window_table, weights, config,
                  start, end, prefill, last_token_logits, tp_rank,
                  expert_weight_shard, compact_greedy, moe_intermediate_shard,
                  marlin_prepacked, padded_prefill):
    if start != 0 or end != config['num_hidden_layers'] or end % 2 or tp_rank < 0:
        raise ValueError('Hybrid KV needs complete alternating TP2 stages')
    x = llm.rounded(x * T.unary_op('embedding', weight_name='embedding.weight'))
    for logical_layer in range(end):
        if logical_layer % 2 == 0:
            keys, values, table = window_keys, window_values, window_table
        else:
            keys, values, table = key_pages, value_pages, page_table
        x = llm.layer(x, position_ids, keys, values, table, context_length,
            slot_position, active_mask, config, logical_layer, prefill,
            logical_layer // 2, tp_rank, expert_weight_shard,
            moe_intermediate_shard, marlin_prepacked)
    if last_token_logits:
        if padded_prefill:
            valid_count = context_length * T.binary_op('sub') * slot_position
            x = x * T.binary_op('select_last_valid_sequence') * valid_count
        else:
            x = x * T.unary_op('last_token')
    x = llm.linear(llm.norm(x, 'norm.scale', config['hidden_size']),
        'unembedding.weight', op='lm_head', tp_mode='vocab', tp_rank=tp_rank)
    if compact_greedy:
        x = x * T.unary_op('gpt_oss_vocab_top1', tp_rank=tp_rank,
                           hidden_size=config['vocab_size'] // 2)
        return x * T.unary_op('gpt_oss_tp_all_gather', tp_rank=tp_rank, hidden_size=2)
    return llm.tp_all_gather_logits(x, tp_rank, config)
