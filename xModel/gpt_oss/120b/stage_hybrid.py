# SPDX-License-Identifier: Apache-2.0
# Separate template: preserve the original stage's eight-input public contract.
import garnet
import garnet_gpt_oss
garnet.bind_operator_module(garnet_gpt_oss)
from .tensor_compat import tensor
from . import gpt_oss_hybrid_llm as llm
T = tensor()
STAGE_START = 0
STAGE_END = 1
STAGE_PREFILL = 1
STAGE_LAST_TOKEN = 0
STAGE_TP_RANK = -1
STAGE_EXPERT_WEIGHT_SHARD = 0
STAGE_COMPACT_GREEDY = 0
STAGE_MOE_INTERMEDIATE_SHARD = 0
STAGE_MARLIN_PREPACKED = 0
STAGE_PADDED_PREFILL = 0
STAGE_OPERATORS = ['gpt_oss_round_bf16', 'gpt_oss_apply_yarn_rope_packed',
                   'gpt_oss_paged_attention', 'gpt_oss_moe_mxfp4',
                   'gpt_oss_moe_tp_reduce_bf16', 'gpt_oss_rms_norm',
                   'gpt_oss_tp_all_reduce', 'gpt_oss_tp_all_gather']
if STAGE_COMPACT_GREEDY:
    STAGE_OPERATORS.append('gpt_oss_vocab_top1')
GARNET_MODEL_SPEC = {'arguments': [
    {'name': name, 'kind': 'tensor'} for name in
    ['hidden_or_ids', 'position_ids', 'key_pages', 'value_pages', 'page_table',
     'context_length', 'slot_position', 'active_mask', 'window_keys',
     'window_values', 'window_table']] + [
    {'name': 'weights', 'kind': 'weights'}, {'name': 'config', 'kind': 'config'},
    {'name': 'start', 'kind': 'int', 'value': STAGE_START},
    {'name': 'end', 'kind': 'int', 'value': STAGE_END},
    {'name': 'prefill', 'kind': 'int', 'value': STAGE_PREFILL},
    {'name': 'last_token_logits', 'kind': 'int', 'value': STAGE_LAST_TOKEN},
    {'name': 'tp_rank', 'kind': 'int', 'value': STAGE_TP_RANK},
    {'name': 'expert_weight_shard', 'kind': 'int', 'value': STAGE_EXPERT_WEIGHT_SHARD},
    {'name': 'compact_greedy', 'kind': 'int', 'value': STAGE_COMPACT_GREEDY},
    {'name': 'moe_intermediate_shard', 'kind': 'int', 'value': STAGE_MOE_INTERMEDIATE_SHARD},
    {'name': 'marlin_prepacked', 'kind': 'int', 'value': STAGE_MARLIN_PREPACKED},
    {'name': 'padded_prefill', 'kind': 'int', 'value': STAGE_PADDED_PREFILL}],
    'requires': {'operator_plugins': [{'id': 'gpt_oss', 'module': 'garnet_gpt_oss',
    'abi': 1, 'backend': 'tensorrt', 'operators': STAGE_OPERATORS}]}}

@T.fusion(name='gpt_oss_stage', role='transformer_stage', boundary='required')
def GptOssStage(x, position_ids, keys, values, table, length, slot, active,
                window_keys, window_values, window_table, weights, config,
                start, end, prefill, last_token_logits, tp_rank,
                expert_weight_shard, compact_greedy, moe_intermediate_shard,
                marlin_prepacked, padded_prefill):
    return llm.forward_stage(x, position_ids, keys, values, table, length, slot,
        active, window_keys, window_values, window_table, weights, config,
        start, end, prefill, last_token_logits, tp_rank, expert_weight_shard,
        compact_greedy, moe_intermediate_shard, marlin_prepacked, padded_prefill)
