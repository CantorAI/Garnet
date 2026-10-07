# SPDX-License-Identifier: Apache-2.0
# Instantiated in the engine cache by tools/gpt_oss/pipeline.py.
import garnet
import garnet_gpt_oss
garnet.bind_operator_module(garnet_gpt_oss)
from .tensor_compat import tensor
from . import gpt_oss_llm as llm
T = tensor()
STAGE_START = 0
STAGE_END = 1
STAGE_PREFILL = 1
STAGE_LAST_TOKEN = 0
STAGE_TP_RANK = -1
STAGE_EXPERT_WEIGHT_SHARD = 0
STAGE_COMPACT_GREEDY = 0
STAGE_OPERATORS = ['gpt_oss_round_bf16', 'gpt_oss_apply_yarn_rope_packed',
                   'gpt_oss_paged_attention', 'gpt_oss_moe_mxfp4', 'gpt_oss_rms_norm']
if STAGE_TP_RANK >= 0:
    STAGE_OPERATORS.extend(['gpt_oss_tp_all_reduce', 'gpt_oss_tp_all_gather'])
if STAGE_COMPACT_GREEDY:
    STAGE_OPERATORS.append('gpt_oss_vocab_top1')
GARNET_MODEL_SPEC = {'arguments': [
    {'name': name, 'kind': 'tensor'} for name in
    ['hidden_or_ids', 'position_ids', 'key_pages', 'value_pages', 'page_table',
     'context_length', 'slot_position', 'active_mask']] + [
    {'name': 'weights', 'kind': 'weights'}, {'name': 'config', 'kind': 'config'},
    {'name': 'start', 'kind': 'int', 'value': STAGE_START},
    {'name': 'end', 'kind': 'int', 'value': STAGE_END},
    {'name': 'prefill', 'kind': 'int', 'value': STAGE_PREFILL},
    {'name': 'last_token_logits', 'kind': 'int', 'value': STAGE_LAST_TOKEN},
    {'name': 'tp_rank', 'kind': 'int', 'value': STAGE_TP_RANK},
    {'name': 'expert_weight_shard', 'kind': 'int', 'value': STAGE_EXPERT_WEIGHT_SHARD},
    {'name': 'compact_greedy', 'kind': 'int', 'value': STAGE_COMPACT_GREEDY}],
    'requires': {'operator_plugins': [{'id': 'gpt_oss', 'module': 'garnet_gpt_oss',
    'abi': 1, 'backend': 'tensorrt', 'operators': STAGE_OPERATORS}]}}

@T.fusion(name='gpt_oss_stage', role='transformer_stage', boundary='required')
def GptOssStage(x, position_ids, keys, values, table, length, slot, active,
                weights, config, start, end, prefill, last_token_logits, tp_rank,
                expert_weight_shard, compact_greedy):
    return llm.forward_stage(x, position_ids, keys, values, table, length, slot,
                            active, weights, config, start, end, prefill, last_token_logits,
                            tp_rank, expert_weight_shard, compact_greedy)
