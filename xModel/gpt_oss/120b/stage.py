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
GARNET_MODEL_SPEC = {'arguments': [
    {'name': name, 'kind': 'tensor'} for name in
    ['hidden_or_ids', 'position_ids', 'key_pages', 'value_pages', 'page_table',
     'context_length', 'slot_position', 'active_mask']] + [
    {'name': 'weights', 'kind': 'weights'}, {'name': 'config', 'kind': 'config'},
    {'name': 'start', 'kind': 'int', 'value': STAGE_START},
    {'name': 'end', 'kind': 'int', 'value': STAGE_END},
    {'name': 'prefill', 'kind': 'int', 'value': STAGE_PREFILL}],
    'requires': {'operator_plugins': [{'id': 'gpt_oss', 'module': 'garnet_gpt_oss',
    'abi': 1, 'backend': 'tensorrt', 'operators': ['gpt_oss_round_bf16',
    'gpt_oss_apply_yarn_rope_packed', 'gpt_oss_paged_attention', 'gpt_oss_moe_mxfp4']}]}}

@T.fusion(name='gpt_oss_stage', role='transformer_stage', boundary='required')
def GptOssStage(x, position_ids, keys, values, table, length, slot, active,
                weights, config, start, end, prefill):
    return llm.forward_stage(x, position_ids, keys, values, table, length, slot,
                            active, weights, config, start, end, prefill)
