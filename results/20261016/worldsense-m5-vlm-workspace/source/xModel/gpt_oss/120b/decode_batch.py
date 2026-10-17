# SPDX-License-Identifier: Apache-2.0
import garnet
import garnet_gpt_oss
garnet.bind_operator_module(garnet_gpt_oss)
from .tensor_compat import tensor
from . import gpt_oss_llm as llm
T = tensor()
GARNET_MODEL_SPEC = {'arguments': [{'name': 'input_ids', 'kind': 'tensor'}, {'name': 'position_ids', 'kind': 'tensor'}, {'name': 'key_pages', 'kind': 'tensor'}, {'name': 'value_pages', 'kind': 'tensor'}, {'name': 'page_table', 'kind': 'tensor'}, {'name': 'context_length', 'kind': 'tensor'}, {'name': 'slot_position', 'kind': 'tensor'}, {'name': 'active_mask', 'kind': 'tensor'}, {'name': 'weights', 'kind': 'weights'}, {'name': 'config', 'kind': 'config'}]}
GARNET_MODEL_SPEC["requires"] = {'operator_plugins': [{'id': 'gpt_oss', 'module': 'garnet_gpt_oss', 'abi': 1, 'backend': 'tensorrt', 'operators': ['gpt_oss_round_bf16', 'gpt_oss_apply_yarn_rope_packed', 'gpt_oss_paged_attention', 'gpt_oss_moe_mxfp4', 'gpt_oss_moe_tp_reduce_bf16', 'gpt_oss_rms_norm']}]}

@T.fusion(name="gpt_oss_120b_decode_batch", role="transformer_decode", boundary="required")
def GptOssDecodeBatch(input_ids, position_ids, key_pages, value_pages, page_table,
           context_length, slot_position, active_mask, weights, config):
    return llm.forward(input_ids, position_ids, key_pages, value_pages, page_table,
                       context_length, slot_position, active_mask, weights, config, 0)
