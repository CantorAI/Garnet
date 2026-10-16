# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import garnet

T = garnet.tensor()

from . import qwen_asr_llm as llm

GARNET_MODEL_SPEC = {
    "arguments": [
        {"name": "input_ids", "kind": "tensor"},
        {"name": "position_ids", "kind": "tensor"},
        {"name": "key_pages", "kind": "tensor"},
        {"name": "value_pages", "kind": "tensor"},
        {"name": "page_table", "kind": "tensor"},
        {"name": "context_length", "kind": "tensor"},
        {"name": "slot_position", "kind": "tensor"},
        {"name": "weights", "kind": "weights"},
        {"name": "config", "kind": "config"}
    ]
}


def attention(x, position_ids, key_pages, value_pages, page_table,
              context_length, slot_position, config, layer_idx):
    prefix = "thinker.model.layers." + str(layer_idx) + ".self_attn"
    qkv = llm.qkv_with_rope(x, position_ids, config, layer_idx)
    layer_keys = key_pages * T.unary_op(
        "paged_kv_select_layer", layer_idx=layer_idx)
    layer_values = value_pages * T.unary_op(
        "paged_kv_select_layer", layer_idx=layer_idx)
    state = qkv * T.binary_op("paged_kv_bind_key_pages") * layer_keys
    state = state * T.binary_op("paged_kv_bind_value_pages") * layer_values
    state = state * T.binary_op("paged_kv_bind_page_table") * page_table
    state = state * T.binary_op("paged_kv_bind_context_length") * context_length
    state = state * T.binary_op("paged_kv_bind_slot_position") * slot_position
    output = state * T.unary_op(
        "paged_kv_decode_bf16", page_size=16,
        q_heads=config['num_attention_heads'],
        kv_heads=config['num_key_value_heads'],
        head_dim=config['head_dim'])
    return llm.linear(output, prefix + ".o_proj.weight", op="o_proj")


def layer(x, position_ids, key_pages, value_pages, page_table,
          context_length, slot_position, config, layer_idx):
    prefix = "thinker.model.layers." + str(layer_idx)
    residual = x
    normalized = llm.rms_norm(
        x, prefix + ".input_layernorm.weight", config['rms_norm_eps'])
    x = residual + attention(
        normalized, position_ids, key_pages, value_pages, page_table,
        context_length, slot_position, config, layer_idx)
    residual = x
    normalized = llm.rms_norm(
        x, prefix + ".post_attention_layernorm.weight", config['rms_norm_eps'])
    return residual + llm.mlp(normalized, config, layer_idx)


@T.fusion(name="qwen3_asr_decode", role="transformer_decode", boundary="required", cuda_graph=True)
def Qwen3ASRDecode(input_ids, position_ids, key_pages, value_pages,
                   page_table, context_length, slot_position, weights, config):
    text_config = config['thinker_config']['text_config']
    x = input_ids * T.unary_op(
        "embedding", weight_name="thinker.model.embed_tokens.weight")
    for layer_idx in range(text_config['num_hidden_layers']):
        x = layer(
            x, position_ids, key_pages, value_pages, page_table,
            context_length, slot_position, text_config, layer_idx)
    return llm.finish(x, text_config)
