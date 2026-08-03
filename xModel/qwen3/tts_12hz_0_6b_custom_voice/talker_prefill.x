from garnet import garnet

T = garnet.tensor()

from "." import talker_common as talker

GARNET_MODEL_SPEC = {
    "arguments": [
        {"name": "aligned_ids", "kind": "tensor"},
        {"name": "position_ids", "kind": "tensor"},
        {"name": "attention_mask", "kind": "tensor"},
        {"name": "key_pages", "kind": "tensor"},
        {"name": "value_pages", "kind": "tensor"},
        {"name": "page_table", "kind": "tensor"},
        {"name": "start_position", "kind": "tensor"},
        {"name": "weights", "kind": "weights"},
        {"name": "config", "kind": "config"}
    ]
}


def attention(x, position_ids, attention_mask, key_pages, value_pages,
              page_table, start_position, config, layer_idx):
    prefix = "talker.model.layers." + str(layer_idx) + ".self_attn"
    qkv = talker.qkv_with_rope(x, position_ids, config, layer_idx)
    keys = key_pages * T.unary_op("paged_kv_select_layer", layer_idx=layer_idx)
    values = value_pages * T.unary_op("paged_kv_select_layer", layer_idx=layer_idx)
    state = qkv * T.binary_op("paged_kv_bind_key_pages") * keys
    state = state * T.binary_op("paged_kv_bind_value_pages") * values
    state = state * T.binary_op("paged_kv_bind_page_table") * page_table
    state = state * T.binary_op("paged_kv_bind_slot_position") * start_position
    qkv = state * T.unary_op(
        "paged_kv_prefill_write_bf16", page_size=16,
        q_heads=config.num_attention_heads,
        kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim)
    x = qkv * T.binary_op(
        "paged_attention_packed",
        num_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        causal=True) * attention_mask
    return talker.linear(x, prefix + ".o_proj.weight", op="o_proj")


def layer(x, position_ids, attention_mask, key_pages, value_pages,
          page_table, start_position, config, layer_idx):
    prefix = "talker.model.layers." + str(layer_idx)
    residual = x
    x = talker.rms_norm(
        x, prefix + ".input_layernorm.weight", config.rms_norm_eps)
    x = residual + attention(
        x, position_ids, attention_mask, key_pages, value_pages,
        page_table, start_position, config, layer_idx)
    residual = x
    x = talker.rms_norm(
        x, prefix + ".post_attention_layernorm.weight", config.rms_norm_eps)
    return residual + talker.mlp(x, config, layer_idx)


@T.fusion(name="qwen3_tts_talker_prefill", role="transformer_prefill", boundary="required")
def Qwen3TTSTalkerPrefill(aligned_ids, position_ids, attention_mask,
                         key_pages, value_pages, page_table, start_position,
                         weights, config):
    talker_config = config.talker_config
    x = talker.prompt_embedding(aligned_ids)
    for layer_idx in range(talker_config.num_hidden_layers):
        x = layer(
            x, position_ids, attention_mask, key_pages, value_pages,
            page_table, start_position, talker_config, layer_idx)
    return talker.finish(x, talker_config)
