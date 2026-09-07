import garnet

T = garnet.tensor()

from . import qwen_llm as llm

GARNET_MODEL_SPEC = {
    "arguments": [
        {"name": "input_ids", "kind": "tensor"},
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
    prefix = "model.layers." + str(layer_idx) + ".self_attn"
    qkv = llm.qkv_with_rope(x, position_ids, config, layer_idx)
    layer_keys = key_pages * T.unary_op(
        "paged_kv_select_layer", layer_idx=layer_idx)
    layer_values = value_pages * T.unary_op(
        "paged_kv_select_layer", layer_idx=layer_idx)
    state = qkv * T.binary_op("paged_kv_bind_key_pages") * layer_keys
    state = state * T.binary_op("paged_kv_bind_value_pages") * layer_values
    state = state * T.binary_op("paged_kv_bind_page_table") * page_table
    state = state * T.binary_op(
        "paged_kv_bind_slot_position") * start_position
    qkv = state * T.unary_op(
        "paged_kv_prefill_write_bf16",
        page_size=16,
        q_heads=config['num_attention_heads'],
        kv_heads=config['num_key_value_heads'],
        head_dim=config['head_dim']
    )
    output = qkv * T.binary_op(
        "paged_attention_packed",
        num_heads=config['num_attention_heads'],
        num_key_value_heads=config['num_key_value_heads'],
        head_dim=config['head_dim'],
        causal=True
    ) * attention_mask
    output = output * T.unary_op("merge_attention_heads")
    return llm.linear(output, prefix + ".o_proj.weight", op="o_proj")


@T.fusion(role="decoder_layer", atomic=True)
def layer(x, position_ids, attention_mask, key_pages, value_pages,
          page_table, start_position, config, layer_idx):
    prefix = "model.layers." + str(layer_idx)
    residual = x
    normalized = llm.rms_norm(
        x, prefix + ".input_layernorm.weight", config['rms_norm_eps'])
    x = residual + attention(
        normalized, position_ids, attention_mask, key_pages, value_pages,
        page_table, start_position, config, layer_idx)
    residual = x
    normalized = llm.rms_norm(
        x, prefix + ".post_attention_layernorm.weight", config['rms_norm_eps'])
    return residual + llm.mlp(normalized, config, layer_idx)


@T.fusion(
    name="qwen3_1_7b_prefill",
    role="transformer_prefill",
    boundary="required"
)
def Qwen3Prefill(input_ids, position_ids, attention_mask, key_pages,
                 value_pages, page_table, start_position, weights, config):
    x = input_ids * T.unary_op(
        "embedding", weight_name="model.embed_tokens.weight")
    for layer_idx in range(config['num_hidden_layers']):
        x = layer(
            x, position_ids, attention_mask, key_pages, value_pages,
            page_table, start_position, config, layer_idx)
    return llm.finish(x, config)
