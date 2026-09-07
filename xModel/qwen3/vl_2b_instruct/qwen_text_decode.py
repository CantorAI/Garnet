import garnet

T = garnet.tensor()

from . import qwen_llm as llm

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


def DecodeAttention(x, position_ids, key_pages, value_pages, page_table,
                    context_length, slot_position, config, layer_idx):
    prefix = "model.language_model.layers." + str(layer_idx) + ".self_attn"
    q_heads = config['text_config']['num_attention_heads']
    kv_heads = config['text_config']['num_key_value_heads']
    head_dim = config['text_config']['head_dim']

    qkv = x * T.unary_op(
        "qwen3_text_qkv_packed",
        q_weight_name=prefix + ".q_proj.weight",
        k_weight_name=prefix + ".k_proj.weight",
        v_weight_name=prefix + ".v_proj.weight",
        q_norm_weight_name=prefix + ".q_norm.weight",
        k_norm_weight_name=prefix + ".k_norm.weight",
        norm_eps=config['text_config']['rms_norm_eps'],
        num_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim
    )
    qkv = qkv * T.binary_op(
        "qwen3_vl_apply_text_rope_packed",
        rope_theta=config['text_config']['rope_theta'],
        mrope_section=config['text_config']['rope_scaling']['mrope_section'],
        num_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim
    ) * position_ids

    layer_keys = key_pages * T.unary_op("paged_kv_select_layer", layer_idx=layer_idx)
    layer_values = value_pages * T.unary_op("paged_kv_select_layer", layer_idx=layer_idx)
    state = qkv * T.binary_op("paged_kv_bind_key_pages") * layer_keys
    state = state * T.binary_op("paged_kv_bind_value_pages") * layer_values
    state = state * T.binary_op("paged_kv_bind_page_table") * page_table
    state = state * T.binary_op("paged_kv_bind_context_length") * context_length
    state = state * T.binary_op("paged_kv_bind_slot_position") * slot_position
    attention = state * T.unary_op(
        "paged_kv_decode_bf16",
        page_size=16,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim
    )
    return llm.linear(attention, prefix + ".o_proj.weight", None, op="o_proj")


@T.fusion(role="decoder_layer", atomic=True)
def DecodeLayer(x, position_ids, key_pages, value_pages, page_table,
                context_length, slot_position, weights, config, layer_idx):
    prefix = "model.language_model.layers." + str(layer_idx)
    residual = x
    normalized = llm.rms_norm(
        x,
        prefix + ".input_layernorm.weight",
        eps=config['text_config']['rms_norm_eps']
    )
    attention = DecodeAttention(
        normalized,
        position_ids,
        key_pages,
        value_pages,
        page_table,
        context_length,
        slot_position,
        config,
        layer_idx
    )
    x = residual + attention
    residual = x
    normalized = llm.rms_norm(
        x,
        prefix + ".post_attention_layernorm.weight",
        eps=config['text_config']['rms_norm_eps']
    )
    return residual + llm.Qwen3TextMLP(normalized, weights, config, layer_idx)


@T.fusion(
    name="text_decode",
    role="transformer_decode",
    boundary="required",
    cuda_graph=True
)
def Qwen3TextDecode(input_ids, position_ids, key_pages, value_pages,
                    page_table, context_length, slot_position, weights, config):
    x = input_ids * T.unary_op(
        "embedding",
        weight_name="model.language_model.embed_tokens.weight"
    )
    for layer_idx in range(config['text_config']['num_hidden_layers']):
        x = DecodeLayer(
            x,
            position_ids,
            key_pages,
            value_pages,
            page_table,
            context_length,
            slot_position,
            weights,
            config,
            layer_idx
        )
    x = llm.rms_norm(
        x,
        "model.language_model.norm.weight",
        eps=config['text_config']['rms_norm_eps']
    )
    return x * T.unary_op(
        "lm_head",
        tied_word_embeddings=config['tie_word_embeddings'],
        weight_name="model.language_model.embed_tokens.weight"
    )
