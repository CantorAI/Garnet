import CpuTensor as T

# Qwen3-VL text decoder.
#
# This follows Qwen3VLTextModel / Qwen3VLTextDecoderLayer:
#   embed_tokens
#   interleaved MRoPE rotary embeddings over temporal/height/width ids
#   decoder layers with q_norm and k_norm on head_dim
#   DeepStack visual feature injection into early decoder hidden states
#   final RMSNorm


def linear(x, weight, bias=None, op="linear"):
    y = x * T.binary_op(op) * weight
    if bias is not None:
        y = y + bias
    return y


def rms_norm(x, weight, eps=1e-6):
    return x * T.unary_op("rms_norm", weight=weight, eps=eps)


def Qwen3TextRotaryEmbedding(hidden_states, position_ids, config):
    # Qwen3-VL text position_ids are 4-way in the full model:
    #   text_position_ids + 3D multimodal ids (temporal, height, width).
    # The text rotary module consumes the 3D ids and applies interleaved MRoPE.
    return position_ids * T.unary_op(
        "qwen3_vl_text_interleaved_mrope",
        rope_theta=config.text_config.rope_theta,
        head_dim=config.text_config.head_dim,
        mrope_section=config.text_config.rope_scaling.mrope_section
    )


def Qwen3TextAttention(x, position_embeddings, attention_mask, past_key_values, weights, config, layer_idx, use_cache=True):
    prefix = "language_model.layers." + str(layer_idx) + ".self_attn"
    head_dim = config.text_config.head_dim
    num_heads = config.text_config.num_attention_heads
    num_kv_heads = config.text_config.num_key_value_heads

    q = linear(x, weights[prefix + ".q_proj.weight"], None, op="q_proj")
    k = linear(x, weights[prefix + ".k_proj.weight"], None, op="k_proj")
    v = linear(x, weights[prefix + ".v_proj.weight"], None, op="v_proj")

    q = q * T.unary_op("reshape_q_heads", num_heads=num_heads, head_dim=head_dim)
    k = k * T.unary_op("reshape_kv_heads", num_heads=num_kv_heads, head_dim=head_dim)
    v = v * T.unary_op("reshape_kv_heads", num_heads=num_kv_heads, head_dim=head_dim)

    # Qwen3-VL normalizes Q and K per head_dim before RoPE.
    q = rms_norm(q, weights[prefix + ".q_norm.weight"], eps=config.text_config.rms_norm_eps)
    k = rms_norm(k, weights[prefix + ".k_norm.weight"], eps=config.text_config.rms_norm_eps)

    q, k = q * T.binary_op("qwen3_vl_apply_text_rope", position_embeddings=position_embeddings) * k

    if use_cache:
        k, v = k * T.binary_op(
            "paged_kv_update",
            past_key_values=past_key_values,
            layer_idx=layer_idx
        ) * v

    attn = q * T.binary_op(
        "paged_attention",
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        layer_idx=layer_idx,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        scale=head_dim ** -0.5,
        causal=True
    ) * v
    attn = attn * T.unary_op("merge_attention_heads")
    attn = linear(attn, weights[prefix + ".o_proj.weight"], None, op="o_proj")
    return attn


def Qwen3TextMLP(x, weights, config, layer_idx):
    prefix = "language_model.layers." + str(layer_idx) + ".mlp"
    gate = linear(x, weights[prefix + ".gate_proj.weight"], None, op="gate_proj")
    up = linear(x, weights[prefix + ".up_proj.weight"], None, op="up_proj")
    hidden = gate * T.unary_op(config.text_config.hidden_act) * up
    return linear(hidden, weights[prefix + ".down_proj.weight"], None, op="down_proj")


def Qwen3TextDecoderLayer(x, position_embeddings, attention_mask, past_key_values, weights, config, layer_idx, use_cache=True):
    prefix = "language_model.layers." + str(layer_idx)

    residual = x
    x = rms_norm(
        x,
        weights[prefix + ".input_layernorm.weight"],
        eps=config.text_config.rms_norm_eps
    )
    x = Qwen3TextAttention(
        x,
        position_embeddings,
        attention_mask,
        past_key_values,
        weights,
        config,
        layer_idx,
        use_cache=use_cache
    )
    x = residual + x

    residual = x
    x = rms_norm(
        x,
        weights[prefix + ".post_attention_layernorm.weight"],
        eps=config.text_config.rms_norm_eps
    )
    x = Qwen3TextMLP(x, weights, config, layer_idx)
    x = residual + x
    return x


@T.fusion()
def Qwen3TextModel(
    input_ids,
    inputs_embeds,
    position_ids,
    attention_mask,
    visual_pos_masks,
    deepstack_visual_embeds,
    past_key_values,
    weights,
    config,
    use_cache=True
):
    if inputs_embeds is None:
        x = input_ids * T.binary_op("embedding") * weights["language_model.embed_tokens.weight"]
    else:
        x = inputs_embeds

    causal_mask = x * T.unary_op(
        "create_causal_mask",
        attention_mask=attention_mask,
        past_key_values=past_key_values
    )
    position_embeddings = Qwen3TextRotaryEmbedding(x, position_ids, config)

    for layer_idx in range(config.text_config.num_hidden_layers):
        x = Qwen3TextDecoderLayer(
            x,
            position_embeddings,
            causal_mask,
            past_key_values,
            weights,
            config,
            layer_idx,
            use_cache=use_cache
        )

        # DeepStack injects selected merged vision features into early decoder
        # hidden states at visual token positions.
        if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
            x = x * T.binary_op(
                "qwen3_vl_deepstack_add",
                visual_pos_masks=visual_pos_masks
            ) * deepstack_visual_embeds[layer_idx]

    x = rms_norm(x, weights["language_model.norm.weight"], eps=config.text_config.rms_norm_eps)
    return {
        "last_hidden_state": x,
        "past_key_values": past_key_values
    }
