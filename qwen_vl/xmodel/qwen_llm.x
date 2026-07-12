from garnet import garnet

T = garnet.tensor()
T.set_backend("TensorRT")

# Qwen3-VL text decoder.
#
# This follows Qwen3VLTextModel / Qwen3VLTextDecoderLayer:
#   embed_tokens
#   interleaved MRoPE rotary embeddings over temporal/height/width ids
#   decoder layers with q_norm and k_norm on head_dim
#   DeepStack visual feature injection into early decoder hidden states
#   final RMSNorm


def linear(x, weight_name, bias_name=None, op="linear"):
    return x * T.unary_op(op, weight_name=weight_name, bias_name=bias_name)


def rms_norm(x, weight_name, eps=1e-6):
    return x * T.unary_op("rms_norm", weight_name=weight_name, eps=eps)


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


def Qwen3TextAttention(x, position_ids, attention_mask, past_key_values, weights, config, layer_idx, use_cache=True):
    prefix = "model.language_model.layers." + str(layer_idx) + ".self_attn"
    head_dim = config.text_config.head_dim
    num_heads = config.text_config.num_attention_heads
    num_kv_heads = config.text_config.num_key_value_heads

    # Keep Q/K/V in one graph value. Tensor expressions have one output; tuple
    # assignment here would incorrectly ask one expression to become two values.
    # The TensorRT backend may lower this sequence as fused QKV/RoPE/KV kernels.
    qkv = x * T.unary_op(
        "qwen3_text_qkv_packed",
        q_weight_name=prefix + ".q_proj.weight",
        k_weight_name=prefix + ".k_proj.weight",
        v_weight_name=prefix + ".v_proj.weight",
        q_norm_weight_name=prefix + ".q_norm.weight",
        k_norm_weight_name=prefix + ".k_norm.weight",
        norm_eps=config.text_config.rms_norm_eps,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim
    )
    qkv = qkv * T.binary_op(
        "qwen3_vl_apply_text_rope_packed",
        rope_theta=config.text_config.rope_theta,
        mrope_section=config.text_config.rope_scaling.mrope_section,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim
    ) * position_ids

    qkv = qkv * T.unary_op(
        "paged_kv_update_packed",
        past_key_values=past_key_values,
        layer_idx=layer_idx,
        enabled=use_cache
    )

    attn = qkv * T.binary_op(
        "paged_attention_packed",
        past_key_values=past_key_values,
        layer_idx=layer_idx,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        causal=True
    ) * attention_mask
    attn = attn * T.unary_op("merge_attention_heads")
    attn = linear(attn, prefix + ".o_proj.weight", None, op="o_proj")
    return attn


def Qwen3TextMLP(x, weights, config, layer_idx):
    prefix = "model.language_model.layers." + str(layer_idx) + ".mlp"
    gate = linear(x, prefix + ".gate_proj.weight", None, op="gate_proj")
    up = linear(x, prefix + ".up_proj.weight", None, op="up_proj")
    hidden = gate * T.unary_op(config.text_config.hidden_act) * up
    return linear(hidden, prefix + ".down_proj.weight", None, op="down_proj")


def Qwen3TextDecoderLayer(x, position_ids, attention_mask, past_key_values, weights, config, layer_idx, use_cache=True):
    prefix = "model.language_model.layers." + str(layer_idx)

    residual = x
    x = rms_norm(
        x,
        prefix + ".input_layernorm.weight",
        eps=config.text_config.rms_norm_eps
    )
    x = Qwen3TextAttention(
        x,
        position_ids,
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
        prefix + ".post_attention_layernorm.weight",
        eps=config.text_config.rms_norm_eps
    )
    x = Qwen3TextMLP(x, weights, config, layer_idx)
    x = residual + x
    return x


def Qwen3TextModel(
    input_ids,
    inputs_embeds,
    position_ids,
    attention_mask,
    deepstack_visual_embeds,
    past_key_values,
    weights,
    config,
    use_cache=True
):
    # The VLM root always supplies merged text/visual embeddings. A text-only
    # entry point can perform embedding lookup before calling this function.
    x = inputs_embeds

    deepstack_count = len(deepstack_visual_embeds)
    for layer_idx in range(deepstack_count):
        x = Qwen3TextDecoderLayer(
            x,
            position_ids,
            attention_mask,
            past_key_values,
            weights,
            config,
            layer_idx,
            use_cache=use_cache
        )

        x = x * T.binary_op(
            "qwen3_vl_deepstack_add",
            input_ids=input_ids,
            image_token_id=config.image_token_id,
            video_token_id=config.video_token_id
        ) * deepstack_visual_embeds[layer_idx]

    for layer_offset in range(config.text_config.num_hidden_layers - deepstack_count):
        layer_idx = deepstack_count + layer_offset
        x = Qwen3TextDecoderLayer(
            x,
            position_ids,
            attention_mask,
            past_key_values,
            weights,
            config,
            layer_idx,
            use_cache=use_cache
        )

    x = rms_norm(x, "model.language_model.norm.weight", eps=config.text_config.rms_norm_eps)
    return {
        "last_hidden_state": x,
        "past_key_values": past_key_values
    }
