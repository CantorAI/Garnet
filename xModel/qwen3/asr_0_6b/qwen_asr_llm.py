import garnet

T = garnet.tensor()


def linear(x, weight_name, op="linear", bias_name=None):
    return x * T.unary_op(
        op, weight_name=weight_name, bias_name=bias_name)


def rms_norm(x, weight_name, eps):
    return x * T.unary_op("rms_norm", weight_name=weight_name, eps=eps)


def qkv_with_rope(x, position_ids, config, layer_idx):
    prefix = "thinker.model.layers." + str(layer_idx) + ".self_attn"
    qkv = x * T.unary_op(
        "qwen3_text_qkv_packed",
        q_weight_name=prefix + ".q_proj.weight",
        k_weight_name=prefix + ".k_proj.weight",
        v_weight_name=prefix + ".v_proj.weight",
        q_norm_weight_name=prefix + ".q_norm.weight",
        k_norm_weight_name=prefix + ".k_norm.weight",
        norm_eps=config['rms_norm_eps'],
        num_heads=config['num_attention_heads'],
        num_kv_heads=config['num_key_value_heads'],
        head_dim=config['head_dim']
    )
    return qkv * T.binary_op(
        "qwen3_vl_apply_text_rope_packed",
        rope_theta=config['rope_theta'],
        mrope_section=config['rope_scaling']['mrope_section'],
        num_heads=config['num_attention_heads'],
        num_kv_heads=config['num_key_value_heads'],
        head_dim=config['head_dim']
    ) * position_ids


def mlp(x, config, layer_idx):
    prefix = "thinker.model.layers." + str(layer_idx) + ".mlp"
    hidden = x * T.unary_op(
        "qwen3_mlp_gate_up_swiglu_packed",
        gate_weight_name=prefix + ".gate_proj.weight",
        up_weight_name=prefix + ".up_proj.weight"
    )
    return linear(hidden, prefix + ".down_proj.weight", op="down_proj")


def finish(x, config):
    x = rms_norm(x, "thinker.model.norm.weight", config['rms_norm_eps'])
    return x * T.unary_op(
        "lm_head", tied_word_embeddings=True,
        weight_name="thinker.lm_head.weight")
