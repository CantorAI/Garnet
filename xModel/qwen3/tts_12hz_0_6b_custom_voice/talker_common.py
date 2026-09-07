import garnet

T = garnet.tensor()


def linear(x, weight_name, op="linear", bias_name=None):
    return x * T.unary_op(
        op, weight_name=weight_name, bias_name=bias_name)


def rms_norm(x, weight_name, eps):
    return x * T.unary_op("rms_norm", weight_name=weight_name, eps=eps)


def qkv_with_rope(x, position_ids, config, layer_idx, prefix="talker.model"):
    base = prefix + ".layers." + str(layer_idx) + ".self_attn"
    qkv = x * T.unary_op(
        "qwen3_text_qkv_packed",
        q_weight_name=base + ".q_proj.weight",
        k_weight_name=base + ".k_proj.weight",
        v_weight_name=base + ".v_proj.weight",
        q_norm_weight_name=base + ".q_norm.weight",
        k_norm_weight_name=base + ".k_norm.weight",
        norm_eps=config['rms_norm_eps'],
        num_heads=config['num_attention_heads'],
        num_kv_heads=config['num_key_value_heads'],
        head_dim=config['head_dim'])
    return qkv * T.binary_op(
        "qwen3_vl_apply_text_rope_packed",
        rope_theta=config['rope_theta'],
        mrope_section=config['rope_scaling']['mrope_section'],
        num_heads=config['num_attention_heads'],
        num_kv_heads=config['num_key_value_heads'],
        head_dim=config['head_dim']) * position_ids


def mlp(x, config, layer_idx, prefix="talker.model"):
    base = prefix + ".layers." + str(layer_idx) + ".mlp"
    hidden = x * T.unary_op(
        "qwen3_mlp_gate_up_swiglu_packed",
        gate_weight_name=base + ".gate_proj.weight",
        up_weight_name=base + ".up_proj.weight")
    return linear(hidden, base + ".down_proj.weight", op="down_proj")


def prompt_embedding(aligned_ids):
    return aligned_ids * T.unary_op(
        "qwen3_tts_aligned_prompt_embedding",
        text_embedding_name="talker.model.text_embedding.weight",
        text_fc1_weight_name="talker.text_projection.linear_fc1.weight",
        text_fc1_bias_name="talker.text_projection.linear_fc1.bias",
        text_fc2_weight_name="talker.text_projection.linear_fc2.weight",
        text_fc2_bias_name="talker.text_projection.linear_fc2.bias",
        codec_embedding_name="talker.model.codec_embedding.weight")


def decode_embedding(codec_ids, config):
    return codec_ids * T.unary_op(
        "qwen3_tts_decode_embedding",
        codec_embedding_name="talker.model.codec_embedding.weight",
        predictor_embedding_prefix=
            "talker.code_predictor.model.codec_embedding.",
        text_embedding_name="talker.model.text_embedding.weight",
        text_fc1_weight_name="talker.text_projection.linear_fc1.weight",
        text_fc1_bias_name="talker.text_projection.linear_fc1.bias",
        text_fc2_weight_name="talker.text_projection.linear_fc2.weight",
        text_fc2_bias_name="talker.text_projection.linear_fc2.bias",
        tts_pad_token_id=config['tts_pad_token_id'],
        num_code_groups=config['talker_config']['code_predictor_config']['num_code_groups'])


def finish(x, config):
    x = rms_norm(x, "talker.model.norm.weight", config['rms_norm_eps'])
    return x * T.unary_op(
        "qwen3_tts_pack_hidden_logits",
        weight_name="talker.codec_head.weight")
