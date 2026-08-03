from garnet import garnet

T = garnet.tensor()

GARNET_MODEL_SPEC = {
    "arguments": [
        {"name": "talker_hidden", "kind": "tensor"},
        {"name": "first_code", "kind": "tensor"},
        {"name": "weights", "kind": "weights"},
        {"name": "config", "kind": "config"}
    ]
}


def linear(x, weight_name, op="linear"):
    return x * T.unary_op(op, weight_name=weight_name, bias_name=None)


def norm(x, weight_name, eps):
    return x * T.unary_op("rms_norm", weight_name=weight_name, eps=eps)


def attention(x, config, layer_idx):
    prefix = "talker.code_predictor.model.layers." + str(layer_idx) + ".self_attn"
    qkv = x * T.unary_op(
        "qwen3_text_qkv_packed",
        q_weight_name=prefix + ".q_proj.weight",
        k_weight_name=prefix + ".k_proj.weight",
        v_weight_name=prefix + ".v_proj.weight",
        q_norm_weight_name=prefix + ".q_norm.weight",
        k_norm_weight_name=prefix + ".k_norm.weight",
        norm_eps=config.rms_norm_eps,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim)
    positions = x * T.unary_op("qwen3_positions", components=1)
    qkv = qkv * T.binary_op(
        "qwen3_apply_text_rope_packed",
        rope_theta=config.rope_theta,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim) * positions
    attended = qkv * T.unary_op(
        "qwen3_tts_dense_attention_packed",
        num_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        sliding_window=0)
    return linear(attended, prefix + ".o_proj.weight", op="o_proj")


def layer(x, config, layer_idx):
    prefix = "talker.code_predictor.model.layers." + str(layer_idx)
    residual = x
    x = norm(x, prefix + ".input_layernorm.weight", config.rms_norm_eps)
    x = residual + attention(x, config, layer_idx)
    residual = x
    x = norm(x, prefix + ".post_attention_layernorm.weight", config.rms_norm_eps)
    mlp = prefix + ".mlp"
    x = x * T.unary_op(
        "qwen3_mlp_gate_up_swiglu_packed",
        gate_weight_name=mlp + ".gate_proj.weight",
        up_weight_name=mlp + ".up_proj.weight")
    x = linear(x, mlp + ".down_proj.weight", op="down_proj")
    return residual + x


def predict(sequence, config, group):
    x = sequence
    for layer_idx in range(config.num_hidden_layers):
        x = layer(x, config, layer_idx)
    x = norm(
        x, "talker.code_predictor.model.norm.weight",
        config.rms_norm_eps)
    x = x * T.unary_op("last_token")
    logits = linear(
        x, "talker.code_predictor.lm_head." + str(group) + ".weight",
        op="lm_head")
    return logits * T.unary_op("argmax_last_dim")


@T.fusion(name="qwen3_tts_code_predictor", role="audio_code_predictor", boundary="required")
def Qwen3TTSCodePredictor(talker_hidden, first_code, weights, config):
    predictor = config.talker_config.code_predictor_config
    first_embedding = first_code * T.unary_op(
        "embedding", weight_name="talker.model.codec_embedding.weight")
    sequence = talker_hidden * T.binary_op("concat_sequence") * first_embedding
    sequence = sequence * T.unary_op(
        "linear",
        weight_name="talker.code_predictor.small_to_mtp_projection.weight",
        bias_name="talker.code_predictor.small_to_mtp_projection.bias")
    codes = first_code
    for group in range(predictor.num_code_groups - 2):
        next_code = predict(sequence, predictor, group)
        codes = codes * T.binary_op("concat_tokens") * next_code
        next_embedding = next_code * T.unary_op(
            "embedding",
            weight_name="talker.code_predictor.model.codec_embedding." +
                str(group) + ".weight")
        next_embedding = next_embedding * T.unary_op(
            "linear",
            weight_name="talker.code_predictor.small_to_mtp_projection.weight",
            bias_name="talker.code_predictor.small_to_mtp_projection.bias")
        sequence = sequence * T.binary_op("concat_sequence") * next_embedding
    final_group = predictor.num_code_groups - 2
    next_code = predict(sequence, predictor, final_group)
    codes = codes * T.binary_op("concat_tokens") * next_code
    return codes
