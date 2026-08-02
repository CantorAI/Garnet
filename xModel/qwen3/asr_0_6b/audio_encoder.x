from garnet import garnet

T = garnet.tensor()


def linear(x, weight_name, bias_name=None, op="linear"):
    return x * T.unary_op(
        op, weight_name=weight_name, bias_name=bias_name)


def audio_attention(x, cu_seqlens, config, layer_idx):
    prefix = "thinker.audio_tower.layers." + str(layer_idx) + ".self_attn"
    qkv = x * T.unary_op(
        "qwen3_asr_audio_qkv_packed",
        q_weight_name=prefix + ".q_proj.weight",
        q_bias_name=prefix + ".q_proj.bias",
        k_weight_name=prefix + ".k_proj.weight",
        k_bias_name=prefix + ".k_proj.bias",
        v_weight_name=prefix + ".v_proj.weight",
        v_bias_name=prefix + ".v_proj.bias",
        num_heads=config.encoder_attention_heads
    )
    attended = qkv * T.unary_op(
        "qwen3_asr_audio_attention_packed",
        num_heads=config.encoder_attention_heads,
        tokens_per_chunk=13,
        cu_seqlens=cu_seqlens)
    return linear(
        attended, prefix + ".out_proj.weight",
        prefix + ".out_proj.bias", op="o_proj")


@T.fusion(role="audio_encoder_layer", atomic=True)
def encoder_layer(x, cu_seqlens, config, layer_idx):
    prefix = "thinker.audio_tower.layers." + str(layer_idx)
    residual = x
    x = x * T.unary_op(
        "layer_norm",
        weight_name=prefix + ".self_attn_layer_norm.weight",
        bias_name=prefix + ".self_attn_layer_norm.bias")
    x = residual + audio_attention(x, cu_seqlens, config, layer_idx)
    residual = x
    x = x * T.unary_op(
        "layer_norm",
        weight_name=prefix + ".final_layer_norm.weight",
        bias_name=prefix + ".final_layer_norm.bias")
    x = linear(x, prefix + ".fc1.weight", prefix + ".fc1.bias")
    x = x * T.unary_op("gelu")
    x = linear(x, prefix + ".fc2.weight", prefix + ".fc2.bias")
    return residual + x


@T.fusion(role="audio_encoder")
def Qwen3ASRAudioEncoder(input_features, cu_seqlens, weights, config):
    x = input_features * T.unary_op(
        "qwen3_asr_conv_subsample",
        conv1_weight_name="thinker.audio_tower.conv2d1.weight",
        conv1_bias_name="thinker.audio_tower.conv2d1.bias",
        conv2_weight_name="thinker.audio_tower.conv2d2.weight",
        conv2_bias_name="thinker.audio_tower.conv2d2.bias",
        conv3_weight_name="thinker.audio_tower.conv2d3.weight",
        conv3_bias_name="thinker.audio_tower.conv2d3.bias",
        out_weight_name="thinker.audio_tower.conv_out.weight",
        position_channels=config.d_model)
    for layer_idx in range(config.encoder_layers):
        x = encoder_layer(x, cu_seqlens, config, layer_idx)
    x = x * T.unary_op(
        "layer_norm",
        weight_name="thinker.audio_tower.ln_post.weight",
        bias_name="thinker.audio_tower.ln_post.bias")
    x = linear(
        x, "thinker.audio_tower.proj1.weight",
        "thinker.audio_tower.proj1.bias")
    x = x * T.unary_op("gelu")
    x = linear(
        x, "thinker.audio_tower.proj2.weight",
        "thinker.audio_tower.proj2.bias")
    return x * T.binary_op(
        "qwen3_asr_compact_audio_tokens", tokens_per_chunk=13) * cu_seqlens
