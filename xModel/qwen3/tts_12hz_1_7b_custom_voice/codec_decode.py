# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import garnet

T = garnet.tensor()

GARNET_MODEL_SPEC = {
    "arguments": [
        {"name": "codes", "kind": "tensor"},
        {"name": "weights", "kind": "weights"},
        {"name": "config", "kind": "config"}
    ]
}


def linear(x, weight_name, op="linear", bias_name=None):
    return x * T.unary_op(op, weight_name=weight_name, bias_name=bias_name)


def rms_norm(x, weight_name, eps):
    return x * T.unary_op("rms_norm", weight_name=weight_name, eps=eps)


def causal_conv(x, prefix, kernel, dilation=1, groups=1):
    return x * T.unary_op(
        "qwen3_tts_causal_conv1d",
        weight_name=prefix + ".conv.weight",
        bias_name=prefix + ".conv.bias",
        kernel_size=kernel, dilation=dilation, groups=groups)


def trans_conv(x, prefix, kernel, stride):
    return x * T.unary_op(
        "qwen3_tts_causal_transconv1d",
        weight_name=prefix + ".conv.weight",
        bias_name=prefix + ".conv.bias",
        kernel_size=kernel, stride=stride)


def snake(x, prefix):
    return x * T.unary_op(
        "qwen3_tts_snake_beta",
        alpha_name=prefix + ".alpha", beta_name=prefix + ".beta")


def transformer_layer(x, config, layer_idx):
    prefix = "decoder.pre_transformer.layers." + str(layer_idx)
    residual = x
    hidden = rms_norm(x, prefix + ".input_layernorm.weight", config['rms_norm_eps'])
    attn = prefix + ".self_attn"
    qkv = hidden * T.unary_op(
        "qwen3_text_qkv_packed",
        q_weight_name=attn + ".q_proj.weight",
        k_weight_name=attn + ".k_proj.weight",
        v_weight_name=attn + ".v_proj.weight",
        q_norm_weight_name=None, k_norm_weight_name=None,
        norm_eps=config['rms_norm_eps'],
        num_heads=config['num_attention_heads'],
        num_kv_heads=config['num_key_value_heads'],
        head_dim=config['head_dim'])
    positions = x * T.unary_op("qwen3_positions", components=1)
    qkv = qkv * T.binary_op(
        "qwen3_apply_text_rope_packed", rope_theta=config['rope_theta'],
        num_heads=config['num_attention_heads'],
        num_kv_heads=config['num_key_value_heads'],
        head_dim=config['head_dim']) * positions
    hidden = qkv * T.unary_op(
        "qwen3_tts_dense_attention_packed",
        num_heads=config['num_attention_heads'],
        num_key_value_heads=config['num_key_value_heads'],
        head_dim=config['head_dim'],
        sliding_window=config['sliding_window'])
    hidden = linear(hidden, attn + ".o_proj.weight", "o_proj")
    hidden = hidden * T.unary_op(
        "qwen3_tts_channel_scale",
        weight_name=prefix + ".self_attn_layer_scale.scale")
    x = residual + hidden
    residual = x
    hidden = rms_norm(x, prefix + ".post_attention_layernorm.weight", config['rms_norm_eps'])
    mlp = prefix + ".mlp"
    hidden = hidden * T.unary_op(
        "qwen3_mlp_gate_up_swiglu_packed",
        gate_weight_name=mlp + ".gate_proj.weight",
        up_weight_name=mlp + ".up_proj.weight")
    hidden = linear(hidden, mlp + ".down_proj.weight", "down_proj")
    hidden = hidden * T.unary_op(
        "qwen3_tts_channel_scale",
        weight_name=prefix + ".mlp_layer_scale.scale")
    return residual + hidden


def convnext(x, prefix, dim):
    residual = x
    x = causal_conv(x, prefix + ".dwconv", 7, groups=dim)
    x = x * T.unary_op(
        "layer_norm", weight_name=prefix + ".norm.weight",
        bias_name=prefix + ".norm.bias", eps=0.000001)
    x = linear(x, prefix + ".pwconv1.weight",
               bias_name=prefix + ".pwconv1.bias")
    x = x * T.unary_op("gelu")
    x = linear(x, prefix + ".pwconv2.weight",
               bias_name=prefix + ".pwconv2.bias")
    x = x * T.unary_op(
        "qwen3_tts_channel_scale", weight_name=prefix + ".gamma")
    return residual + x


def residual_unit(x, prefix, dilation):
    residual = x
    x = snake(x, prefix + ".act1")
    x = causal_conv(x, prefix + ".conv1", 7, dilation=dilation)
    x = snake(x, prefix + ".act2")
    x = causal_conv(x, prefix + ".conv2", 1)
    return residual + x


@T.fusion(name="qwen3_tts_codec_decode", role="audio_codec_decoder", boundary="required")
def Qwen3TTSCodecDecode(codes, weights, config):
    decoder = config['decoder_config']
    x = codes * T.unary_op(
        "qwen3_tts_rvq_decode", prefix="decoder.quantizer",
        num_quantizers=decoder['num_quantizers'], semantic_quantizers=1)
    x = causal_conv(x, "decoder.pre_conv", 3)
    x = linear(x, "decoder.pre_transformer.input_proj.weight",
               bias_name="decoder.pre_transformer.input_proj.bias")
    for layer_idx in range(decoder['num_hidden_layers']):
        x = transformer_layer(x, decoder, layer_idx)
    x = rms_norm(x, "decoder.pre_transformer.norm.weight", decoder['rms_norm_eps'])
    x = linear(x, "decoder.pre_transformer.output_proj.weight",
               bias_name="decoder.pre_transformer.output_proj.bias")

    for stage in range(len(decoder['upsampling_ratios'])):
        factor = decoder['upsampling_ratios'][stage]
        prefix = "decoder.upsample." + str(stage)
        x = trans_conv(x, prefix + ".0", factor, factor)
        x = convnext(x, prefix + ".1", decoder['latent_dim'])

    x = causal_conv(x, "decoder.decoder.0", 7)
    for stage in range(len(decoder['upsample_rates'])):
        rate = decoder['upsample_rates'][stage]
        block = "decoder.decoder." + str(stage + 1) + ".block"
        x = snake(x, block + ".0")
        x = trans_conv(x, block + ".1", 2 * rate, rate)
        dilations = [1, 3, 9]
        for unit in range(3):
            dilation = dilations[unit]
            x = residual_unit(x, block + "." + str(unit + 2), dilation)
    final = "decoder.decoder." + str(len(decoder['upsample_rates']) + 1)
    x = snake(x, final)
    x = causal_conv(
        x, "decoder.decoder." + str(len(decoder['upsample_rates']) + 2), 7)
    return x * T.unary_op("qwen3_tts_waveform")
