# SPDX-License-Identifier: Apache-2.0
import garnet
from .tensor_compat import tensor
T = tensor()


def rounded(x):
    return x * T.unary_op("gpt_oss_round_bf16")


def norm(x, name):
    return rounded(x * T.unary_op("rms_norm", weight_name=name, eps=0.00001))


def linear(x, name, bias=None, op="linear"):
    return rounded(x * T.unary_op(op, weight_name=name, bias_name=bias,
                                 compute_dtype="bfloat16",
                                 accumulation_dtype="float32"))


@T.fusion(role="decoder_layer", atomic=True)
def layer(x, position_ids, key_pages, value_pages, page_table,
          context_length, slot_position, active_mask, config, layer_idx, prefill, kv_layer_idx=None):
    prefix = "block." + str(layer_idx)
    residual = x
    x = norm(x, prefix + ".attn.norm.scale")
    qkv = linear(x, prefix + ".attn.qkv.weight", prefix + ".attn.qkv.bias")
    qkv = qkv * T.binary_op(
        "gpt_oss_apply_yarn_rope_packed",
        num_heads=config['num_attention_heads'], num_kv_heads=config['num_key_value_heads'],
        head_dim=config['head_dim'], rope_theta=config['rope_theta'],
        rope_scaling_factor=config['rope_scaling_factor'],
        initial_context_length=config['initial_context_length'],
        rope_ntk_alpha=config['rope_ntk_alpha'], rope_ntk_beta=config['rope_ntk_beta']) * position_ids
    if kv_layer_idx is None:
        kv_layer_idx = layer_idx
    keys = key_pages * T.unary_op("paged_kv_select_layer", layer_idx=kv_layer_idx)
    values = value_pages * T.unary_op("paged_kv_select_layer", layer_idx=kv_layer_idx)
    state = qkv * T.binary_op("paged_kv_bind_key_pages") * keys
    state = state * T.binary_op("paged_kv_bind_value_pages") * values
    state = state * T.binary_op("paged_kv_bind_page_table") * page_table
    state = state * T.binary_op("paged_kv_bind_context_length") * context_length
    state = state * T.binary_op("paged_kv_bind_slot_position") * slot_position
    state = state * T.binary_op("paged_kv_bind_active_mask") * active_mask
    window = 0
    if layer_idx % 2 == 0:
        window = config['sliding_window']
    attention = state * T.unary_op(
        "gpt_oss_paged_attention", page_size=16, prefill=prefill,
        num_heads=config['num_attention_heads'], num_kv_heads=config['num_key_value_heads'],
        head_dim=config['head_dim'], sliding_window=window,
        sinks_weight_name=prefix + ".attn.sinks")
    x = rounded(residual + linear(attention, prefix + ".attn.out.weight", prefix + ".attn.out.bias"))
    residual = x
    x = norm(x, prefix + ".mlp.norm.scale")
    x = x * T.unary_op(
        "gpt_oss_moe_mxfp4", hidden_size=config['hidden_size'],
        intermediate_size=config['intermediate_size'], num_experts=config['num_experts'],
        experts_per_token=config['experts_per_token'], swiglu_limit=config['swiglu_limit'],
        router_weight_name=prefix + ".mlp.gate.weight", router_bias_name=prefix + ".mlp.gate.bias",
        gate_up_blocks_name=prefix + ".mlp.mlp1_weight.blocks",
        gate_up_scales_name=prefix + ".mlp.mlp1_weight.scales",
        gate_up_bias_name=prefix + ".mlp.mlp1_bias",
        down_blocks_name=prefix + ".mlp.mlp2_weight.blocks",
        down_scales_name=prefix + ".mlp.mlp2_weight.scales", down_bias_name=prefix + ".mlp.mlp2_bias")
    return rounded(residual + x)


def forward(input_ids, position_ids, key_pages, value_pages, page_table,
            context_length, slot_position, active_mask, weights, config, prefill):
    x = rounded(input_ids * T.unary_op("embedding", weight_name="embedding.weight"))
    for layer_idx in range(config['num_hidden_layers']):
        x = layer(x, position_ids, key_pages, value_pages, page_table,
                  context_length, slot_position, active_mask, config, layer_idx, prefill)
    return linear(norm(x, "norm.scale"), "unembedding.weight", op="lm_head")


def forward_stage(x, position_ids, key_pages, value_pages, page_table,
                  context_length, slot_position, active_mask, weights, config,
                  start, end, prefill, last_token_logits=False):
    if start == 0:
        x = rounded(x * T.unary_op("embedding", weight_name="embedding.weight"))
    for layer_idx in range(start, end):
        x = layer(x, position_ids, key_pages, value_pages, page_table,
                  context_length, slot_position, active_mask, config, layer_idx,
                  prefill, layer_idx - start)
    if end == config['num_hidden_layers']:
        if last_token_logits:
            x = x * T.unary_op("last_token")
        x = linear(norm(x, "norm.scale"), "unembedding.weight", op="lm_head")
    return x
