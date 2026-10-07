# SPDX-License-Identifier: Apache-2.0
import garnet
from .tensor_compat import tensor
T = tensor()


def rounded(x):
    return x * T.unary_op("gpt_oss_round_bf16")


def norm(x, name, hidden_size):
    return x * T.unary_op("gpt_oss_rms_norm", weight_name=name,
                          hidden_size=hidden_size, eps=0.00001)


def linear(x, name, bias=None, op="linear", tp_mode=None, tp_rank=-1,
           tp_heads=0, tp_kv_heads=0, tp_head_dim=0):
    attributes = dict(weight_name=name, bias_name=bias,
                      compute_dtype="bfloat16", accumulation_dtype="float32")
    if tp_mode is not None and tp_rank >= 0:
        attributes.update(tp_mode=tp_mode, tp_rank=tp_rank,
                          tp_heads=tp_heads, tp_kv_heads=tp_kv_heads,
                          tp_head_dim=tp_head_dim)
    return rounded(x * T.unary_op(op, **attributes))


def tp_all_reduce(x, rank, config, bf16_communication=False):
    """Sum a rank-local partial hidden state across both GPT-OSS TP ranks."""
    return x * T.unary_op("gpt_oss_tp_all_reduce",
                          hidden_size=config['hidden_size'], tp_rank=rank,
                          bf16_communication=1 if bf16_communication else 0)


def tp_all_gather_logits(x, rank, config):
    return x * T.unary_op("gpt_oss_tp_all_gather", tp_rank=rank,
                          hidden_size=config['vocab_size'] // 2)


@T.fusion(role="decoder_layer", atomic=True, cuda_graph=True)
def layer(x, position_ids, key_pages, value_pages, page_table,
          context_length, slot_position, active_mask, config, layer_idx, prefill,
          kv_layer_idx=None, tp_rank=-1, expert_weight_shard=0):
    prefix = "block." + str(layer_idx)
    residual = x
    x = norm(x, prefix + ".attn.norm.scale", config['hidden_size'])
    q_heads, kv_heads = config['num_attention_heads'], config['num_key_value_heads']
    if tp_rank >= 0:
        if q_heads % 2 or kv_heads % 2:
            raise ValueError('GPT-OSS TP2 requires even query and KV head counts')
        qkv = linear(x, prefix + ".attn.qkv.weight", prefix + ".attn.qkv.bias",
                     tp_mode='qkv', tp_rank=tp_rank, tp_heads=q_heads,
                     tp_kv_heads=kv_heads, tp_head_dim=config['head_dim'])
        q_heads //= 2
        kv_heads //= 2
    else:
        qkv = linear(x, prefix + ".attn.qkv.weight", prefix + ".attn.qkv.bias")
    qkv = qkv * T.binary_op(
        "gpt_oss_apply_yarn_rope_packed",
        num_heads=q_heads, num_kv_heads=kv_heads, tp_rank=tp_rank,
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
        num_heads=q_heads, num_kv_heads=kv_heads, tp_rank=tp_rank,
        head_dim=config['head_dim'], sliding_window=window,
        sinks_weight_name=prefix + ".attn.sinks")
    if tp_rank >= 0:
        attention_output = linear(attention, prefix + ".attn.out.weight",
                                  prefix + ".attn.out.bias", tp_mode='row', tp_rank=tp_rank)
        attention_output = rounded(tp_all_reduce(
            attention_output, tp_rank, config, bf16_communication=True))
    else:
        attention_output = linear(attention, prefix + ".attn.out.weight", prefix + ".attn.out.bias")
    x = rounded(residual + attention_output)
    residual = x
    x = norm(x, prefix + ".mlp.norm.scale", config['hidden_size'])
    x = x * T.unary_op(
        "gpt_oss_moe_mxfp4", hidden_size=config['hidden_size'],
        intermediate_size=config['intermediate_size'], num_experts=config['num_experts'],
        experts_per_token=config['experts_per_token'], swiglu_limit=config['swiglu_limit'],
        tp_rank=tp_rank, expert_weight_shard=expert_weight_shard,
        router_weight_name=prefix + ".mlp.gate.weight", router_bias_name=prefix + ".mlp.gate.bias",
        gate_up_blocks_name=prefix + ".mlp.mlp1_weight.blocks",
        gate_up_scales_name=prefix + ".mlp.mlp1_weight.scales",
        gate_up_bias_name=prefix + ".mlp.mlp1_bias",
        down_blocks_name=prefix + ".mlp.mlp2_weight.blocks",
        down_scales_name=prefix + ".mlp.mlp2_weight.scales", down_bias_name=prefix + ".mlp.mlp2_bias")
    if tp_rank >= 0:
        x = rounded(tp_all_reduce(x, tp_rank, config))
    return rounded(residual + x)


def forward(input_ids, position_ids, key_pages, value_pages, page_table,
            context_length, slot_position, active_mask, weights, config, prefill):
    x = rounded(input_ids * T.unary_op("embedding", weight_name="embedding.weight"))
    for layer_idx in range(config['num_hidden_layers']):
        x = layer(x, position_ids, key_pages, value_pages, page_table,
                  context_length, slot_position, active_mask, config, layer_idx, prefill)
    return linear(norm(x, "norm.scale", config['hidden_size']), "unembedding.weight", op="lm_head")


def forward_stage(x, position_ids, key_pages, value_pages, page_table,
                  context_length, slot_position, active_mask, weights, config,
                  start, end, prefill, last_token_logits=False, tp_rank=-1,
                  expert_weight_shard=0):
    if start == 0:
        x = rounded(x * T.unary_op("embedding", weight_name="embedding.weight"))
    for layer_idx in range(start, end):
        x = layer(x, position_ids, key_pages, value_pages, page_table,
                  context_length, slot_position, active_mask, config, layer_idx,
                  prefill, layer_idx - start, tp_rank, expert_weight_shard)
    if end == config['num_hidden_layers']:
        if last_token_logits:
            x = x * T.unary_op("last_token")
        if tp_rank >= 0:
            x = linear(norm(x, "norm.scale", config['hidden_size']), "unembedding.weight", op="lm_head",
                       tp_mode='vocab', tp_rank=tp_rank)
            x = tp_all_gather_logits(x, tp_rank, config)
        else:
            x = linear(norm(x, "norm.scale", config['hidden_size']), "unembedding.weight", op="lm_head")
    return x
