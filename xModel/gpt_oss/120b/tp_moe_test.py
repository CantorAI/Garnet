# SPDX-License-Identifier: Apache-2.0
import garnet
import garnet_gpt_oss
garnet.bind_operator_module(garnet_gpt_oss)


class TensorCompat:
    def __init__(self, native):
        self._native = native

    def __getattr__(self, name):
        return getattr(self._native, name)

    def unary_op(self, name, **attributes):
        return self._native.unary_op(name, **attributes)


T = TensorCompat(garnet.tensor())
TP_RANK = -1
EXPERT_WEIGHT_SHARD = 0
operators = ['gpt_oss_moe_mxfp4', 'gpt_oss_round_bf16']
if TP_RANK >= 0:
    operators.append('gpt_oss_tp_all_reduce')
GARNET_MODEL_SPEC = {
    'arguments': [
        {'name': 'x', 'kind': 'tensor'},
        {'name': 'tp_rank', 'kind': 'int', 'value': TP_RANK},
        {'name': 'expert_weight_shard', 'kind': 'int', 'value': EXPERT_WEIGHT_SHARD},
        {'name': 'weights', 'kind': 'weights'},
    ],
    'requires': {'operator_plugins': [{
        'id': 'gpt_oss', 'module': 'garnet_gpt_oss', 'abi': 1,
        'backend': 'tensorrt', 'operators': operators,
    }]},
}


@T.fusion(name='gpt_oss_tp_moe_test', role='decoder_layer', boundary='required')
def GptOssTpMoe(x, tp_rank, expert_weight_shard, weights):
    y = x * T.unary_op(
        'gpt_oss_moe_mxfp4', hidden_size=32, intermediate_size=32,
        num_experts=5, experts_per_token=2, swiglu_limit=7,
        tp_rank=tp_rank, expert_weight_shard=expert_weight_shard,
        router_weight_name='block.0.mlp.gate.weight',
        router_bias_name='block.0.mlp.gate.bias',
        gate_up_blocks_name='block.0.mlp.mlp1_weight.blocks',
        gate_up_scales_name='block.0.mlp.mlp1_weight.scales',
        gate_up_bias_name='block.0.mlp.mlp1_bias',
        down_blocks_name='block.0.mlp.mlp2_weight.blocks',
        down_scales_name='block.0.mlp.mlp2_weight.scales',
        down_bias_name='block.0.mlp.mlp2_bias')
    if TP_RANK >= 0:
        y = y * T.unary_op('gpt_oss_tp_all_reduce', hidden_size=32, tp_rank=tp_rank)
        y = y * T.unary_op('gpt_oss_round_bf16')
    return y
