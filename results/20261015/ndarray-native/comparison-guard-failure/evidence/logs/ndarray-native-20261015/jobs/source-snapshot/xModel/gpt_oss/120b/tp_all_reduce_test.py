# SPDX-License-Identifier: Apache-2.0
import garnet
import garnet_gpt_oss
garnet.bind_operator_module(garnet_gpt_oss)

from .tensor_compat import tensor

T = tensor()
TP_RANK = 0
GARNET_MODEL_SPEC = {
    'arguments': [
        {'name': 'x', 'kind': 'tensor'},
        {'name': 'tp_rank', 'kind': 'int', 'value': TP_RANK},
    ],
    'requires': {'operator_plugins': [{
        'id': 'gpt_oss', 'module': 'garnet_gpt_oss', 'abi': 1,
        'backend': 'tensorrt', 'operators': ['gpt_oss_tp_all_reduce'],
    }]},
}


@T.fusion(name='gpt_oss_tp_all_reduce_test', role='decoder_layer', boundary='required')
def GptOssTpCollective(x, tp_rank):
    return x * T.unary_op('gpt_oss_tp_all_reduce', hidden_size=8, tp_rank=tp_rank)
