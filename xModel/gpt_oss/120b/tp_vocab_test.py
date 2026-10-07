# SPDX-License-Identifier: Apache-2.0
import garnet
import garnet_gpt_oss
garnet.bind_operator_module(garnet_gpt_oss)
from .tensor_compat import tensor
T = tensor()
TP_RANK = 0
LOCAL_WIDTH = 7
COMPACT = 0
GARNET_MODEL_SPEC = {'arguments': [
    {'name': 'x', 'kind': 'tensor'},
    {'name': 'rank', 'kind': 'int', 'value': TP_RANK},
    {'name': 'width', 'kind': 'int', 'value': LOCAL_WIDTH},
    {'name': 'compact', 'kind': 'int', 'value': COMPACT}],
    'requires': {'operator_plugins': [{'id': 'gpt_oss', 'module': 'garnet_gpt_oss',
        'abi': 1, 'backend': 'tensorrt',
        'operators': ['gpt_oss_vocab_top1', 'gpt_oss_tp_all_gather']}]}}

@T.fusion(name='gpt_oss_vocab_test', boundary='required')
def GptOssVocabTest(x, rank, width, compact):
    if compact:
        x = x * T.unary_op('gpt_oss_vocab_top1', tp_rank=rank, hidden_size=width)
        width = 2
    return x * T.unary_op('gpt_oss_tp_all_gather', tp_rank=rank, hidden_size=width)
