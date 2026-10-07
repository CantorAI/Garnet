# SPDX-License-Identifier: Apache-2.0
import garnet
T = garnet.tensor()
GARNET_MODEL_SPEC = {'arguments': [
    {'name': 'x', 'kind': 'tensor'}, {'name': 'count', 'kind': 'tensor'}]}


@T.fusion(name='sequence_selection', role='output_projection', boundary='required')
def SelectLastValid(x, count):
    return x * T.binary_op('select_last_valid_sequence') * count
