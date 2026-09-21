# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

from garnet import garnet

T = garnet.tensor()

GARNET_MODEL_SPEC = {
    "arguments": [
        {"name": "x", "kind": "tensor"},
        {"name": "weights", "kind": "weights"}
    ]
}


@T.fusion()
def Model(x, weights):
    return x * T.unary_op("linear", weight_name="weight")
