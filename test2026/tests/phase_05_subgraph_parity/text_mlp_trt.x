# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

from garnet import garnet

T = garnet.tensor()

x = tensor(shape=[3, 8], dtype=tensor.float32)
weights = {}
T.set_weights(weights)

prefix = "language_model.layers.0.mlp"
gate = x * T.binary_op("gate_proj") * weights[prefix + ".gate_proj.weight"]
up = x * T.binary_op("up_proj") * weights[prefix + ".up_proj.weight"]
hidden = gate * T.unary_op("silu") * up
output = hidden * T.binary_op("down_proj") * weights[prefix + ".down_proj.weight"]

print("Qwen3TextMLP TRT expression built.")
