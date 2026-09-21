# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

from garnet import garnet

T = garnet.tensor()

x = tensor(shape=[5, 8], dtype=tensor.float32)
weights = {}
T.set_weights(weights)

prefix = "visual.blocks.0.mlp"
hidden = x * T.binary_op("linear_fc1") * weights[prefix + ".linear_fc1.weight"]
hidden = hidden + weights[prefix + ".linear_fc1.bias"]
hidden = hidden * T.unary_op("gelu")
output = hidden * T.binary_op("linear_fc2") * weights[prefix + ".linear_fc2.weight"]
output = output + weights[prefix + ".linear_fc2.bias"]

print("VisionMLP TRT expression built.")
