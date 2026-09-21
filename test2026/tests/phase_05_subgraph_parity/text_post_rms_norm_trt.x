# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

from garnet import garnet

T = garnet.tensor()

x = tensor(shape=[4, 2048], dtype=tensor.float32)
weights = {}
T.set_weights(weights)

output = x * T.unary_op(
    "rms_norm",
    weight=weights["language_model.layers.0.input_layernorm.weight"],
    eps=1e-6
)

print("Text post-attention RMSNorm TRT expression built.")
