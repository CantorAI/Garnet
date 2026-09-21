# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

from garnet import garnet

T = garnet.tensor()

qkv = tensor(shape=[4, 4096], dtype=tensor.float32)

output = qkv * T.unary_op(
    "scaled_dot_product_attention",
    num_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    causal=True
)

print("Text attention core TRT expression built.")
