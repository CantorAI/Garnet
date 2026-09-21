# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

from garnet import garnet

T = garnet.tensor()

x = tensor(shape=[4, 8], dtype=tensor.float32)
weights = {}
T.set_weights(weights)

prefix = "language_model.layers.0.self_attn"
q = x * T.binary_op("q_proj") * weights[prefix + ".q_proj.weight"]
k = x * T.binary_op("k_proj") * weights[prefix + ".k_proj.weight"]
v = x * T.binary_op("v_proj") * weights[prefix + ".v_proj.weight"]
q = q * T.unary_op("reshape_q_heads", num_heads=16, head_dim=128)
k = k * T.unary_op("reshape_kv_heads", num_heads=8, head_dim=128)
q = q * T.unary_op("rms_norm", weight=weights[prefix + ".q_norm.weight"], eps=1e-6)
k = k * T.unary_op("rms_norm", weight=weights[prefix + ".k_norm.weight"], eps=1e-6)
output = q * T.binary_op("concat_qkv_head_norm") * k * v

print("Text QKV head-norm TRT expression built.")
