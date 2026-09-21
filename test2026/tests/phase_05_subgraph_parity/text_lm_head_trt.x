# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

from garnet import garnet

T = garnet.tensor()

x = tensor(shape=[4, 8], dtype=tensor.float32)
weights = {}
T.set_weights(weights)

output = x * T.binary_op("lm_head") * weights["language_model.embed_tokens.weight"]

print("Text tied LM head TRT expression built.")
