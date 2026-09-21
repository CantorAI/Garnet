# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

from garnet import garnet

T = garnet.tensor()

a = tensor(shape=[1, 4], dtype=tensor.float32)
weights = {}
T.set_weights(weights)

output = a
for layer_idx in range(5):
    output = output * T.binary_op("trt_matmul") * weights["W"]

print("TRT repeat expression built.")
