# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

from garnet import garnet

T = garnet.tensor()

a = tensor(shape=[1, 4], dtype=tensor.float32)
weights = {}
T.set_weights(weights)

def project(x, weight):
    return x * T.binary_op("trt_matmul") * weight

def block(x, weight):
    return project(x, weight)

def model(x, weight):
    return block(x, weight)

output = model(a, weights["W"])

print("TRT nested function expression built.")
