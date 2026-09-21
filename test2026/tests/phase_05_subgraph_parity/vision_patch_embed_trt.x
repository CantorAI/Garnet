# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

from garnet import garnet

T = garnet.tensor()

x = tensor(shape=[4, 1536], dtype=tensor.float32)
weights = {}
T.set_weights(weights)

output = x * T.binary_op("vision_patch_embed") * weights["visual.patch_embed.proj.weight"]
output = output + weights["visual.patch_embed.proj.bias"]

print("Vision patch embed TRT expression built.")
