# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

from garnet import garnet
import CpuTensor as T
# m = garnet.loadModel("models/Meta-Llama-3-8B/model_weights.bin")
m001 = garnet.loadModel("models/DeepSeek-V3-Base/model_weights_from_safetensor.bin")

def softmax(m):
    return m
model_embed_tokens_weight = m001["model.embed_tokens.weight"]
model_layers_0_input_layernorm_weight = m001["model.layers.0.input_layernorm.weight"]
Y = model_embed_tokens_weight*model_layers_0_input_layernorm_weight
z = Y+10

y_graph = T.graph(Z)

y_graph.run()


print("Done")
