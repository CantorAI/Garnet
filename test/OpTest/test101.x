import cantor thru 'lrpc:1000'
from garnet import garnet
garnet.cantor = cantor
T = garnet.tensor()
# modelPath = "D:/CantorAIProjects/DeepSeek-V3/DeepSeek-V3-Base/model_weights_from_safetensor.bin"
modelPath = "D:/CantorAIProjects/DeepSeek-V3/deepseek-moe-16b-base"
model = garnet.loadModel(modelPath)
model_embed_tokens_weight = model["model.embed_tokens.weight"]
model_layers_0_input_layernorm_weight = model["model.layers.0.input_layernorm.weight"]
Y = model_embed_tokens_weight * model_layers_0_input_layernorm_weight
y_graph = T.graph(Y)
print("y_graph:",y_graph)
y_graph.run()
print("Y=",Y)
print("Done")
