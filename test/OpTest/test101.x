from garnet import garnet
T = garnet.tensor()
modelPath = "D:/CantorAIProjects/DeepSeek-V3/DeepSeek-V3-Base/model_weights_from_safetensor.bin"
m001 = garnet.loadModel(modelPath)
model_embed_tokens_weight = m001["model.embed_tokens.weight"]
model_layers_0_input_layernorm_weight = m001["model.layers.0.input_layernorm.weight"]
Y = model_embed_tokens_weight * model_layers_0_input_layernorm_weight
y_graph = T.graph(Y)
print("y_graph:",y_graph)
y_graph.run()
print("Y=",Y)
print("Done")
