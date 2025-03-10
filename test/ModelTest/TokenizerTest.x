import cantor thru 'lrpc:1000'
from garnet import garnet
T = garnet.tensor()
garnet.cantor = cantor
modelPath = "D:/CantorAIProjects/DeepSeek-V3/deepseek-moe-16b-base/*.bin"
model = garnet.loadModel(modelPath)
text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
inputs = model.tokenizer(text, return_tensors = "pt")
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask
print(inputs)

model_embed_tokens_weight = model["model.embed_tokens.weight"]
s_embed = model_embed_tokens_weight*T.gather()*input_ids
s_graph = T.graph(s_embed)
s_graph.run()
print("s_embed:",s_embed)
model_layers_0_input_layernorm_weight = model["model.layers.0.input_layernorm.weight"]
Y = model_embed_tokens_weight * model_layers_0_input_layernorm_weight
y_graph = T.graph(Y)
print("y_graph:",y_graph)
y_graph.run()
print("Y=",Y)
print("Done")

