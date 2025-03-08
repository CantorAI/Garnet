import cantor thru 'lrpc:1000'
from garnet import garnet
garnet.cantor = cantor
modelPath = "D:/CantorAIProjects/DeepSeek-V3/deepseek-moe-16b-base/*.xyz"
model = garnet.loadModel(modelPath)
text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
inputs = model.tokenizer(text, return_tensors = "pt")
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask
print(inputs)
