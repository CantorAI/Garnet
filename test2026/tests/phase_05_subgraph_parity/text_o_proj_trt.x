from garnet import garnet

T = garnet.tensor()
T.set_backend("TensorRT")

x = tensor(shape=[4, 8], dtype=tensor.float32)
weights = {}
T.set_weights(weights)

prefix = "language_model.layers.0.self_attn"
output = x * T.binary_op("o_proj") * weights[prefix + ".o_proj.weight"]

print("Text output projection TRT expression built.")
