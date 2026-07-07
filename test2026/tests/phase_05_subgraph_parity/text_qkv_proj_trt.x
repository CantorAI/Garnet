from garnet import garnet

T = garnet.tensor()
T.set_backend("TensorRT")

x = tensor(shape=[3, 8], dtype=tensor.float32)
weights = {}
T.set_weights(weights)

prefix = "language_model.layers.0.self_attn"
q = x * T.binary_op("q_proj") * weights[prefix + ".q_proj.weight"]
k = x * T.binary_op("k_proj") * weights[prefix + ".k_proj.weight"]
v = x * T.binary_op("v_proj") * weights[prefix + ".v_proj.weight"]
output = q * T.binary_op("concat_qkv") * k * v

print("Text QKV projection TRT expression built.")
