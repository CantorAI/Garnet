from garnet import garnet

T = garnet.tensor()
T.set_backend("TensorRT")

x = tensor(shape=[5, 8], dtype=tensor.float32)
weights = {}
T.set_weights(weights)

output = x * T.unary_op(
    "layer_norm",
    weight=weights["visual.blocks.0.norm1.weight"],
    bias=weights["visual.blocks.0.norm1.bias"],
    eps=1e-6
)

print("LayerNorm TRT expression built.")
