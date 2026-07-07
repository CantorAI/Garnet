from garnet import garnet

T = garnet.tensor()
T.set_backend("TensorRT")

x = tensor(shape=[4, 2048], dtype=tensor.float32)
weights = {}
T.set_weights(weights)

output = x * T.unary_op(
    "rms_norm",
    weight=weights["language_model.layers.0.input_layernorm.weight"],
    eps=1e-6
)

print("Text post-attention RMSNorm TRT expression built.")
