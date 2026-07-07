from garnet import garnet

T = garnet.tensor()
T.set_backend("TensorRT")

x = tensor(shape=[4, 8], dtype=tensor.float32)
weights = {}
T.set_weights(weights)

output = x * T.binary_op("linear_bias") * weights["W"]
output = output + weights["B"]

print("Generic linear+bias TRT expression built.")
