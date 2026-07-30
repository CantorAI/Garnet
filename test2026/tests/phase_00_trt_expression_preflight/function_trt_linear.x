from garnet import garnet

T = garnet.tensor()

a = tensor(shape=[1, 4], dtype=tensor.float32)
weights = {}
T.set_weights(weights)

def trt_linear(x, weight):
    return x * T.binary_op("trt_matmul") * weight

output = trt_linear(a, weights["W"])

print("TRT function expression built.")
