from garnet import garnet

T = garnet.tensor()

a = tensor(shape=[1, 4], dtype=tensor.float32)
weights = {}
T.set_weights(weights)

use_trt_linear = True

if use_trt_linear:
    output = a * T.binary_op("trt_matmul") * weights["W"]
else:
    output = a

print("TRT branch expression built.")
