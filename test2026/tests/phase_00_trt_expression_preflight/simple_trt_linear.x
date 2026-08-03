from garnet import garnet

T = garnet.tensor()

# Stage 0 preflight expression:
# input [1, 4] * weight [4, 3] -> output [1, 3]
# This intentionally stays tiny so failures identify xlang/Garnet/TRT plumbing,
# not Qwen-VL model complexity.
a = tensor(shape=[1, 4], dtype=tensor.float32)
weights = {}
T.set_weights(weights)

output = a * T.binary_op("trt_matmul") * weights["W"]

print("TRT preflight tensor expression built.")
