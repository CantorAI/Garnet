from garnet import garnet
T = garnet.tensor()

T.set_backend("TensorRT")

# Just declare shapes using standard XLang tensor()
a = tensor(shape=[1, 128], dtype=tensor.float32)

# Bridge the weights natively using XLang dictionary
weights = {}
T.set_weights(weights)

output = a * T.binary_op("trt_matmul") * weights["W"]

# The graph is injected by XLang natively, so we just return the AST output node
# if we were inside a function, but here we can just evaluate it.
print("AST output built successfully.")
