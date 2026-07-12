from garnet import garnet

T = garnet.tensor()
T.set_backend("TensorRT")


@T.fusion()
def Model(left, right):
    return left * T.binary_op("matmul") * right
