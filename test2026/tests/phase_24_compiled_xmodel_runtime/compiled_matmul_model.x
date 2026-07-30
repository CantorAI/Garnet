from garnet import garnet

T = garnet.tensor()


@T.fusion()
def Model(left, right):
    return left * T.binary_op("matmul") * right
