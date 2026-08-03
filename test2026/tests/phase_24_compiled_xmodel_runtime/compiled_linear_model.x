from garnet import garnet

T = garnet.tensor()


@T.fusion()
def Model(x, weight):
    return x * T.binary_op("linear") * weight
