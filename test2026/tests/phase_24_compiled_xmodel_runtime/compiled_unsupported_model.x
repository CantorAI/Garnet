from garnet import garnet

T = garnet.tensor()


@T.fusion()
def Model(x):
    return x * T.unary_op("unsupported_probe")
