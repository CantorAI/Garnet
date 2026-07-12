from garnet import garnet

T = garnet.tensor()
T.set_backend("TensorRT")


@T.fusion()
def Model(x):
    return x * T.unary_op("unsupported_probe")
