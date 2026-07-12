from garnet import garnet

T = garnet.tensor()
T.set_backend("TensorRT")


@T.fusion()
def Model(x):
    doubled = x + x
    scaled = doubled * x
    return scaled - x
