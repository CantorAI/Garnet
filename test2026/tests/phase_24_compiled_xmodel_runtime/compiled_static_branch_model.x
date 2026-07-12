from garnet import garnet

T = garnet.tensor()
T.set_backend("TensorRT")


@T.fusion()
def Model(x):
    if True:
        output = x + x
    else:
        output = x - x
    return output
