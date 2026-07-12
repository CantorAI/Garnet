from garnet import garnet

T = garnet.tensor()
T.set_backend("TensorRT")


def DoubleBlock(x):
    return x + x


@T.fusion()
def Model(x):
    output = x
    for layer_index in range(5):
        output = DoubleBlock(output)
    return output
