from garnet import garnet

T = garnet.tensor()


@T.fusion()
def Model(x):
    if True:
        output = x + x
    else:
        output = x - x
    return output
