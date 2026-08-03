from garnet import garnet

T = garnet.tensor()


@T.fusion()
def Model(x):
    packed = x * T.unary_op("packed_state_fixture", lanes=3)
    return packed * T.unary_op("packed_state_consumer_fixture", lanes=3)
