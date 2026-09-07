import garnet

T = garnet.tensor()


@T.fusion(name="tiny_add", role="integration")
def forward(left, right):
    return left + right
