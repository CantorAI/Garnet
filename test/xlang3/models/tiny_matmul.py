import garnet

T = garnet.tensor()


@T.fusion(name="tiny_matmul", role="integration")
def forward(left, right):
    return left @ right
