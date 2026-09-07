import garnet

T = garnet.tensor()

GARNET_MODEL_SPEC = {"arguments": [{"name": "value", "kind": "tensor"}]}


@T.fusion(name="tiny_registered", role="integration")
def forward(value):
    return value * T.unary_op("relu")
