from garnet import garnet

T = garnet.tensor()
T.set_backend("TensorRT")


@T.fusion(role="decoder_layer", atomic=True)
def AtomicBlock(x):
    return x + x


@T.fusion(
    name="input_projection",
    role="projection",
    boundary="preferred"
)
def InputProjection(x):
    return x + x


@T.fusion(
    name="final_projection",
    role="projection",
    boundary="required"
)
def FinalProjection(x):
    return x + x


@T.fusion(
    name="fixture_root",
    role="transformer_prefill",
    boundary="required",
    cuda_graph=True
)
def Model(x):
    output = InputProjection(x)
    for layer_index in range(5):
        output = AtomicBlock(output)
    return FinalProjection(output)
