from garnet import garnet

T = garnet.tensor()

GARNET_MODEL_SPEC = {
    "arguments": [
        {"name": "hidden_states", "kind": "tensor"},
        {"name": "weights", "kind": "weights"}
    ]
}


@T.fusion(name="openvino_qwen3_vl_vision_mlp", role="model_subgraph")
def Model(hidden_states, weights):
    prefix = "model.visual.blocks.0.mlp"
    projected = hidden_states * T.unary_op(
        "linear",
        weight_name=prefix + ".linear_fc1.weight",
        bias_name=prefix + ".linear_fc1.bias"
    )
    activated = projected * T.unary_op("gelu_pytorch_tanh")
    return activated * T.unary_op(
        "linear",
        weight_name=prefix + ".linear_fc2.weight",
        bias_name=prefix + ".linear_fc2.bias"
    )
