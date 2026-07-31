from garnet import garnet

T = garnet.tensor()

GARNET_MODEL_SPEC = {
    "arguments": [
        {"name": "hidden_states", "kind": "tensor"},
        {"name": "weights", "kind": "weights"}
    ]
}


@T.fusion(name="openvino_qwen3_mlp", role="model_subgraph")
def Model(hidden_states, weights):
    prefix = "model.layers.0"
    normalized = hidden_states * T.unary_op(
        "rms_norm",
        weight_name=prefix + ".post_attention_layernorm.weight",
        eps=1.0e-6
    )
    activated = normalized * T.unary_op(
        "qwen3_mlp_gate_up_swiglu_packed",
        gate_weight_name=prefix + ".mlp.gate_proj.weight",
        up_weight_name=prefix + ".mlp.up_proj.weight"
    )
    return activated * T.unary_op(
        "down_proj",
        weight_name=prefix + ".mlp.down_proj.weight",
        bias_name=None
    )
