import garnet

T = garnet.tensor()

from . import vision_encoder as vision

GARNET_MODEL_SPEC = {
    "arguments": [
        {"name": "pixel_values", "kind": "tensor"},
        {"name": "grid_thw", "kind": "tensor"},
        {"name": "bilinear_indices", "kind": "tensor"},
        {"name": "bilinear_weights", "kind": "tensor"},
        {"name": "vision_position_ids", "kind": "tensor"},
        {"name": "cu_seqlens", "kind": "tensor"},
        {"name": "weights", "kind": "weights"},
        {"name": "config", "kind": "config"}
    ]
}


@T.fusion(name="debug_vision_pooler", role="debug_probe", boundary="required")
def Qwen3VisionPoolerProbe(
    pixel_values,
    grid_thw,
    bilinear_indices,
    bilinear_weights,
    vision_position_ids,
    cu_seqlens,
    weights,
    config
):
    outputs = vision.Qwen3VisionEncoder(
        pixel_values,
        grid_thw,
        bilinear_indices,
        bilinear_weights,
        vision_position_ids,
        cu_seqlens,
        weights,
        config
    )
    return outputs["pooler_output"]
