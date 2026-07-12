from garnet import garnet

T = garnet.tensor()
T.set_backend("TensorRT")

from "." import vision_encoder as vision
from "." import vl_adapter as adapter
from "." import qwen_llm as llm
from "." import qwen_text_prefill as prefill

GARNET_MODEL_SPEC = {
    "arguments": [
        {"name": "input_ids", "kind": "tensor"},
        {"name": "pixel_values", "kind": "tensor"},
        {"name": "image_grid_thw", "kind": "tensor"},
        {"name": "vision_bilinear_indices", "kind": "tensor"},
        {"name": "vision_bilinear_weights", "kind": "tensor"},
        {"name": "vision_position_ids", "kind": "tensor"},
        {"name": "vision_cu_seqlens", "kind": "tensor"},
        {"name": "video_grid_thw", "kind": "none"},
        {"name": "mm_token_type_ids", "kind": "tensor"},
        {"name": "attention_mask", "kind": "tensor"},
        {"name": "position_ids", "kind": "tensor"},
        {"name": "mrope_position_deltas", "kind": "tensor"},
        {"name": "key_pages", "kind": "tensor"},
        {"name": "value_pages", "kind": "tensor"},
        {"name": "page_table", "kind": "tensor"},
        {"name": "start_position", "kind": "tensor"},
        {"name": "weights", "kind": "weights"},
        {"name": "config", "kind": "config"}
    ]
}


@T.fusion()
def Qwen3VLPrefill(
    input_ids,
    pixel_values,
    image_grid_thw,
    vision_bilinear_indices,
    vision_bilinear_weights,
    vision_position_ids,
    vision_cu_seqlens,
    video_grid_thw,
    mm_token_type_ids,
    attention_mask,
    position_ids,
    mrope_position_deltas,
    key_pages,
    value_pages,
    page_table,
    start_position,
    weights,
    config
):
    visual_outputs = vision.Qwen3VisionEncoder(
        pixel_values,
        image_grid_thw,
        vision_bilinear_indices,
        vision_bilinear_weights,
        vision_position_ids,
        vision_cu_seqlens,
        weights,
        config
    )
    prepared = adapter.Qwen3PrepareInputsEmbeds(input_ids, visual_outputs, weights, config)
    x = prepared["inputs_embeds"]
    deepstack = prepared["deepstack_visual_embeds"]
    deepstack_count = len(deepstack)
    for layer_idx in range(deepstack_count):
        x = prefill.PrefillLayer(
            x, position_ids, attention_mask, key_pages, value_pages,
            page_table, start_position, weights, config, layer_idx
        )
        x = x * T.binary_op(
            "qwen3_vl_deepstack_add",
            input_ids=input_ids,
            image_token_id=config.image_token_id,
            video_token_id=config.video_token_id
        ) * deepstack[layer_idx]
    for layer_offset in range(config.text_config.num_hidden_layers - deepstack_count):
        layer_idx = deepstack_count + layer_offset
        x = prefill.PrefillLayer(
            x, position_ids, attention_mask, key_pages, value_pages,
            page_table, start_position, weights, config, layer_idx
        )
    x = llm.rms_norm(
        x, "model.language_model.norm.weight", eps=config.text_config.rms_norm_eps
    )
    return x * T.unary_op(
        "lm_head",
        tied_word_embeddings=config.tie_word_embeddings,
        weight_name="model.language_model.embed_tokens.weight"
    )
