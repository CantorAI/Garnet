from garnet import garnet

T = garnet.tensor()
T.set_backend("TensorRT")

from "." import vision_encoder as vision
from "." import vl_adapter as adapter
from "." import qwen_llm as llm

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
        {"name": "weights", "kind": "weights"},
        {"name": "config", "kind": "config"},
        {"name": "past_key_values", "kind": "none"},
        {"name": "use_cache", "kind": "bool", "value": False}
    ]
}

# Qwen3-VL dense model forward skeleton.
#
# This xlang source is intended to be the canonical Garnet model expression for
# Workbench and backend lowering. It follows the current Qwen3-VL structure:
#   visual = Qwen3VLVisionModel(pixel_values, grid_thw)
#   inputs_embeds = text embeddings with visual placeholder spans replaced
#   position_ids = multimodal MRoPE positions from token types + grid_thw
#   language_model(..., visual_pos_masks, deepstack_features, kv_cache)
#   logits = tied lm_head / embed_tokens projection


@T.fusion(
    name="qwen3_vl_model",
    role="multimodal_prefill",
    boundary="required"
)
def Qwen3VLModel(
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
    weights,
    config,
    past_key_values=None,
    use_cache=True
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

    prepared = adapter.Qwen3PrepareInputsEmbeds(
        input_ids,
        visual_outputs,
        weights,
        config
    )

    text_outputs = llm.Qwen3TextModel(
        input_ids=input_ids,
        inputs_embeds=prepared["inputs_embeds"],
        position_ids=position_ids,
        attention_mask=attention_mask,
        deepstack_visual_embeds=prepared["deepstack_visual_embeds"],
        past_key_values=past_key_values,
        weights=weights,
        config=config,
        use_cache=use_cache
    )

    hidden_states = text_outputs["last_hidden_state"]

    # Qwen3-VL-2B config ties word embeddings. If a model provides an untied
    # lm_head, backend can map this op to lm_head.weight instead.
    logits = hidden_states * T.unary_op(
        "lm_head",
        tied_word_embeddings=config.tie_word_embeddings,
        weight_name="model.language_model.embed_tokens.weight"
    )

    return {
        "logits": logits,
        "past_key_values": text_outputs["past_key_values"],
        "rope_deltas": mrope_position_deltas,
        "visual_outputs": visual_outputs
    }


# Alias used by some scripts/docs.
QWenVLModel = Qwen3VLModel
