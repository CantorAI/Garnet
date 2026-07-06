import CpuTensor as T

from vision_encoder import Qwen3VisionEncoder
from vl_adapter import Qwen3GetRopeIndex, Qwen3PrepareInputsEmbeds
from qwen_llm import Qwen3TextModel

# Qwen3-VL dense model forward skeleton.
#
# This xlang source is intended to be the canonical Garnet model expression for
# Workbench and backend lowering. It follows the current Qwen3-VL structure:
#   visual = Qwen3VLVisionModel(pixel_values, grid_thw)
#   inputs_embeds = text embeddings with visual placeholder spans replaced
#   position_ids = multimodal MRoPE positions from token types + grid_thw
#   language_model(..., visual_pos_masks, deepstack_features, kv_cache)
#   logits = tied lm_head / embed_tokens projection


@T.fusion()
def Qwen3VLModel(
    input_ids,
    pixel_values,
    image_grid_thw,
    video_grid_thw,
    mm_token_type_ids,
    attention_mask,
    weights,
    config,
    past_key_values=None,
    use_cache=True
):
    visual_outputs = Qwen3VisionEncoder(
        pixel_values,
        image_grid_thw,
        weights,
        config
    )

    prepared = Qwen3PrepareInputsEmbeds(
        input_ids,
        visual_outputs,
        weights,
        config
    )

    rope = Qwen3GetRopeIndex(
        input_ids,
        mm_token_type_ids,
        image_grid_thw,
        video_grid_thw,
        attention_mask,
        config
    )

    text_outputs = Qwen3TextModel(
        input_ids=None,
        inputs_embeds=prepared["inputs_embeds"],
        position_ids=rope["position_ids"],
        attention_mask=attention_mask,
        visual_pos_masks=prepared["visual_pos_masks"],
        deepstack_visual_embeds=prepared["deepstack_visual_embeds"],
        past_key_values=past_key_values,
        weights=weights,
        config=config,
        use_cache=use_cache
    )

    hidden_states = text_outputs["last_hidden_state"]

    # Qwen3-VL-2B config ties word embeddings. If a model provides an untied
    # lm_head, backend can map this op to lm_head.weight instead.
    logits = hidden_states * T.binary_op(
        "lm_head",
        tied_word_embeddings=config.tie_word_embeddings
    ) * weights["language_model.embed_tokens.weight"]

    return {
        "logits": logits,
        "past_key_values": text_outputs["past_key_values"],
        "rope_deltas": rope["mrope_position_deltas"],
        "visual_outputs": visual_outputs
    }


# Alias used by some scripts/docs.
QWenVLModel = Qwen3VLModel
