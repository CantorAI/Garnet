from garnet import garnet

T = garnet.tensor()

# Qwen3-VL multimodal glue.
#
# Qwen3-VL does not use the old learned-query cross-attention adapter sketch.
# The vision tower returns merged visual tokens and DeepStack features. The main
# model replaces image/video placeholder embedding positions with merged visual
# embeddings, builds 3D MRoPE position ids, and passes visual masks/features into
# the text decoder.


def Qwen3VisionPositionIds(start_position, grid_thw, config, time_interval=1):
    return grid_thw * T.unary_op(
        "qwen3_vl_llm_vision_position_ids",
        start_position=start_position,
        temp_merge_size=1,
        spatial_merge_size=config.vision_config.spatial_merge_size,
        time_interval=time_interval
    )


def Qwen3GetRopeIndex(input_ids, mm_token_type_ids, image_grid_thw, video_grid_thw, attention_mask, config):
    # Produces multimodal position_ids and mrope_position_deltas.
    # position_ids shape follows HF: (3, batch, sequence), while the text model
    # internally expands/handles the text position channel as well.
    position_ids = input_ids * T.unary_op(
        "qwen3_vl_get_position_ids",
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        spatial_merge_size=config.vision_config.spatial_merge_size
    )
    mrope_position_deltas = input_ids * T.unary_op(
        "qwen3_vl_get_mrope_position_deltas",
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        spatial_merge_size=config.vision_config.spatial_merge_size
    )
    return {
        "position_ids": position_ids,
        "mrope_position_deltas": mrope_position_deltas
    }


def Qwen3MergeVisualEmbeddings(input_ids, text_embeddings, visual_embeds, image_token_id, video_token_id):
    # Replace positions corresponding to image/video placeholder tokens with
    # merged visual embeddings. Also returns a visual_pos_mask for DeepStack.
    return text_embeddings * T.binary_op(
        "qwen3_vl_merge_visual_embeddings",
        input_ids=input_ids,
        image_token_id=image_token_id,
        video_token_id=video_token_id
    ) * visual_embeds


def Qwen3PrepareInputsEmbeds(input_ids, visual_outputs, weights, config):
    text_embeddings = input_ids * T.unary_op(
        "embedding",
        weight_name="model.language_model.embed_tokens.weight"
    )
    inputs_embeds = Qwen3MergeVisualEmbeddings(
        input_ids,
        text_embeddings,
        visual_outputs["pooler_output"],
        image_token_id=config.image_token_id,
        video_token_id=config.video_token_id
    )
    return {
        "inputs_embeds": inputs_embeds,
        "deepstack_visual_embeds": visual_outputs["deepstack_features"]
    }
