from garnet import garnet

T = garnet.tensor()
T.set_backend("TensorRT")


@T.fusion(name="vision", role="encoder", boundary="preferred")
def Vision(visual_embeddings):
    return visual_embeddings + visual_embeddings


@T.fusion(name="merge", role="multimodal_merge", boundary="required")
def Merge(input_ids, text_embeddings, visual_embeddings):
    return text_embeddings * T.binary_op(
        "qwen3_vl_merge_visual_embeddings",
        input_ids=input_ids,
        image_token_id=1,
        video_token_id=2
    ) * visual_embeddings


@T.fusion(name="keyword_boundary_root", boundary="required")
def Model(input_ids, text_embeddings, visual_embeddings):
    visual_output = Vision(visual_embeddings)
    return Merge(input_ids, text_embeddings, visual_output)
