import CpuTensor as T

# Qwen3-VL vision tower.
#
# This file follows the current Hugging Face Qwen3-VL structure:
#   PatchEmbed(Conv3d) -> interpolated absolute pos embed -> vision blocks
#   -> final patch merger, plus DeepStack mergers from selected vision layers.
#
# Many operations are intentionally expressed as open ops because Garnet does not
# implement them yet. Workbench and backend handlers should map these names.


def linear(x, weight, bias=None, op="linear"):
    y = x * T.binary_op(op) * weight
    if bias is not None:
        y = y + bias
    return y


def layer_norm(x, weight, bias, eps=1e-6):
    return x * T.unary_op("layer_norm", weight=weight, bias=bias, eps=eps)


def gelu(x):
    return x * T.unary_op("gelu_pytorch_tanh")


def VisionPatchMerger(x, weights, config, prefix, use_postshuffle_norm=False):
    # HF Qwen3VLVisionPatchMerger:
    #   hidden_size = vision_hidden_size * spatial_merge_size^2
    #   norm -> linear_fc1 -> GELU -> linear_fc2(out_hidden_size)
    x = x * T.unary_op(
        "qwen3_vl_patch_merger_shuffle",
        spatial_merge_size=config.vision_config.spatial_merge_size,
        use_postshuffle_norm=use_postshuffle_norm
    )
    x = layer_norm(
        x,
        weights[prefix + ".norm.weight"],
        weights[prefix + ".norm.bias"],
        eps=1e-6
    )
    x = linear(
        x,
        weights[prefix + ".linear_fc1.weight"],
        weights[prefix + ".linear_fc1.bias"]
    )
    x = gelu(x)
    x = linear(
        x,
        weights[prefix + ".linear_fc2.weight"],
        weights[prefix + ".linear_fc2.bias"]
    )
    return x


def VisionMLP(x, weights, prefix, hidden_act):
    x = linear(
        x,
        weights[prefix + ".linear_fc1.weight"],
        weights[prefix + ".linear_fc1.bias"]
    )
    x = x * T.unary_op(hidden_act)
    x = linear(
        x,
        weights[prefix + ".linear_fc2.weight"],
        weights[prefix + ".linear_fc2.bias"]
    )
    return x


def VisionAttention(x, cu_seqlens, position_embeddings, weights, config, prefix):
    # HF Qwen3VLVisionAttention uses one qkv projection with bias, vision RoPE,
    # variable-length non-causal attention, then output projection.
    qkv = linear(
        x,
        weights[prefix + ".qkv.weight"],
        weights[prefix + ".qkv.bias"],
        op="qkv_linear"
    )
    q, k, v = qkv * T.unary_op(
        "qwen3_vl_split_vision_qkv",
        num_heads=config.vision_config.num_heads,
        head_dim=config.vision_config.hidden_size // config.vision_config.num_heads
    )
    q, k = q * T.binary_op("qwen3_vl_apply_vision_rope") * k
    attn_out = q * T.binary_op(
        "vision_varlen_attention",
        cu_seqlens=cu_seqlens,
        position_embeddings=position_embeddings,
        causal=False,
        scale=(config.vision_config.hidden_size // config.vision_config.num_heads) ** -0.5
    ) * v
    attn_out = attn_out * T.unary_op("merge_attention_heads")
    attn_out = linear(
        attn_out,
        weights[prefix + ".proj.weight"],
        weights[prefix + ".proj.bias"]
    )
    return attn_out


def VisionBlock(x, cu_seqlens, position_embeddings, weights, config, layer_idx):
    prefix = "visual.blocks." + str(layer_idx)

    residual = x
    x_norm = layer_norm(
        x,
        weights[prefix + ".norm1.weight"],
        weights[prefix + ".norm1.bias"],
        eps=1e-6
    )
    x = residual + VisionAttention(
        x_norm,
        cu_seqlens,
        position_embeddings,
        weights,
        config,
        prefix + ".attn"
    )

    residual = x
    x_norm = layer_norm(
        x,
        weights[prefix + ".norm2.weight"],
        weights[prefix + ".norm2.bias"],
        eps=1e-6
    )
    x = residual + VisionMLP(
        x_norm,
        weights,
        prefix + ".mlp",
        hidden_act=config.vision_config.hidden_act
    )
    return x


@T.fusion()
def Qwen3VisionEncoder(pixel_values, grid_thw, weights, config):
    # Build vision sequence metadata.
    bilinear = grid_thw * T.unary_op(
        "qwen3_vl_vision_bilinear_indices_and_weights",
        num_position_embeddings=config.vision_config.num_position_embeddings,
        spatial_merge_size=config.vision_config.spatial_merge_size
    )
    vision_position_ids = grid_thw * T.unary_op(
        "qwen3_vl_vision_position_ids",
        spatial_merge_size=config.vision_config.spatial_merge_size
    )
    cu_seqlens = grid_thw * T.unary_op("qwen3_vl_vision_cu_seqlens")

    # PatchEmbed is Conv3d with kernel/stride:
    # (temporal_patch_size, patch_size, patch_size).
    x = pixel_values * T.unary_op(
        "qwen3_vl_patch_embed_conv3d",
        weight=weights["visual.patch_embed.proj.weight"],
        bias=weights["visual.patch_embed.proj.bias"],
        patch_size=config.vision_config.patch_size,
        temporal_patch_size=config.vision_config.temporal_patch_size,
        in_channels=config.vision_config.in_channels
    )

    # Interpolated absolute position embedding.
    pos_embeds = weights["visual.pos_embed.weight"] * T.binary_op(
        "qwen3_vl_pos_embed_interpolate"
    ) * bilinear
    x = x + pos_embeds

    # Vision rotary embeddings used by all vision attention blocks.
    position_embeddings = vision_position_ids * T.unary_op(
        "qwen3_vl_vision_rotary_embedding",
        head_dim=(config.vision_config.hidden_size // config.vision_config.num_heads) // 2
    )

    deepstack_features = []
    for layer_idx in range(config.vision_config.depth):
        x = VisionBlock(x, cu_seqlens, position_embeddings, weights, config, layer_idx)

        if layer_idx in config.vision_config.deepstack_visual_indexes:
            deepstack_slot = config.vision_config.deepstack_visual_indexes.index(layer_idx)
            deepstack_feature = VisionPatchMerger(
                x,
                weights,
                config,
                "visual.deepstack_merger_list." + str(deepstack_slot),
                use_postshuffle_norm=True
            )
            deepstack_features.append(deepstack_feature)

    merged_visual_tokens = VisionPatchMerger(
        x,
        weights,
        config,
        "visual.merger",
        use_postshuffle_norm=False
    )

    return {
        "last_hidden_state": x,
        "pooler_output": merged_visual_tokens,
        "deepstack_features": deepstack_features,
        "grid_thw": grid_thw
    }
