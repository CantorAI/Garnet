from garnet import garnet

T = garnet.tensor()

# Qwen3-VL vision tower.
#
# This file follows the current Hugging Face Qwen3-VL structure:
#   PatchEmbed(Conv3d) -> interpolated absolute pos embed -> vision blocks
#   -> final patch merger, plus DeepStack mergers from selected vision layers.
#
# Many operations are intentionally expressed as open ops because Garnet does not
# implement them yet. Workbench and backend handlers should map these names.


def linear(x, weight_name, bias_name=None, op="linear"):
    return x * T.unary_op(op, weight_name=weight_name, bias_name=bias_name)


def layer_norm(x, weight_name, bias_name, eps=1e-6):
    return x * T.unary_op(
        "layer_norm",
        weight_name=weight_name,
        bias_name=bias_name,
        eps=eps
    )


def gelu(x):
    return x * T.unary_op("gelu_pytorch_tanh")


def VisionPatchMergerProjection(x, prefix):
    x = linear(
        x,
        prefix + ".linear_fc1.weight",
        prefix + ".linear_fc1.bias"
    )
    x = gelu(x)
    x = linear(
        x,
        prefix + ".linear_fc2.weight",
        prefix + ".linear_fc2.bias"
    )
    return x


def VisionPatchMergerPostShuffleNorm(x, config, prefix):
    x = x * T.unary_op(
        "qwen3_vl_patch_merger_shuffle",
        spatial_merge_size=config.vision_config.spatial_merge_size
    )
    x = layer_norm(x, prefix + ".norm.weight", prefix + ".norm.bias", eps=1e-6)
    return VisionPatchMergerProjection(x, prefix)


def VisionPatchMergerPreShuffleNorm(x, config, prefix):
    x = layer_norm(x, prefix + ".norm.weight", prefix + ".norm.bias", eps=1e-6)
    x = x * T.unary_op(
        "qwen3_vl_patch_merger_shuffle",
        spatial_merge_size=config.vision_config.spatial_merge_size
    )
    return VisionPatchMergerProjection(x, prefix)


def VisionMLP(x, weights, prefix, hidden_act):
    x = linear(
        x,
        prefix + ".linear_fc1.weight",
        prefix + ".linear_fc1.bias"
    )
    x = x * T.unary_op(hidden_act)
    x = linear(
        x,
        prefix + ".linear_fc2.weight",
        prefix + ".linear_fc2.bias"
    )
    return x


def VisionAttention(x, cu_seqlens, vision_position_ids, weights, config, prefix):
    # HF Qwen3VLVisionAttention uses one qkv projection with bias, vision RoPE,
    # variable-length non-causal attention, then output projection.
    qkv = linear(
        x,
        prefix + ".qkv.weight",
        prefix + ".qkv.bias",
        op="qkv_linear"
    )
    qkv = qkv * T.binary_op(
        "qwen3_vl_apply_vision_rope_packed",
        num_heads=config.vision_config.num_heads,
        head_dim=config.vision_config.hidden_size // config.vision_config.num_heads
    ) * vision_position_ids
    attn_out = qkv * T.binary_op(
        "vision_varlen_attention_packed",
        num_heads=config.vision_config.num_heads,
        head_dim=config.vision_config.hidden_size // config.vision_config.num_heads,
        causal=False
    ) * cu_seqlens
    attn_out = linear(
        attn_out,
        prefix + ".proj.weight",
        prefix + ".proj.bias"
    )
    return attn_out


@T.fusion(role="vision_layer", atomic=True)
def VisionBlock(x, cu_seqlens, vision_position_ids, weights, config, layer_idx):
    prefix = "model.visual.blocks." + str(layer_idx)

    residual = x
    x_norm = layer_norm(
        x,
        prefix + ".norm1.weight",
        prefix + ".norm1.bias",
        eps=1e-6
    )
    x = residual + VisionAttention(
        x_norm,
        cu_seqlens,
        vision_position_ids,
        weights,
        config,
        prefix + ".attn"
    )

    residual = x
    x_norm = layer_norm(
        x,
        prefix + ".norm2.weight",
        prefix + ".norm2.bias",
        eps=1e-6
    )
    x = residual + VisionMLP(
        x_norm,
        weights,
        prefix + ".mlp",
        hidden_act=config.vision_config.hidden_act
    )
    return x


@T.fusion(name="vision", role="encoder", boundary="preferred")
def Qwen3VisionEncoder(pixel_values, grid_thw, bilinear_indices, bilinear_weights, vision_position_ids, cu_seqlens, weights, config):
    # PatchEmbed is Conv3d with kernel/stride:
    # (temporal_patch_size, patch_size, patch_size).
    x = pixel_values * T.unary_op(
        "qwen3_vl_patch_embed_conv3d",
        weight_name="model.visual.patch_embed.proj.weight",
        bias_name="model.visual.patch_embed.proj.bias",
        patch_size=config.vision_config.patch_size,
        temporal_patch_size=config.vision_config.temporal_patch_size,
        in_channels=config.vision_config.in_channels
    )

    # Interpolated absolute position embedding.
    pos_embeds = bilinear_indices * T.binary_op(
        "qwen3_vl_pos_embed_interpolate",
        weight_name="model.visual.pos_embed.weight"
    ) * bilinear_weights
    x = x + pos_embeds

    # DeepStack indexes are compile-time model configuration. Segmenting the
    # loop keeps them out of TensorGraph branch metadata.
    deepstack_features = []
    next_layer = 0
    for deepstack_slot in range(len(config.vision_config.deepstack_visual_indexes)):
        stop_layer = config.vision_config.deepstack_visual_indexes[deepstack_slot] + 1
        for layer_offset in range(stop_layer - next_layer):
            layer_idx = next_layer + layer_offset
            x = VisionBlock(x, cu_seqlens, vision_position_ids, weights, config, layer_idx)
        deepstack_feature = VisionPatchMergerPostShuffleNorm(
            x,
            config,
            "model.visual.deepstack_merger_list." + str(deepstack_slot)
        )
        deepstack_features.append(deepstack_feature)
        next_layer = stop_layer

    for layer_offset in range(config.vision_config.depth - next_layer):
        layer_idx = next_layer + layer_offset
        x = VisionBlock(x, cu_seqlens, vision_position_ids, weights, config, layer_idx)

    merged_visual_tokens = VisionPatchMergerPreShuffleNorm(
        x,
        config,
        "model.visual.merger"
    )

    return {
        "last_hidden_state": x,
        "pooler_output": merged_visual_tokens,
        "deepstack_features": deepstack_features,
        "grid_thw": grid_thw
    }
