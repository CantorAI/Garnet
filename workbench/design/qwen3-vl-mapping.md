# Qwen3-VL Mapping For Workbench

This file maps the current Garnet `.x` model source to the Qwen3-VL dense model
structure. It is the first model target for Workbench.

## Source Files

```text
xModel/qwen3/vl_2b_instruct/qwen_vl_model.x
xModel/qwen3/vl_2b_instruct/vision_encoder.x
xModel/qwen3/vl_2b_instruct/vl_adapter.x
xModel/qwen3/vl_2b_instruct/qwen_llm.x
```

## Top-Level Flow

```text
Qwen3VLModel
  -> Qwen3VisionEncoder
  -> Qwen3PrepareInputsEmbeds
  -> Qwen3GetRopeIndex
  -> Qwen3TextModel
  -> lm_head
```

## Vision Tower

Workbench blocks:

- `VisionPatchEmbed`
- `VisionPositionInterpolation`
- `VisionRotaryEmbedding`
- `VisionBlock[0..depth-1]`
- `VisionAttention[i]`
- `VisionMLP[i]`
- `VisionPatchMerger`
- `DeepStackMerger[0..2]`

Important open ops:

- `qwen3_vl_vision_bilinear_indices_and_weights`
- `qwen3_vl_vision_position_ids`
- `qwen3_vl_vision_cu_seqlens`
- `qwen3_vl_patch_embed_conv3d`
- `qwen3_vl_pos_embed_interpolate`
- `qwen3_vl_vision_rotary_embedding`
- `qwen3_vl_split_vision_qkv`
- `qwen3_vl_apply_vision_rope`
- `vision_varlen_attention`
- `qwen3_vl_patch_merger_shuffle`
- `gelu_pytorch_tanh`
- `layer_norm`

## Multimodal Merge

Workbench blocks:

- `TextEmbedding`
- `VisualPlaceholderMerge`
- `VisualPositionMask`
- `DeepStackFeatureRoute`
- `MRoPEPositionBuilder`

Important open ops:

- `embedding`
- `qwen3_vl_merge_visual_embeddings`
- `qwen3_vl_get_rope_index`
- `qwen3_vl_llm_vision_position_ids`

## Text Decoder

Workbench blocks:

- `TextEmbedding`
- `CausalMask`
- `TextMRoPE`
- `TextDecoderLayer[0..num_hidden_layers-1]`
- `TextAttention[i]`
- `TextMLP[i]`
- `DeepStackAdd[i]`
- `FinalRMSNorm`
- `LMHead`

Important open ops:

- `qwen3_vl_text_interleaved_mrope`
- `reshape_q_heads`
- `reshape_kv_heads`
- `rms_norm`
- `qwen3_vl_apply_text_rope`
- `paged_kv_update`
- `paged_attention`
- `merge_attention_heads`
- `create_causal_mask`
- `qwen3_vl_deepstack_add`
- `lm_head`

## Backend Ownership Draft

| Area | Preferred backend | Reason |
| --- | --- | --- |
| dense linear / MLP | TensorRT or custom GEMM path | stable dense math |
| layer norm / RMSNorm | custom CUDA or TRT plugin | common fusion target |
| vision patch embed Conv3d | TensorRT first | standard conv |
| vision varlen attention | custom CUDA / flash attention style | dynamic sequence lengths |
| text paged attention | custom CUDA | KV page table and decode path |
| MRoPE | custom CUDA | Qwen-specific indexing |
| visual placeholder merge | custom CUDA or CPU debug first | dynamic scatter/replace |
| KV cache update | custom CUDA | serving runtime state |
| sampling | custom CUDA later | generation hot path |

## MVP Risk

The `.x` files are now closer to the real model architecture, but Workbench
should not depend on full xlang execution for the first version. The first parser
can statically extract model blocks and open ops, then later connect to real
TensorExpression/TensorGraph once the frontend execution path is stable.

