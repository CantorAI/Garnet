window.GARNET_DEFAULT_GRAPH = {
  "schema_version": "0.1",
  "model": {
    "name": "Qwen3-VL from xmodel",
    "family": "qwen3_vl",
    "source": "qwen_vl/xmodel/qwen_vl_model.x"
  },
  "nodes": [
    {
      "id": "block.Qwen3VLModel",
      "kind": "block",
      "label": "Qwen3VLModel",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/qwen_vl_model.x",
        "function": "Qwen3VLModel",
        "line": 19
      },
      "metadata": {
        "extractor": "static_xmodel",
        "expression": "def Qwen3VLModel(\n    input_ids,\n    pixel_values,\n    image_grid_thw,\n    video_grid_thw,\n    mm_token_type_ids,\n    attention_mask,\n    weights,\n    config,\n    past_key_values=None,\n    use_cache=True"
      }
    },
    {
      "id": "op.qwen3vlmodel.lm_head.71",
      "kind": "op",
      "label": "lm_head",
      "op": "lm_head",
      "backend": "trt",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/qwen_vl_model.x",
        "function": "Qwen3VLModel",
        "line": 71
      },
      "metadata": {
        "xlang_op_kind": "binary_op",
        "extractor": "static_xmodel",
        "expression": "logits = hidden_states * T.binary_op("
      }
    },
    {
      "id": "matrix.language_model_embed_tokens_weight",
      "kind": "matrix",
      "label": "language_model.embed_tokens.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[151936, 2048]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/qwen_vl_model.x",
        "function": "Qwen3VLModel",
        "line": 74
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "embedding_matrix",
        "dimensions": [
          151936,
          2048
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 311164928,
        "bytes_fp16": 622329856,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"language_model.embed_tokens.weight\"]"
      }
    },
    {
      "id": "block.linear",
      "kind": "block",
      "label": "linear",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "linear",
        "line": 13
      },
      "metadata": {
        "extractor": "static_xmodel",
        "expression": "def linear(x, weight, bias=None, op=\"linear\"):\n    y = x * T.binary_op(op) * weight\n    if bias is not None:\n        y = y + bias\n    return y\n\n"
      }
    },
    {
      "id": "block.layer_norm",
      "kind": "block",
      "label": "layer_norm",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "layer_norm",
        "line": 20
      },
      "metadata": {
        "extractor": "static_xmodel",
        "expression": "def layer_norm(x, weight, bias, eps=1e-6):\n    return x * T.unary_op(\"layer_norm\", weight=weight, bias=bias, eps=eps)\n\n"
      }
    },
    {
      "id": "op.layer_norm.layer_norm.21",
      "kind": "op",
      "label": "layer_norm",
      "op": "layer_norm",
      "backend": "cuda",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "layer_norm",
        "line": 21
      },
      "metadata": {
        "xlang_op_kind": "unary_op",
        "extractor": "static_xmodel",
        "expression": "return x * T.unary_op(\"layer_norm\", weight=weight, bias=bias, eps=eps)"
      }
    },
    {
      "id": "block.gelu",
      "kind": "block",
      "label": "gelu",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "gelu",
        "line": 24
      },
      "metadata": {
        "extractor": "static_xmodel",
        "expression": "def gelu(x):\n    return x * T.unary_op(\"gelu_pytorch_tanh\")\n\n"
      }
    },
    {
      "id": "op.gelu.gelu_pytorch_tanh.25",
      "kind": "op",
      "label": "gelu_pytorch_tanh",
      "op": "gelu_pytorch_tanh",
      "backend": "cuda",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "gelu",
        "line": 25
      },
      "metadata": {
        "xlang_op_kind": "unary_op",
        "extractor": "static_xmodel",
        "expression": "return x * T.unary_op(\"gelu_pytorch_tanh\")"
      }
    },
    {
      "id": "block.VisionPatchMerger",
      "kind": "block",
      "label": "VisionPatchMerger",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionPatchMerger",
        "line": 28
      },
      "metadata": {
        "extractor": "static_xmodel",
        "expression": "def VisionPatchMerger(x, weights, config, prefix, use_postshuffle_norm=False):\n    # HF Qwen3VLVisionPatchMerger:\n    #   hidden_size = vision_hidden_size * spatial_merge_size^2\n    #   norm -> linear_fc1 -> GELU -> linear_fc2(out_hidden_size)\n    x = x * T.unary_op(\n        \"qwen3_vl_patch_merger_shuffle\",\n        spatial_merge_size=config.vision_config.spatial_merge_size,\n        use_postshuffle_norm=use_postshuffle_norm\n    )\n    x = layer_norm(\n        x,\n        weights[prefix + \".norm.weight\"],\n        weights[prefix + \".norm.bias\"],\n        eps=1e-6\n    )\n    x = linear(\n        x,\n        weights[prefix + \".linear_fc1.weight\"],\n        weights[prefix + \".linear_fc1.bias\"]\n    )\n    x = gelu(x)\n    x = linear(\n        x,\n        weights[prefix + \".linear_fc2.weight\"],\n        weights[prefix + \".linear_fc2.bias\"]\n    )\n    return x\n\n"
      }
    },
    {
      "id": "op.visionpatchmerger.qwen3_vl_patch_merger_shuffle.32",
      "kind": "op",
      "label": "qwen3_vl_patch_merger_shuffle",
      "op": "qwen3_vl_patch_merger_shuffle",
      "backend": "unknown",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionPatchMerger",
        "line": 32
      },
      "metadata": {
        "xlang_op_kind": "unary_op",
        "extractor": "static_xmodel",
        "expression": "x = x * T.unary_op("
      }
    },
    {
      "id": "matrix.visionpatchmerger_norm_weight",
      "kind": "matrix",
      "label": "VisionPatchMerger.norm.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[4096]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionPatchMerger",
        "line": 39
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "normalization_vector",
        "dimensions": [
          4096
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 4096,
        "bytes_fp16": 8192,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"VisionPatchMerger.norm.weight\"]"
      }
    },
    {
      "id": "matrix.visionpatchmerger_norm_bias",
      "kind": "matrix",
      "label": "VisionPatchMerger.norm.bias",
      "status": "defined",
      "shape": {
        "outputs": [
          "[4096]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionPatchMerger",
        "line": 40
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "normalization_vector",
        "dimensions": [
          4096
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 4096,
        "bytes_fp16": 8192,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"VisionPatchMerger.norm.bias\"]"
      }
    },
    {
      "id": "matrix.visionpatchmerger_linear_fc1_weight",
      "kind": "matrix",
      "label": "VisionPatchMerger.linear_fc1.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[4096, 4096]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionPatchMerger",
        "line": 45
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "mlp_matrix",
        "dimensions": [
          4096,
          4096
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 16777216,
        "bytes_fp16": 33554432,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"VisionPatchMerger.linear_fc1.weight\"]"
      }
    },
    {
      "id": "matrix.visionpatchmerger_linear_fc1_bias",
      "kind": "matrix",
      "label": "VisionPatchMerger.linear_fc1.bias",
      "status": "defined",
      "shape": {
        "outputs": [
          "[4096]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionPatchMerger",
        "line": 46
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "mlp_matrix",
        "dimensions": [
          4096
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 4096,
        "bytes_fp16": 8192,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"VisionPatchMerger.linear_fc1.bias\"]"
      }
    },
    {
      "id": "matrix.visionpatchmerger_linear_fc2_weight",
      "kind": "matrix",
      "label": "VisionPatchMerger.linear_fc2.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[2048, 4096]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionPatchMerger",
        "line": 51
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "mlp_matrix",
        "dimensions": [
          2048,
          4096
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 8388608,
        "bytes_fp16": 16777216,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"VisionPatchMerger.linear_fc2.weight\"]"
      }
    },
    {
      "id": "matrix.visionpatchmerger_linear_fc2_bias",
      "kind": "matrix",
      "label": "VisionPatchMerger.linear_fc2.bias",
      "status": "defined",
      "shape": {
        "outputs": [
          "[2048]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionPatchMerger",
        "line": 52
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "mlp_matrix",
        "dimensions": [
          2048
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 2048,
        "bytes_fp16": 4096,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"VisionPatchMerger.linear_fc2.bias\"]"
      }
    },
    {
      "id": "block.VisionMLP",
      "kind": "block",
      "label": "VisionMLP",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionMLP",
        "line": 57
      },
      "metadata": {
        "extractor": "static_xmodel",
        "expression": "def VisionMLP(x, weights, prefix, hidden_act):\n    x = linear(\n        x,\n        weights[prefix + \".linear_fc1.weight\"],\n        weights[prefix + \".linear_fc1.bias\"]\n    )\n    x = x * T.unary_op(hidden_act)\n    x = linear(\n        x,\n        weights[prefix + \".linear_fc2.weight\"],\n        weights[prefix + \".linear_fc2.bias\"]\n    )\n    return x\n\n"
      }
    },
    {
      "id": "matrix.visionmlp_linear_fc1_weight",
      "kind": "matrix",
      "label": "VisionMLP.linear_fc1.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[4096, 1024]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionMLP",
        "line": 60
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "mlp_matrix",
        "dimensions": [
          4096,
          1024
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 4194304,
        "bytes_fp16": 8388608,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"VisionMLP.linear_fc1.weight\"]"
      }
    },
    {
      "id": "matrix.visionmlp_linear_fc1_bias",
      "kind": "matrix",
      "label": "VisionMLP.linear_fc1.bias",
      "status": "defined",
      "shape": {
        "outputs": [
          "[4096]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionMLP",
        "line": 61
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "mlp_matrix",
        "dimensions": [
          4096
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 4096,
        "bytes_fp16": 8192,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"VisionMLP.linear_fc1.bias\"]"
      }
    },
    {
      "id": "matrix.visionmlp_linear_fc2_weight",
      "kind": "matrix",
      "label": "VisionMLP.linear_fc2.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[1024, 4096]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionMLP",
        "line": 66
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "mlp_matrix",
        "dimensions": [
          1024,
          4096
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 4194304,
        "bytes_fp16": 8388608,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"VisionMLP.linear_fc2.weight\"]"
      }
    },
    {
      "id": "matrix.visionmlp_linear_fc2_bias",
      "kind": "matrix",
      "label": "VisionMLP.linear_fc2.bias",
      "status": "defined",
      "shape": {
        "outputs": [
          "[1024]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionMLP",
        "line": 67
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "mlp_matrix",
        "dimensions": [
          1024
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 1024,
        "bytes_fp16": 2048,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"VisionMLP.linear_fc2.bias\"]"
      }
    },
    {
      "id": "block.VisionAttention",
      "kind": "block",
      "label": "VisionAttention",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionAttention",
        "line": 72
      },
      "metadata": {
        "extractor": "static_xmodel",
        "expression": "def VisionAttention(x, cu_seqlens, position_embeddings, weights, config, prefix):\n    # HF Qwen3VLVisionAttention uses one qkv projection with bias, vision RoPE,\n    # variable-length non-causal attention, then output projection.\n    qkv = linear(\n        x,\n        weights[prefix + \".qkv.weight\"],\n        weights[prefix + \".qkv.bias\"],\n        op=\"qkv_linear\"\n    )\n    q, k, v = qkv * T.unary_op(\n        \"qwen3_vl_split_vision_qkv\",\n        num_heads=config.vision_config.num_heads,\n        head_dim=config.vision_config.hidden_size // config.vision_config.num_heads\n    )\n    q, k = q * T.binary_op(\"qwen3_vl_apply_vision_rope\") * k\n    attn_out = q * T.binary_op(\n        \"vision_varlen_attention\",\n        cu_seqlens=cu_seqlens,\n        position_embeddings=position_embeddings,\n        causal=False,\n        scale=(config.vision_config.hidden_size // config.vision_config.num_heads) ** -0.5\n    ) * v\n    attn_out = attn_out * T.unary_op(\"merge_attention_heads\")\n    attn_out = linear(\n        attn_out,\n        weights[prefix + \".proj.weight\"],\n        weights[prefix + \".proj.bias\"]\n    )\n    return attn_out\n\n"
      }
    },
    {
      "id": "matrix.visionattention_qkv_weight",
      "kind": "matrix",
      "label": "VisionAttention.qkv.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[3072, 1024]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionAttention",
        "line": 77
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "attention_matrix",
        "dimensions": [
          3072,
          1024
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 3145728,
        "bytes_fp16": 6291456,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"VisionAttention.qkv.weight\"]"
      }
    },
    {
      "id": "matrix.visionattention_qkv_bias",
      "kind": "matrix",
      "label": "VisionAttention.qkv.bias",
      "status": "defined",
      "shape": {
        "outputs": [
          "[3072]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionAttention",
        "line": 78
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "attention_matrix",
        "dimensions": [
          3072
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 3072,
        "bytes_fp16": 6144,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"VisionAttention.qkv.bias\"]"
      }
    },
    {
      "id": "op.visionattention.qwen3_vl_split_vision_qkv.81",
      "kind": "op",
      "label": "qwen3_vl_split_vision_qkv",
      "op": "qwen3_vl_split_vision_qkv",
      "backend": "unknown",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionAttention",
        "line": 81
      },
      "metadata": {
        "xlang_op_kind": "unary_op",
        "extractor": "static_xmodel",
        "expression": "q, k, v = qkv * T.unary_op("
      }
    },
    {
      "id": "op.visionattention.qwen3_vl_apply_vision_rope.86",
      "kind": "op",
      "label": "qwen3_vl_apply_vision_rope",
      "op": "qwen3_vl_apply_vision_rope",
      "backend": "cuda",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionAttention",
        "line": 86
      },
      "metadata": {
        "xlang_op_kind": "binary_op",
        "extractor": "static_xmodel",
        "expression": "q, k = q * T.binary_op(\"qwen3_vl_apply_vision_rope\") * k"
      }
    },
    {
      "id": "op.visionattention.vision_varlen_attention.87",
      "kind": "op",
      "label": "vision_varlen_attention",
      "op": "vision_varlen_attention",
      "backend": "cuda",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionAttention",
        "line": 87
      },
      "metadata": {
        "xlang_op_kind": "binary_op",
        "extractor": "static_xmodel",
        "expression": "attn_out = q * T.binary_op("
      }
    },
    {
      "id": "op.visionattention.merge_attention_heads.94",
      "kind": "op",
      "label": "merge_attention_heads",
      "op": "merge_attention_heads",
      "backend": "cuda",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionAttention",
        "line": 94
      },
      "metadata": {
        "xlang_op_kind": "unary_op",
        "extractor": "static_xmodel",
        "expression": "attn_out = attn_out * T.unary_op(\"merge_attention_heads\")"
      }
    },
    {
      "id": "matrix.visionattention_proj_weight",
      "kind": "matrix",
      "label": "VisionAttention.proj.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[1024, 1024]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionAttention",
        "line": 97
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "matrix",
        "dimensions": [
          1024,
          1024
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 1048576,
        "bytes_fp16": 2097152,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"VisionAttention.proj.weight\"]"
      }
    },
    {
      "id": "matrix.visionattention_proj_bias",
      "kind": "matrix",
      "label": "VisionAttention.proj.bias",
      "status": "defined",
      "shape": {
        "outputs": [
          "[1024]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionAttention",
        "line": 98
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "matrix",
        "dimensions": [
          1024
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 1024,
        "bytes_fp16": 2048,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"VisionAttention.proj.bias\"]"
      }
    },
    {
      "id": "block.VisionBlock",
      "kind": "block",
      "label": "VisionBlock",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionBlock",
        "line": 103
      },
      "metadata": {
        "extractor": "static_xmodel",
        "expression": "def VisionBlock(x, cu_seqlens, position_embeddings, weights, config, layer_idx):\n    prefix = \"visual.blocks.\" + str(layer_idx)\n\n    residual = x\n    x_norm = layer_norm(\n        x,\n        weights[prefix + \".norm1.weight\"],\n        weights[prefix + \".norm1.bias\"],\n        eps=1e-6\n    )\n    x = residual + VisionAttention(\n        x_norm,\n        cu_seqlens,\n        position_embeddings,\n        weights,\n        config,\n        prefix + \".attn\"\n    )\n\n    residual = x\n    x_norm = layer_norm(\n        x,\n        weights[prefix + \".norm2.weight\"],\n        weights[prefix + \".norm2.bias\"],\n        eps=1e-6\n    )\n    x = residual + VisionMLP(\n        x_norm,\n        weights,\n        prefix + \".mlp\",\n        hidden_act=config.vision_config.hidden_act\n    )\n    return x\n\n"
      }
    },
    {
      "id": "matrix.visionblock_norm1_weight",
      "kind": "matrix",
      "label": "VisionBlock.norm1.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[1024]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionBlock",
        "line": 109
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "normalization_vector",
        "dimensions": [
          1024
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 1024,
        "bytes_fp16": 2048,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"VisionBlock.norm1.weight\"]"
      }
    },
    {
      "id": "matrix.visionblock_norm1_bias",
      "kind": "matrix",
      "label": "VisionBlock.norm1.bias",
      "status": "defined",
      "shape": {
        "outputs": [
          "[1024]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionBlock",
        "line": 110
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "normalization_vector",
        "dimensions": [
          1024
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 1024,
        "bytes_fp16": 2048,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"VisionBlock.norm1.bias\"]"
      }
    },
    {
      "id": "matrix.visionblock_norm2_weight",
      "kind": "matrix",
      "label": "VisionBlock.norm2.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[1024]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionBlock",
        "line": 125
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "normalization_vector",
        "dimensions": [
          1024
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 1024,
        "bytes_fp16": 2048,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"VisionBlock.norm2.weight\"]"
      }
    },
    {
      "id": "matrix.visionblock_norm2_bias",
      "kind": "matrix",
      "label": "VisionBlock.norm2.bias",
      "status": "defined",
      "shape": {
        "outputs": [
          "[1024]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "VisionBlock",
        "line": 126
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "normalization_vector",
        "dimensions": [
          1024
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 1024,
        "bytes_fp16": 2048,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"VisionBlock.norm2.bias\"]"
      }
    },
    {
      "id": "block.Qwen3VisionEncoder",
      "kind": "block",
      "label": "Qwen3VisionEncoder",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "Qwen3VisionEncoder",
        "line": 139
      },
      "metadata": {
        "extractor": "static_xmodel",
        "expression": "def Qwen3VisionEncoder(pixel_values, grid_thw, weights, config):\n    # Build vision sequence metadata.\n    bilinear = grid_thw * T.unary_op(\n        \"qwen3_vl_vision_bilinear_indices_and_weights\",\n        num_position_embeddings=config.vision_config.num_position_embeddings,\n        spatial_merge_size=config.vision_config.spatial_merge_size\n    )\n    vision_position_ids = grid_thw * T.unary_op(\n        \"qwen3_vl_vision_position_ids\",\n        spatial_merge_size=config.vision_config.spatial_merge_size\n    )\n    cu_seqlens = grid_thw * T.unary_op(\"qwen3_vl_vision_cu_seqlens\")\n\n    # PatchEmbed is Conv3d with kernel/stride:\n    # (temporal_patch_size, patch_size, patch_size).\n    x = pixel_values * T.unary_op(\n        \"qwen3_vl_patch_embed_conv3d\",\n        weight=weights[\"visual.patch_embed.proj.weight\"],\n        bias=weights[\"visual.patch_embed.proj.bias\"],\n        patch_size=config.vision_config.patch_size,\n        temporal_patch_size=config.vision_config.temporal_patch_size,\n        in_channels=config.vision_config.in_channels\n    )\n\n    # Interpolated absolute position embedding.\n    pos_embeds = weights[\"visual.pos_embed.weight\"] * T.binary_op(\n        \"qwen3_vl_pos_embed_interpolate\"\n    ) * bilinear\n    x = x + pos_embeds\n\n    # Vision rotary embeddings used by all vision attention blocks.\n    position_embeddings = vision_position_ids * T.unary_op(\n        \"qwen3_vl_vision_rotary_embedding\",\n        head_dim=(config.vision_config.hidden_size // config.vision_config.num_heads) // 2\n    )\n\n    deepstack_features = []\n    for layer_idx in range(config.vision_config.depth):\n        x = VisionBlock(x, cu_seqlens, position_embeddings, weights, config, layer_idx)\n\n        if layer_idx in config.vision_config.deepstack_visual_indexes:\n            deepstack_slot = config.vision_config.deepstack_visual_indexes.index(layer_idx)\n            deepstack_feature = VisionPatchMerger(\n                x,\n                weights,\n                config,\n                \"visual.deepstack_merger_list.\" + str(deepstack_slot),\n                use_postshuffle_norm=True\n            )\n            deepstack_features.append(deepstack_feature)\n\n    merged_visual_tokens = VisionPatchMerger(\n        x,\n        weights,\n        config,\n        \"visual.merger\",\n        use_postshuffle_norm=False\n    )\n\n    return {\n        \"last_hidden_state\": x,\n        \"pooler_output\": merged_visual_tokens,\n        \"deepstack_features\": deepstack_features,\n        \"grid_thw\": grid_thw\n    }\n"
      }
    },
    {
      "id": "op.qwen3visionencoder.qwen3_vl_vision_bilinear_indices_and_weights.141",
      "kind": "op",
      "label": "qwen3_vl_vision_bilinear_indices_and_weights",
      "op": "qwen3_vl_vision_bilinear_indices_and_weights",
      "backend": "trt",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "Qwen3VisionEncoder",
        "line": 141
      },
      "metadata": {
        "xlang_op_kind": "unary_op",
        "extractor": "static_xmodel",
        "expression": "bilinear = grid_thw * T.unary_op("
      }
    },
    {
      "id": "op.qwen3visionencoder.qwen3_vl_vision_position_ids.146",
      "kind": "op",
      "label": "qwen3_vl_vision_position_ids",
      "op": "qwen3_vl_vision_position_ids",
      "backend": "unknown",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "Qwen3VisionEncoder",
        "line": 146
      },
      "metadata": {
        "xlang_op_kind": "unary_op",
        "extractor": "static_xmodel",
        "expression": "vision_position_ids = grid_thw * T.unary_op("
      }
    },
    {
      "id": "op.qwen3visionencoder.qwen3_vl_vision_cu_seqlens.150",
      "kind": "op",
      "label": "qwen3_vl_vision_cu_seqlens",
      "op": "qwen3_vl_vision_cu_seqlens",
      "backend": "unknown",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "Qwen3VisionEncoder",
        "line": 150
      },
      "metadata": {
        "xlang_op_kind": "unary_op",
        "extractor": "static_xmodel",
        "expression": "cu_seqlens = grid_thw * T.unary_op(\"qwen3_vl_vision_cu_seqlens\")"
      }
    },
    {
      "id": "op.qwen3visionencoder.qwen3_vl_patch_embed_conv3d.154",
      "kind": "op",
      "label": "qwen3_vl_patch_embed_conv3d",
      "op": "qwen3_vl_patch_embed_conv3d",
      "backend": "trt",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "Qwen3VisionEncoder",
        "line": 154
      },
      "metadata": {
        "xlang_op_kind": "unary_op",
        "extractor": "static_xmodel",
        "expression": "x = pixel_values * T.unary_op("
      }
    },
    {
      "id": "matrix.visual_patch_embed_proj_weight",
      "kind": "matrix",
      "label": "visual.patch_embed.proj.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[1024, 3, 2, 16, 16]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "Qwen3VisionEncoder",
        "line": 156
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "patch_embed_kernel",
        "dimensions": [
          1024,
          3,
          2,
          16,
          16
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 1572864,
        "bytes_fp16": 3145728,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"visual.patch_embed.proj.weight\"]"
      }
    },
    {
      "id": "matrix.visual_patch_embed_proj_bias",
      "kind": "matrix",
      "label": "visual.patch_embed.proj.bias",
      "status": "defined",
      "shape": {
        "outputs": [
          "[1024]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "Qwen3VisionEncoder",
        "line": 157
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "patch_embed_kernel",
        "dimensions": [
          1024
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 1024,
        "bytes_fp16": 2048,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"visual.patch_embed.proj.bias\"]"
      }
    },
    {
      "id": "op.qwen3visionencoder.qwen3_vl_pos_embed_interpolate.164",
      "kind": "op",
      "label": "qwen3_vl_pos_embed_interpolate",
      "op": "qwen3_vl_pos_embed_interpolate",
      "backend": "unknown",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "Qwen3VisionEncoder",
        "line": 164
      },
      "metadata": {
        "xlang_op_kind": "binary_op",
        "extractor": "static_xmodel",
        "expression": "pos_embeds = weights[\"visual.pos_embed.weight\"] * T.binary_op("
      }
    },
    {
      "id": "matrix.visual_pos_embed_weight",
      "kind": "matrix",
      "label": "visual.pos_embed.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[2304, 1024]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "Qwen3VisionEncoder",
        "line": 164
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "matrix",
        "dimensions": [
          2304,
          1024
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 2359296,
        "bytes_fp16": 4718592,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"visual.pos_embed.weight\"]"
      }
    },
    {
      "id": "op.qwen3visionencoder.qwen3_vl_vision_rotary_embedding.170",
      "kind": "op",
      "label": "qwen3_vl_vision_rotary_embedding",
      "op": "qwen3_vl_vision_rotary_embedding",
      "backend": "cuda",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vision_encoder.x",
        "function": "Qwen3VisionEncoder",
        "line": 170
      },
      "metadata": {
        "xlang_op_kind": "unary_op",
        "extractor": "static_xmodel",
        "expression": "position_embeddings = vision_position_ids * T.unary_op("
      }
    },
    {
      "id": "block.Qwen3VisionPositionIds",
      "kind": "block",
      "label": "Qwen3VisionPositionIds",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vl_adapter.x",
        "function": "Qwen3VisionPositionIds",
        "line": 12
      },
      "metadata": {
        "extractor": "static_xmodel",
        "expression": "def Qwen3VisionPositionIds(start_position, grid_thw, config, time_interval=1):\n    return grid_thw * T.unary_op(\n        \"qwen3_vl_llm_vision_position_ids\",\n        start_position=start_position,\n        temp_merge_size=1,\n        spatial_merge_size=config.vision_config.spatial_merge_size,\n        time_interval=time_interval\n    )\n\n"
      }
    },
    {
      "id": "op.qwen3visionpositionids.qwen3_vl_llm_vision_position_ids.13",
      "kind": "op",
      "label": "qwen3_vl_llm_vision_position_ids",
      "op": "qwen3_vl_llm_vision_position_ids",
      "backend": "unknown",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vl_adapter.x",
        "function": "Qwen3VisionPositionIds",
        "line": 13
      },
      "metadata": {
        "xlang_op_kind": "unary_op",
        "extractor": "static_xmodel",
        "expression": "return grid_thw * T.unary_op("
      }
    },
    {
      "id": "block.Qwen3GetRopeIndex",
      "kind": "block",
      "label": "Qwen3GetRopeIndex",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vl_adapter.x",
        "function": "Qwen3GetRopeIndex",
        "line": 22
      },
      "metadata": {
        "extractor": "static_xmodel",
        "expression": "def Qwen3GetRopeIndex(input_ids, mm_token_type_ids, image_grid_thw, video_grid_thw, attention_mask, config):\n    # Produces multimodal position_ids and mrope_position_deltas.\n    # position_ids shape follows HF: (3, batch, sequence), while the text model\n    # internally expands/handles the text position channel as well.\n    return input_ids * T.unary_op(\n        \"qwen3_vl_get_rope_index\",\n        mm_token_type_ids=mm_token_type_ids,\n        image_grid_thw=image_grid_thw,\n        video_grid_thw=video_grid_thw,\n        attention_mask=attention_mask,\n        spatial_merge_size=config.vision_config.spatial_merge_size\n    )\n\n"
      }
    },
    {
      "id": "op.qwen3getropeindex.qwen3_vl_get_rope_index.26",
      "kind": "op",
      "label": "qwen3_vl_get_rope_index",
      "op": "qwen3_vl_get_rope_index",
      "backend": "cuda",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vl_adapter.x",
        "function": "Qwen3GetRopeIndex",
        "line": 26
      },
      "metadata": {
        "xlang_op_kind": "unary_op",
        "extractor": "static_xmodel",
        "expression": "return input_ids * T.unary_op("
      }
    },
    {
      "id": "block.Qwen3MergeVisualEmbeddings",
      "kind": "block",
      "label": "Qwen3MergeVisualEmbeddings",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vl_adapter.x",
        "function": "Qwen3MergeVisualEmbeddings",
        "line": 36
      },
      "metadata": {
        "extractor": "static_xmodel",
        "expression": "def Qwen3MergeVisualEmbeddings(input_ids, text_embeddings, visual_embeds, image_token_id, video_token_id):\n    # Replace positions corresponding to image/video placeholder tokens with\n    # merged visual embeddings. Also returns a visual_pos_mask for DeepStack.\n    return text_embeddings * T.binary_op(\n        \"qwen3_vl_merge_visual_embeddings\",\n        input_ids=input_ids,\n        image_token_id=image_token_id,\n        video_token_id=video_token_id\n    ) * visual_embeds\n\n"
      }
    },
    {
      "id": "op.qwen3mergevisualembeddings.qwen3_vl_merge_visual_embeddings.39",
      "kind": "op",
      "label": "qwen3_vl_merge_visual_embeddings",
      "op": "qwen3_vl_merge_visual_embeddings",
      "backend": "cuda",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vl_adapter.x",
        "function": "Qwen3MergeVisualEmbeddings",
        "line": 39
      },
      "metadata": {
        "xlang_op_kind": "binary_op",
        "extractor": "static_xmodel",
        "expression": "return text_embeddings * T.binary_op("
      }
    },
    {
      "id": "block.Qwen3PrepareInputsEmbeds",
      "kind": "block",
      "label": "Qwen3PrepareInputsEmbeds",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vl_adapter.x",
        "function": "Qwen3PrepareInputsEmbeds",
        "line": 47
      },
      "metadata": {
        "extractor": "static_xmodel",
        "expression": "def Qwen3PrepareInputsEmbeds(input_ids, visual_outputs, weights, config):\n    text_embeddings = input_ids * T.binary_op(\"embedding\") * weights[\"language_model.embed_tokens.weight\"]\n    merged = Qwen3MergeVisualEmbeddings(\n        input_ids,\n        text_embeddings,\n        visual_outputs[\"pooler_output\"],\n        image_token_id=config.image_token_id,\n        video_token_id=config.video_token_id\n    )\n    return {\n        \"inputs_embeds\": merged[\"inputs_embeds\"],\n        \"visual_pos_masks\": merged[\"visual_pos_masks\"],\n        \"deepstack_visual_embeds\": visual_outputs[\"deepstack_features\"]\n    }\n"
      }
    },
    {
      "id": "op.qwen3prepareinputsembeds.embedding.48",
      "kind": "op",
      "label": "embedding",
      "op": "embedding",
      "backend": "cuda",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/vl_adapter.x",
        "function": "Qwen3PrepareInputsEmbeds",
        "line": 48
      },
      "metadata": {
        "xlang_op_kind": "binary_op",
        "extractor": "static_xmodel",
        "expression": "text_embeddings = input_ids * T.binary_op(\"embedding\") * weights[\"language_model.embed_tokens.weight\"]"
      }
    },
    {
      "id": "block.rms_norm",
      "kind": "block",
      "label": "rms_norm",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "rms_norm",
        "line": 20
      },
      "metadata": {
        "extractor": "static_xmodel",
        "expression": "def rms_norm(x, weight, eps=1e-6):\n    return x * T.unary_op(\"rms_norm\", weight=weight, eps=eps)\n\n"
      }
    },
    {
      "id": "op.rms_norm.rms_norm.21",
      "kind": "op",
      "label": "rms_norm",
      "op": "rms_norm",
      "backend": "cuda",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "rms_norm",
        "line": 21
      },
      "metadata": {
        "xlang_op_kind": "unary_op",
        "extractor": "static_xmodel",
        "expression": "return x * T.unary_op(\"rms_norm\", weight=weight, eps=eps)"
      }
    },
    {
      "id": "block.Qwen3TextRotaryEmbedding",
      "kind": "block",
      "label": "Qwen3TextRotaryEmbedding",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextRotaryEmbedding",
        "line": 24
      },
      "metadata": {
        "extractor": "static_xmodel",
        "expression": "def Qwen3TextRotaryEmbedding(hidden_states, position_ids, config):\n    # Qwen3-VL text position_ids are 4-way in the full model:\n    #   text_position_ids + 3D multimodal ids (temporal, height, width).\n    # The text rotary module consumes the 3D ids and applies interleaved MRoPE.\n    return position_ids * T.unary_op(\n        \"qwen3_vl_text_interleaved_mrope\",\n        rope_theta=config.text_config.rope_theta,\n        head_dim=config.text_config.head_dim,\n        mrope_section=config.text_config.rope_scaling.mrope_section\n    )\n\n"
      }
    },
    {
      "id": "op.qwen3textrotaryembedding.qwen3_vl_text_interleaved_mrope.28",
      "kind": "op",
      "label": "qwen3_vl_text_interleaved_mrope",
      "op": "qwen3_vl_text_interleaved_mrope",
      "backend": "cuda",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextRotaryEmbedding",
        "line": 28
      },
      "metadata": {
        "xlang_op_kind": "unary_op",
        "extractor": "static_xmodel",
        "expression": "return position_ids * T.unary_op("
      }
    },
    {
      "id": "block.Qwen3TextAttention",
      "kind": "block",
      "label": "Qwen3TextAttention",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextAttention",
        "line": 36
      },
      "metadata": {
        "extractor": "static_xmodel",
        "expression": "def Qwen3TextAttention(x, position_embeddings, attention_mask, past_key_values, weights, config, layer_idx, use_cache=True):\n    prefix = \"language_model.layers.\" + str(layer_idx) + \".self_attn\"\n    head_dim = config.text_config.head_dim\n    num_heads = config.text_config.num_attention_heads\n    num_kv_heads = config.text_config.num_key_value_heads\n\n    q = linear(x, weights[prefix + \".q_proj.weight\"], None, op=\"q_proj\")\n    k = linear(x, weights[prefix + \".k_proj.weight\"], None, op=\"k_proj\")\n    v = linear(x, weights[prefix + \".v_proj.weight\"], None, op=\"v_proj\")\n\n    q = q * T.unary_op(\"reshape_q_heads\", num_heads=num_heads, head_dim=head_dim)\n    k = k * T.unary_op(\"reshape_kv_heads\", num_heads=num_kv_heads, head_dim=head_dim)\n    v = v * T.unary_op(\"reshape_kv_heads\", num_heads=num_kv_heads, head_dim=head_dim)\n\n    # Qwen3-VL normalizes Q and K per head_dim before RoPE.\n    q = rms_norm(q, weights[prefix + \".q_norm.weight\"], eps=config.text_config.rms_norm_eps)\n    k = rms_norm(k, weights[prefix + \".k_norm.weight\"], eps=config.text_config.rms_norm_eps)\n\n    q, k = q * T.binary_op(\"qwen3_vl_apply_text_rope\", position_embeddings=position_embeddings) * k\n\n    if use_cache:\n        k, v = k * T.binary_op(\n            \"paged_kv_update\",\n            past_key_values=past_key_values,\n            layer_idx=layer_idx\n        ) * v\n\n    attn = q * T.binary_op(\n        \"paged_attention\",\n        attention_mask=attention_mask,\n        past_key_values=past_key_values,\n        layer_idx=layer_idx,\n        num_heads=num_heads,\n        num_key_value_heads=num_kv_heads,\n        scale=head_dim ** -0.5,\n        causal=True\n    ) * v\n    attn = attn * T.unary_op(\"merge_attention_heads\")\n    attn = linear(attn, weights[prefix + \".o_proj.weight\"], None, op=\"o_proj\")\n    return attn\n\n"
      }
    },
    {
      "id": "matrix.qwen3textattention_q_proj_weight",
      "kind": "matrix",
      "label": "Qwen3TextAttention.q_proj.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[2048, 2048]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextAttention",
        "line": 42
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "attention_matrix",
        "dimensions": [
          2048,
          2048
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 4194304,
        "bytes_fp16": 8388608,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"Qwen3TextAttention.q_proj.weight\"]"
      }
    },
    {
      "id": "matrix.qwen3textattention_k_proj_weight",
      "kind": "matrix",
      "label": "Qwen3TextAttention.k_proj.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[1024, 2048]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextAttention",
        "line": 43
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "attention_matrix",
        "dimensions": [
          1024,
          2048
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 2097152,
        "bytes_fp16": 4194304,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"Qwen3TextAttention.k_proj.weight\"]"
      }
    },
    {
      "id": "matrix.qwen3textattention_v_proj_weight",
      "kind": "matrix",
      "label": "Qwen3TextAttention.v_proj.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[1024, 2048]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextAttention",
        "line": 44
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "attention_matrix",
        "dimensions": [
          1024,
          2048
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 2097152,
        "bytes_fp16": 4194304,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"Qwen3TextAttention.v_proj.weight\"]"
      }
    },
    {
      "id": "op.qwen3textattention.reshape_q_heads.46",
      "kind": "op",
      "label": "reshape_q_heads",
      "op": "reshape_q_heads",
      "backend": "unknown",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextAttention",
        "line": 46
      },
      "metadata": {
        "xlang_op_kind": "unary_op",
        "extractor": "static_xmodel",
        "expression": "q = q * T.unary_op(\"reshape_q_heads\", num_heads=num_heads, head_dim=head_dim)"
      }
    },
    {
      "id": "op.qwen3textattention.reshape_kv_heads.47",
      "kind": "op",
      "label": "reshape_kv_heads",
      "op": "reshape_kv_heads",
      "backend": "unknown",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextAttention",
        "line": 47
      },
      "metadata": {
        "xlang_op_kind": "unary_op",
        "extractor": "static_xmodel",
        "expression": "k = k * T.unary_op(\"reshape_kv_heads\", num_heads=num_kv_heads, head_dim=head_dim)"
      }
    },
    {
      "id": "op.qwen3textattention.reshape_kv_heads.48",
      "kind": "op",
      "label": "reshape_kv_heads",
      "op": "reshape_kv_heads",
      "backend": "unknown",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextAttention",
        "line": 48
      },
      "metadata": {
        "xlang_op_kind": "unary_op",
        "extractor": "static_xmodel",
        "expression": "v = v * T.unary_op(\"reshape_kv_heads\", num_heads=num_kv_heads, head_dim=head_dim)"
      }
    },
    {
      "id": "matrix.qwen3textattention_q_norm_weight",
      "kind": "matrix",
      "label": "Qwen3TextAttention.q_norm.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[128]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextAttention",
        "line": 51
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "normalization_vector",
        "dimensions": [
          128
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 128,
        "bytes_fp16": 256,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"Qwen3TextAttention.q_norm.weight\"]"
      }
    },
    {
      "id": "matrix.qwen3textattention_k_norm_weight",
      "kind": "matrix",
      "label": "Qwen3TextAttention.k_norm.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[128]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextAttention",
        "line": 52
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "normalization_vector",
        "dimensions": [
          128
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 128,
        "bytes_fp16": 256,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"Qwen3TextAttention.k_norm.weight\"]"
      }
    },
    {
      "id": "op.qwen3textattention.qwen3_vl_apply_text_rope.54",
      "kind": "op",
      "label": "qwen3_vl_apply_text_rope",
      "op": "qwen3_vl_apply_text_rope",
      "backend": "cuda",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextAttention",
        "line": 54
      },
      "metadata": {
        "xlang_op_kind": "binary_op",
        "extractor": "static_xmodel",
        "expression": "q, k = q * T.binary_op(\"qwen3_vl_apply_text_rope\", position_embeddings=position_embeddings) * k"
      }
    },
    {
      "id": "op.qwen3textattention.paged_kv_update.57",
      "kind": "op",
      "label": "paged_kv_update",
      "op": "paged_kv_update",
      "backend": "cuda",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextAttention",
        "line": 57
      },
      "metadata": {
        "xlang_op_kind": "binary_op",
        "extractor": "static_xmodel",
        "expression": "k, v = k * T.binary_op("
      }
    },
    {
      "id": "op.qwen3textattention.paged_attention.63",
      "kind": "op",
      "label": "paged_attention",
      "op": "paged_attention",
      "backend": "cuda",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextAttention",
        "line": 63
      },
      "metadata": {
        "xlang_op_kind": "binary_op",
        "extractor": "static_xmodel",
        "expression": "attn = q * T.binary_op("
      }
    },
    {
      "id": "op.qwen3textattention.merge_attention_heads.73",
      "kind": "op",
      "label": "merge_attention_heads",
      "op": "merge_attention_heads",
      "backend": "cuda",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextAttention",
        "line": 73
      },
      "metadata": {
        "xlang_op_kind": "unary_op",
        "extractor": "static_xmodel",
        "expression": "attn = attn * T.unary_op(\"merge_attention_heads\")"
      }
    },
    {
      "id": "matrix.qwen3textattention_o_proj_weight",
      "kind": "matrix",
      "label": "Qwen3TextAttention.o_proj.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[2048, 2048]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextAttention",
        "line": 74
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "attention_matrix",
        "dimensions": [
          2048,
          2048
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 4194304,
        "bytes_fp16": 8388608,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"Qwen3TextAttention.o_proj.weight\"]"
      }
    },
    {
      "id": "block.Qwen3TextMLP",
      "kind": "block",
      "label": "Qwen3TextMLP",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextMLP",
        "line": 78
      },
      "metadata": {
        "extractor": "static_xmodel",
        "expression": "def Qwen3TextMLP(x, weights, config, layer_idx):\n    prefix = \"language_model.layers.\" + str(layer_idx) + \".mlp\"\n    gate = linear(x, weights[prefix + \".gate_proj.weight\"], None, op=\"gate_proj\")\n    up = linear(x, weights[prefix + \".up_proj.weight\"], None, op=\"up_proj\")\n    hidden = gate * T.unary_op(config.text_config.hidden_act) * up\n    return linear(hidden, weights[prefix + \".down_proj.weight\"], None, op=\"down_proj\")\n\n"
      }
    },
    {
      "id": "matrix.qwen3textmlp_gate_proj_weight",
      "kind": "matrix",
      "label": "Qwen3TextMLP.gate_proj.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[6144, 2048]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextMLP",
        "line": 80
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "mlp_matrix",
        "dimensions": [
          6144,
          2048
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 12582912,
        "bytes_fp16": 25165824,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"Qwen3TextMLP.gate_proj.weight\"]"
      }
    },
    {
      "id": "matrix.qwen3textmlp_up_proj_weight",
      "kind": "matrix",
      "label": "Qwen3TextMLP.up_proj.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[6144, 2048]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextMLP",
        "line": 81
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "mlp_matrix",
        "dimensions": [
          6144,
          2048
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 12582912,
        "bytes_fp16": 25165824,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"Qwen3TextMLP.up_proj.weight\"]"
      }
    },
    {
      "id": "matrix.qwen3textmlp_down_proj_weight",
      "kind": "matrix",
      "label": "Qwen3TextMLP.down_proj.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[2048, 6144]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextMLP",
        "line": 83
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "mlp_matrix",
        "dimensions": [
          2048,
          6144
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 12582912,
        "bytes_fp16": 25165824,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"Qwen3TextMLP.down_proj.weight\"]"
      }
    },
    {
      "id": "block.Qwen3TextDecoderLayer",
      "kind": "block",
      "label": "Qwen3TextDecoderLayer",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextDecoderLayer",
        "line": 86
      },
      "metadata": {
        "extractor": "static_xmodel",
        "expression": "def Qwen3TextDecoderLayer(x, position_embeddings, attention_mask, past_key_values, weights, config, layer_idx, use_cache=True):\n    prefix = \"language_model.layers.\" + str(layer_idx)\n\n    residual = x\n    x = rms_norm(\n        x,\n        weights[prefix + \".input_layernorm.weight\"],\n        eps=config.text_config.rms_norm_eps\n    )\n    x = Qwen3TextAttention(\n        x,\n        position_embeddings,\n        attention_mask,\n        past_key_values,\n        weights,\n        config,\n        layer_idx,\n        use_cache=use_cache\n    )\n    x = residual + x\n\n    residual = x\n    x = rms_norm(\n        x,\n        weights[prefix + \".post_attention_layernorm.weight\"],\n        eps=config.text_config.rms_norm_eps\n    )\n    x = Qwen3TextMLP(x, weights, config, layer_idx)\n    x = residual + x\n    return x\n\n"
      }
    },
    {
      "id": "matrix.qwen3textdecoderlayer_input_layernorm_weight",
      "kind": "matrix",
      "label": "Qwen3TextDecoderLayer.input_layernorm.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[2048]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextDecoderLayer",
        "line": 92
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "normalization_vector",
        "dimensions": [
          2048
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 2048,
        "bytes_fp16": 4096,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"Qwen3TextDecoderLayer.input_layernorm.weight\"]"
      }
    },
    {
      "id": "matrix.qwen3textdecoderlayer_post_attention_layernorm_weight",
      "kind": "matrix",
      "label": "Qwen3TextDecoderLayer.post_attention_layernorm.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[2048]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextDecoderLayer",
        "line": 110
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "normalization_vector",
        "dimensions": [
          2048
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 2048,
        "bytes_fp16": 4096,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"Qwen3TextDecoderLayer.post_attention_layernorm.weight\"]"
      }
    },
    {
      "id": "block.Qwen3TextModel",
      "kind": "block",
      "label": "Qwen3TextModel",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextModel",
        "line": 119
      },
      "metadata": {
        "extractor": "static_xmodel",
        "expression": "def Qwen3TextModel(\n    input_ids,\n    inputs_embeds,\n    position_ids,\n    attention_mask,\n    visual_pos_masks,\n    deepstack_visual_embeds,\n    past_key_values,\n    weights,\n    config,\n    use_cache=True"
      }
    },
    {
      "id": "op.qwen3textmodel.embedding.132",
      "kind": "op",
      "label": "embedding",
      "op": "embedding",
      "backend": "cuda",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextModel",
        "line": 132
      },
      "metadata": {
        "xlang_op_kind": "binary_op",
        "extractor": "static_xmodel",
        "expression": "x = input_ids * T.binary_op(\"embedding\") * weights[\"language_model.embed_tokens.weight\"]"
      }
    },
    {
      "id": "op.qwen3textmodel.create_causal_mask.136",
      "kind": "op",
      "label": "create_causal_mask",
      "op": "create_causal_mask",
      "backend": "unknown",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextModel",
        "line": 136
      },
      "metadata": {
        "xlang_op_kind": "unary_op",
        "extractor": "static_xmodel",
        "expression": "causal_mask = x * T.unary_op("
      }
    },
    {
      "id": "op.qwen3textmodel.qwen3_vl_deepstack_add.158",
      "kind": "op",
      "label": "qwen3_vl_deepstack_add",
      "op": "qwen3_vl_deepstack_add",
      "backend": "unknown",
      "status": "defined",
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextModel",
        "line": 158
      },
      "metadata": {
        "xlang_op_kind": "binary_op",
        "extractor": "static_xmodel",
        "expression": "x = x * T.binary_op("
      }
    },
    {
      "id": "matrix.language_model_norm_weight",
      "kind": "matrix",
      "label": "language_model.norm.weight",
      "status": "defined",
      "shape": {
        "outputs": [
          "[2048]"
        ]
      },
      "source": {
        "file": "qwen_vl/xmodel/qwen_llm.x",
        "function": "Qwen3TextModel",
        "line": 163
      },
      "metadata": {
        "extractor": "static_xmodel",
        "role": "normalization_vector",
        "dimensions": [
          2048
        ],
        "dtype": "fp16/bf16",
        "parameter_count": 2048,
        "bytes_fp16": 4096,
        "dimension_source": "qwen3_vl_2b_config_inferred",
        "expression": "weights[\"language_model.norm.weight\"]"
      }
    }
  ],
  "edges": [
    {
      "from": "block.Qwen3VLModel",
      "to": "op.qwen3vlmodel.lm_head.71",
      "label": "binary_op"
    },
    {
      "from": "matrix.language_model_embed_tokens_weight",
      "to": "block.Qwen3VLModel",
      "label": "weight"
    },
    {
      "from": "block.Qwen3VLModel",
      "to": "block.linear",
      "label": "next block"
    },
    {
      "from": "block.linear",
      "to": "block.layer_norm",
      "label": "next block"
    },
    {
      "from": "block.layer_norm",
      "to": "op.layer_norm.layer_norm.21",
      "label": "unary_op"
    },
    {
      "from": "block.layer_norm",
      "to": "block.gelu",
      "label": "next block"
    },
    {
      "from": "block.gelu",
      "to": "op.gelu.gelu_pytorch_tanh.25",
      "label": "unary_op"
    },
    {
      "from": "block.gelu",
      "to": "block.VisionPatchMerger",
      "label": "next block"
    },
    {
      "from": "block.VisionPatchMerger",
      "to": "op.visionpatchmerger.qwen3_vl_patch_merger_shuffle.32",
      "label": "unary_op"
    },
    {
      "from": "matrix.visionpatchmerger_norm_weight",
      "to": "block.VisionPatchMerger",
      "label": "weight"
    },
    {
      "from": "matrix.visionpatchmerger_norm_bias",
      "to": "block.VisionPatchMerger",
      "label": "weight"
    },
    {
      "from": "matrix.visionpatchmerger_linear_fc1_weight",
      "to": "block.VisionPatchMerger",
      "label": "weight"
    },
    {
      "from": "matrix.visionpatchmerger_linear_fc1_bias",
      "to": "block.VisionPatchMerger",
      "label": "weight"
    },
    {
      "from": "matrix.visionpatchmerger_linear_fc2_weight",
      "to": "block.VisionPatchMerger",
      "label": "weight"
    },
    {
      "from": "matrix.visionpatchmerger_linear_fc2_bias",
      "to": "block.VisionPatchMerger",
      "label": "weight"
    },
    {
      "from": "block.VisionPatchMerger",
      "to": "block.VisionMLP",
      "label": "next block"
    },
    {
      "from": "matrix.visionmlp_linear_fc1_weight",
      "to": "block.VisionMLP",
      "label": "weight"
    },
    {
      "from": "matrix.visionmlp_linear_fc1_bias",
      "to": "block.VisionMLP",
      "label": "weight"
    },
    {
      "from": "matrix.visionmlp_linear_fc2_weight",
      "to": "block.VisionMLP",
      "label": "weight"
    },
    {
      "from": "matrix.visionmlp_linear_fc2_bias",
      "to": "block.VisionMLP",
      "label": "weight"
    },
    {
      "from": "block.VisionMLP",
      "to": "block.VisionAttention",
      "label": "next block"
    },
    {
      "from": "matrix.visionattention_qkv_weight",
      "to": "block.VisionAttention",
      "label": "weight"
    },
    {
      "from": "matrix.visionattention_qkv_bias",
      "to": "block.VisionAttention",
      "label": "weight"
    },
    {
      "from": "block.VisionAttention",
      "to": "op.visionattention.qwen3_vl_split_vision_qkv.81",
      "label": "unary_op"
    },
    {
      "from": "block.VisionAttention",
      "to": "op.visionattention.qwen3_vl_apply_vision_rope.86",
      "label": "binary_op"
    },
    {
      "from": "block.VisionAttention",
      "to": "op.visionattention.vision_varlen_attention.87",
      "label": "binary_op"
    },
    {
      "from": "block.VisionAttention",
      "to": "op.visionattention.merge_attention_heads.94",
      "label": "unary_op"
    },
    {
      "from": "matrix.visionattention_proj_weight",
      "to": "block.VisionAttention",
      "label": "weight"
    },
    {
      "from": "matrix.visionattention_proj_bias",
      "to": "block.VisionAttention",
      "label": "weight"
    },
    {
      "from": "block.VisionAttention",
      "to": "block.VisionBlock",
      "label": "next block"
    },
    {
      "from": "matrix.visionblock_norm1_weight",
      "to": "block.VisionBlock",
      "label": "weight"
    },
    {
      "from": "matrix.visionblock_norm1_bias",
      "to": "block.VisionBlock",
      "label": "weight"
    },
    {
      "from": "matrix.visionblock_norm2_weight",
      "to": "block.VisionBlock",
      "label": "weight"
    },
    {
      "from": "matrix.visionblock_norm2_bias",
      "to": "block.VisionBlock",
      "label": "weight"
    },
    {
      "from": "block.VisionBlock",
      "to": "block.Qwen3VisionEncoder",
      "label": "next block"
    },
    {
      "from": "block.Qwen3VisionEncoder",
      "to": "op.qwen3visionencoder.qwen3_vl_vision_bilinear_indices_and_weights.141",
      "label": "unary_op"
    },
    {
      "from": "block.Qwen3VisionEncoder",
      "to": "op.qwen3visionencoder.qwen3_vl_vision_position_ids.146",
      "label": "unary_op"
    },
    {
      "from": "block.Qwen3VisionEncoder",
      "to": "op.qwen3visionencoder.qwen3_vl_vision_cu_seqlens.150",
      "label": "unary_op"
    },
    {
      "from": "block.Qwen3VisionEncoder",
      "to": "op.qwen3visionencoder.qwen3_vl_patch_embed_conv3d.154",
      "label": "unary_op"
    },
    {
      "from": "matrix.visual_patch_embed_proj_weight",
      "to": "block.Qwen3VisionEncoder",
      "label": "weight"
    },
    {
      "from": "matrix.visual_patch_embed_proj_bias",
      "to": "block.Qwen3VisionEncoder",
      "label": "weight"
    },
    {
      "from": "block.Qwen3VisionEncoder",
      "to": "op.qwen3visionencoder.qwen3_vl_pos_embed_interpolate.164",
      "label": "binary_op"
    },
    {
      "from": "matrix.visual_pos_embed_weight",
      "to": "op.qwen3visionencoder.qwen3_vl_pos_embed_interpolate.164",
      "label": "weight"
    },
    {
      "from": "block.Qwen3VisionEncoder",
      "to": "op.qwen3visionencoder.qwen3_vl_vision_rotary_embedding.170",
      "label": "unary_op"
    },
    {
      "from": "block.Qwen3VisionEncoder",
      "to": "block.Qwen3VisionPositionIds",
      "label": "next block"
    },
    {
      "from": "block.Qwen3VisionPositionIds",
      "to": "op.qwen3visionpositionids.qwen3_vl_llm_vision_position_ids.13",
      "label": "unary_op"
    },
    {
      "from": "block.Qwen3VisionPositionIds",
      "to": "block.Qwen3GetRopeIndex",
      "label": "next block"
    },
    {
      "from": "block.Qwen3GetRopeIndex",
      "to": "op.qwen3getropeindex.qwen3_vl_get_rope_index.26",
      "label": "unary_op"
    },
    {
      "from": "block.Qwen3GetRopeIndex",
      "to": "block.Qwen3MergeVisualEmbeddings",
      "label": "next block"
    },
    {
      "from": "block.Qwen3MergeVisualEmbeddings",
      "to": "op.qwen3mergevisualembeddings.qwen3_vl_merge_visual_embeddings.39",
      "label": "binary_op"
    },
    {
      "from": "block.Qwen3MergeVisualEmbeddings",
      "to": "block.Qwen3PrepareInputsEmbeds",
      "label": "next block"
    },
    {
      "from": "block.Qwen3PrepareInputsEmbeds",
      "to": "op.qwen3prepareinputsembeds.embedding.48",
      "label": "binary_op"
    },
    {
      "from": "matrix.language_model_embed_tokens_weight",
      "to": "op.qwen3prepareinputsembeds.embedding.48",
      "label": "weight"
    },
    {
      "from": "block.Qwen3PrepareInputsEmbeds",
      "to": "block.linear",
      "label": "next block"
    },
    {
      "from": "block.linear",
      "to": "block.rms_norm",
      "label": "next block"
    },
    {
      "from": "block.rms_norm",
      "to": "op.rms_norm.rms_norm.21",
      "label": "unary_op"
    },
    {
      "from": "block.rms_norm",
      "to": "block.Qwen3TextRotaryEmbedding",
      "label": "next block"
    },
    {
      "from": "block.Qwen3TextRotaryEmbedding",
      "to": "op.qwen3textrotaryembedding.qwen3_vl_text_interleaved_mrope.28",
      "label": "unary_op"
    },
    {
      "from": "block.Qwen3TextRotaryEmbedding",
      "to": "block.Qwen3TextAttention",
      "label": "next block"
    },
    {
      "from": "matrix.qwen3textattention_q_proj_weight",
      "to": "block.Qwen3TextAttention",
      "label": "weight"
    },
    {
      "from": "matrix.qwen3textattention_k_proj_weight",
      "to": "block.Qwen3TextAttention",
      "label": "weight"
    },
    {
      "from": "matrix.qwen3textattention_v_proj_weight",
      "to": "block.Qwen3TextAttention",
      "label": "weight"
    },
    {
      "from": "block.Qwen3TextAttention",
      "to": "op.qwen3textattention.reshape_q_heads.46",
      "label": "unary_op"
    },
    {
      "from": "block.Qwen3TextAttention",
      "to": "op.qwen3textattention.reshape_kv_heads.47",
      "label": "unary_op"
    },
    {
      "from": "block.Qwen3TextAttention",
      "to": "op.qwen3textattention.reshape_kv_heads.48",
      "label": "unary_op"
    },
    {
      "from": "matrix.qwen3textattention_q_norm_weight",
      "to": "block.Qwen3TextAttention",
      "label": "weight"
    },
    {
      "from": "matrix.qwen3textattention_k_norm_weight",
      "to": "block.Qwen3TextAttention",
      "label": "weight"
    },
    {
      "from": "block.Qwen3TextAttention",
      "to": "op.qwen3textattention.qwen3_vl_apply_text_rope.54",
      "label": "binary_op"
    },
    {
      "from": "block.Qwen3TextAttention",
      "to": "op.qwen3textattention.paged_kv_update.57",
      "label": "binary_op"
    },
    {
      "from": "block.Qwen3TextAttention",
      "to": "op.qwen3textattention.paged_attention.63",
      "label": "binary_op"
    },
    {
      "from": "block.Qwen3TextAttention",
      "to": "op.qwen3textattention.merge_attention_heads.73",
      "label": "unary_op"
    },
    {
      "from": "matrix.qwen3textattention_o_proj_weight",
      "to": "block.Qwen3TextAttention",
      "label": "weight"
    },
    {
      "from": "block.Qwen3TextAttention",
      "to": "block.Qwen3TextMLP",
      "label": "next block"
    },
    {
      "from": "matrix.qwen3textmlp_gate_proj_weight",
      "to": "block.Qwen3TextMLP",
      "label": "weight"
    },
    {
      "from": "matrix.qwen3textmlp_up_proj_weight",
      "to": "block.Qwen3TextMLP",
      "label": "weight"
    },
    {
      "from": "matrix.qwen3textmlp_down_proj_weight",
      "to": "block.Qwen3TextMLP",
      "label": "weight"
    },
    {
      "from": "block.Qwen3TextMLP",
      "to": "block.Qwen3TextDecoderLayer",
      "label": "next block"
    },
    {
      "from": "matrix.qwen3textdecoderlayer_input_layernorm_weight",
      "to": "block.Qwen3TextDecoderLayer",
      "label": "weight"
    },
    {
      "from": "matrix.qwen3textdecoderlayer_post_attention_layernorm_weight",
      "to": "block.Qwen3TextDecoderLayer",
      "label": "weight"
    },
    {
      "from": "block.Qwen3TextDecoderLayer",
      "to": "block.Qwen3TextModel",
      "label": "next block"
    },
    {
      "from": "block.Qwen3TextModel",
      "to": "op.qwen3textmodel.embedding.132",
      "label": "binary_op"
    },
    {
      "from": "matrix.language_model_embed_tokens_weight",
      "to": "op.qwen3textmodel.embedding.132",
      "label": "weight"
    },
    {
      "from": "block.Qwen3TextModel",
      "to": "op.qwen3textmodel.create_causal_mask.136",
      "label": "unary_op"
    },
    {
      "from": "block.Qwen3TextModel",
      "to": "op.qwen3textmodel.qwen3_vl_deepstack_add.158",
      "label": "binary_op"
    },
    {
      "from": "matrix.language_model_norm_weight",
      "to": "block.Qwen3TextModel",
      "label": "weight"
    }
  ]
};
