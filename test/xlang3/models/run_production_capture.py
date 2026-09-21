# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

"""Capture migrated Qwen Python programs without loading pretrained weights."""
import importlib
import os
import sys

import garnet
import tensor


ROOTS = [
    ("text_1_7b.prefill", "Qwen3Prefill"),
    ("text_1_7b.decode", "Qwen3Decode"),
    ("text_1_7b.decode_batch", "Qwen3DecodeBatch"),
    ("vl_2b_instruct.qwen_vl_model", "Qwen3VLModel"),
    ("vl_2b_instruct.qwen_vl_prefill", "Qwen3VLPrefill"),
    ("vl_2b_instruct.qwen_text_prefill", "Qwen3TextPrefill"),
    ("vl_2b_instruct.qwen_text_decode", "Qwen3TextDecode"),
    ("vl_2b_instruct.qwen_text_decode_batch", "Qwen3TextDecodeBatch"),
    ("vl_2b_instruct.debug_vision_pooler", "Qwen3VisionPoolerProbe"),
    ("asr_0_6b.prefill", "Qwen3ASRPrefill"),
    ("asr_0_6b.decode", "Qwen3ASRDecode"),
]
for family in ("tts_12hz_0_6b_custom_voice", "tts_12hz_1_7b_custom_voice"):
    ROOTS.extend([
        (family + ".talker_prefill", "Qwen3TTSTalkerPrefill"),
        (family + ".talker_decode", "Qwen3TTSTalkerDecode"),
        (family + ".code_predictor", "Qwen3TTSCodePredictor"),
        (family + ".codec_decode", "Qwen3TTSCodecDecode"),
    ])


def config(layers):
    base = {
        "num_hidden_layers": layers, "hidden_size": 8,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
        "rms_norm_eps": 0.000001, "rope_theta": 10000.0,
        "rope_scaling": {"mrope_section": [1, 1, 0]},
        "sliding_window": 4, "tie_word_embeddings": False,
        "tts_pad_token_id": 0, "num_code_groups": 3,
    }
    result = dict(base)
    result["text_config"] = dict(base)
    result["vision_config"] = {
        "depth": layers, "hidden_size": 8, "num_heads": 2,
        "in_channels": 3, "patch_size": 2, "temporal_patch_size": 1,
        "spatial_merge_size": 2, "deepstack_visual_indexes": [0],
        "hidden_act": "gelu",
    }
    result["image_token_id"] = 1
    result["video_token_id"] = 2
    result["thinker_config"] = {
        "text_config": dict(base), "audio_token_id": 1,
        "audio_config": {"d_model": 8, "encoder_layers": layers,
                         "encoder_attention_heads": 2},
    }
    result["talker_config"] = dict(base)
    result["talker_config"]["code_predictor_config"] = dict(base)
    result["decoder_config"] = dict(base)
    result["decoder_config"].update({
        "latent_dim": 8, "num_quantizers": 3,
        "upsampling_ratios": [2], "upsample_rates": [2, 2],
    })
    return result


def symbolic(name, serial):
    shapes = {
        "input_ids": [1, 2], "aligned_ids": [1, 2, 2], "codec_ids": [1, 3],
        "codes": [1, 3, 2], "first_code": [1, 1], "talker_hidden": [1, 1, 8],
        "position_ids": [3, 1, 2], "attention_mask": [1, 2],
        "key_pages": [2, 2, 2, 1, 4], "value_pages": [2, 2, 2, 1, 4],
        "page_table": [2], "context_lengths": [1], "slot_positions": [1],
        "context_length": [1], "slot_position": [1], "active_mask": [1],
        "start_position": [1], "batch_row_offsets": [2],
        "pixel_values": [4, 12], "image_grid_thw": [1, 3], "grid_thw": [1, 3],
        "vision_bilinear_indices": [4, 4], "bilinear_indices": [4, 4],
        "vision_bilinear_weights": [4, 4], "bilinear_weights": [4, 4],
        "vision_position_ids": [4, 2], "vision_cu_seqlens": [2], "cu_seqlens": [2],
        "mm_token_type_ids": [1, 2], "mrope_position_deltas": [1, 1],
        "input_features": [1, 8, 8], "audio_cu_seqlens": [2],
    }
    assert name in shapes, "unhandled production input: " + name
    floating = name in ("pixel_values", "vision_bilinear_weights", "bilinear_weights",
                        "talker_hidden", "input_features", "key_pages", "value_pages")
    return tensor.input(str(serial) + "_" + name, shape=shapes[name],
                        dtype=tensor.float32 if floating else tensor.int64)


def capture(module, entry, layers):
    arguments = []
    for item in module.GARNET_MODEL_SPEC["arguments"]:
        kind = item["kind"]
        if kind == "tensor":
            arguments.append(symbolic(item["name"], layers))
        elif kind == "config":
            arguments.append(config(layers))
        elif kind == "weights":
            arguments.append({})
        elif kind == "none":
            arguments.append(None)
        elif kind == "bool":
            arguments.append(item["value"])
        else:
            raise AssertionError("unhandled production argument kind: " + kind)
    graph = getattr(module, entry)(*arguments)
    nodes = graph.inspect()
    seen = set()
    root_ids = set()
    for node in nodes:
        assert node["id"] not in seen, str(node)
        for dependency in node["inputs"]:
            assert dependency in seen, (entry, "non-topological dependency", node)
        seen.add(node["id"])
    operations = [node for node in nodes if node["provider"] != ""]
    assert operations, entry
    assert any(node["provider"] == "garnet" for node in operations), entry
    for node in operations:
        assert node["provider"] in ("garnet", "cpu"), str(node)
        assert node["regions"], str(node)
        root = node["regions"][0]
        assert root["boundary"] == "required", str(root)
        root_ids.add(root["id"])
        atomic = False
        region_ids = set()
        for region in node["regions"]:
            assert region["id"] not in region_ids, str(node)
            region_ids.add(region["id"])
            assert not (atomic and region.get("boundary") == "required"), str(node)
            atomic = atomic or region.get("atomic", False)
    assert len(root_ids) == 1, (entry, root_ids)
    return operations


def main():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.insert(0, repo)
    imported = []
    for directory, _, files in os.walk(os.path.join(repo, "xModel", "qwen3")):
        for filename in sorted(files):
            assert not filename.endswith(".x"), "legacy model source remains: " + filename
            if filename.endswith(".py") and filename != "__init__.py":
                relative = os.path.relpath(os.path.join(directory, filename), repo)
                name = relative[:-3].replace("\\", ".").replace("/", ".")
                importlib.import_module(name)
                imported.append(name)
    assert len(imported) == 27, str(imported)
    print("garnet-production-modules-imported", len(imported), flush=True)
    all_ops = set()
    for name, entry in ROOTS:
        module = importlib.import_module("xModel.qwen3." + name)
        one = capture(module, entry, 1)
        two = capture(module, entry, 2)
        assert len(two) > len(one), (entry, len(one), len(two))
        assert one[0]["regions"][0]["id"] != two[0]["regions"][0]["id"], entry
        for node in two:
            all_ops.add(node["provider"] + ":" + node["name"])
        print("garnet-production-root-captured", name, entry, len(one), len(two), flush=True)
    adapter = importlib.import_module("xModel.qwen3.vl_2b_instruct.vl_adapter")
    position = adapter.Qwen3VisionPositionIds(0, symbolic("grid_thw", 3), config(2))
    rope = adapter.Qwen3GetRopeIndex(
        symbolic("input_ids", 3), symbolic("mm_token_type_ids", 3),
        symbolic("image_grid_thw", 3), None, symbolic("attention_mask", 3), config(2))
    for label, outputs, expected_count in (("vision_position_ids", position, 1), ("rope_index", rope, 2)):
        nodes = tensor.graph(outputs).inspect()
        operations = [node for node in nodes if node["provider"]]
        assert len(operations) == expected_count, (label, nodes)
        for node in operations:
            all_ops.add(node["provider"] + ":" + node["name"])
        print("garnet-production-helper-captured", label, expected_count, flush=True)
    print("garnet-production-operators", len(all_ops), sorted(all_ops), flush=True)
    print("garnet-production-capture-passed", len(imported), len(ROOTS) * 2, "roots", 2, "helpers", flush=True)


if __name__ == "__main__":
    main()
