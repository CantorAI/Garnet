# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-free serving selection checks against the real Garnet DLL."""
import os
import json
import garnet


def require(condition, message):
    if not condition:
        raise AssertionError(message)


root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "xModel", "qwen3"))
cases = [
    ("text_1_7b", "prefill.py", ""),
    ("text_1_7b", "prefill.py", "Qwen3-1.7B"),
    ("vl_2b_instruct", "qwen_vl_prefill.py", ""),
    ("vl_2b_instruct", "qwen_vl_prefill.py", "Qwen3-VL-2B-Instruct"),
    ("asr_0_6b", "prefill.py", "Qwen3-ASR-0.6B"),
    ("tts_12hz_0_6b_custom_voice", "talker_prefill.py", "Qwen3-TTS-12Hz-0.6B-CustomVoice"),
    ("tts_12hz_1_7b_custom_voice", "talker_prefill.py", "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
]

for folder, entry, model_id in cases:
    source = os.path.join(root, folder)
    require(os.path.isfile(os.path.join(source, entry)), "missing Python test entry: " + entry)
    require(not os.path.isfile(os.path.join(source, "config.json")), "test requires checkpoint-free source folder")
    result = json.loads(garnet.serve_model(source, source, "", "", model_id))
    require(result["error_code"] == "model_incomplete", str(result))
    require("configuration or tokenizer" in result["error_message"], str(result))

# The family directory contains no entry scripts. Errors must name the selected
# Python filename, including the distinct TTS talker entry, not an obsolete .x.
for model_id, entry in [
    ("Qwen3-1.7B", "prefill.py"),
    ("Qwen3-VL-2B-Instruct", "qwen_vl_prefill.py"),
    ("Qwen3-ASR-0.6B", "prefill.py"),
    ("Qwen3-TTS-12Hz-0.6B-CustomVoice", "talker_prefill.py"),
    ("Qwen3-TTS-12Hz-1.7B-CustomVoice", "talker_prefill.py"),
]:
    require(not os.path.isfile(os.path.join(root, entry)), "unexpected root entry")
    result = json.loads(garnet.serve_model(root, root, "", "", model_id))
    require(result["error_code"] == "xmodel_missing", str(result))
    require(result["error_message"].endswith(entry), str(result))

result = json.loads(garnet.serve_model(root))
require(result["error_code"] == "xmodel_missing", str(result))
require(result["error_message"].endswith(os.path.join("xmodel", "qwen_vl_prefill.py")), str(result))
require(json.loads(garnet.list_loaded_models_json())["models"] == [], "validation must not load a model")
print("garnet-serving-python-defaults-passed", flush=True)
