# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / "test2026" / "artifacts" / "qwen_vl_cached_prompt_image"
PHASE05_SCRIPT = REPO_ROOT / "test2026" / "tests" / "phase_05_subgraph_parity" / "test_real_qwen_mlp_subgraphs.py"
DEFAULT_IMAGE = REPO_ROOT / "data" / "Dataset.1980Love" / "imgs" / "frame_0.jpg"
DEFAULT_PROMPT = "Describe this picture in one short sentence."


def extract_decode_json(text):
    marker = '"processor_npz"'
    marker_index = text.find(marker)
    if marker_index < 0:
        raise AssertionError("Garnet cached prompt-image JSON was not found in subprocess output")
    start = text.rfind("{", 0, marker_index)
    if start < 0:
        raise AssertionError("Garnet cached prompt-image JSON start was not found")
    decoder = json.JSONDecoder()
    result, _ = decoder.raw_decode(text[start:])
    return result


def main():
    image = Path(os.environ.get("GARNET_NATIVE_PROMPT_IMAGE_TEST_IMAGE", DEFAULT_IMAGE))
    prompt = os.environ.get("GARNET_NATIVE_PROMPT_IMAGE_TEST_PROMPT", DEFAULT_PROMPT)
    max_pixels = os.environ.get("GARNET_NATIVE_PROMPT_IMAGE_TEST_PIXELS", "65536")
    max_new_tokens = os.environ.get("GARNET_NATIVE_PROMPT_IMAGE_TEST_TOKENS", "3")
    text_layer_count = os.environ.get("GARNET_CACHED_PROMPT_IMAGE_TEXT_LAYERS", "4")
    if not image.exists():
        raise AssertionError(f"test image does not exist: {image}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "RUN_GARNET_REAL_QWEN_MLP_PARITY": "1",
            "RUN_GARNET_REAL_QWEN_MODEL_FORWARD_NATIVE_ROPE_DECODE": "1",
            "GARNET_QWEN_NATIVE_ROPE_DECODE_ONLY": "1",
            "GARNET_QWEN_NATIVE_ROPE_DECODE_ONE_CALL_PREPARE": "1",
            "GARNET_QWEN_NATIVE_ROPE_DECODE_GARNET_TOKENIZER": "1",
            "GARNET_QWEN_NATIVE_ROPE_DECODE_USE_KV_MANAGER": "1",
            "GARNET_QWEN_NATIVE_ROPE_DECODE_USE_CACHED_TEXT": "1",
            "GARNET_QWEN_NATIVE_ROPE_DECODE_TEXT_LAYER_COUNT": text_layer_count,
            "GARNET_QWEN_NATIVE_ROPE_DECODE_TEXT_KV_PAGE_SIZE": "16",
            "GARNET_QWEN_NATIVE_ROPE_DECODE_PROCESSOR_PIXELS": max_pixels,
            "GARNET_QWEN_NATIVE_ROPE_DECODE_TOKENS": max_new_tokens,
            "GARNET_QWEN_NATIVE_ROPE_DECODE_PROMPT": prompt,
            "GARNET_QWEN_NATIVE_ROPE_DECODE_IMAGE": str(image),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-u", str(PHASE05_SCRIPT)],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=int(os.environ.get("GARNET_CACHED_PROMPT_IMAGE_TEST_TIMEOUT_SECONDS", "900")),
    )

    combined_output = completed.stdout + "\n" + completed.stderr
    (ARTIFACT_DIR / "last_stdout_stderr.log").write_text(combined_output, encoding="utf-8", errors="ignore")
    if completed.returncode != 0:
        raise AssertionError(
            f"Garnet cached prompt-image subprocess failed with exit code {completed.returncode}. "
            f"See {ARTIFACT_DIR / 'last_stdout_stderr.log'}"
        )

    result = extract_decode_json(combined_output)
    result.update(
        {
            "input_mode": "jpeg_file_and_prompt_to_cached_garnet_qwen_vl",
            "phase": "phase_18_qwen_vl_cached_prompt_image",
        }
    )
    (ARTIFACT_DIR / "last_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result.get("text_decode_mode") != "cached_prefill_decode":
        raise AssertionError(f"expected cached_prefill_decode, got {result.get('text_decode_mode')}")
    if not result.get("raw_generated_text", "").strip():
        raise AssertionError("Garnet cached prompt-image decode produced empty raw_generated_text")

    timing = result.get("timing_ms", {})
    print("Phase 18: cached prompt+image Garnet generate passed.")
    print(f"image={image}")
    print(f"prompt={prompt}")
    print(f"text_layers={result.get('text_layer_count')}/{result.get('full_text_layer_count')}")
    print(f"generated_text={result.get('generated_text')}")
    print(f"raw_generated_text={result.get('raw_generated_text')}")
    print(
        "timing_ms="
        f"visual={timing.get('visual_tokens_once_ms', 0.0):.2f}, "
        f"text_prefill={timing.get('text_prefill_ms', 0.0):.2f}, "
        f"decode_token_loop={timing.get('decode_token_loop_ms', 0.0):.2f}, "
        f"total_after_frontend={timing.get('total_after_frontend_ms', 0.0):.2f}"
    )
    print(f"result={ARTIFACT_DIR / 'last_result.json'}")


if __name__ == "__main__":
    main()
