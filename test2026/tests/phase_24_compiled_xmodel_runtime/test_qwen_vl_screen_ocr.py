# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
GARNET_DLL = REPO_ROOT.parent / "out" / "build" / "x64-Release" / "bin" / "garnet.dll"
SCREENSHOT_DIR = REPO_ROOT / "test2026" / "fixtures" / "vlm_screenshots"
ARTIFACT_DIR = REPO_ROOT / "test2026" / "artifacts" / "qwen_vl_screen_ocr"

dll_handles = []
for directory in [
    GARNET_DLL.parent,
    REPO_ROOT.parent / "out" / "build" / "x64-Release" / "bin",
    REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "bin",
    Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin"),
]:
    if directory.exists() and hasattr(os, "add_dll_directory"):
        dll_handles.append(os.add_dll_directory(str(directory)))

import xlang3


snapshots = sorted((
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3-VL-2B-Instruct"
    / "snapshots"
).glob("*"))
assert snapshots, "Qwen3-VL-2B-Instruct weights are not available in the HF cache"

screenshots = sorted(SCREENSHOT_DIR.glob("*.jpg"))
assert len(screenshots) == 2, f"expected two JPG screenshots in {SCREENSHOT_DIR}, got {screenshots}"

# This profile matches the regular-resolution benchmark: 1312x736 (or the
# nearest aspect-preserving 28-pixel grid), 3,772 patches, and 943 visual tokens.
patch_count = 3772
visual_token_count = 943
max_prefill_tokens = 1024
kv_pages = 96
max_new_tokens = int(os.environ.get("GARNET_SCREEN_OCR_MAX_NEW_TOKENS", "512"))
prompt = os.environ.get(
    "GARNET_SCREEN_OCR_PROMPT",
    (
        "Perform OCR on this screenshot. Transcribe all clearly readable text in the "
        "LeetCode problem description panel, from top to bottom. Preserve the title, "
        "problem statement, numbered lists, examples, formulas, punctuation, and line "
        "breaks. Do not summarize, explain, or add markdown fences. Output only the "
        "transcribed text."
    ),
)

garnet = xlang3.importModule("garnet", fromPath=str(GARNET_DLL))
load_start = time.perf_counter()
model = garnet.load_model(
    str(REPO_ROOT / "xModel" / "qwen3" / "vl_2b_instruct" / "qwen_vl_prefill.py"),
    runtime_mode="compiled_xmodel",
    entry_function="Qwen3VLPrefill",
    frontend="qwen3_vl",
    weights=str(snapshots[-1]),
    input_shapes=[
        [1, max_prefill_tokens],
        [patch_count, 1536],
        [1, 3],
        [patch_count, 4],
        [patch_count, 4],
        [patch_count, 2],
        [2],
        [1, max_prefill_tokens],
        [1, max_prefill_tokens],
        [3, 1, max_prefill_tokens],
        [1, 1],
        [28, kv_pages, 16, 8, 128],
        [28, kv_pages, 16, 8, 128],
        [kv_pages],
        [1],
    ],
    input_dtypes=[
        "int64",
        "bfloat16",
        "int64",
        "int64",
        "bfloat16",
        "int64",
        "int32",
        "int64",
        "int64",
        "int64",
        "int64",
        "bfloat16",
        "bfloat16",
        "int32",
        "int32",
    ],
    compile={"builder_workspace_mb": 4096},
    cache_dir=os.environ.get(
        "GARNET_SCREEN_OCR_CACHE_DIR",
        str(SCRIPT_DIR / "qwen_vl_screen_ocr_cache"),
    ),
)
load_ms = (time.perf_counter() - load_start) * 1000.0
status = model.runtime_status()
assert bool(status["ready"]), status
assert bool(status["engines_prepared"]), status
assert bool(status["frontend_prepared"]), status


def make_request(image_path, token_limit):
    return {
        "image_path": str(image_path),
        "prompt": prompt,
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1280 * 28 * 28,
        "max_new_tokens": token_limit,
        "ignore_eos": 0,
    }


# Run a short request first so reported OCR timings exclude one-time CUDA and
# execution-context initialization while still exercising the same image shape.
warmup_start = time.perf_counter()
warmup = model.forward(make_request(screenshots[0], 1))
warmup_ms = (time.perf_counter() - warmup_start) * 1000.0
assert warmup["status"] == "ok", warmup
assert int(warmup["visual_token_count"]) == visual_token_count, warmup

results = []
for screenshot in screenshots:
    request_start = time.perf_counter()
    result = model.forward(make_request(screenshot, max_new_tokens))
    total_ms = (time.perf_counter() - request_start) * 1000.0
    assert result["status"] == "ok", result
    assert int(result["visual_token_count"]) == visual_token_count, result
    assert str(result["text"]).strip(), result

    results.append({
        "image": screenshot.name,
        "source_width": int(result["source_width"]),
        "source_height": int(result["source_height"]),
        "model_width": int(result["width"]),
        "model_height": int(result["height"]),
        "visual_tokens": int(result["visual_token_count"]),
        "prompt_tokens": int(result["prompt_token_count"]),
        "generated_tokens": int(result["generated_token_count"]),
        "max_new_tokens": max_new_tokens,
        "time_to_first_token_ms": float(result["time_to_first_token_ms"]),
        "decode_ms": float(result["decode_ms"]),
        "decode_tokens_per_second": float(result["decode_tokens_per_second"]),
        "total_ms": total_ms,
        "text": str(result["text"]),
        "token_ids": list(result["token_ids"]),
    })

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
artifact_path = ARTIFACT_DIR / "results.json"
artifact_path.write_text(
    json.dumps(
        {
            "model": "Qwen3-VL-2B-Instruct",
            "runtime": "Garnet compiled_xmodel",
            "prompt": prompt,
            "load_ms": load_ms,
            "warmup_ms": warmup_ms,
            "results": results,
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print(
    "Qwen VLM screen OCR passed: "
    f"load_ms={load_ms:.2f}, warmup_ms={warmup_ms:.2f}, artifact={artifact_path}"
)
for item in results:
    print(
        f"\n=== {item['image']} ===\n"
        f"source={item['source_width']}x{item['source_height']}, "
        f"model={item['model_width']}x{item['model_height']}, "
        f"visual_tokens={item['visual_tokens']}, prompt_tokens={item['prompt_tokens']}, "
        f"generated={item['generated_tokens']}, total_ms={item['total_ms']:.2f}, "
        f"ttft_ms={item['time_to_first_token_ms']:.2f}, "
        f"decode_ms={item['decode_ms']:.2f}, "
        f"decode_tokens_per_second={item['decode_tokens_per_second']:.2f}\n"
        f"{item['text']}"
    )
