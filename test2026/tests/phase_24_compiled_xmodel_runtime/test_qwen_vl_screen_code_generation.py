# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import os
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
GARNET_DLL = REPO_ROOT.parent / "out" / "build" / "x64-Release" / "bin" / "garnet.dll"
SCREENSHOT_DIR = REPO_ROOT / "test2026" / "fixtures" / "vlm_screenshots"
ARTIFACT_DIR = REPO_ROOT / "test2026" / "artifacts" / "qwen_vl_screen_code_generation"

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
expected_methods = {
    "Q1.jpg": "minimizeResult",
    "Q2.jpg": "getResults",
}

patch_count = 3772
visual_token_count = 943
max_prefill_tokens = 1024
kv_pages = 96
max_new_tokens = int(os.environ.get("GARNET_SCREEN_CODE_MAX_NEW_TOKENS", "512"))
prompt = os.environ.get(
    "GARNET_SCREEN_CODE_PROMPT",
    (
        "Use only the LeetCode problem in the left panel; ignore the right editor. "
        "Solve it with the shortest correct Python 3 submission using class Solution "
        "and its required method. Code only: no fences, comments, or explanation."
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
        "GARNET_SCREEN_CODE_CACHE_DIR",
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


def remove_markdown_fence(text):
    code = text.strip()
    if code.startswith("```python"):
        code = code[len("```python"):].lstrip()
    elif code.startswith("```"):
        code = code[3:].lstrip()
    if code.endswith("```"):
        code = code[:-3].rstrip()
    return code


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
    raw_text = str(result["text"]).strip()
    code = remove_markdown_fence(raw_text)
    assert code, result
    try:
        ast.parse(code)
        syntax_valid = True
        syntax_error = ""
    except SyntaxError as error:
        syntax_valid = False
        syntax_error = f"{error.msg} at line {error.lineno}"
    expected_method = expected_methods[screenshot.name]
    method_found = f"def {expected_method}(" in code
    hit_token_limit = int(result["generated_token_count"]) >= max_new_tokens

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
        "expected_method": expected_method,
        "expected_method_found": method_found,
        "syntax_valid": syntax_valid,
        "syntax_error": syntax_error,
        "hit_token_limit": hit_token_limit,
        "raw_text": raw_text,
        "code": code,
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

quality_passed = all(
    item["syntax_valid"]
    and item["expected_method_found"]
    and not item["hit_token_limit"]
    for item in results
)
print(
    "Qwen VLM screenshot code generation completed: "
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
        f"decode_tokens_per_second={item['decode_tokens_per_second']:.2f}, "
        f"expected_method={item['expected_method']}, "
        f"method_found={item['expected_method_found']}, "
        f"syntax_valid={item['syntax_valid']}, "
        f"hit_token_limit={item['hit_token_limit']}\n"
        f"{item['code']}"
    )
assert quality_passed, f"generated code did not pass quality checks; inspect {artifact_path}"
