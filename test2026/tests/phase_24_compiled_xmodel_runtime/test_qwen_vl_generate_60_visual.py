# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
GARNET_DLL = REPO_ROOT.parent / "out" / "build" / "x64-Release" / "bin" / "garnet.dll"
handles = []
for directory in [
    GARNET_DLL.parent,
    REPO_ROOT.parent / "out" / "build" / "x64-Release" / "bin",
    REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "bin",
    Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin"),
]:
    if directory.exists() and hasattr(os, "add_dll_directory"):
        handles.append(os.add_dll_directory(str(directory)))

import xlang3

snapshots = sorted((
    Path.home() / ".cache" / "huggingface" / "hub" /
    "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots"
).glob("*"))
assert snapshots
image_path = REPO_ROOT / "data" / "Dataset.1980Love" / "imgs" / "frame_0.jpg"
assert image_path.exists()
garnet = xlang3.importModule("garnet", fromPath=str(GARNET_DLL))
load_start = time.perf_counter()
model = garnet.load_model(
    str(
        REPO_ROOT
        / "xModel"
        / "qwen3"
        / "vl_2b_instruct"
        / "qwen_vl_prefill.py"
    ),
    runtime_mode="compiled_xmodel",
    entry_function="Qwen3VLPrefill",
    frontend="qwen3_vl",
    weights=str(snapshots[-1]),
    input_shapes=[
        [1, 96], [240, 1536], [1, 3], [240, 4], [240, 4], [240, 2], [2],
        [1, 96], [1, 96], [3, 1, 96], [1, 1],
        [28, 8, 16, 8, 128], [28, 8, 16, 8, 128], [8], [1],
    ],
    input_dtypes=[
        "int64", "bfloat16", "int64", "int64", "bfloat16", "int64", "int32",
        "int64", "int64", "int64", "int64", "bfloat16", "bfloat16", "int32", "int32",
    ],
    cache_dir=os.environ.get(
        "GARNET_60_CACHE_DIR",
        str(SCRIPT_DIR / "qwen_vl_generate_60_visual_cache"),
    ),
)
status = model.runtime_status()
assert bool(status["ready"]), status
assert bool(status["engines_prepared"]), status
assert float(status["engine_prepare_ms"]) > 0.0, status
assert bool(status["frontend_prepared"]), status
plan = json.loads(status["execution_plan_json"])
assert status["engine_partition_count"] >= 2, status
assert any(
    operation["partition_reason"].startswith("preferred:")
    for operation in plan["operations"]
), plan
assert all(Path(partition["engine_path"]).exists() for partition in plan["engine_partitions"])
load_ms = (time.perf_counter() - load_start) * 1000.0
request = {
    "image_path": str(image_path),
    "prompt": "Describe the image in one sentence.",
    "min_pixels": 65536,
    "max_pixels": 65536,
    "max_new_tokens": 10,
}
cold_start = time.perf_counter()
cold = model.forward(request)
assert cold["status"] == "ok", cold
cold_ms = (time.perf_counter() - cold_start) * 1000.0
assert int(cold["visual_token_count"]) == 60, cold
assert cold_ms < 1000.0, (
    "ready model performed lazy engine refit/context creation during first request",
    cold_ms,
    status,
)
warm_start = time.perf_counter()
warm = model.forward(request)
assert warm["status"] == "ok", warm
warm_ms = (time.perf_counter() - warm_start) * 1000.0
assert int(warm["visual_token_count"]) == 60, warm
assert int(warm["prompt_token_count"]) == 77, warm
assert str(warm["text"]).strip(), warm
frame_results = []
frames = sorted((REPO_ROOT / "data" / "Dataset.1980Love" / "imgs").glob("*.jpg"))[:4]
assert len(frames) == 4
for frame in frames:
    frame_request = dict(request)
    frame_request["image_path"] = str(frame)
    frame_start = time.perf_counter()
    frame_result = model.forward(frame_request)
    assert frame_result["status"] == "ok", frame_result
    frame_results.append({
        "frame": frame.stem,
        "ms": (time.perf_counter() - frame_start) * 1000.0,
        "text": str(frame_result["text"]),
    })
print(
    "Qwen VLM 60-visual-token generation passed: "
    f"state={status['state']}, load_ms={load_ms:.2f}, cold_ms={cold_ms:.2f}, "
    f"engine_prepare_ms={float(status['engine_prepare_ms']):.2f}, "
    f"frontend_prepare_ms={float(status['frontend_prepare_ms']):.2f}, "
    f"warm_ms={warm_ms:.2f}, prompt_tokens={warm['prompt_token_count']}, "
    f"generated={warm['generated_token_count']}, token_ids={list(warm['token_ids'])}, "
    f"text={warm['text']!r}"
)
for frame_result in frame_results:
    print(
        f"{frame_result['frame']}: {frame_result['ms']:.2f} ms, "
        f"text={frame_result['text']!r}"
    )
