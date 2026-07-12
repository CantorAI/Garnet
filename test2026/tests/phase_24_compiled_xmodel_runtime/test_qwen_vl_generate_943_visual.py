import json
import os
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
GARNET_DLL = REPO_ROOT / "out" / "build" / "x64-Release" / "bin" / "garnet.dll"
handles = []
for directory in [
    GARNET_DLL.parent,
    REPO_ROOT.parent / "xlang" / "out" / "build" / "x64-Release" / "bin",
    REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "bin",
    Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin"),
]:
    if directory.exists() and hasattr(os, "add_dll_directory"):
        handles.append(os.add_dll_directory(str(directory)))

import xlang


snapshots = sorted((
    Path.home() / ".cache" / "huggingface" / "hub" /
    "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots"
).glob("*"))
assert snapshots
image_path = REPO_ROOT / "data" / "Dataset.1980Love" / "imgs" / "frame_0.jpg"
assert image_path.exists()

patch_count = 3772
visual_token_count = 943
max_tokens = 1024
kv_pages = 80
garnet = xlang.importModule("garnet", fromPath=str(GARNET_DLL))
load_start = time.perf_counter()
model = garnet.load_model(
    str(REPO_ROOT / "qwen_vl" / "xmodel" / "qwen_vl_prefill.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Qwen3VLPrefill",
    frontend="qwen3_vl",
    weights=str(snapshots[-1]),
    input_shapes=[
        [1, max_tokens], [patch_count, 1536], [1, 3],
        [patch_count, 4], [patch_count, 4], [patch_count, 2], [2],
        [1, max_tokens], [1, max_tokens], [3, 1, max_tokens], [1, 1],
        [28, kv_pages, 16, 8, 128], [28, kv_pages, 16, 8, 128],
        [kv_pages], [1],
    ],
    input_dtypes=[
        "int64", "bfloat16", "int64", "int64", "bfloat16", "int64", "int32",
        "int64", "int64", "int64", "int64", "bfloat16", "bfloat16", "int32", "int32",
    ],
    compile={"builder_workspace_mb": 4096},
    cache_dir=str(SCRIPT_DIR / "qwen_vl_generate_943_visual_cache"),
)
status = model.runtime_status()
assert bool(status["ready"]), status
assert bool(status["engines_prepared"]), status
assert bool(status["frontend_prepared"]), status
assert int(status["partition_options"]["builder_workspace_bytes"]) == 4096 << 20, status
plan = json.loads(status["execution_plan_json"])
assert status["engine_partition_count"] >= 2, status
assert any(
    operation["partition_reason"].startswith("preferred:")
    for operation in plan["operations"]
), plan
load_ms = (time.perf_counter() - load_start) * 1000.0

request = {
    "image_path": str(image_path),
    "prompt": "Describe the image in one sentence.",
    "min_pixels": 256 * 28 * 28,
    "max_pixels": 1280 * 28 * 28,
    "max_new_tokens": int(os.environ.get("GARNET_BALANCED_MAX_NEW_TOKENS", "100")),
    "ignore_eos": int(os.environ.get("GARNET_BALANCED_IGNORE_EOS", "1")),
}

cold_start = time.perf_counter()
cold = model.forward(request)
assert cold["status"] == "ok", cold
cold_ms = (time.perf_counter() - cold_start) * 1000.0
assert int(cold["visual_token_count"]) == visual_token_count, cold
assert int(cold["width"]) == 1312, cold
assert int(cold["height"]) == 736, cold
if os.environ.get("GARNET_BALANCED_PROBE_ONLY", "0") == "1":
    print(
        "Qwen VLM balanced single-request probe passed: "
        f"load_ms={load_ms:.2f}, request_ms={cold_ms:.2f}, "
        f"prompt_tokens={cold['prompt_token_count']}, "
        f"visual_tokens={cold['visual_token_count']}, "
        f"generated={cold['generated_token_count']}, text={cold['text']!r}"
    )
    raise SystemExit(0)

warm_start = time.perf_counter()
warm = model.forward(request)
assert warm["status"] == "ok", warm
warm_ms = (time.perf_counter() - warm_start) * 1000.0
assert int(warm["visual_token_count"]) == visual_token_count, warm
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
    assert int(frame_result["visual_token_count"]) == visual_token_count, frame_result
    assert int(frame_result["generated_token_count"]) == request["max_new_tokens"], frame_result
    frame_results.append({
        "frame": frame.stem,
        "ms": (time.perf_counter() - frame_start) * 1000.0,
        "generated": int(frame_result["generated_token_count"]),
        "ttft_ms": float(frame_result["time_to_first_token_ms"]),
        "decode_ms": float(frame_result["decode_ms"]),
        "decode_tokens_per_second": float(frame_result["decode_tokens_per_second"]),
        "text": str(frame_result["text"]),
    })

print(
    "Qwen VLM balanced 943-visual-token generation passed: "
    f"state={status['state']}, partitions={status['engine_partition_count']}, "
    f"load_ms={load_ms:.2f}, engine_prepare_ms={float(status['engine_prepare_ms']):.2f}, "
    f"cold_ms={cold_ms:.2f}, warm_ms={warm_ms:.2f}, "
    f"prompt_tokens={warm['prompt_token_count']}, generated={warm['generated_token_count']}, "
    f"text={warm['text']!r}"
)
for frame_result in frame_results:
    print(
        f"{frame_result['frame']}: {frame_result['ms']:.2f} ms, "
        f"generated={frame_result['generated']}, "
        f"ttft_ms={frame_result['ttft_ms']:.2f}, "
        f"decode_ms={frame_result['decode_ms']:.2f}, "
        f"decode_tokens_per_second={frame_result['decode_tokens_per_second']:.2f}, "
        f"text={frame_result['text']!r}"
    )
