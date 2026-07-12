import os
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
GARNET_DLL = REPO_ROOT / "out" / "build" / "x64-Release" / "bin" / "garnet.dll"
CACHE_DIR = SCRIPT_DIR / "qwen_prefill_cache"

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

snapshot_root = (
    Path.home() / ".cache" / "huggingface" / "hub" /
    "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots"
)
snapshots = sorted(snapshot_root.glob("*"))
assert snapshots, "local Qwen3-VL-2B-Instruct snapshot is required"

garnet = xlang.importModule("garnet", fromPath=str(GARNET_DLL))
start = time.perf_counter()
model = garnet.load_model(
    str(REPO_ROOT / "qwen_vl" / "xmodel" / "qwen_text_prefill.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Qwen3TextPrefill",
    weights=str(snapshots[-1]),
    input_shapes=[
        [1, 4],
        [3, 1, 4],
        [1, 4],
        [28, 1, 16, 8, 128],
        [28, 1, 16, 8, 128],
        [1],
        [1],
    ],
    input_dtypes=[
        "int64", "int64", "int64", "bfloat16", "bfloat16", "int32", "int32",
    ],
    cache_dir=str(CACHE_DIR),
)
status = model.runtime_status()
assert bool(status["ready"]), status
load_ms = (time.perf_counter() - start) * 1000.0

zero_pages = np.zeros((28, 1, 16, 8, 128), dtype=np.uint16)
key_pages = garnet.tensor_from_bfloat16_bits(zero_pages)
value_pages = garnet.tensor_from_bfloat16_bits(zero_pages.copy())
token_ids = [151644, 8948, 198, 1986]
positions = list(range(4))
request = {"inputs": [
    garnet.tensor_from_host(token_ids, dtype="int64", shape=[1, 4]),
    garnet.tensor_from_host(positions * 3, dtype="int64", shape=[3, 1, 4]),
    garnet.tensor_from_host([1, 1, 1, 1], dtype="int64", shape=[1, 4]),
    key_pages,
    value_pages,
    garnet.tensor_from_host([0], dtype="int32"),
    garnet.tensor_from_host([0], dtype="int32"),
]}
forward_start = time.perf_counter()
result = model.forward(request)
assert result["status"] == "ok", result
logits = np.asarray(garnet.tensor_to_cpu(result["output"]).toarray()).reshape(1, 4, 151936)
cold_forward_ms = (time.perf_counter() - forward_start) * 1000.0
assert np.isfinite(logits).all()

sample_request = dict(request)
sample_request["sample"] = "greedy"
sample_start = time.perf_counter()
sample_result = model.forward(sample_request)
assert sample_result["status"] == "ok", sample_result
warm_sample_ms = (time.perf_counter() - sample_start) * 1000.0
next_token = int(sample_result["token_id"])

decode_model = garnet.load_model(
    str(REPO_ROOT / "qwen_vl" / "xmodel" / "qwen_text_decode.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Qwen3TextDecode",
    weights=str(snapshots[-1]),
    input_shapes=[
        [1, 1], [3, 1, 1], [28, 1, 16, 8, 128], [28, 1, 16, 8, 128],
        [1], [1], [1],
    ],
    input_dtypes=[
        "int64", "int64", "bfloat16", "bfloat16", "int32", "int32", "int32",
    ],
    cache_dir=str(SCRIPT_DIR / "qwen_decode_cache"),
)
decode_status = decode_model.runtime_status()
assert bool(decode_status["ready"]), decode_status
decode_times = []
generated = []
for position in range(4, 8):
    decode_request = {"inputs": [
        garnet.tensor_from_host([next_token], dtype="int64", shape=[1, 1]),
        garnet.tensor_from_host(
            [position, position, position], dtype="int64", shape=[3, 1, 1]
        ),
        key_pages,
        value_pages,
        garnet.tensor_from_host([0], dtype="int32"),
        garnet.tensor_from_host([position + 1], dtype="int32"),
        garnet.tensor_from_host([position], dtype="int32"),
    ], "sample": "greedy"}
    decode_start = time.perf_counter()
    decode_result = decode_model.forward(decode_request)
    assert decode_result["status"] == "ok", decode_result
    decode_times.append((time.perf_counter() - decode_start) * 1000.0)
    next_token = int(decode_result["token_id"])
    generated.append(next_token)
steady_decode = np.asarray(decode_times[1:], dtype=np.float64)
print(
    "Qwen BF16 paged prefill passed: "
    f"state={status['state']}, load_ms={load_ms:.2f}, "
    f"cold_refit_forward_ms={cold_forward_ms:.2f}, "
    f"warm_gpu_sampled_prefill_ms={warm_sample_ms:.2f}, "
    f"handoff_decode_avg_ms={steady_decode.mean():.2f}, "
    f"generated={generated}"
)
