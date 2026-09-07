import os
import json
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
GARNET_DLL = REPO_ROOT.parent / "out" / "build" / "x64-Release" / "bin" / "garnet.dll"
CACHE_DIR = SCRIPT_DIR / "qwen_decode_cache"

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

snapshot_root = (
    Path.home() / ".cache" / "huggingface" / "hub" /
    "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots"
)
snapshots = sorted(snapshot_root.glob("*"))
assert snapshots, "local Qwen3-VL-2B-Instruct snapshot is required"

garnet = xlang3.importModule("garnet", fromPath=str(GARNET_DLL))
start = time.perf_counter()
model = garnet.load_model(
    str(REPO_ROOT / "xModel" / "qwen3" / "vl_2b_instruct" / "qwen_text_decode.py"),
    runtime_mode="compiled_xmodel",
    entry_function="Qwen3TextDecode",
    weights=str(snapshots[-1]),
    input_shapes=[
        [1, 1],
        [3, 1, 1],
        [28, 1, 16, 8, 128],
        [28, 1, 16, 8, 128],
        [1],
        [1],
        [1],
    ],
    input_dtypes=[
        "int64", "int64", "bfloat16", "bfloat16", "int32", "int32", "int32",
    ],
    cache_dir=str(CACHE_DIR),
)
status = model.runtime_status()
assert bool(status["ready"]), status
assert status["scheduler"] == "cpu_control_gpu_execution", status
execution_plan = json.loads(status["execution_plan_json"])
decode_regions = [
    region for region in execution_plan["regions"]
    if region["role"] == "transformer_decode"
]
layer_regions = [
    region for region in execution_plan["regions"]
    if region["role"] == "decoder_layer"
]
assert len(decode_regions) == 1, execution_plan
assert decode_regions[0]["boundary"] == "required", execution_plan
assert decode_regions[0]["cuda_graph"] is True, execution_plan
assert len(layer_regions) == 28, execution_plan
assert all(region["atomic"] for region in layer_regions), execution_plan
assert all(region["operation_count"] > 0 for region in layer_regions), execution_plan
assert [region["invocation"] for region in layer_regions] == list(range(28)), execution_plan
load_ms = (time.perf_counter() - start) * 1000.0

zero_page_bits = np.zeros((28, 1, 16, 8, 128), dtype=np.uint16)
key_pages = garnet.tensor_from_bfloat16_bits(zero_page_bits)
value_pages = garnet.tensor_from_bfloat16_bits(zero_page_bits.copy())
request = {"inputs": [
    garnet.tensor_from_host([1986], dtype="int64", shape=[1, 1]),
    garnet.tensor_from_host([14, 14, 14], dtype="int64", shape=[3, 1, 1]),
    key_pages,
    value_pages,
    garnet.tensor_from_host([0], dtype="int32"),
    garnet.tensor_from_host([1], dtype="int32"),
    garnet.tensor_from_host([0], dtype="int32"),
]}

forward_start = time.perf_counter()
result = model.forward(request)
assert result["status"] == "ok", result
logits = np.asarray(garnet.tensor_to_cpu(result["output"]).tolist()).reshape(1, 1, 151936)
forward_ms = (time.perf_counter() - forward_start) * 1000.0
assert np.isfinite(logits).all()
next_token = int(np.argmax(logits[0, 0]))
warm_start = time.perf_counter()
warm_result = model.forward(request)
assert warm_result["status"] == "ok", warm_result
warm_logits = np.asarray(
    garnet.tensor_to_cpu(warm_result["output"]).tolist()
).reshape(1, 1, 151936)
warm_ms = (time.perf_counter() - warm_start) * 1000.0
assert np.isfinite(warm_logits).all()
assert int(np.argmax(warm_logits[0, 0])) == next_token
sampled_request = dict(request)
sampled_request["sample"] = "greedy"
sampled_start = time.perf_counter()
sampled_result = model.forward(sampled_request)
assert sampled_result["status"] == "ok", sampled_result
sampled_ms = (time.perf_counter() - sampled_start) * 1000.0
assert int(sampled_result["token_id"]) == next_token

sequence_keys = garnet.tensor_from_bfloat16_bits(zero_page_bits.copy())
sequence_values = garnet.tensor_from_bfloat16_bits(zero_page_bits.copy())
sequence_token = 1986
decode_times = []
generated_tokens = []
for position in range(16):
    step_request = {"inputs": [
        garnet.tensor_from_host([sequence_token], dtype="int64", shape=[1, 1]),
        garnet.tensor_from_host(
            [position, position, position], dtype="int64", shape=[3, 1, 1]
        ),
        sequence_keys,
        sequence_values,
        garnet.tensor_from_host([0], dtype="int32"),
        garnet.tensor_from_host([position + 1], dtype="int32"),
        garnet.tensor_from_host([position], dtype="int32"),
    ], "sample": "greedy"}
    step_start = time.perf_counter()
    step_result = model.forward(step_request)
    assert step_result["status"] == "ok", step_result
    decode_times.append((time.perf_counter() - step_start) * 1000.0)
    sequence_token = int(step_result["token_id"])
    generated_tokens.append(sequence_token)
steady_times = np.asarray(decode_times[1:], dtype=np.float64)
engine_bytes = Path(status["engine_path"]).stat().st_size
print(
    "Qwen BF16 paged decode passed: "
    f"state={status['state']}, engine_bytes={engine_bytes}, load_ms={load_ms:.2f}, "
    f"cold_refit_forward_ms={forward_ms:.2f}, warm_decode_ms={warm_ms:.2f}, "
    f"warm_gpu_sampled_decode_ms={sampled_ms:.2f}, "
    f"page_decode_avg_ms={steady_times.mean():.2f}, "
    f"page_decode_p95_ms={np.percentile(steady_times, 95):.2f}, "
    f"next_token={next_token}, generated_tail={generated_tokens[-4:]}"
)
