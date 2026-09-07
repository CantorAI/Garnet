import os
import time
from pathlib import Path

import numpy as np


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

snapshot_root = (
    Path.home() / ".cache" / "huggingface" / "hub" /
    "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots"
)
snapshots = sorted(snapshot_root.glob("*"))
assert snapshots, "local Qwen3-VL-2B-Instruct snapshot is required"

garnet = xlang3.importModule("garnet", fromPath=str(GARNET_DLL))
load_start = time.perf_counter()
model = garnet.load_model(
    str(REPO_ROOT / "xModel" / "qwen3" / "vl_2b_instruct" / "qwen_vl_prefill.py"),
    runtime_mode="compiled_xmodel",
    entry_function="Qwen3VLPrefill",
    weights=str(snapshots[-1]),
    input_shapes=[
        [1, 16], [4, 1536], [1, 3], [4, 4], [4, 4], [4, 2], [2],
        [1, 16], [1, 16], [3, 1, 16], [1, 1],
        [28, 2, 16, 8, 128], [28, 2, 16, 8, 128], [2], [1],
    ],
    input_dtypes=[
        "int64", "bfloat16", "int64", "int64", "bfloat16", "int64", "int32",
        "int64", "int64", "int64", "int64",
        "bfloat16", "bfloat16", "int32", "int32",
    ],
    cache_dir=str(SCRIPT_DIR / "qwen_vl_prefill_cache"),
)
status = model.runtime_status()
assert bool(status["ready"]), status
load_ms = (time.perf_counter() - load_start) * 1000.0

input_ids = np.zeros((1, 16), dtype=np.int64)
input_ids[0, 4] = 151655
pixel_bits = np.zeros((4, 1536), dtype=np.uint16)
bilinear_bits = np.zeros((4, 4), dtype=np.uint16)
bilinear_bits[0, :] = np.uint16(0x3F80)
positions = np.broadcast_to(np.arange(16, dtype=np.int64), (3, 1, 16)).copy()
zero_pages = np.zeros((28, 2, 16, 8, 128), dtype=np.uint16)
key_pages = garnet.tensor_from_bfloat16_bits(zero_pages)
value_pages = garnet.tensor_from_bfloat16_bits(zero_pages.copy())
request = {"inputs": [
    garnet.tensor_from_host(input_ids.reshape(-1).tolist(), dtype="int64", shape=[1, 16]),
    garnet.tensor_from_bfloat16_bits(pixel_bits),
    garnet.tensor_from_host([1, 2, 2], dtype="int64", shape=[1, 3]),
    garnet.tensor_from_host([0] * 16, dtype="int64", shape=[4, 4]),
    garnet.tensor_from_bfloat16_bits(bilinear_bits),
    garnet.tensor_from_host([0, 0, 0, 1, 1, 0, 1, 1], dtype="int64", shape=[4, 2]),
    garnet.tensor_from_host([0, 4], dtype="int32", shape=[2]),
    garnet.tensor_from_host([0] * 16, dtype="int64", shape=[1, 16]),
    garnet.tensor_from_host([1] * 16, dtype="int64", shape=[1, 16]),
    garnet.tensor_from_host(positions.reshape(-1).tolist(), dtype="int64", shape=[3, 1, 16]),
    garnet.tensor_from_host([0], dtype="int64", shape=[1, 1]),
    key_pages,
    value_pages,
    garnet.tensor_from_host([0, 1], dtype="int32"),
    garnet.tensor_from_host([0], dtype="int32"),
], "sample": "greedy"}
cold_start = time.perf_counter()
cold_result = model.forward(request)
assert cold_result["status"] == "ok", cold_result
cold_ms = (time.perf_counter() - cold_start) * 1000.0
prefill_token = int(cold_result["token_id"])
warm_start = time.perf_counter()
warm_result = model.forward(request)
assert warm_result["status"] == "ok", warm_result
warm_ms = (time.perf_counter() - warm_start) * 1000.0
assert int(warm_result["token_id"]) == prefill_token

decode_model = garnet.load_model(
    str(REPO_ROOT / "xModel" / "qwen3" / "vl_2b_instruct" / "qwen_text_decode.py"),
    runtime_mode="compiled_xmodel",
    entry_function="Qwen3TextDecode",
    weights=str(snapshots[-1]),
    input_shapes=[
        [1, 1], [3, 1, 1], [28, 2, 16, 8, 128], [28, 2, 16, 8, 128],
        [2], [1], [1],
    ],
    input_dtypes=[
        "int64", "int64", "bfloat16", "bfloat16", "int32", "int32", "int32",
    ],
    cache_dir=str(SCRIPT_DIR / "qwen_decode_two_page_cache"),
)
decode_status = decode_model.runtime_status()
assert bool(decode_status["ready"]), decode_status
decode_start = time.perf_counter()
decode_result = decode_model.forward({"inputs": [
    garnet.tensor_from_host([prefill_token], dtype="int64", shape=[1, 1]),
    garnet.tensor_from_host([16, 16, 16], dtype="int64", shape=[3, 1, 1]),
    key_pages,
    value_pages,
    garnet.tensor_from_host([0, 1], dtype="int32"),
    garnet.tensor_from_host([17], dtype="int32"),
    garnet.tensor_from_host([16], dtype="int32"),
], "sample": "greedy"})
assert decode_result["status"] == "ok", decode_result
decode_ms = (time.perf_counter() - decode_start) * 1000.0
warm_decode_start = time.perf_counter()
warm_decode_result = decode_model.forward({"inputs": [
    garnet.tensor_from_host([prefill_token], dtype="int64", shape=[1, 1]),
    garnet.tensor_from_host([16, 16, 16], dtype="int64", shape=[3, 1, 1]),
    key_pages,
    value_pages,
    garnet.tensor_from_host([0, 1], dtype="int32"),
    garnet.tensor_from_host([17], dtype="int32"),
    garnet.tensor_from_host([16], dtype="int32"),
], "sample": "greedy"})
assert warm_decode_result["status"] == "ok", warm_decode_result
warm_decode_ms = (time.perf_counter() - warm_decode_start) * 1000.0
print(
    "Qwen VLM BF16 paged prefill/decode passed: "
    f"state={status['state']}, load_ms={load_ms:.2f}, cold_ms={cold_ms:.2f}, "
    f"warm_prefill_ms={warm_ms:.2f}, cold_decode_ms={decode_ms:.2f}, "
    f"warm_decode_ms={warm_decode_ms:.2f}, "
    f"tokens=[{prefill_token}, {int(decode_result['token_id'])}]"
)
