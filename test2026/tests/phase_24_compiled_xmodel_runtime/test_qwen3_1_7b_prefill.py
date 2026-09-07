import os
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
GARNET_DLL = (
    REPO_ROOT.parent / "out" / "build" / "x64-Release" / "bin" / "garnet.dll"
)

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
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3-1.7B"
    / "snapshots"
)
snapshots = sorted(snapshot_root.glob("*"))
assert snapshots, "local Qwen3-1.7B snapshot is required"
weights = snapshots[-1]

layers = 28
tokens = 4
garnet = xlang3.importModule("garnet", fromPath=str(GARNET_DLL))
load_start = time.perf_counter()
model = garnet.load_model(
    str(REPO_ROOT / "xModel" / "qwen3" / "text_1_7b" / "prefill.py"),
    runtime_mode="compiled_xmodel",
    backend="tensorrt",
    entry_function="Qwen3Prefill",
    weights=str(weights),
    input_shapes=[
        [1, tokens],
        [1, 1, tokens],
        [1, tokens],
        [layers, 1, 16, 8, 128],
        [layers, 1, 16, 8, 128],
        [1],
        [1],
    ],
    input_dtypes=[
        "int64",
        "int64",
        "int64",
        "bfloat16",
        "bfloat16",
        "int32",
        "int32",
    ],
    cache_dir=str(SCRIPT_DIR / "qwen3_1_7b_prefill_cache"),
    compile={
        "builder_workspace_mb": 4096,
        "builder_optimization_level": 5,
    },
)
status = model.runtime_status()
assert bool(status["ready"]), status
load_ms = (time.perf_counter() - load_start) * 1000.0

zero_pages = np.zeros((layers, 1, 16, 8, 128), dtype=np.uint16)
key_pages = garnet.tensor_from_bfloat16_bits(zero_pages)
value_pages = garnet.tensor_from_bfloat16_bits(zero_pages.copy())
request = {
    "inputs": [
        garnet.tensor_from_host(
            [151644, 8948, 198, 1986], dtype="int64", shape=[1, tokens]
        ),
        garnet.tensor_from_host(
            list(range(tokens)), dtype="int64", shape=[1, 1, tokens]
        ),
        garnet.tensor_from_host(
            [1] * tokens, dtype="int64", shape=[1, tokens]
        ),
        key_pages,
        value_pages,
        garnet.tensor_from_host([0], dtype="int32", shape=[1]),
        garnet.tensor_from_host([0], dtype="int32", shape=[1]),
    ],
    "sample": "greedy",
}

start = time.perf_counter()
result = model.forward(request)
assert result["status"] == "ok", result
forward_ms = (time.perf_counter() - start) * 1000.0
next_token = int(result["token_id"])

decode = garnet.load_model(
    str(REPO_ROOT / "xModel" / "qwen3" / "text_1_7b" / "decode.py"),
    runtime_mode="compiled_xmodel",
    backend="tensorrt",
    entry_function="Qwen3Decode",
    weights=str(weights),
    input_shapes=[
        [1, 1],
        [1, 1, 1],
        [layers, 1, 16, 8, 128],
        [layers, 1, 16, 8, 128],
        [1],
        [1],
        [1],
    ],
    input_dtypes=[
        "int64",
        "int64",
        "bfloat16",
        "bfloat16",
        "int32",
        "int32",
        "int32",
    ],
    cache_dir=str(SCRIPT_DIR / "qwen3_1_7b_decode_cache"),
)
decode_status = decode.runtime_status()
assert bool(decode_status["ready"]), decode_status
generated = []
for position in range(tokens, tokens + 4):
    decode_result = decode.forward(
        {
            "inputs": [
                garnet.tensor_from_host(
                    [next_token], dtype="int64", shape=[1, 1]
                ),
                garnet.tensor_from_host(
                    [position], dtype="int64", shape=[1, 1, 1]
                ),
                key_pages,
                value_pages,
                garnet.tensor_from_host([0], dtype="int32", shape=[1]),
                garnet.tensor_from_host(
                    [position + 1], dtype="int32", shape=[1]
                ),
                garnet.tensor_from_host(
                    [position], dtype="int32", shape=[1]
                ),
            ],
            "sample": "greedy",
        }
    )
    assert decode_result["status"] == "ok", decode_result
    next_token = int(decode_result["token_id"])
    generated.append(next_token)

print(
    "Qwen3-1.7B paged prefill/decode handoff passed: "
    f"load_ms={load_ms:.2f}, prefill_ms={forward_ms:.3f}, "
    f"generated={generated}"
)
