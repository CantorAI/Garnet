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
assert (weights / "model.safetensors.index.json").is_file()
assert len(list(weights.glob("model-*-of-*.safetensors"))) == 2

layers = 28
page_size = 16
kv_heads = 8
head_dim = 128

garnet = xlang3.importModule("garnet", fromPath=str(GARNET_DLL))
load_start = time.perf_counter()
model = garnet.load_model(
    str(REPO_ROOT / "xModel" / "qwen3" / "text_1_7b" / "decode.py"),
    runtime_mode="compiled_xmodel",
    backend="tensorrt",
    entry_function="Qwen3Decode",
    weights=str(weights),
    input_shapes=[
        [1, 1],
        [1, 1, 1],
        [layers, 1, page_size, kv_heads, head_dim],
        [layers, 1, page_size, kv_heads, head_dim],
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
    compile={
        "builder_workspace_mb": 4096,
        "builder_optimization_level": 5,
    },
)
status = model.runtime_status()
assert bool(status["ready"]), status
load_ms = (time.perf_counter() - load_start) * 1000.0
weight_probe = model.debug_probe(
    "weight", "model.layers.0.input_layernorm.weight"
)
assert weight_probe["status"] == "ok", weight_probe
probed_weight = np.asarray(
    garnet.tensor_to_cpu(weight_probe["tensor"]).tolist()
).reshape(2048)
assert np.isfinite(probed_weight).all()

page_bits = np.zeros(
    (layers, 1, page_size, kv_heads, head_dim), dtype=np.uint16
)
key_pages = garnet.tensor_from_bfloat16_bits(page_bits)
value_pages = garnet.tensor_from_bfloat16_bits(page_bits.copy())


def request(token, position):
    return {
        "inputs": [
            garnet.tensor_from_host([token], dtype="int64", shape=[1, 1]),
            garnet.tensor_from_host(
                [position], dtype="int64", shape=[1, 1, 1]
            ),
            key_pages,
            value_pages,
            garnet.tensor_from_host([0], dtype="int32", shape=[1]),
            garnet.tensor_from_host(
                [position + 1], dtype="int32", shape=[1]
            ),
            garnet.tensor_from_host([position], dtype="int32", shape=[1]),
        ],
        "sample": "greedy",
    }


token = 1986
times = []
generated = []
for position in range(12):
    start = time.perf_counter()
    result = model.forward(request(token, position))
    assert result["status"] == "ok", result
    token = int(result["token_id"])
    generated.append(token)
    times.append((time.perf_counter() - start) * 1000.0)

steady = np.asarray(times[2:], dtype=np.float64)
assert np.isfinite(steady).all()
assert np.all(steady > 0.0)
print(
    "Qwen3-1.7B sharded BF16 decode passed: "
    f"load_ms={load_ms:.2f}, avg_ms={steady.mean():.3f}, "
    f"p95_ms={np.percentile(steady, 95):.3f}, "
    f"tokens_per_second={1000.0 / steady.mean():.2f}, "
    f"generated_tail={generated[-4:]}"
)
