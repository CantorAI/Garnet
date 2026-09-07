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
    / "models--Qwen--Qwen3-VL-2B-Instruct"
    / "snapshots"
)
snapshots = sorted(snapshot_root.glob("*"))
assert snapshots, "local Qwen3-VL-2B-Instruct snapshot is required"

batch_size = 4
layers = 28
page_size = 16
kv_heads = 8
head_dim = 128
physical_pages = batch_size
logical_pages = 1

garnet = xlang3.importModule("garnet", fromPath=str(GARNET_DLL))
load_start = time.perf_counter()
model = garnet.load_model(
    str(
        REPO_ROOT
        / "xModel"
        / "qwen3"
        / "vl_2b_instruct"
        / "qwen_text_decode_batch.py"
    ),
    runtime_mode="compiled_xmodel",
    backend="tensorrt",
    entry_function="Qwen3TextDecodeBatch",
    weights=str(snapshots[-1]),
    input_shapes=[
        [batch_size, 1],
        [3, batch_size, 1],
        [layers, physical_pages, page_size, kv_heads, head_dim],
        [layers, physical_pages, page_size, kv_heads, head_dim],
        [batch_size, logical_pages],
        [batch_size],
        [batch_size],
        [batch_size],
    ],
    input_dtypes=[
        "int64",
        "int64",
        "bfloat16",
        "bfloat16",
        "int32",
        "int32",
        "int32",
        "int32",
    ],
    cache_dir=str(SCRIPT_DIR / "qwen_decode_batch_b4_cache"),
    compile={
        "builder_workspace_mb": 4096,
        "builder_optimization_level": 5,
    },
)
status = model.runtime_status()
assert bool(status["ready"]), status
load_ms = (time.perf_counter() - load_start) * 1000.0

page_bits = np.zeros(
    (layers, physical_pages, page_size, kv_heads, head_dim),
    dtype=np.uint16,
)
key_pages = garnet.tensor_from_bfloat16_bits(page_bits)
value_pages = garnet.tensor_from_bfloat16_bits(page_bits.copy())
page_table = garnet.tensor_from_host(
    list(range(batch_size)),
    dtype="int32",
    shape=[batch_size, logical_pages],
)
active_mask = garnet.tensor_from_host(
    [1] * batch_size, dtype="int32", shape=[batch_size]
)
input_ids = garnet.tensor_from_host(
    [0] * batch_size, dtype="int64", shape=[batch_size, 1]
)
position_ids = garnet.tensor_from_host(
    [0] * (3 * batch_size),
    dtype="int64",
    shape=[3, batch_size, 1],
)
context_lengths = garnet.tensor_from_host(
    [1] * batch_size, dtype="int32", shape=[batch_size]
)
slot_positions = garnet.tensor_from_host(
    [0] * batch_size, dtype="int32", shape=[batch_size]
)


def make_request(token_ids, position):
    positions = []
    for _ in range(3):
        positions.extend([position] * batch_size)
    assert garnet.tensor_update_from_host(input_ids, token_ids)
    assert garnet.tensor_update_from_host(position_ids, positions)
    assert garnet.tensor_update_from_host(
        context_lengths, [position + 1] * batch_size
    )
    assert garnet.tensor_update_from_host(
        slot_positions, [position] * batch_size
    )
    return {
        "inputs": [
            input_ids,
            position_ids,
            key_pages,
            value_pages,
            page_table,
            context_lengths,
            slot_positions,
            active_mask,
        ],
        "sample": "greedy_batch",
        "reuse_output": 1,
    }


tokens = [1986] * batch_size
warmup_times = []
for position in range(3):
    start = time.perf_counter()
    result = model.forward(make_request(tokens, position))
    assert result["status"] == "ok", result
    tokens = [int(value) for value in result["token_ids"]]
    assert len(tokens) == batch_size
    assert len(set(tokens)) == 1
    warmup_times.append((time.perf_counter() - start) * 1000.0)

decode_times = []
iterations = 12
for offset in range(iterations):
    position = offset + 3
    start = time.perf_counter()
    result = model.forward(make_request(tokens, position))
    assert result["status"] == "ok", result
    tokens = [int(value) for value in result["token_ids"]]
    decode_times.append((time.perf_counter() - start) * 1000.0)

times = np.asarray(decode_times, dtype=np.float64)
aggregate_tokens_per_second = batch_size * 1000.0 / times.mean()
assert np.isfinite(times).all()
assert aggregate_tokens_per_second > 0.0
print(
    "Qwen B4 masked continuous-decode graph passed: "
    f"state={status['state']}, load_ms={load_ms:.2f}, "
    f"warmup_ms={warmup_times}, avg_step_ms={times.mean():.3f}, "
    f"p95_step_ms={np.percentile(times, 95):.3f}, "
    f"aggregate_tokens_per_second={aggregate_tokens_per_second:.2f}, "
    f"tokens={tokens}"
)
