import os
from pathlib import Path

import numpy as np


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
garnet = xlang.importModule("garnet", fromPath=str(GARNET_DLL))
model = garnet.load_model(
    str(REPO_ROOT / "qwen_vl" / "xmodel" / "qwen_vl_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Qwen3VLModel",
    weights=str(snapshots[-1]),
    input_shapes=[
        [1, 16], [4, 1536], [1, 3], [4, 4], [4, 4], [4, 2],
        [2], [1, 16], [1, 16], [3, 1, 16], [1, 1],
    ],
    input_dtypes=[
        "int64", "bfloat16", "int64", "int64", "bfloat16",
        "int64", "int32", "int64", "int64", "int64", "int64",
    ],
    cache_dir=str(SCRIPT_DIR / "qwen_root_cache"),
)
assert bool(model.runtime_status()["ready"])
prepared = garnet.qwen_vl_prepare_request(
    model_dir=str(snapshots[-1]),
    image_path=str(REPO_ROOT / "data" / "Dataset.1980Love" / "imgs" / "frame_0.jpg"),
    prompt="Describe.",
    min_pixels=1024,
    max_pixels=1024,
)
ids = list(prepared["input_ids"])
types = list(prepared["mm_token_type_ids"])
prompt_tokens = len(ids)
positions = np.asarray(
    garnet.tensor_to_cpu(prepared["position_ids"]).toarray(), dtype=np.int64
).reshape(3, 1, prompt_tokens)
delta = int(np.asarray(
    garnet.tensor_to_cpu(prepared["mrope_position_deltas"]).toarray(), dtype=np.int64
).reshape(-1)[0])
pixel_values = garnet.tensor_to_bfloat16(prepared["pixel_values"])
generated = []
for _ in range(3):
    padded_ids = np.zeros((1, 16), dtype=np.int64)
    padded_ids[0, :len(ids)] = ids
    padded_types = np.zeros((1, 16), dtype=np.int64)
    padded_types[0, :len(types)] = types
    attention = np.zeros((1, 16), dtype=np.int64)
    attention[0, :len(ids)] = 1
    padded_positions = np.zeros((3, 1, 16), dtype=np.int64)
    padded_positions[:, :, :positions.shape[2]] = positions
    result = model.forward({"inputs": [
        padded_ids,
        pixel_values,
        np.array([[1, 2, 2]], dtype=np.int64),
        prepared["vision_bilinear_indices"],
        prepared["vision_bilinear_weights"],
        prepared["vision_position_ids"],
        prepared["vision_cu_seqlens"],
        padded_types,
        attention,
        padded_positions,
        prepared["mrope_position_deltas"],
    ], "sample": "greedy", "sample_row": len(ids) - 1})
    assert result["status"] == "ok", result
    token = int(result["token_id"])
    generated.append(token)
    if len(ids) >= 16:
        break
    ids.append(token)
    types.append(0)
    next_position = len(ids) - 1 + delta
    positions = np.concatenate([
        positions,
        np.full((3, 1, 1), next_position, dtype=np.int64),
    ], axis=2)
print(f"Qwen VLM cacheless token decisions: prompt_tokens={prompt_tokens}, tokens={generated}")
