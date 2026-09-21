# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


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
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3-1.7B"
    / "snapshots"
)
snapshots = sorted(snapshot_root.glob("*"))
assert snapshots, "local Qwen3-1.7B snapshot is required"
weights_root = snapshots[-1]

index = json.loads(
    (weights_root / "model.safetensors.index.json").read_text(encoding="utf-8")
)


def weight(name):
    path = weights_root / index["weight_map"][name]
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


rng = np.random.default_rng(31)
source_fp32 = rng.normal(0.0, 0.15, size=(1, 1, 2048)).astype(np.float32)
source_bits = (source_fp32.view(np.uint32) >> 16).astype(np.uint16)
source_bf16 = (source_bits.astype(np.uint32) << 16).view(np.float32)

garnet = xlang3.importModule("garnet", fromPath=str(GARNET_DLL))
start = time.perf_counter()
model = garnet.load_model(
    str(SCRIPT_DIR / "openvino_qwen3_mlp_model.x"),
    runtime_mode="compiled_xmodel",
    backend="openvino",
    entry_function="Model",
    weights=str(weights_root),
    input_shapes=[[1, 1, 2048]],
    input_dtypes=["bfloat16"],
    cache_dir=str(SCRIPT_DIR / "cache" / "openvino" / "qwen3_1_7b_mlp"),
)
load_ms = (time.perf_counter() - start) * 1000.0
status = model.runtime_status()
assert status["state"] in ("compiled_engine_ready", "engine_cache_loaded"), status

source = garnet.tensor_from_bfloat16_bits(source_bits)
start = time.perf_counter()
result = model.forward({"inputs": [source]})
inference_ms = (time.perf_counter() - start) * 1000.0
assert result["status"] == "ok", result
actual = np.asarray(
    garnet.tensor_to_cpu(result["output"]).tolist(), dtype=np.float32
).reshape(1, 1, 2048)

x = torch.from_numpy(source_bf16).to(torch.bfloat16)
norm_weight = weight(
    "model.layers.0.post_attention_layernorm.weight"
).to(torch.bfloat16)
gate_weight = weight("model.layers.0.mlp.gate_proj.weight").to(torch.bfloat16)
up_weight = weight("model.layers.0.mlp.up_proj.weight").to(torch.bfloat16)
down_weight = weight("model.layers.0.mlp.down_proj.weight").to(torch.bfloat16)
normalized = (
    x.float()
    * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1.0e-6)
).to(torch.bfloat16) * norm_weight
expected = F.linear(
    F.silu(F.linear(normalized, gate_weight)) *
    F.linear(normalized, up_weight),
    down_weight,
).float().numpy()

np.testing.assert_allclose(actual, expected, rtol=6e-2, atol=8e-2)
assert np.isfinite(actual).all()
print(
    "OpenVINO Qwen3-1.7B real-weight RMSNorm+SwiGLU MLP passed: "
    f"state={status['state']}, load_ms={load_ms:.2f}, "
    f"inference_ms={inference_ms:.2f}, max_abs_error="
    f"{np.max(np.abs(actual - expected)):.6f}"
)
