import json
import os
import struct
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


def tensor_array(garnet, value, shape):
    cpu = garnet.tensor_to_cpu(value)
    return np.asarray(cpu.toarray()).reshape(shape)


def make_tiny_model(garnet):
    fixture_root = REPO_ROOT / "test2026" / "artifacts" / "paged_flash_attention"
    weights = fixture_root / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    header = json.dumps({
        "weight": {"dtype": "F32", "shape": [1, 4], "data_offsets": [0, 16]},
    }, separators=(",", ":")).encode("utf-8")
    with (weights / "model.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q", len(header)))
        stream.write(header)
        stream.write(np.arange(4, dtype=np.float32).tobytes())
    (weights / "config.json").write_text("{}", encoding="utf-8")
    return garnet.load_model(
        str(SCRIPT_DIR / "compiled_add_model.x"),
        runtime_mode="compiled_xmodel",
        entry_function="Model",
        input_shapes=[[1, 4]],
        weights=str(weights),
        cache_dir=str(fixture_root / "cache"),
    )


def bfloat16_bits(values):
    return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


garnet = xlang.importModule("garnet", fromPath=str(GARNET_DLL))
model = make_tiny_model(garnet)
assert bool(model.runtime_status()["ready"]), model.runtime_status()

rng = np.random.default_rng(20260712)
page_size = 16
q_heads = 16
kv_heads = 8
head_dim = 128
q_width = q_heads * head_dim
kv_width = kv_heads * head_dim
qkv_width = q_width + 2 * kv_width

for token_count in [64, 256, 1024, 1280]:
    page_count = (token_count + page_size - 1) // page_size
    qkv = rng.normal(0.0, 0.12, size=(token_count, qkv_width)).astype(np.float32)
    qkv_bits = bfloat16_bits(qkv)
    page_shape = (page_count, page_size, kv_heads, head_dim)
    page_bits = np.zeros(page_shape, dtype=np.uint16)
    page_table = list(reversed(range(page_count)))

    outputs = {}
    for implementation in ["reference", "split", "flash"]:
        probe = model.debug_probe("paged_kv_bf16", {
            "qkv": garnet.tensor_from_bfloat16_bits(qkv_bits),
            "key_pages": garnet.tensor_from_bfloat16_bits(page_bits),
            "value_pages": garnet.tensor_from_bfloat16_bits(page_bits),
            "page_table": page_table,
            "token_count": token_count,
            "start_position": 0,
            "sequence_length": token_count,
            "page_size": page_size,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "implementation": implementation,
        })
        assert probe["status"] == "ok", probe
        outputs[implementation] = tensor_array(
            garnet, probe["output"], (q_heads, head_dim)
        ).astype(np.float32)

    np.testing.assert_allclose(outputs["split"], outputs["reference"], rtol=0, atol=0.002)
    np.testing.assert_allclose(outputs["flash"], outputs["reference"], rtol=0, atol=0.004)
    print(
        f"paged flash parity: tokens={token_count}, "
        f"split_max={np.max(np.abs(outputs['split'] - outputs['reference'])):.6f}, "
        f"flash_max={np.max(np.abs(outputs['flash'] - outputs['reference'])):.6f}"
    )

print("Paged FlashAttention batch-one context-length parity passed")
