# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]


def add_windows_dll_dirs(garnet_dll):
    handles = []
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return handles
    for path in [
        garnet_dll.parent,
        REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "bin",
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin"),
    ]:
        if path.exists():
            handles.append(os.add_dll_directory(str(path)))
    return handles


def cpu_array(garnet, tensor):
    cpu = garnet.tensor_to_cpu(tensor)
    if cpu is None:
        raise AssertionError("tensor_to_cpu failed")
    return np.asarray(cpu.tolist())


print("Phase 23: GPU X::Tensor model orchestration operations")

import xlang3

garnet_dll = Path(os.environ.get(
    "GARNET_DLL_PATH",
    REPO_ROOT.parent / "out" / "build" / "x64-Release" / "bin" / "garnet.dll",
))
if not garnet_dll.exists():
    raise AssertionError(f"garnet.dll not found: {garnet_dll}")

_dll_handles = add_windows_dll_dirs(garnet_dll)
garnet = xlang3.importModule("garnet", fromPath=str(garnet_dll))

lhs = np.arange(12, dtype=np.float32).reshape(3, 4)
rhs = np.full((3, 4), 0.25, dtype=np.float32)
lhs_gpu = garnet.tensor_to_gpu(lhs)
add_actual = cpu_array(garnet, garnet.tensor_add(lhs_gpu, rhs)).reshape(lhs.shape)
np.testing.assert_allclose(add_actual, lhs + rhs, rtol=0.0, atol=0.0)
last_row_actual = cpu_array(garnet, garnet.tensor_last_row(lhs)).reshape(1, 4)
np.testing.assert_allclose(last_row_actual, lhs[-1:], rtol=0.0, atol=0.0)

weight = np.arange(40, dtype=np.float32).reshape(10, 4) / 10.0
token_ids = np.asarray([7, 2, 9], dtype=np.int64)
embedding_actual = cpu_array(garnet, garnet.embedding(weight, token_ids)).reshape(3, 4)
np.testing.assert_allclose(embedding_actual, weight[token_ids], rtol=0.0, atol=0.0)

import torch
bf16_source = torch.from_numpy(weight).to(torch.bfloat16)
bf16_bits = bf16_source.view(torch.uint16).numpy()
bf16_weight = garnet.tensor_from_bfloat16_bits(bf16_bits)
if bf16_weight is None:
    raise AssertionError("tensor_from_bfloat16_bits failed")
bf16_embedding_actual = cpu_array(garnet, garnet.embedding(bf16_weight, token_ids)).reshape(3, 4)
np.testing.assert_allclose(
    bf16_embedding_actual,
    bf16_source.float().numpy()[token_ids],
    rtol=0.0,
    atol=0.0,
)

base = np.arange(20, dtype=np.float32).reshape(5, 4)
mask = np.asarray([0, 1, 0, 1, 0], dtype=np.int64)
replacements = np.asarray([[100, 101, 102, 103], [200, 201, 202, 203]], dtype=np.float32)
replace_expected = base.copy()
replace_expected[mask == 1] = replacements
replace_actual = cpu_array(
    garnet,
    garnet.replace_rows_by_mask(base, mask, replacements, 1),
).reshape(base.shape)
np.testing.assert_allclose(replace_actual, replace_expected, rtol=0.0, atol=0.0)

gelu_input = np.linspace(-3.0, 3.0, 17, dtype=np.float32).reshape(1, 17)
gelu_expected = 0.5 * gelu_input * (
    1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (gelu_input + 0.044715 * gelu_input**3))
)
gelu_actual = cpu_array(garnet, garnet.gelu_tanh(gelu_input)).reshape(gelu_input.shape)
np.testing.assert_allclose(gelu_actual, gelu_expected, rtol=2e-6, atol=2e-6)

rope_tokens = 2
rope_heads = 2
rope_head_dim = 4
rope_hidden = rope_heads * rope_head_dim
qkv = np.arange(rope_tokens * 3 * rope_hidden, dtype=np.float32).reshape(rope_tokens, 3 * rope_hidden) / 20.0
angles = np.asarray([[0.1, 0.2, 0.1, 0.2], [0.3, 0.4, 0.3, 0.4]], dtype=np.float32)
cos = np.cos(angles).astype(np.float32)
sin = np.sin(angles).astype(np.float32)
rope_expected = qkv.copy().reshape(rope_tokens, 3, rope_heads, rope_head_dim)
for tensor_index in (0, 1):
    value = rope_expected[:, tensor_index].copy()
    rotated = np.concatenate([-value[..., rope_head_dim // 2 :], value[..., : rope_head_dim // 2]], axis=-1)
    rope_expected[:, tensor_index] = value * cos[:, None, :] + rotated * sin[:, None, :]
rope_expected = rope_expected.reshape(qkv.shape)
rope_actual = cpu_array(garnet, garnet.vision_rope(qkv, cos, sin, rope_heads)).reshape(qkv.shape)
np.testing.assert_allclose(rope_actual, rope_expected, rtol=2e-6, atol=2e-6)

print(f"garnet={garnet_dll}")
print("tensor_to_gpu=pass tensor_add=pass tensor_last_row=pass embedding_fp32=pass embedding_bf16=pass replace_rows_by_mask=pass")
print("gelu_tanh=pass vision_rope=pass")
print("Phase 23 GPU tensor ops passed.")
