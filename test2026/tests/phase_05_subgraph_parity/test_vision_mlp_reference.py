# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
ARTIFACT_DIR = REPO_ROOT / "test2026" / "artifacts" / "qwen_vl_subgraphs"


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


print("Phase 05: VisionMLP reference dump")

rng = np.random.default_rng(20260707)
tokens = 5
hidden_size = 8
intermediate_size = 20

x = rng.normal(0.0, 0.25, size=(tokens, hidden_size)).astype(np.float32)
w_fc1 = rng.normal(0.0, 0.20, size=(intermediate_size, hidden_size)).astype(np.float32)
b_fc1 = rng.normal(0.0, 0.05, size=(intermediate_size,)).astype(np.float32)
w_fc2 = rng.normal(0.0, 0.20, size=(hidden_size, intermediate_size)).astype(np.float32)
b_fc2 = rng.normal(0.0, 0.05, size=(hidden_size,)).astype(np.float32)

fc1 = x @ w_fc1.T + b_fc1
hidden = gelu_tanh(fc1)
output = hidden @ w_fc2.T + b_fc2

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
npz_path = ARTIFACT_DIR / "vision_mlp_tiny_reference.npz"
json_path = ARTIFACT_DIR / "vision_mlp_tiny_reference.json"

np.savez_compressed(
    npz_path,
    x=x,
    linear_fc1_weight=w_fc1,
    linear_fc1_bias=b_fc1,
    linear_fc2_weight=w_fc2,
    linear_fc2_bias=b_fc2,
    fc1=fc1,
    hidden=hidden,
    output=output,
)

metadata = {
    "schema_version": "0.1",
    "subgraph": "VisionMLP",
    "source": "deterministic_numpy_reference",
    "formula": "output = gelu_tanh(x @ W_fc1.T + b_fc1) @ W_fc2.T + b_fc2",
    "dtype": "float32",
    "tolerance": {
        "rtol": 1e-4,
        "atol": 1e-5,
    },
    "arrays": {
        "x": {"shape": list(x.shape), "dtype": str(x.dtype)},
        "linear_fc1_weight": {"shape": list(w_fc1.shape), "dtype": str(w_fc1.dtype)},
        "linear_fc1_bias": {"shape": list(b_fc1.shape), "dtype": str(b_fc1.dtype)},
        "linear_fc2_weight": {"shape": list(w_fc2.shape), "dtype": str(w_fc2.dtype)},
        "linear_fc2_bias": {"shape": list(b_fc2.shape), "dtype": str(b_fc2.dtype)},
        "output": {"shape": list(output.shape), "dtype": str(output.dtype)},
    },
    "npz": str(npz_path),
}
json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

reloaded = np.load(npz_path)
np.testing.assert_allclose(reloaded["output"], output, rtol=1e-6, atol=1e-7)
assert reloaded["output"].shape == (tokens, hidden_size)

print(f"reference_json={json_path}")
print(f"reference_npz={npz_path}")
print(f"output_shape={list(output.shape)}")
print("Phase 05 VisionMLP reference dump passed.")
