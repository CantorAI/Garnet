# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
ARTIFACT_DIR = REPO_ROOT / "test2026" / "artifacts" / "qwen_vl_subgraphs"


def silu(x):
    return x / (1.0 + np.exp(-x))


print("Phase 05: Qwen3TextMLP reference dump")

rng = np.random.default_rng(20260706)
tokens = 3
hidden_size = 8
intermediate_size = 16

x = rng.normal(0.0, 0.25, size=(tokens, hidden_size)).astype(np.float32)
w_gate = rng.normal(0.0, 0.20, size=(intermediate_size, hidden_size)).astype(np.float32)
w_up = rng.normal(0.0, 0.20, size=(intermediate_size, hidden_size)).astype(np.float32)
w_down = rng.normal(0.0, 0.20, size=(hidden_size, intermediate_size)).astype(np.float32)

gate = x @ w_gate.T
up = x @ w_up.T
hidden = silu(gate) * up
output = hidden @ w_down.T

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
npz_path = ARTIFACT_DIR / "qwen3_text_mlp_tiny_reference.npz"
json_path = ARTIFACT_DIR / "qwen3_text_mlp_tiny_reference.json"

np.savez_compressed(
    npz_path,
    x=x,
    gate_proj_weight=w_gate,
    up_proj_weight=w_up,
    down_proj_weight=w_down,
    gate=gate,
    up=up,
    hidden=hidden,
    output=output,
)

metadata = {
    "schema_version": "0.1",
    "subgraph": "Qwen3TextMLP",
    "source": "deterministic_numpy_reference",
    "formula": "output = (silu(x @ W_gate.T) * (x @ W_up.T)) @ W_down.T",
    "dtype": "float32",
    "tolerance": {
        "rtol": 1e-4,
        "atol": 1e-5,
    },
    "arrays": {
        "x": {"shape": list(x.shape), "dtype": str(x.dtype)},
        "gate_proj_weight": {"shape": list(w_gate.shape), "dtype": str(w_gate.dtype)},
        "up_proj_weight": {"shape": list(w_up.shape), "dtype": str(w_up.dtype)},
        "down_proj_weight": {"shape": list(w_down.shape), "dtype": str(w_down.dtype)},
        "gate": {"shape": list(gate.shape), "dtype": str(gate.dtype)},
        "up": {"shape": list(up.shape), "dtype": str(up.dtype)},
        "hidden": {"shape": list(hidden.shape), "dtype": str(hidden.dtype)},
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
print("Phase 05 Qwen3TextMLP reference dump passed.")
