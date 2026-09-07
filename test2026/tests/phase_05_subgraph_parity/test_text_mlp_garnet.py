import json
import os
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
ARTIFACT_DIR = REPO_ROOT / "test2026" / "artifacts" / "qwen_vl_subgraphs"


def env_flag(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def skip(message):
    print(f"SKIP: {message}")
    raise SystemExit(0)


print("Phase 05: Garnet Qwen3TextMLP subgraph parity")

if not env_flag("RUN_GARNET_TEXT_MLP_PARITY"):
    skip("set RUN_GARNET_TEXT_MLP_PARITY=1 after Garnet multi-op subgraph execution exists")

try:
    import xlang3
except Exception as exc:
    skip(f"xlang Python module not available: {exc}")

json_path = Path(os.environ.get(
    "GARNET_TEXT_MLP_REFERENCE_JSON",
    ARTIFACT_DIR / "qwen3_text_mlp_tiny_reference.json",
))
if not json_path.exists():
    skip("reference dump missing; run test_text_mlp_reference.py first")

metadata = json.loads(json_path.read_text(encoding="utf-8"))
npz_path = Path(metadata["npz"])
arrays = np.load(npz_path)

required = ["x", "gate_proj_weight", "up_proj_weight", "down_proj_weight", "output"]
missing = [key for key in required if key not in arrays]
if missing:
    raise AssertionError(f"reference dump missing arrays: {missing}")

garnet_dll = Path(os.environ.get(
    "GARNET_DLL_PATH",
    REPO_ROOT / "out" / "build" / "x64-Debug" / "bin" / "garnet.dll",
))
if not garnet_dll.exists():
    skip(f"garnet.dll not found: {garnet_dll}")

if os.name == "nt" and hasattr(os, "add_dll_directory"):
    for dll_dir in [
        garnet_dll.parent,
        REPO_ROOT.parent / "xlang" / "out" / "build" / "x64-Debug" / "bin",
    ]:
        if dll_dir.exists():
            os.add_dll_directory(str(dll_dir))

try:
    garnet = xlang3.importModule("garnet", fromPath=str(garnet_dll))
except Exception as exc:
    skip(f"failed to import Garnet from {garnet_dll}: {exc}")

weights = {
    "language_model.layers.0.mlp.gate_proj.weight": arrays["gate_proj_weight"],
    "language_model.layers.0.mlp.up_proj.weight": arrays["up_proj_weight"],
    "language_model.layers.0.mlp.down_proj.weight": arrays["down_proj_weight"],
}

xmodel_path = SCRIPT_DIR / "text_mlp_trt.x"
engine = garnet.load_model(
    str(xmodel_path),
    weights=weights,
    cache_dir=str(SCRIPT_DIR / "cache"),
    input_shapes=[list(arrays["x"].shape)],
    subgraph="qwen3_text_mlp",
)
if engine is None:
    raise AssertionError("Garnet load_model returned None for Qwen3TextMLP")

output = engine.forward(arrays["x"])
if output is None:
    raise AssertionError("Garnet Qwen3TextMLP forward returned None")

to_numpy = getattr(output, "numpy", None)
to_list = getattr(output, "tolist", None)
if callable(to_numpy):
    actual = to_numpy()
elif callable(to_list):
    actual = np.array(to_list(), dtype=np.float32)
else:
    raise AssertionError("Garnet Qwen3TextMLP output has no numpy/tolist conversion")

expected = arrays["output"]
actual = actual.reshape(expected.shape)
diff = np.abs(actual - expected)
max_error = float(diff.max())
mean_error = float(diff.mean())
try:
    np.testing.assert_allclose(actual, expected, rtol=metadata["tolerance"]["rtol"], atol=metadata["tolerance"]["atol"])
except AssertionError:
    first_idx = np.argwhere(diff > metadata["tolerance"]["atol"])[0].tolist()
    raise AssertionError(
        f"Qwen3TextMLP parity mismatch max_error={max_error} mean_error={mean_error} first_idx={first_idx}"
    )

print(f"garnet={garnet_dll}")
print(f"xmodel={xmodel_path}")
print(f"reference={json_path}")
print(f"output_shape={list(actual.shape)}")
print(f"max_error={max_error}")
print(f"mean_error={mean_error}")
print("Phase 05 Garnet Qwen3TextMLP parity passed.")
