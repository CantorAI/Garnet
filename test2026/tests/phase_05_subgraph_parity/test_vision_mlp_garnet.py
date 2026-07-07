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


print("Phase 05: Garnet VisionMLP subgraph parity")

if not env_flag("RUN_GARNET_VISION_MLP_PARITY"):
    skip("set RUN_GARNET_VISION_MLP_PARITY=1 to run Garnet VisionMLP parity")

try:
    import xlang
except Exception as exc:
    skip(f"xlang Python module not available: {exc}")

json_path = Path(os.environ.get(
    "GARNET_VISION_MLP_REFERENCE_JSON",
    ARTIFACT_DIR / "vision_mlp_tiny_reference.json",
))
if not json_path.exists():
    skip("reference dump missing; run test_vision_mlp_reference.py first")

metadata = json.loads(json_path.read_text(encoding="utf-8"))
arrays = np.load(Path(metadata["npz"]))

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
    garnet = xlang.importModule("garnet", fromPath=str(garnet_dll))
except Exception as exc:
    skip(f"failed to import Garnet from {garnet_dll}: {exc}")

weights = {
    "visual.blocks.0.mlp.linear_fc1.weight": arrays["linear_fc1_weight"],
    "visual.blocks.0.mlp.linear_fc1.bias": arrays["linear_fc1_bias"],
    "visual.blocks.0.mlp.linear_fc2.weight": arrays["linear_fc2_weight"],
    "visual.blocks.0.mlp.linear_fc2.bias": arrays["linear_fc2_bias"],
}

xmodel_path = SCRIPT_DIR / "vision_mlp_trt.x"
engine = garnet.load_model(
    str(xmodel_path),
    weights=weights,
    cache_dir=str(SCRIPT_DIR / "cache"),
    input_shapes=[list(arrays["x"].shape)],
    subgraph="vision_mlp",
)
if engine is None:
    raise AssertionError("Garnet load_model returned None for VisionMLP")

output = engine.forward(arrays["x"])
if output is None:
    raise AssertionError("Garnet VisionMLP forward returned None")

to_numpy = getattr(output, "numpy", None)
to_list = getattr(output, "tolist", None)
if callable(to_numpy):
    actual = to_numpy()
elif callable(to_list):
    actual = np.array(to_list(), dtype=np.float32)
else:
    raise AssertionError("Garnet VisionMLP output has no numpy/tolist conversion")

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
        f"VisionMLP parity mismatch max_error={max_error} mean_error={mean_error} first_idx={first_idx}"
    )

print(f"garnet={garnet_dll}")
print(f"xmodel={xmodel_path}")
print(f"reference={json_path}")
print(f"output_shape={list(actual.shape)}")
print(f"max_error={max_error}")
print(f"mean_error={mean_error}")
print("Phase 05 Garnet VisionMLP parity passed.")
