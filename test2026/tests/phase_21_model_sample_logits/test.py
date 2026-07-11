import os
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]


def env_flag(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def skip(message):
    print(f"SKIP: {message}")
    raise SystemExit(0)


def add_windows_dll_dirs(garnet_dll):
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return []
    handles = []
    for dll_dir in [
        garnet_dll.parent,
        REPO_ROOT.parent / "xlang" / "out" / "build" / "x64-Debug" / "bin",
        REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "bin",
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin"),
        Path("C:/Program Files/Microsoft Visual Studio/18/Community/VC/Redist/MSVC/14.51.36231/debug_nonredist/x64/Microsoft.VC145.DebugCRT"),
    ]:
        if dll_dir.exists():
            handles.append(os.add_dll_directory(str(dll_dir)))
    return handles


print("Phase 21: Garnet Model.debug_probe logits_top1 GPU tensor smoke")

if not env_flag("RUN_GARNET_MODEL_SAMPLE_LOGITS"):
    skip("set RUN_GARNET_MODEL_SAMPLE_LOGITS=1 to verify xlang-facing Model.debug_probe logits_top1")

try:
    import xlang
except Exception as exc:
    skip(f"xlang Python module not available: {exc}")

garnet_dll = Path(os.environ.get(
    "GARNET_DLL_PATH",
    REPO_ROOT / "out" / "build" / "x64-Debug" / "bin" / "garnet.dll",
))
if not garnet_dll.exists():
    skip(f"garnet.dll not found: {garnet_dll}")

_dll_handles = add_windows_dll_dirs(garnet_dll)
garnet = xlang.importModule("garnet", fromPath=str(garnet_dll))

xmodel_path = REPO_ROOT / "test2026" / "tests" / "phase_00_trt_expression_preflight" / "simple_trt_linear.x"
weights = {"W": np.zeros((4, 11), dtype=np.float32)}
weights["W"][0, 4] = 100.0
weights["W"][1, 5] = 200.0
weights["W"][2, 7] = 42.5
weights["W"][2, 9] = 41.5
model = garnet.load_model(
    str(xmodel_path),
    weights=weights,
    cache_dir=str(SCRIPT_DIR / "cache"),
    input_shapes=[[3, 4]],
    weight_shape=[4, 11],
    subgraph="matmul",
)
if model is None:
    raise AssertionError("Garnet load_model returned None")

probe_input = np.zeros((3, 4), dtype=np.float32)
probe_input[0, 0] = 1.0
probe_input[1, 1] = 1.0
probe_input[2, 2] = 1.0
logits = model.forward(probe_input)
if logits is None:
    raise AssertionError("model.forward returned None; cannot verify debug_probe on Garnet tensor output")

cpu_logits = garnet.tensor_to_cpu(logits)
if cpu_logits is None:
    raise AssertionError("garnet.tensor_to_cpu returned None")
cpu_logits_np = cpu_logits.toarray()
np.testing.assert_allclose(cpu_logits_np, probe_input @ weights["W"], rtol=1e-6, atol=1e-6)

sample = model.debug_probe("logits_top1", logits)
if sample is None:
    raise AssertionError("model.debug_probe('logits_top1', logits) returned None")

probe = str(sample["probe"])
if probe != "logits_top1":
    raise AssertionError(f"expected probe=logits_top1, got {probe}")
token_id = int(sample["token_id"])
token_value = float(sample["token_value"])
if token_id != 7:
    raise AssertionError(f"expected token_id=7 from last logits row, got {token_id}")
if abs(token_value - 42.5) > 1e-4:
    raise AssertionError(f"expected token_value=42.5, got {token_value}")

print(f"garnet={garnet_dll}")
print(f"xmodel={xmodel_path}")
print(f"probe={probe}")
print(f"token_id={token_id}")
print(f"token_value={token_value}")
print("Phase 21 Model.debug_probe logits_top1 GPU tensor smoke passed.")
