import os
import time
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


garnet = xlang.importModule("garnet", fromPath=str(GARNET_DLL))
cache_root = SCRIPT_DIR / "cache" / "openvino"


def load(name, shapes, cache_name):
    return garnet.load_model(
        str(SCRIPT_DIR / name),
        runtime_mode="compiled_xmodel",
        backend="openvino",
        entry_function="Model",
        input_shapes=shapes,
        cache_dir=str(cache_root / cache_name),
    )


source = np.array([[1.0, -2.0, 3.5, 4.0]], dtype=np.float32)

start = time.perf_counter()
add_model = load("compiled_add_model.x", [[1, 4]], "add")
compile_ms = (time.perf_counter() - start) * 1000.0
status = add_model.runtime_status()
assert status["state"] in ("compiled_engine_ready", "engine_cache_loaded"), status
assert status["backend"] == "openvino", status
result = add_model.forward({"inputs": [source]})
assert result["status"] == "ok", result
actual = np.asarray(
    garnet.tensor_to_cpu(result["output"]).toarray(), dtype=np.float32
).reshape(source.shape)
np.testing.assert_allclose(
    actual, (source + source) * source - source, rtol=1e-6, atol=1e-6
)

left = np.arange(6, dtype=np.float32).reshape(2, 3)
right = (np.arange(6, dtype=np.float32).reshape(3, 2) + 1.0) / 4.0
matmul_model = load("compiled_matmul_model.x", [[2, 3], [3, 2]], "matmul")
result = matmul_model.forward({"inputs": [left, right]})
assert result["status"] == "ok", result
actual = np.asarray(
    garnet.tensor_to_cpu(result["output"]).toarray(), dtype=np.float32
).reshape(2, 2)
np.testing.assert_allclose(actual, left @ right, rtol=1e-6, atol=1e-6)

relu_model = load("compiled_unary_model.x", [[1, 4]], "relu")
result = relu_model.forward({"inputs": [source]})
assert result["status"] == "ok", result
actual = np.asarray(
    garnet.tensor_to_cpu(result["output"]).toarray(), dtype=np.float32
).reshape(source.shape)
np.testing.assert_allclose(actual, np.maximum(source, 0), rtol=1e-6, atol=1e-6)

cache_start = time.perf_counter()
cached_model = load("compiled_add_model.x", [[1, 4]], "add")
cache_load_ms = (time.perf_counter() - cache_start) * 1000.0
cached_status = cached_model.runtime_status()
assert cached_status["state"] == "engine_cache_loaded", cached_status
cached_result = cached_model.forward({"inputs": [source]})
assert cached_result["status"] == "ok", cached_result

print(
    "OpenVINO TensorGraph backend passed: "
    f"device={os.environ.get('GARNET_OPENVINO_DEVICE', 'CPU')}, "
    f"first_compile_ms={compile_ms:.2f}, cache_load_ms={cache_load_ms:.2f}"
)
