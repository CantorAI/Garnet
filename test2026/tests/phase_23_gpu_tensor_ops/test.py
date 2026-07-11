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
        Path("C:/Program Files/Microsoft Visual Studio/18/Community/VC/Redist/MSVC/14.51.36231/debug_nonredist/x64/Microsoft.VC145.DebugCRT"),
    ]:
        if path.exists():
            handles.append(os.add_dll_directory(str(path)))
    return handles


def cpu_array(garnet, tensor):
    cpu = garnet.tensor_to_cpu(tensor)
    if cpu is None:
        raise AssertionError("tensor_to_cpu failed")
    return np.asarray(cpu.toarray())


print("Phase 23: GPU X::Tensor model orchestration operations")

import xlang

garnet_dll = Path(os.environ.get(
    "GARNET_DLL_PATH",
    REPO_ROOT / "out" / "build" / "x64-Debug" / "bin" / "garnet.dll",
))
if not garnet_dll.exists():
    raise AssertionError(f"garnet.dll not found: {garnet_dll}")

_dll_handles = add_windows_dll_dirs(garnet_dll)
garnet = xlang.importModule("garnet", fromPath=str(garnet_dll))

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

print(f"garnet={garnet_dll}")
print("tensor_to_gpu=pass tensor_add=pass tensor_last_row=pass embedding=pass replace_rows_by_mask=pass")
print("Phase 23 GPU tensor ops passed.")
