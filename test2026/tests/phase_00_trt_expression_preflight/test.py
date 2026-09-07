import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]


def env_flag(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def skip(message):
    print(f"SKIP: {message}")
    raise SystemExit(0)


def discover_garnet_dll():
    explicit = os.environ.get("GARNET_DLL_PATH", "").strip()
    candidates = []
    if explicit:
        candidates.append(Path(explicit))

    candidates.extend([
        REPO_ROOT / "out" / "build" / "x64-Debug" / "garnet.dll",
        REPO_ROOT / "out" / "build" / "x64-Release" / "garnet.dll",
        REPO_ROOT / "out" / "build" / "x64-Debug" / "bin" / "garnet.dll",
        REPO_ROOT / "out" / "build" / "x64-Release" / "bin" / "garnet.dll",
        REPO_ROOT / "out" / "install" / "x64-Debug" / "bin" / "garnet.dll",
        REPO_ROOT / "out" / "install" / "x64-Release" / "bin" / "garnet.dll",
        REPO_ROOT.parent / "out" / "build" / "x64-Debug" / "bin" / "garnet.dll",
        REPO_ROOT.parent / "out" / "build" / "x64-Release" / "bin" / "garnet.dll",
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def add_windows_dll_dirs(garnet_dll):
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return []

    handles = []
    candidates = [
        garnet_dll.parent if garnet_dll else None,
        REPO_ROOT / "out" / "build" / "x64-Debug" / "bin",
        REPO_ROOT / "out" / "build" / "x64-Release" / "bin",
        REPO_ROOT.parent / "xlang" / "out" / "build" / "x64-Debug" / "bin",
        REPO_ROOT.parent / "xlang" / "out" / "build" / "x64-Release" / "bin",
    ]
    for path in candidates:
        if path and path.exists():
            handles.append(os.add_dll_directory(str(path)))
    return handles


print("Phase 00: TRT tensor-expression preflight")

if not env_flag("RUN_GARNET_TRT_PREFLIGHT"):
    skip("set RUN_GARNET_TRT_PREFLIGHT=1 to verify xlang -> Garnet -> TRT preflight")

try:
    import numpy as np
except Exception as exc:
    skip(f"numpy is required for enabled TRT preflight: {exc}")

garnet_dll = discover_garnet_dll()
if garnet_dll is None:
    skip("garnet.dll not found; build Garnet or set GARNET_DLL_PATH")

_dll_dir_handles = add_windows_dll_dirs(garnet_dll)

try:
    import xlang3
except Exception as exc:
    skip(f"xlang Python module not available: {exc}")

try:
    garnet = xlang3.importModule("garnet", fromPath=str(garnet_dll))
except Exception as exc:
    print(f"FAIL_STAGE=dll_import")
    print(f"garnet_dll={garnet_dll}")
    print(f"error={exc}")
    raise

print(f"garnet_dll={garnet_dll}")
print("input_shape=[1, 4]")
print("weight_shape=[4, 3]")

CASES = [
    ("simple", SCRIPT_DIR / "simple_trt_linear.x", (4, 3)),
    ("function", SCRIPT_DIR / "function_trt_linear.x", (4, 3)),
    ("branch", SCRIPT_DIR / "branch_trt_linear.x", (4, 3)),
    ("repeat", SCRIPT_DIR / "repeat_trt_linear.x", (4, 4)),
    ("nested_function", SCRIPT_DIR / "nested_function_trt_linear.x", (4, 3)),
]


def tensor_to_numpy(value):
    to_numpy = getattr(value, "numpy", None)
    to_list = getattr(value, "tolist", None)
    if callable(to_numpy):
        return to_numpy()
    if callable(to_list):
        return np.array(to_list(), dtype=np.float32)
    raise AssertionError("FAIL_STAGE=trt_engine_output_missing_numpy_or_tolist")


def run_case(case_name, xmodel_path, weight_dims):
    engine_path = SCRIPT_DIR / "cache" / f"{xmodel_path.stem}.engine"
    weight_rows, weight_cols = weight_dims
    weights = {"W": np.arange(weight_rows * weight_cols, dtype=np.float32).reshape(weight_rows, weight_cols) / 10.0}

    if engine_path.exists():
        engine_path.unlink()

    print(f"\nCASE={case_name}")
    print(f"xmodel={xmodel_path}")

    try:
        model = garnet.load_model(
            str(xmodel_path),
            weights=weights,
            cache_dir=str(SCRIPT_DIR / "cache"),
            input_shapes=[[1, 4]],
            weight_shape=[weight_rows, weight_cols],
        )
    except Exception as exc:
        print("FAIL_STAGE=load_model_or_expression_build")
        print(f"case={case_name}")
        print(f"error={exc}")
        raise

    if model is None:
        raise AssertionError(f"FAIL_STAGE=load_model_returned_none case={case_name}")

    if not engine_path.exists():
        raise AssertionError(f"FAIL_STAGE=trt_engine_export_missing case={case_name} path={engine_path}")
    if engine_path.stat().st_size <= 0:
        raise AssertionError(f"FAIL_STAGE=trt_engine_export_empty case={case_name} path={engine_path}")

    print(f"trt_engine={engine_path}")
    print(f"trt_engine_bytes={engine_path.stat().st_size}")

    engine = getattr(model, "engine", None)
    if engine is None:
        engine = getattr(model, "m_engine", None)
    if engine is None:
        print("WARN_STAGE=compiled_engine_not_attached")
    else:
        print(f"model_engine={engine}")

    try:
        a_np = np.arange(4, dtype=np.float32).reshape(1, 4)
        out = model.forward(a_np)
    except Exception as exc:
        print("FAIL_STAGE=trt_engine_run")
        print(f"case={case_name}")
        print(f"error={exc}")
        raise

    if out is None:
        raise AssertionError(f"FAIL_STAGE=trt_engine_run_returned_none case={case_name}")

    expected = a_np @ weights["W"]
    actual = tensor_to_numpy(out)
    actual = actual.reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    print(f"trt_output={actual.tolist()}")
    print(f"expected_output={expected.tolist()}")
    print(f"CASE_PASS={case_name}")


for case_name, xmodel_path, weight_dims in CASES:
    run_case(case_name, xmodel_path, weight_dims)

print("Phase 00 TRT preflight loaded function/branch/repeat/nested .x files, ran model.forward, and matched NumPy.")
