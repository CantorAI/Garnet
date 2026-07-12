import os
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
GARNET_DLL = REPO_ROOT / "out" / "build" / "x64-Release" / "bin" / "garnet.dll"

_dll_handles = []
for directory in [
    GARNET_DLL.parent,
    REPO_ROOT.parent / "xlang" / "out" / "build" / "x64-Release" / "bin",
    REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "bin",
    Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin"),
]:
    if directory.exists() and hasattr(os, "add_dll_directory"):
        _dll_handles.append(os.add_dll_directory(str(directory)))

import xlang

snapshot_root = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3-VL-2B-Instruct"
    / "snapshots"
)
snapshots = sorted(snapshot_root.glob("*"))
if not snapshots:
    raise AssertionError("local Qwen3-VL-2B-Instruct snapshot is required")

cache_dir = SCRIPT_DIR / "qwen_root_cache"
rebuild_root = os.environ.get("GARNET_REBUILD_QWEN_ROOT") == "1"
if rebuild_root:
    shutil.rmtree(cache_dir, ignore_errors=True)
elif not (cache_dir / "model.engine").exists():
    print("Qwen root compile skipped; set GARNET_REBUILD_QWEN_ROOT=1 for the large integration build")
    raise SystemExit(0)

garnet = xlang.importModule("garnet", fromPath=str(GARNET_DLL))
model = garnet.load_model(
    str(REPO_ROOT / "qwen_vl" / "xmodel" / "qwen_vl_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Qwen3VLModel",
    weights=str(snapshots[-1]),
    input_shapes=[
        [1, 16],
        [4, 1536],
        [1, 3],
        [4, 4],
        [4, 4],
        [4, 2],
        [2],
        [1, 16],
        [1, 16],
        [3, 1, 16],
        [1, 1],
    ],
    input_dtypes=[
        "int64", "bfloat16", "int64", "int64", "bfloat16",
        "int64", "int32", "int64", "int64", "int64", "int64"
    ],
    cache_dir=str(cache_dir),
)
status = model.runtime_status()
assert status["state"] in {"compiled_engine_ready", "engine_cache_loaded"}, status
assert bool(status["ready"]), status
assert status["error_code"] == "", status
assert status["forbidden_path_counters"]["hardcoded_qwen_runner_calls"] == 0, status
assert status["graph_summary"], status
assert Path(cache_dir / "runtime_graph.cache").exists(), status
assert Path(status["engine_path"]).exists(), status

print("Qwen root .x compiled without native assertions or fallback runners")
print(f"state: {status['state']}")
print(f"engine bytes: {Path(status['engine_path']).stat().st_size}")
