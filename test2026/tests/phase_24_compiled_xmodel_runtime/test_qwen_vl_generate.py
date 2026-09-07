import os
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
GARNET_DLL = REPO_ROOT.parent / "out" / "build" / "x64-Release" / "bin" / "garnet.dll"

handles = []
for directory in [
    GARNET_DLL.parent,
    REPO_ROOT.parent / "out" / "build" / "x64-Release" / "bin",
    REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "bin",
    Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin"),
]:
    if directory.exists() and hasattr(os, "add_dll_directory"):
        handles.append(os.add_dll_directory(str(directory)))

import xlang3

snapshot_root = (
    Path.home() / ".cache" / "huggingface" / "hub" /
    "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots"
)
snapshots = sorted(snapshot_root.glob("*"))
assert snapshots, "local Qwen3-VL-2B-Instruct snapshot is required"
image_path = REPO_ROOT / "data" / "Dataset.1980Love" / "imgs" / "frame_0.jpg"
assert image_path.exists(), image_path

garnet = xlang3.importModule("garnet", fromPath=str(GARNET_DLL))
load_start = time.perf_counter()
model = garnet.load_model(
    str(
        REPO_ROOT
        / "xModel"
        / "qwen3"
        / "vl_2b_instruct"
        / "qwen_vl_prefill.py"
    ),
    runtime_mode="compiled_xmodel",
    entry_function="Qwen3VLPrefill",
    frontend="qwen3_vl",
    weights=str(snapshots[-1]),
    input_shapes=[
        [1, 96], [4, 1536], [1, 3], [4, 4], [4, 4], [4, 2], [2],
        [1, 96], [1, 96], [3, 1, 96], [1, 1],
        [28, 8, 16, 8, 128], [28, 8, 16, 8, 128], [8], [1],
    ],
    input_dtypes=[
        "int64", "bfloat16", "int64", "int64", "bfloat16", "int64", "int32",
        "int64", "int64", "int64", "int64",
        "bfloat16", "bfloat16", "int32", "int32",
    ],
    cache_dir=str(SCRIPT_DIR / "qwen_vl_generate_96_cache"),
)
status = model.runtime_status()
assert bool(status["ready"]), status
load_ms = (time.perf_counter() - load_start) * 1000.0

request = {
    "image_path": str(image_path),
    "prompt": "Describe.",
    "min_pixels": 1024,
    "max_pixels": 1024,
    "max_new_tokens": 4,
}
first_start = time.perf_counter()
first = model.forward(request)
assert first["status"] == "ok", first
first_ms = (time.perf_counter() - first_start) * 1000.0
assert int(first["generated_token_count"]) > 0, first
assert isinstance(first["text"], str), first
assert list(first["token_ids"])[:2] == [1986, 2168], first
assert str(first["text"]).startswith("This image"), first

warm_start = time.perf_counter()
warm = model.forward(request)
assert warm["status"] == "ok", warm
assert list(warm["token_ids"])[:2] == [1986, 2168], warm
assert str(warm["text"]).startswith("This image"), warm
warm_ms = (time.perf_counter() - warm_start) * 1000.0
print(
    "Qwen VLM native one-call generation passed: "
    f"state={status['state']}, load_ms={load_ms:.2f}, first_ms={first_ms:.2f}, "
    f"warm_ms={warm_ms:.2f}, prompt_tokens={warm['prompt_token_count']}, "
    f"visual_tokens={warm['visual_token_count']}, token_ids={list(warm['token_ids'])}, "
    f"text={warm['text']!r}"
)
