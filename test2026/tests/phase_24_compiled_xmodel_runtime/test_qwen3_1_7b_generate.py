import os
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
GARNET_DLL = (
    REPO_ROOT / "out" / "build" / "x64-Release" / "bin" / "garnet.dll"
)

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


snapshot_root = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3-1.7B"
    / "snapshots"
)
snapshots = sorted(snapshot_root.glob("*"))
assert snapshots, "local Qwen3-1.7B snapshot is required"

layers = 28
max_prompt_tokens = 64
pages = 8
page_size = 16

garnet = xlang.importModule("garnet", fromPath=str(GARNET_DLL))
load_start = time.perf_counter()
model = garnet.load_model(
    str(REPO_ROOT / "xModel" / "qwen3" / "text_1_7b" / "prefill.x"),
    runtime_mode="compiled_xmodel",
    backend="tensorrt",
    entry_function="Qwen3Prefill",
    frontend="qwen3_text",
    weights=str(snapshots[-1]),
    input_shapes=[
        [1, max_prompt_tokens],
        [1, 1, max_prompt_tokens],
        [1, max_prompt_tokens],
        [layers, pages, page_size, 8, 128],
        [layers, pages, page_size, 8, 128],
        [pages],
        [1],
    ],
    input_dtypes=[
        "int64",
        "int64",
        "int64",
        "bfloat16",
        "bfloat16",
        "int32",
        "int32",
    ],
    cache_dir=str(SCRIPT_DIR / "qwen3_1_7b_generate_64_cache"),
    compile={
        "builder_workspace_mb": 4096,
        "builder_optimization_level": 5,
    },
)
status = model.runtime_status()
assert bool(status["ready"]), status
assert bool(status["frontend_prepared"]), status
assert bool(status["engines_prepared"]), status
load_ms = (time.perf_counter() - load_start) * 1000.0

request = {
    "prompt": "Say: Garnet works.",
    "enable_thinking": 0,
    "max_new_tokens": 12,
}
start = time.perf_counter()
result = model.forward(request)
elapsed_ms = (time.perf_counter() - start) * 1000.0
assert result["status"] == "ok", result
text = str(result["text"])
token_ids = [int(token) for token in result["token_ids"]]
assert int(result["prompt_token_count"]) == 18, result
assert int(result["generated_token_count"]) > 0, result
assert token_ids, result
assert text.strip(), result
assert text.startswith("Garnet works."), result
print(
    "Qwen3-1.7B native real-prompt generation passed: "
    f"load_ms={load_ms:.2f}, request_ms={elapsed_ms:.2f}, "
    f"prompt_tokens={result['prompt_token_count']}, "
    f"decode_tokens_per_second={result['decode_tokens_per_second']:.2f}, "
    f"token_ids={token_ids}, text={text!r}"
)
