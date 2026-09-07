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
        REPO_ROOT.parent / "out" / "build" / "x64-Debug" / "bin",
        REPO_ROOT.parent / "out" / "build" / "x64-debug" / "bin",
        REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "bin",
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin"),
        Path("C:/Program Files/Microsoft Visual Studio/18/Community/VC/Redist/MSVC/14.51.36231/debug_nonredist/x64/Microsoft.VC145.DebugCRT"),
    ]:
        if dll_dir.exists():
            handles.append(os.add_dll_directory(str(dll_dir)))
    return handles


print("Phase 22: Garnet model.forward text RoPE -> device KV -> attention bridge")

if not env_flag("RUN_GARNET_MODEL_DEVICE_KV_BRIDGE"):
    skip("set RUN_GARNET_MODEL_DEVICE_KV_BRIDGE=1 to verify model-owned device KV bridge")

try:
    import xlang3
except Exception as exc:
    skip(f"xlang Python module not available: {exc}")

garnet_dll = Path(os.environ.get(
    "GARNET_DLL_PATH",
    REPO_ROOT / "out" / "build" / "x64-Debug" / "bin" / "garnet.dll",
))
if not garnet_dll.exists():
    skip(f"garnet.dll not found: {garnet_dll}")

_dll_handles = add_windows_dll_dirs(garnet_dll)
garnet = xlang3.importModule("garnet", fromPath=str(garnet_dll))

os.environ["GARNET_TRT_SYNC_CPU_OUTPUTS"] = "0"

tokens = 3
q_heads = 16
kv_heads = 8
head_dim = 128
hidden = q_heads * head_dim
qkv_width = hidden + 2 * kv_heads * head_dim

rng = np.random.default_rng(20260722)
qkv = rng.normal(0.0, 0.05, size=(tokens, qkv_width)).astype(np.float32)
cos = np.ones((tokens, head_dim), dtype=np.float32)
sin = np.zeros((tokens, head_dim), dtype=np.float32)

rope_model = garnet.load_model(
    str(REPO_ROOT / "test2026" / "tests" / "phase_05_subgraph_parity" / "text_rope_apply_trt.x"),
    weights={},
    cache_dir=str(SCRIPT_DIR / "cache" / "rope"),
    input_shapes=[[tokens, qkv_width], [tokens, head_dim], [tokens, head_dim]],
    subgraph="text_rope_apply",
)
if rope_model is None:
    raise AssertionError("failed to load text_rope_apply model")

kv_cache = rope_model.create_device_kv_cache(tokens, 8, q_heads, kv_heads, head_dim, 1)
if kv_cache is None:
    raise AssertionError("create_device_kv_cache returned None")

rope_result = rope_model.forward(qkv, cos, sin, kv_cache, tokens, 0)
if rope_result is None:
    raise AssertionError("text_rope_apply forward with kv_cache returned None")
if str(rope_result["status"]) != "ok":
    raise AssertionError(f"expected rope status ok, got {rope_result['status']}")
if not bool(rope_result["kv_written"]):
    raise AssertionError("text_rope_apply did not report kv_written")
if int(rope_result["kv_logical_length"]) != tokens:
    raise AssertionError(f"expected kv logical length {tokens}, got {rope_result['kv_logical_length']}")

qkv_gpu = rope_result["output_tensor"]
if qkv_gpu is None:
    raise AssertionError("text_rope_apply did not return output_tensor")

attention_model = garnet.load_model(
    str(REPO_ROOT / "test2026" / "tests" / "phase_05_subgraph_parity" / "text_attention_core_trt.x"),
    weights={},
    cache_dir=str(SCRIPT_DIR / "cache" / "attention"),
    input_shapes=[[tokens, qkv_width]],
    subgraph="text_attention_core",
)
if attention_model is None:
    raise AssertionError("failed to load text_attention_core model")

attention_result = attention_model.forward(qkv_gpu, kv_cache, tokens, hidden)
if attention_result is None:
    raise AssertionError("text_attention_core forward with kv_cache returned None")
if str(attention_result["status"]) != "ok":
    raise AssertionError(f"expected attention status ok, got {attention_result['status']}")
if not bool(attention_result["kv_read"]):
    raise AssertionError("text_attention_core did not report kv_read")

attention_output = attention_result["output_tensor"]
if attention_output is None:
    raise AssertionError("text_attention_core did not return output_tensor")

probe = attention_model.debug_probe("logits_top1", attention_output)
if probe is None or str(probe["probe"]) != "logits_top1":
    raise AssertionError("debug_probe failed on device attention output tensor")
token_id = int(probe["token_id"])
if token_id < 0 or token_id >= hidden:
    raise AssertionError(f"debug_probe token id out of range: {token_id}")

destroyed = bool(rope_model.destroy_device_kv_cache(int(kv_cache["handle"])))
if not destroyed:
    raise AssertionError("destroy_device_kv_cache failed")

print(f"garnet={garnet_dll}")
print(f"tokens={tokens} qkv_width={qkv_width} hidden={hidden}")
print(f"kv_logical_length={int(rope_result['kv_logical_length'])}")
print(f"attention_probe_token={token_id}")
print("Phase 22 model device-KV bridge passed.")
