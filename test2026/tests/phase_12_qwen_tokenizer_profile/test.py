import ctypes
import json
import os
import statistics
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / "test2026" / "artifacts" / "qwen_tokenizer_profile"
DEFAULT_PROMPT = "Describe this picture in one short sentence."


def skip(message):
    print(f"SKIP: {message}")
    raise SystemExit(0)


def default_model_dir():
    explicit = os.environ.get("HF_QWEN_VL_MODEL_DIR") or os.environ.get("GARNET_QWEN_VL_WEIGHT_INDEX_OR_DIR")
    if explicit:
        return Path(explicit)
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots"
    if cache_root.exists():
        snapshots = sorted(cache_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if snapshots:
            return snapshots[0]
    return None


def add_dll_dirs(garnet_dll):
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    for dll_dir in [
        garnet_dll.parent,
        REPO_ROOT.parent / "xlang" / "out" / "build" / "x64-Debug" / "bin",
        REPO_ROOT.parent / "out" / "build" / "x64-Debug" / "bin",
        REPO_ROOT.parent / "out" / "build" / "x64-debug" / "bin",
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin"),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64"),
        REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "bin",
        REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "lib",
        Path("C:/Program Files/Microsoft Visual Studio/18/Community/VC/Redist/MSVC/14.51.36231/debug_nonredist/x64/Microsoft.VC145.DebugCRT"),
    ]:
        if dll_dir.exists():
            os.add_dll_directory(str(dll_dir))


def load_dll():
    garnet_dll = Path(os.environ.get("GARNET_DLL_PATH", REPO_ROOT / "out" / "build" / "x64-Debug" / "bin" / "garnet.dll"))
    if not garnet_dll.exists():
        skip(f"garnet.dll not found: {garnet_dll}")
    add_dll_dirs(garnet_dll)
    dll = ctypes.CDLL(str(garnet_dll))
    dll.GarnetQwenTokenizerEncode.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_longlong), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_int,
    ]
    dll.GarnetQwenTokenizerEncode.restype = ctypes.c_int
    dll.GarnetQwenTokenizerDecode.argtypes = [
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_longlong), ctypes.c_int, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_int,
    ]
    dll.GarnetQwenTokenizerDecode.restype = ctypes.c_int
    dll.GarnetQwenVLBuildSingleImagePromptIds.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_longlong), ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong), ctypes.c_int, ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p, ctypes.c_int,
    ]
    dll.GarnetQwenVLBuildSingleImagePromptIds.restype = ctypes.c_int
    dll.GarnetQwenTokenizerTokenId.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    dll.GarnetQwenTokenizerTokenId.restype = ctypes.c_longlong
    return dll


def percentile(values, p):
    values = sorted(values)
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * p)))
    return values[idx]


def summarize_ms(values):
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values) if values else 0.0,
        "median_ms": statistics.median(values) if values else 0.0,
        "p90_ms": percentile(values, 0.90),
        "min_ms": min(values) if values else 0.0,
        "max_ms": max(values) if values else 0.0,
    }


def call_encode(dll, model_dir_b, text_b):
    count = ctypes.c_int(0)
    error = ctypes.create_string_buffer(512)
    rc = dll.GarnetQwenTokenizerEncode(model_dir_b, text_b, None, 0, ctypes.byref(count), error, len(error))
    if rc not in (0, 2):
        raise AssertionError(f"encode count failed rc={rc}: {error.value.decode(errors='ignore')}")
    out = (ctypes.c_longlong * count.value)()
    rc = dll.GarnetQwenTokenizerEncode(model_dir_b, text_b, out, count.value, ctypes.byref(count), error, len(error))
    if rc != 0:
        raise AssertionError(f"encode failed rc={rc}: {error.value.decode(errors='ignore')}")
    return [int(out[i]) for i in range(count.value)]


def call_prompt(dll, model_dir_b, prompt_b, grid):
    count = ctypes.c_int(0)
    error = ctypes.create_string_buffer(512)
    grid_arr = (ctypes.c_longlong * 3)(*grid)
    rc = dll.GarnetQwenVLBuildSingleImagePromptIds(model_dir_b, prompt_b, grid_arr, 2, None, 0, ctypes.byref(count), error, len(error))
    if rc not in (0, 3):
        raise AssertionError(f"prompt count failed rc={rc}: {error.value.decode(errors='ignore')}")
    out = (ctypes.c_longlong * count.value)()
    rc = dll.GarnetQwenVLBuildSingleImagePromptIds(model_dir_b, prompt_b, grid_arr, 2, out, count.value, ctypes.byref(count), error, len(error))
    if rc != 0:
        raise AssertionError(f"prompt failed rc={rc}: {error.value.decode(errors='ignore')}")
    return [int(out[i]) for i in range(count.value)]


def call_decode(dll, model_dir_b, ids):
    arr = (ctypes.c_longlong * len(ids))(*ids)
    count = ctypes.c_int(0)
    error = ctypes.create_string_buffer(512)
    rc = dll.GarnetQwenTokenizerDecode(model_dir_b, arr, len(ids), 1, None, 0, ctypes.byref(count), error, len(error))
    if rc not in (0, 3):
        raise AssertionError(f"decode count failed rc={rc}: {error.value.decode(errors='ignore')}")
    out = ctypes.create_string_buffer(count.value + 1)
    rc = dll.GarnetQwenTokenizerDecode(model_dir_b, arr, len(ids), 1, out, len(out), ctypes.byref(count), error, len(error))
    if rc != 0:
        raise AssertionError(f"decode failed rc={rc}: {error.value.decode(errors='ignore')}")
    return out.value.decode("utf-8")


def time_calls(fn, iterations):
    timings = []
    last = None
    for _ in range(iterations):
        start = time.perf_counter()
        last = fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings, last


def main():
    model_dir = default_model_dir()
    if model_dir is None or not model_dir.exists():
        skip("Qwen3-VL model directory not found")
    dll = load_dll()
    model_dir_b = str(model_dir).encode("utf-8")
    prompt = os.environ.get("GARNET_QWEN_TOKENIZER_PROFILE_PROMPT", DEFAULT_PROMPT)
    prompt_b = prompt.encode("utf-8")
    grid = [1, 12, 20]
    generated_ids = [32, 883, 23011, 304, 264, 57272, 22360, 75879, 11, 22865]
    iterations = int(os.environ.get("GARNET_QWEN_TOKENIZER_PROFILE_ITERS", "5"))

    first_start = time.perf_counter()
    first_ids = call_prompt(dll, model_dir_b, prompt_b, grid)
    first_ms = (time.perf_counter() - first_start) * 1000.0

    encode_times, encode_ids = time_calls(lambda: call_encode(dll, model_dir_b, prompt_b), iterations)
    prompt_times, prompt_ids = time_calls(lambda: call_prompt(dll, model_dir_b, prompt_b, grid), iterations)
    decode_times, decoded = time_calls(lambda: call_decode(dll, model_dir_b, generated_ids), iterations)

    token_id_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        token_id = int(dll.GarnetQwenTokenizerTokenId(model_dir_b, b"<|image_pad|>"))
        token_id_times.append((time.perf_counter() - start) * 1000.0)
    if token_id != 151655:
        raise AssertionError(f"unexpected image_pad id: {token_id}")

    result = {
        "phase": "phase_12_qwen_tokenizer_profile",
        "model_dir": str(model_dir),
        "iterations": iterations,
        "first_prompt_call_ms": first_ms,
        "encode_plain": summarize_ms(encode_times),
        "build_vl_prompt": summarize_ms(prompt_times),
        "decode_generated": summarize_ms(decode_times),
        "token_id_lookup": summarize_ms(token_id_times),
        "plain_token_count": len(encode_ids),
        "vl_prompt_token_count": len(prompt_ids),
        "first_prompt_token_count": len(first_ids),
        "decoded_text": decoded,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "last_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Phase 12: Qwen tokenizer profile complete.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
