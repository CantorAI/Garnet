import ctypes
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / "test2026" / "artifacts" / "qwen_tokenizer_parity"
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
    garnet_dll = Path(os.environ.get("GARNET_DLL_PATH", REPO_ROOT / "out" / "build" / "x64-Release" / "bin" / "garnet.dll"))
    if not garnet_dll.exists():
        skip(f"garnet.dll not found: {garnet_dll}")
    add_dll_dirs(garnet_dll)
    dll = ctypes.CDLL(str(garnet_dll))
    dll.GarnetQwenTokenizerEncode.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    dll.GarnetQwenTokenizerEncode.restype = ctypes.c_int
    dll.GarnetQwenTokenizerDecode.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    dll.GarnetQwenTokenizerDecode.restype = ctypes.c_int
    dll.GarnetQwenVLBuildSingleImagePromptIds.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    dll.GarnetQwenVLBuildSingleImagePromptIds.restype = ctypes.c_int
    dll.GarnetQwenTokenizerTokenId.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    dll.GarnetQwenTokenizerTokenId.restype = ctypes.c_longlong
    return dll


def encode_native(dll, model_dir, text):
    count = ctypes.c_int(0)
    error = ctypes.create_string_buffer(512)
    rc = dll.GarnetQwenTokenizerEncode(
        str(model_dir).encode("utf-8"),
        text.encode("utf-8"),
        None,
        0,
        ctypes.byref(count),
        error,
        len(error),
    )
    if rc not in (0, 2):
        raise AssertionError(f"count encode failed rc={rc}: {error.value.decode(errors='ignore')}")
    output = (ctypes.c_longlong * count.value)()
    rc = dll.GarnetQwenTokenizerEncode(
        str(model_dir).encode("utf-8"),
        text.encode("utf-8"),
        output,
        count.value,
        ctypes.byref(count),
        error,
        len(error),
    )
    if rc != 0:
        raise AssertionError(f"encode failed rc={rc}: {error.value.decode(errors='ignore')}")
    return [int(output[i]) for i in range(count.value)]


def build_prompt_native(dll, model_dir, prompt, grid):
    count = ctypes.c_int(0)
    error = ctypes.create_string_buffer(512)
    grid_arr = (ctypes.c_longlong * 3)(*grid)
    rc = dll.GarnetQwenVLBuildSingleImagePromptIds(
        str(model_dir).encode("utf-8"),
        prompt.encode("utf-8"),
        grid_arr,
        2,
        None,
        0,
        ctypes.byref(count),
        error,
        len(error),
    )
    if rc not in (0, 3):
        raise AssertionError(f"count build prompt failed rc={rc}: {error.value.decode(errors='ignore')}")
    output = (ctypes.c_longlong * count.value)()
    rc = dll.GarnetQwenVLBuildSingleImagePromptIds(
        str(model_dir).encode("utf-8"),
        prompt.encode("utf-8"),
        grid_arr,
        2,
        output,
        count.value,
        ctypes.byref(count),
        error,
        len(error),
    )
    if rc != 0:
        raise AssertionError(f"build prompt failed rc={rc}: {error.value.decode(errors='ignore')}")
    return [int(output[i]) for i in range(count.value)]


def decode_native(dll, model_dir, ids, skip_special=True):
    arr = (ctypes.c_longlong * len(ids))(*ids)
    byte_count = ctypes.c_int(0)
    error = ctypes.create_string_buffer(512)
    rc = dll.GarnetQwenTokenizerDecode(
        str(model_dir).encode("utf-8"),
        arr,
        len(ids),
        1 if skip_special else 0,
        None,
        0,
        ctypes.byref(byte_count),
        error,
        len(error),
    )
    if rc not in (0, 3):
        raise AssertionError(f"count decode failed rc={rc}: {error.value.decode(errors='ignore')}")
    output = ctypes.create_string_buffer(byte_count.value + 1)
    rc = dll.GarnetQwenTokenizerDecode(
        str(model_dir).encode("utf-8"),
        arr,
        len(ids),
        1 if skip_special else 0,
        output,
        len(output),
        ctypes.byref(byte_count),
        error,
        len(error),
    )
    if rc != 0:
        raise AssertionError(f"decode failed rc={rc}: {error.value.decode(errors='ignore')}")
    return output.value.decode("utf-8")


def main():
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        skip(f"phase 11 requires transformers for oracle: {exc}")

    model_dir = default_model_dir()
    if model_dir is None or not model_dir.exists():
        skip("Qwen3-VL model directory not found")
    dll = load_dll()
    hf = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    samples = [
        "Describe this picture in one short sentence.",
        "user\nDescribe this picture in one short sentence.\n",
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
        "A man sits in a rustic wooden hut, surrounded",
    ]
    sample_results = []
    for text in samples:
        native = encode_native(dll, model_dir, text)
        expected = hf.encode(text, add_special_tokens=False)
        if native != expected:
            raise AssertionError(f"encode mismatch for {text!r}\nnative={native}\nhf={expected}")
        sample_results.append({"text": text, "ids": native})

    special_tokens = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]
    special_ids = {}
    for token in special_tokens:
        native_id = int(dll.GarnetQwenTokenizerTokenId(str(model_dir).encode("utf-8"), token.encode("utf-8")))
        hf_id = hf.encode(token, add_special_tokens=False)[0]
        if native_id != hf_id:
            raise AssertionError(f"special id mismatch for {token}: {native_id} != {hf_id}")
        special_ids[token] = native_id

    grid = [1, 12, 20]
    prompt = os.environ.get("GARNET_QWEN_TOKENIZER_TEST_PROMPT", DEFAULT_PROMPT)
    visual_count = grid[0] * grid[1] * grid[2] // 4
    prompt_text = (
        "<|im_start|>user\n"
        "<|vision_start|>"
        + "<|image_pad|>" * visual_count
        + "<|vision_end|>"
        + prompt
        + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    native_prompt = build_prompt_native(dll, model_dir, prompt, grid)
    hf_prompt = hf.encode(prompt_text, add_special_tokens=False)
    if native_prompt != hf_prompt:
        raise AssertionError(
            f"VL prompt mismatch len native={len(native_prompt)} hf={len(hf_prompt)}\n"
            f"first diff={next((i for i,(a,b) in enumerate(zip(native_prompt,hf_prompt)) if a!=b), None)}"
        )

    generated_ids = [32, 883, 23011, 304, 264, 57272, 22360, 75879, 11, 22865]
    native_decode = decode_native(dll, model_dir, generated_ids, skip_special=True)
    hf_decode = hf.decode(generated_ids, skip_special_tokens=True)
    if native_decode != hf_decode:
        raise AssertionError(f"decode mismatch: {native_decode!r} != {hf_decode!r}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "phase": "phase_11_qwen_tokenizer_parity",
        "model_dir": str(model_dir),
        "samples": sample_results,
        "special_ids": special_ids,
        "image_grid_thw": [grid],
        "visual_token_count": visual_count,
        "vl_prompt_token_count": len(native_prompt),
        "generated_ids": generated_ids,
        "decoded_text": native_decode,
    }
    (ARTIFACT_DIR / "last_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Phase 11: Qwen tokenizer parity passed.")
    print(f"vl_prompt_token_count={len(native_prompt)}")
    print(f"decoded_text={native_decode}")
    print(f"result={ARTIFACT_DIR / 'last_result.json'}")


if __name__ == "__main__":
    main()
