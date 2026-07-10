import ctypes
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGE = REPO_ROOT / "data" / "Dataset.1980Love" / "imgs" / "frame_0.jpg"
DEFAULT_PROMPT = b"Describe this picture in one short sentence."


def add_dll_dirs():
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    for dll_dir in [
        REPO_ROOT / "out" / "build" / "x64-Debug" / "bin",
        REPO_ROOT.parent / "xlang" / "out" / "build" / "x64-Debug" / "bin",
        REPO_ROOT.parent / "out" / "build" / "x64-Debug" / "bin",
        REPO_ROOT.parent / "out" / "build" / "x64-debug" / "bin",
        REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "bin",
        REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "lib",
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin"),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64"),
        Path("C:/Program Files/Microsoft Visual Studio/18/Community/VC/Redist/MSVC/14.51.36231/debug_nonredist/x64/Microsoft.VC145.DebugCRT"),
    ]:
        if dll_dir.exists():
            os.add_dll_directory(str(dll_dir))


def default_model_dir():
    explicit = os.environ.get("HF_QWEN_VL_MODEL_DIR")
    if explicit and Path(explicit).exists():
        return Path(explicit)
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots"
    if cache_root.exists():
        snapshots = sorted(cache_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if snapshots:
            return snapshots[0]
    return None


def configure_prepare(dll):
    fn = dll.GarnetQwenVLPrepareJpegPrompt
    fn.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    fn.restype = ctypes.c_int
    return fn


def main():
    model_dir = default_model_dir()
    if model_dir is None:
        print("SKIP: Qwen3-VL model snapshot not found; set HF_QWEN_VL_MODEL_DIR")
        raise SystemExit(0)
    if not DEFAULT_IMAGE.exists():
        raise AssertionError(f"test image missing: {DEFAULT_IMAGE}")

    add_dll_dirs()
    garnet_dll = REPO_ROOT / "out" / "build" / "x64-Debug" / "bin" / "garnet.dll"
    if not garnet_dll.exists():
        print(f"SKIP: garnet.dll not found: {garnet_dll}")
        raise SystemExit(0)

    dll = ctypes.CDLL(str(garnet_dll))
    prepare = configure_prepare(dll)

    input_count = ctypes.c_int(0)
    pixel_count = ctypes.c_int(0)
    grid = (ctypes.c_longlong * 3)()
    source_h = ctypes.c_int(0)
    source_w = ctypes.c_int(0)
    resized_h = ctypes.c_int(0)
    resized_w = ctypes.c_int(0)
    error = ctypes.create_string_buffer(512)

    rc = prepare(
        str(model_dir).encode("utf-8"),
        str(DEFAULT_IMAGE).encode("utf-8"),
        DEFAULT_PROMPT,
        65536,
        65536,
        None,
        0,
        ctypes.byref(input_count),
        None,
        0,
        None,
        0,
        ctypes.byref(pixel_count),
        grid,
        ctypes.byref(source_h),
        ctypes.byref(source_w),
        ctypes.byref(resized_h),
        ctypes.byref(resized_w),
        error,
        len(error),
    )
    assert rc == 3
    assert input_count.value == 79
    assert pixel_count.value == 240 * 1536
    assert list(grid) == [1, 12, 20]

    input_ids = (ctypes.c_longlong * input_count.value)()
    mm_types = (ctypes.c_longlong * input_count.value)()
    pixel_values = (ctypes.c_float * pixel_count.value)()
    error2 = ctypes.create_string_buffer(512)
    rc = prepare(
        str(model_dir).encode("utf-8"),
        str(DEFAULT_IMAGE).encode("utf-8"),
        DEFAULT_PROMPT,
        65536,
        65536,
        input_ids,
        input_count.value,
        ctypes.byref(input_count),
        mm_types,
        input_count.value,
        pixel_values,
        pixel_count.value,
        ctypes.byref(pixel_count),
        grid,
        ctypes.byref(source_h),
        ctypes.byref(source_w),
        ctypes.byref(resized_h),
        ctypes.byref(resized_w),
        error2,
        len(error2),
    )
    assert rc == 0, error2.value.decode(errors="ignore")
    assert input_count.value == 79
    assert pixel_count.value == 240 * 1536
    assert list(grid) == [1, 12, 20]
    assert source_h.value == 1080
    assert source_w.value == 1920
    assert resized_h.value == 192
    assert resized_w.value == 320
    assert sum(1 for item in mm_types if item == 1) == 60
    assert [input_ids[i] for i in range(4)] == [151644, 872, 198, 151652]
    assert abs(float(pixel_values[0]) - (-1.0)) < 1e-6

    try:
        import xlang
    except Exception as exc:
        print(f"SKIP xlang API check: xlang is not available: {exc}")
        print("Phase 14: one-call Qwen-VL JPEG+prompt request prepare passed.")
        return

    garnet = xlang.importModule("garnet", fromPath=str(garnet_dll))
    prepared = garnet.qwen_vl_prepare_request(
        model_dir=str(model_dir),
        image_path=str(DEFAULT_IMAGE),
        prompt=DEFAULT_PROMPT.decode("utf-8"),
        min_pixels=65536,
        max_pixels=65536,
    )
    assert prepared["backend"] == "qwen_vl_request_native_tokenizer_nvjpeg_cuda"
    assert prepared["prompt_token_count"] == 79
    assert prepared["visual_token_count"] == 60
    assert prepared["pixel_value_count"] == 240 * 1536
    assert prepared["source_height"] == 1080
    assert prepared["source_width"] == 1920
    assert prepared["height"] == 192
    assert prepared["width"] == 320
    assert list(prepared["input_ids"])[:4] == [151644, 872, 198, 151652]
    assert list(prepared["image_grid_thw"]) == [1, 12, 20]
    assert sum(1 for item in prepared["mm_token_type_ids"] if item == 1) == 60
    assert list(prepared["pixel_values_shape"]) == [240, 1536]

    print("Phase 14: one-call Qwen-VL JPEG+prompt request prepare passed.")


if __name__ == "__main__":
    main()
