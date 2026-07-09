import json
import os
import sys
from pathlib import Path
import ctypes

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / "test2026" / "artifacts" / "qwen_vl_gpu_image_preprocess"
DEFAULT_IMAGE = REPO_ROOT / "data" / "Dataset.1980Love" / "imgs" / "frame_0.jpg"


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


def load_garnet_dll():
    garnet_dll = Path(os.environ.get("GARNET_DLL_PATH", REPO_ROOT / "out" / "build" / "x64-Debug" / "bin" / "garnet.dll"))
    if not garnet_dll.exists():
        skip(f"garnet.dll not found: {garnet_dll}")
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
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
    dll = ctypes.CDLL(str(garnet_dll))
    dll.GarnetQwenVLPreprocessRGBF32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    dll.GarnetQwenVLPreprocessRGBF32.restype = ctypes.c_int
    return dll


def to_numpy(value, dtype=None):
    if hasattr(value, "numpy"):
        array = value.numpy()
    elif hasattr(value, "tolist"):
        array = np.asarray(value.tolist())
    else:
        array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def smart_resize(height, width, factor=32, min_pixels=65536, max_pixels=65536):
    import math

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def qwen_patch_layout_numpy(raw_rgb, patch_size=16, temporal_patch_size=2, merge_size=2):
    patches = (raw_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    patches = np.transpose(patches, (2, 0, 1))[None, ...]
    batch_size, channel, resized_h, resized_w = patches.shape
    grid_h, grid_w = resized_h // patch_size, resized_w // patch_size
    patches = patches.reshape(
        batch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = np.transpose(patches, (0, 2, 5, 3, 6, 1, 4, 7))
    flatten_patches = (
        np.repeat(patches[:, :, :, :, :, :, None, :, :], temporal_patch_size, axis=6)
        .reshape(batch_size, grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)
    )
    return flatten_patches.reshape(grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size).astype(np.float32)


def main():
    try:
        from PIL import Image
        from transformers import AutoProcessor
    except Exception as exc:
        skip(f"phase 07 requires PIL and transformers for the test oracle/upstream image provider: {exc}")

    image_path = Path(os.environ.get("GARNET_QWEN_IMAGE_PREPROCESS_TEST_IMAGE", DEFAULT_IMAGE))
    if not image_path.exists():
        raise AssertionError(f"test image does not exist: {image_path}")
    model_dir = default_model_dir()
    if model_dir is None or not model_dir.exists():
        skip("Qwen3-VL model directory not found")

    max_pixels = int(os.environ.get("GARNET_QWEN_IMAGE_PREPROCESS_TEST_PIXELS", "65536"))
    image = Image.open(image_path).convert("RGB")
    resized_h, resized_w = smart_resize(image.height, image.width, min_pixels=max_pixels, max_pixels=max_pixels)

    resized = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
    raw_rgb = np.asarray(resized, dtype=np.float32)

    dll = load_garnet_dll()
    patch_count = (resized_h // 16) * (resized_w // 16)
    feature_dim = 3 * 2 * 16 * 16
    actual_pixel_values = np.empty((patch_count, feature_dim), dtype=np.float32)
    actual_grid = np.zeros((1, 3), dtype=np.int64)
    error = ctypes.create_string_buffer(512)
    rc = dll.GarnetQwenVLPreprocessRGBF32(
        raw_rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        resized_h,
        resized_w,
        3,
        0,
        ctypes.c_float(255.0),
        actual_pixel_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        actual_grid.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        error,
        len(error),
    )
    if rc != 0:
        raise AssertionError(f"GarnetQwenVLPreprocessRGBF32 failed rc={rc}: {error.value.decode(errors='ignore')}")

    expected_pixel_values = qwen_patch_layout_numpy(raw_rgb)
    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        min_pixels=max_pixels,
        max_pixels=max_pixels,
    )
    hf_inputs = processor(text=[""], images=[image], return_tensors="pt")
    expected_grid = hf_inputs["image_grid_thw"].detach().cpu().numpy().astype(np.int64)
    hf_pixel_values = hf_inputs["pixel_values"].detach().cpu().numpy().astype(np.float32)

    if actual_pixel_values.shape != expected_pixel_values.shape:
        raise AssertionError(f"pixel_values shape mismatch: {actual_pixel_values.shape} != {expected_pixel_values.shape}")
    np.testing.assert_array_equal(actual_grid, expected_grid)

    diff = np.abs(actual_pixel_values - expected_pixel_values)
    max_abs_error = float(diff.max()) if diff.size else 0.0
    mean_abs_error = float(diff.mean()) if diff.size else 0.0
    hf_resize_diff = np.abs(actual_pixel_values - hf_pixel_values)
    hf_resize_max_abs_error = float(hf_resize_diff.max()) if hf_resize_diff.size else 0.0
    tolerance = float(os.environ.get("GARNET_QWEN_IMAGE_PREPROCESS_TOLERANCE", "0.0001"))
    if max_abs_error > tolerance:
        raise AssertionError(f"pixel_values mismatch: max_abs_error={max_abs_error} > {tolerance}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "phase": "phase_07_qwen_vl_gpu_image_preprocess",
        "image": str(image_path),
        "model_dir": str(model_dir),
        "backend": "cuda_raw_tensor_c_abi",
        "input_mode": "upstream_resized_rgb_float32_hwc_to_garnet_cuda_preprocess",
        "resized_height": resized_h,
        "resized_width": resized_w,
        "pixel_values_shape": list(actual_pixel_values.shape),
        "image_grid_thw": actual_grid.tolist(),
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "hf_end_to_end_resize_max_abs_error": hf_resize_max_abs_error,
    }
    (ARTIFACT_DIR / "last_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Phase 07: Qwen-VL Garnet CUDA image preprocess passed.")
    print(f"image={image_path}")
    print(f"resized={resized_h}x{resized_w}")
    print(f"pixel_values_shape={list(actual_pixel_values.shape)}")
    print(f"max_abs_error={max_abs_error}")
    print(f"result={ARTIFACT_DIR / 'last_result.json'}")


if __name__ == "__main__":
    main()
