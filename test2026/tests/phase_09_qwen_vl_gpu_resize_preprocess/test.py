# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import json
import os
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / "test2026" / "artifacts" / "qwen_vl_gpu_resize_preprocess"
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


def add_dll_dirs(garnet_dll):
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    for dll_dir in [
        garnet_dll.parent,
        REPO_ROOT.parent / "out" / "build" / "x64-Debug" / "bin",
        REPO_ROOT.parent / "out" / "build" / "x64-debug" / "bin",
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin"),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64"),
        REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "bin",
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
    dll.GarnetQwenVLResizePreprocessRGBF32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    dll.GarnetQwenVLResizePreprocessRGBF32.restype = ctypes.c_int
    return dll


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


def main():
    try:
        from PIL import Image
        from transformers import AutoProcessor
    except Exception as exc:
        skip(f"phase 09 requires PIL and transformers for test oracle: {exc}")

    image_path = Path(os.environ.get("GARNET_QWEN_IMAGE_RESIZE_TEST_IMAGE", DEFAULT_IMAGE))
    if not image_path.exists():
        raise AssertionError(f"test image does not exist: {image_path}")
    model_dir = default_model_dir()
    if model_dir is None or not model_dir.exists():
        skip("Qwen3-VL model directory not found")

    max_pixels = int(os.environ.get("GARNET_QWEN_IMAGE_RESIZE_TEST_PIXELS", "65536"))
    image = Image.open(image_path).convert("RGB")
    raw_rgb = np.ascontiguousarray(np.asarray(image, dtype=np.float32))
    resized_h, resized_w = smart_resize(image.height, image.width, min_pixels=max_pixels, max_pixels=max_pixels)
    patch_count = (resized_h // 16) * (resized_w // 16)
    feature_dim = 3 * 2 * 16 * 16
    pixel_values = np.empty((patch_count, feature_dim), dtype=np.float32)
    image_grid = np.zeros((1, 3), dtype=np.int64)
    out_h = ctypes.c_int(0)
    out_w = ctypes.c_int(0)
    error = ctypes.create_string_buffer(512)

    dll = load_dll()
    rc = dll.GarnetQwenVLResizePreprocessRGBF32(
        raw_rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        image.height,
        image.width,
        3,
        0,
        max_pixels,
        max_pixels,
        ctypes.c_float(255.0),
        pixel_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        image_grid.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        ctypes.byref(out_h),
        ctypes.byref(out_w),
        error,
        len(error),
    )
    if rc != 0:
        raise AssertionError(f"GarnetQwenVLResizePreprocessRGBF32 failed rc={rc}: {error.value.decode(errors='ignore')}")

    if out_h.value != resized_h or out_w.value != resized_w:
        raise AssertionError(f"resize mismatch: got {out_h.value}x{out_w.value}, expected {resized_h}x{resized_w}")

    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True, min_pixels=max_pixels, max_pixels=max_pixels)
    hf_inputs = processor(text=[""], images=[image], return_tensors="pt")
    hf_pixel_values = hf_inputs["pixel_values"].detach().cpu().numpy().astype(np.float32)
    hf_grid = hf_inputs["image_grid_thw"].detach().cpu().numpy().astype(np.int64)
    np.testing.assert_array_equal(image_grid, hf_grid)
    if pixel_values.shape != hf_pixel_values.shape:
        raise AssertionError(f"shape mismatch: {pixel_values.shape} != {hf_pixel_values.shape}")

    diff = np.abs(pixel_values - hf_pixel_values)
    max_abs_error = float(diff.max()) if diff.size else 0.0
    mean_abs_error = float(diff.mean()) if diff.size else 0.0
    mean_tolerance = float(os.environ.get("GARNET_QWEN_IMAGE_RESIZE_MEAN_TOLERANCE", "0.08"))
    if mean_abs_error > mean_tolerance:
        raise AssertionError(f"resize+preprocess mismatch: mean_abs_error={mean_abs_error} > {mean_tolerance}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "phase": "phase_09_qwen_vl_gpu_resize_preprocess",
        "image": str(image_path),
        "backend": "cuda_resize_preprocess_c_abi",
        "input_mode": "original_rgb_float32_hwc_to_garnet_cuda_resize_preprocess",
        "source_height": image.height,
        "source_width": image.width,
        "resized_height": out_h.value,
        "resized_width": out_w.value,
        "pixel_values_shape": list(pixel_values.shape),
        "image_grid_thw": image_grid.tolist(),
        "max_abs_error_vs_hf_processor": max_abs_error,
        "mean_abs_error_vs_hf_processor": mean_abs_error,
        "mean_tolerance": mean_tolerance,
    }
    (ARTIFACT_DIR / "last_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Phase 09: Qwen-VL Garnet CUDA resize+preprocess passed.")
    print(f"image={image_path}")
    print(f"source={image.height}x{image.width}")
    print(f"resized={out_h.value}x{out_w.value}")
    print(f"pixel_values_shape={list(pixel_values.shape)}")
    print(f"max_abs_error_vs_hf={max_abs_error}")
    print(f"result={ARTIFACT_DIR / 'last_result.json'}")


if __name__ == "__main__":
    main()
