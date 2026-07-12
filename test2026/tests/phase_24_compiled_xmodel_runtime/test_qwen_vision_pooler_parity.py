import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
ARTIFACT_DIR = REPO_ROOT / "test2026" / "artifacts" / "qwen_vision_pooler_parity"
IMAGE_PATH = REPO_ROOT / "data" / "Dataset.1980Love" / "imgs" / "frame_0.jpg"
SNAPSHOTS = sorted((
    Path.home() / ".cache" / "huggingface" / "hub" /
    "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots"
).glob("*"))


def run_garnet():
    dll = REPO_ROOT / "out" / "build" / "x64-Release" / "bin" / "garnet.dll"
    handles = []
    for directory in [
        dll.parent,
        REPO_ROOT.parent / "xlang" / "out" / "build" / "x64-Release" / "bin",
        REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "bin",
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin"),
    ]:
        if directory.exists() and hasattr(os, "add_dll_directory"):
            handles.append(os.add_dll_directory(str(directory)))
    import xlang

    garnet = xlang.importModule("garnet", fromPath=str(dll))
    prepared = garnet.qwen_vl_prepare_request(
        model_dir=str(SNAPSHOTS[-1]),
        image_path=str(IMAGE_PATH),
        prompt="Describe.",
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    pixel_values = np.asarray(
        garnet.tensor_to_cpu(prepared["pixel_values"]).toarray(), dtype=np.float32
    ).reshape(3772, 1536)
    np.save(ARTIFACT_DIR / "pixel_values.npy", pixel_values)
    model = garnet.load_model(
        str(REPO_ROOT / "qwen_vl" / "xmodel" / "debug_vision_pooler.x"),
        runtime_mode="compiled_xmodel",
        entry_function="Qwen3VisionPoolerProbe",
        weights=str(SNAPSHOTS[-1]),
        input_shapes=[
            [3772, 1536], [1, 3], [3772, 4], [3772, 4], [3772, 2], [2],
        ],
        input_dtypes=[
            "bfloat16", "int64", "int64", "bfloat16", "int64", "int32",
        ],
        cache_dir=str(SCRIPT_DIR / "qwen_vision_pooler_parity_cache"),
    )
    result = model.forward({"inputs": [
        garnet.tensor_to_bfloat16(prepared["pixel_values"]),
        garnet.tensor_from_host([1, 46, 82], dtype="int64", shape=[1, 3]),
        prepared["vision_bilinear_indices"],
        prepared["vision_bilinear_weights"],
        prepared["vision_position_ids"],
        prepared["vision_cu_seqlens"],
    ]})
    assert result["status"] == "ok", result
    output = np.asarray(garnet.tensor_to_cpu(result["output"]).toarray())
    if output.dtype == np.uint16:
        output = (output.astype(np.uint32) << 16).view(np.float32)
    output = output.astype(np.float32).reshape(943, 2048)
    np.save(ARTIFACT_DIR / "garnet_pooler.npy", output)


def run_hf():
    import torch
    from transformers import AutoModelForImageTextToText

    pixels = np.load(ARTIFACT_DIR / "pixel_values.npy")
    model = AutoModelForImageTextToText.from_pretrained(
        str(SNAPSHOTS[-1]), dtype=torch.bfloat16
    ).to("cuda").eval()
    with torch.inference_mode():
        output = model.model.visual(
            torch.from_numpy(pixels).to("cuda", dtype=torch.bfloat16),
            grid_thw=torch.tensor([[1, 46, 82]], device="cuda"),
        ).pooler_output.float().cpu().numpy()
    actual = np.load(ARTIFACT_DIR / "garnet_pooler.npy")
    difference = np.abs(actual - output)
    result = {
        "shape": list(actual.shape),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "cosine_similarity": float(
            np.sum(actual * output) /
            (np.linalg.norm(actual) * np.linalg.norm(output))
        ),
    }
    assert result["cosine_similarity"] >= 0.995, result
    assert result["mean_abs_error"] <= 0.03, result
    (ARTIFACT_DIR / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    assert SNAPSHOTS and IMAGE_PATH.exists()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) == 2 and sys.argv[1] == "--garnet":
        run_garnet()
    elif len(sys.argv) == 2 and sys.argv[1] == "--hf":
        run_hf()
    else:
        subprocess.run([sys.executable, __file__, "--garnet"], check=True)
        subprocess.run([sys.executable, __file__, "--hf"], check=True)
