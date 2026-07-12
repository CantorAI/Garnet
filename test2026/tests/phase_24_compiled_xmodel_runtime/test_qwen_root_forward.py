import os
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
GARNET_DLL = REPO_ROOT / "out" / "build" / "x64-Release" / "bin" / "garnet.dll"
ENGINE_PATH = SCRIPT_DIR / "qwen_root_cache" / "model.engine"

if not ENGINE_PATH.exists():
    print("Qwen root forward skipped; build it with GARNET_REBUILD_QWEN_ROOT=1 first")
    raise SystemExit(0)

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


def bfloat16_tensor(garnet, array):
    fp32 = np.asarray(array, dtype=np.float32)
    bits = (fp32.view(np.uint32) >> 16).astype(np.uint16)
    return garnet.tensor_from_bfloat16_bits(bits)


def tensor_array(garnet, tensor, shape):
    return np.asarray(garnet.tensor_to_cpu(tensor).toarray()).reshape(shape)


snapshot_root = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3-VL-2B-Instruct"
    / "snapshots"
)
snapshots = sorted(snapshot_root.glob("*"))
assert snapshots, "local Qwen3-VL-2B-Instruct snapshot is required"

garnet = xlang.importModule("garnet", fromPath=str(GARNET_DLL))
model = garnet.load_model(
    str(REPO_ROOT / "qwen_vl" / "xmodel" / "qwen_vl_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Qwen3VLModel",
    frontend="qwen3_vl",
    weights=str(snapshots[-1]),
    input_shapes=[
        [1, 16], [4, 1536], [1, 3], [4, 4], [4, 4], [4, 2],
        [2], [1, 16], [1, 16], [3, 1, 16], [1, 1],
    ],
    input_dtypes=[
        "int64", "bfloat16", "int64", "int64", "bfloat16",
        "int64", "int32", "int64", "int64", "int64", "int64",
    ],
    cache_dir=str(SCRIPT_DIR / "qwen_root_cache"),
)
status = model.runtime_status()
assert bool(status["ready"]), status

# Debug-only BF16 paged-KV fixture with a non-identity physical page table.
rng = np.random.default_rng(17)
kv_tokens, kv_page_size, kv_q_heads, kv_heads, kv_head_dim = 5, 4, 4, 2, 8
kv_q_width = kv_q_heads * kv_head_dim
kv_width = kv_heads * kv_head_dim
qkv_source = rng.normal(
    0.0, 0.2, size=(kv_tokens, kv_q_width + 2 * kv_width)
).astype(np.float32)
qkv_bits = (qkv_source.view(np.uint32) >> 16).astype(np.uint16)
qkv_bf16 = (qkv_bits.astype(np.uint32) << 16).view(np.float32)
page_shape = (2, kv_page_size, kv_heads, kv_head_dim)
key_pages = bfloat16_tensor(garnet, np.zeros(page_shape, dtype=np.float32))
value_pages = bfloat16_tensor(garnet, np.zeros(page_shape, dtype=np.float32))
page_table = [1, 0]
kv_probe = model.debug_probe("paged_kv_bf16", {
    "qkv": garnet.tensor_from_bfloat16_bits(qkv_bits),
    "key_pages": key_pages,
    "value_pages": value_pages,
    "page_table": page_table,
    "token_count": kv_tokens,
    "start_position": 0,
    "sequence_length": kv_tokens,
    "page_size": kv_page_size,
    "q_heads": kv_q_heads,
    "kv_heads": kv_heads,
    "head_dim": kv_head_dim,
})
assert kv_probe["status"] == "ok", kv_probe
actual_kv = tensor_array(garnet, kv_probe["output"], (kv_q_heads, kv_head_dim)).astype(np.float32)
logical_keys = qkv_bf16[:, kv_q_width:kv_q_width + kv_width].reshape(
    kv_tokens, kv_heads, kv_head_dim
)
logical_values = qkv_bf16[:, kv_q_width + kv_width:].reshape(
    kv_tokens, kv_heads, kv_head_dim
)
expected_kv = np.empty((kv_q_heads, kv_head_dim), dtype=np.float32)
for q_head in range(kv_q_heads):
    kv_head = q_head // (kv_q_heads // kv_heads)
    query = qkv_bf16[-1, q_head * kv_head_dim:(q_head + 1) * kv_head_dim]
    scores = logical_keys[:, kv_head] @ query / np.sqrt(kv_head_dim)
    probabilities = np.exp(scores - scores.max())
    probabilities /= probabilities.sum()
    expected_kv[q_head] = probabilities @ logical_values[:, kv_head]
expected_bits = (expected_kv.view(np.uint32) >> 16).astype(np.uint16)
expected_kv_bf16 = (expected_bits.astype(np.uint32) << 16).view(np.float32)
np.testing.assert_allclose(actual_kv, expected_kv_bf16, rtol=0, atol=0.002)
print("BF16 paged-KV write/read debug probe passed")

input_ids = np.zeros((1, 16), dtype=np.int64)
input_ids[0, 4] = 151655
pixel_values = bfloat16_tensor(garnet, np.zeros((4, 1536), dtype=np.float32))
bilinear_weights = np.zeros((4, 4), dtype=np.float32)
bilinear_weights[0, :] = 1.0
position_ids = np.broadcast_to(np.arange(16, dtype=np.int64), (3, 1, 16)).copy()

request = {"inputs": [
    input_ids,
    pixel_values,
    np.array([[1, 2, 2]], dtype=np.int64),
    np.zeros((4, 4), dtype=np.int64),
    bfloat16_tensor(garnet, bilinear_weights),
    np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64),
    np.array([0, 4], dtype=np.int64),
    np.zeros((1, 16), dtype=np.int64),
    np.ones((1, 16), dtype=np.int64),
    position_ids,
    np.zeros((1, 1), dtype=np.int64),
]}

cold_start = time.perf_counter()
result = model.forward(request)
assert result["status"] == "ok", result
logits = garnet.tensor_to_cpu(result["output"])
cold_ms = (time.perf_counter() - cold_start) * 1000.0
logits_array = np.asarray(logits.toarray()).reshape(1, 16, 151936)
assert np.isfinite(logits_array).all()

warm_start = time.perf_counter()
warm_result = model.forward(request)
assert warm_result["status"] == "ok", warm_result
warm_logits = garnet.tensor_to_cpu(warm_result["output"])
warm_ms = (time.perf_counter() - warm_start) * 1000.0
warm_array = np.asarray(warm_logits.toarray()).reshape(1, 16, 151936)
assert np.isfinite(warm_array).all()
print(
    f"Qwen stripped-plan forward passed: logits_shape={logits_array.shape}, "
    f"cold_refit_and_forward_ms={cold_ms:.2f}, warm_forward_ms={warm_ms:.2f}"
)

frame_path = REPO_ROOT / "data" / "Dataset.1980Love" / "imgs" / "frame_0.jpg"
if frame_path.exists():
    metadata_fixture = garnet.qwen_vl_prepare_request(
        model_dir=str(snapshots[-1]),
        image_path=str(frame_path),
        prompt="Describe.",
        min_pixels=4096,
        max_pixels=4096,
    )
    fixture_types = np.asarray(list(metadata_fixture["mm_token_type_ids"]), dtype=np.int64)
    fixture_count = int(fixture_types.size)
    fixture_positions = tensor_array(
        garnet, metadata_fixture["position_ids"], (3, 1, fixture_count)
    ).astype(np.int64)[:, 0]
    visual_indices = np.flatnonzero(fixture_types == 1)
    fixture_grid = np.asarray(list(metadata_fixture["image_grid_thw"]), dtype=np.int64)
    llm_grid_t = int(fixture_grid[0])
    llm_grid_h = int(fixture_grid[1]) // 2
    llm_grid_w = int(fixture_grid[2]) // 2
    assert visual_indices.size == llm_grid_t * llm_grid_h * llm_grid_w
    assert visual_indices.size > 1
    visual_start = int(visual_indices[0])
    expected_visual_positions = []
    for t in range(llm_grid_t):
        for h in range(llm_grid_h):
            for w in range(llm_grid_w):
                expected_visual_positions.append(
                    [visual_start + t, visual_start + h, visual_start + w]
                )
    np.testing.assert_array_equal(
        fixture_positions[:, visual_indices],
        np.asarray(expected_visual_positions, dtype=np.int64).T,
    )
    fixture_delta = int(
        tensor_array(garnet, metadata_fixture["mrope_position_deltas"], (1, 1))[0, 0]
    )
    assert fixture_delta == int(fixture_positions.max() + 1 - fixture_count)
    print("native Qwen text MRoPE multi-token fixture passed")

    real_start = time.perf_counter()
    prepared = garnet.qwen_vl_prepare_request(
        model_dir=str(snapshots[-1]),
        image_path=str(frame_path),
        prompt="Describe.",
        min_pixels=1024,
        max_pixels=1024,
    )
    assert prepared is not None
    real_ids = np.asarray(list(prepared["input_ids"]), dtype=np.int64)
    real_mm_types = np.asarray(list(prepared["mm_token_type_ids"]), dtype=np.int64)
    assert real_ids.size <= 16
    token_count = int(real_ids.size)
    padded_ids = np.zeros((1, 16), dtype=np.int64)
    padded_ids[0, :token_count] = real_ids
    padded_mm_types = np.zeros((1, 16), dtype=np.int64)
    padded_mm_types[0, :token_count] = real_mm_types
    real_attention_mask = np.zeros((1, 16), dtype=np.int64)
    real_attention_mask[0, :token_count] = 1
    real_pixels = garnet.tensor_to_bfloat16(prepared["pixel_values"])
    assert real_pixels is not None
    metadata_positions = tensor_array(
        garnet, prepared["vision_position_ids"], (4, 2)
    ).astype(np.int64)
    metadata_cu_seqlens = tensor_array(
        garnet, prepared["vision_cu_seqlens"], (2,)
    ).astype(np.int32)
    metadata_weights = tensor_array(
        garnet, prepared["vision_bilinear_weights"], (4, 4)
    ).astype(np.float32)
    np.testing.assert_array_equal(
        metadata_positions,
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64),
    )
    np.testing.assert_array_equal(metadata_cu_seqlens, np.array([0, 4], dtype=np.int32))
    np.testing.assert_allclose(metadata_weights[0], np.ones(4), rtol=0, atol=0)
    np.testing.assert_allclose(metadata_weights[1:], np.zeros((3, 4)), rtol=0, atol=0)
    native_position_ids = tensor_array(
        garnet, prepared["position_ids"], (3, 1, token_count)
    ).astype(np.int64)
    real_position_ids = np.zeros((3, 1, 16), dtype=np.int64)
    real_position_ids[:, :, :token_count] = native_position_ids
    real_inputs = [
        padded_ids,
        real_pixels,
        np.array([[1, 2, 2]], dtype=np.int64),
        prepared["vision_bilinear_indices"],
        prepared["vision_bilinear_weights"],
        prepared["vision_position_ids"],
        prepared["vision_cu_seqlens"],
        padded_mm_types,
        real_attention_mask,
        real_position_ids,
        prepared["mrope_position_deltas"],
    ]
    real_result = model.forward({"inputs": real_inputs})
    assert real_result["status"] == "ok", real_result
    real_logits = garnet.tensor_to_cpu(real_result["output"])
    real_array = np.asarray(real_logits.toarray()).reshape(1, 16, 151936)
    assert np.isfinite(real_array).all()
    next_token_id = int(np.argmax(real_array[0, token_count - 1]))
    real_ms = (time.perf_counter() - real_start) * 1000.0
    print(
        f"real JPEG native frontend to logits passed: prompt_tokens={token_count}, "
        f"next_token_id={next_token_id}, total_ms={real_ms:.2f}"
    )

    steady_start = time.perf_counter()
    prepared_again = garnet.qwen_vl_prepare_request(
        model_dir=str(snapshots[-1]),
        image_path=str(frame_path),
        prompt="Describe.",
        min_pixels=1024,
        max_pixels=1024,
    )
    real_inputs[1] = garnet.tensor_to_bfloat16(prepared_again["pixel_values"])
    real_inputs[3] = prepared_again["vision_bilinear_indices"]
    real_inputs[4] = prepared_again["vision_bilinear_weights"]
    real_inputs[5] = prepared_again["vision_position_ids"]
    real_inputs[6] = prepared_again["vision_cu_seqlens"]
    next_position_ids = tensor_array(
        garnet, prepared_again["position_ids"], (3, 1, token_count)
    ).astype(np.int64)
    real_position_ids[:, :, :token_count] = next_position_ids
    real_inputs[9] = real_position_ids
    real_inputs[10] = prepared_again["mrope_position_deltas"]
    steady_result = model.forward({"inputs": real_inputs})
    assert steady_result["status"] == "ok", steady_result
    steady_logits = garnet.tensor_to_cpu(steady_result["output"])
    steady_array = np.asarray(steady_logits.toarray()).reshape(1, 16, 151936)
    assert np.isfinite(steady_array).all()
    steady_ms = (time.perf_counter() - steady_start) * 1000.0
    print(f"real JPEG steady frontend to logits: total_ms={steady_ms:.2f}")

    one_call_start = time.perf_counter()
    one_call_result = model.forward({
        "image_path": str(frame_path),
        "prompt": "Describe.",
        "min_pixels": 1024,
        "max_pixels": 1024,
        "sample": "greedy",
        "return_logits": True,
    })
    assert one_call_result["status"] == "ok", one_call_result
    assert int(one_call_result["prompt_token_count"]) == token_count
    one_call_logits = tensor_array(
        garnet, one_call_result["output"], (1, 16, 151936)
    ).astype(np.float32)
    assert np.isfinite(one_call_logits).all()
    one_call_token_id = int(np.argmax(one_call_logits[0, token_count - 1]))
    assert one_call_token_id == next_token_id
    assert int(one_call_result["token_id"]) == next_token_id
    np.testing.assert_allclose(one_call_logits, steady_array, rtol=0, atol=0)
    one_call_ms = (time.perf_counter() - one_call_start) * 1000.0
    print(
        f"one-call JPEG+prompt to GPU logits passed: next_token_id={one_call_token_id}, "
        f"total_ms={one_call_ms:.2f}"
    )

    sampled_start = time.perf_counter()
    sampled_result = model.forward({
        "image_path": str(frame_path),
        "prompt": "Describe.",
        "min_pixels": 1024,
        "max_pixels": 1024,
        "sample": "greedy",
    })
    assert sampled_result["status"] == "ok", sampled_result
    assert int(sampled_result["token_id"]) == next_token_id
    sampled_ms = (time.perf_counter() - sampled_start) * 1000.0
    print(f"one-call GPU-sampled first token: token_id={next_token_id}, total_ms={sampled_ms:.2f}")
