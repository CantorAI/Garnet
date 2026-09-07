import os
import json
import shutil
import struct
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]


def discover_garnet_dll():
    explicit = os.environ.get("GARNET_DLL_PATH", "").strip()
    candidates = [Path(explicit)] if explicit else []
    candidates.extend([
        REPO_ROOT.parent / "out" / "build" / "x64-Release" / "bin" / "garnet.dll",
        REPO_ROOT.parent / "out" / "build" / "x64-Release" / "bin" / "garnet.dll",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise AssertionError("garnet.dll not found; build Garnet or set GARNET_DLL_PATH")


def add_windows_dll_dirs(garnet_dll):
    handles = []
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return handles
    for path in [
        garnet_dll.parent,
        REPO_ROOT.parent / "out" / "build" / "x64-Release" / "bin",
        REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "bin",
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin"),
    ]:
        if path.exists():
            handles.append(os.add_dll_directory(str(path)))
    return handles


print("Phase 24: compiled xmodel production-path guard")

garnet_dll = discover_garnet_dll()
_dll_handles = add_windows_dll_dirs(garnet_dll)

import xlang3

garnet = xlang3.importModule("garnet", fromPath=str(garnet_dll))
root_xmodel = REPO_ROOT / "xModel" / "qwen3" / "vl_2b_instruct" / "qwen_vl_model.py"
cache_dir = REPO_ROOT / "test2026" / "tests" / "phase_24_compiled_xmodel_runtime" / "cache"
shutil.rmtree(cache_dir, ignore_errors=True)

model = garnet.load_model(
    str(root_xmodel),
    runtime_mode="compiled_xmodel",
    cache_dir=str(cache_dir),
)
if model is None:
    raise AssertionError("compiled load_model returned None")

status = model.runtime_status()
assert status["mode"] == "compiled_xmodel", status
assert status["state"] == "root_function_loaded", status
assert not bool(status["ready"]), status
assert status["error_code"] == "symbolic_input_contract_not_implemented", status
assert Path(status["root_xmodel"]).resolve() == root_xmodel.resolve(), status

counters = status["forbidden_path_counters"]
assert counters["root_x_executions"] == 1, counters
for name in [
    "hardcoded_qwen_runner_calls",
    "python_subgraph_calls",
    "direct_internal_export_calls",
    "cpu_tensor_intermediates",
]:
    assert counters[name] == 0, (name, counters)

forward_result = model.forward({"stage": "guard"})
assert forward_result["status"] == "error", forward_result
assert forward_result["error_code"] == "compiled_graph_not_ready", forward_result

print(f"garnet={garnet_dll}")
print(f"root_xmodel={root_xmodel}")
print("compiled path loaded the root fusion function and used no forbidden fallback")

hf_model_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct"
hf_snapshots = sorted((hf_model_root / "snapshots").glob("*")) if hf_model_root.exists() else []
if hf_snapshots:
    checkpoint_model = garnet.load_model(
        str(root_xmodel),
        runtime_mode="compiled_xmodel",
        weights=str(hf_snapshots[-1]),
        cache_dir=str(cache_dir / "qwen_checkpoint_index"),
    )
    checkpoint_status = checkpoint_model.runtime_status()
    assert checkpoint_status["weight_tensor_count"] > 100, checkpoint_status
    assert checkpoint_status["weight_tensor_bytes"] > 4_000_000_000, checkpoint_status
    print(
        "native Qwen safetensors index: "
        f"{checkpoint_status['weight_tensor_count']} tensors, "
        f"{checkpoint_status['weight_tensor_bytes']} bytes"
    )

tiny_weights_dir = cache_dir / "tiny_weights"
tiny_weights_dir.mkdir(parents=True)
tiny_header = json.dumps({
    "weight": {
        "dtype": "F32",
        "shape": [1, 4],
        "data_offsets": [0, 16],
    }
}, separators=(",", ":")).encode("utf-8")
with (tiny_weights_dir / "model.safetensors").open("wb") as stream:
    stream.write(struct.pack("<Q", len(tiny_header)))
    stream.write(tiny_header)
    stream.write(np.arange(4, dtype=np.float32).tobytes())
(tiny_weights_dir / "config.json").write_text("{}", encoding="utf-8")
tiny_weight_model = garnet.load_model(
    str(SCRIPT_DIR / "compiled_add_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[1, 4]],
    weights=str(tiny_weights_dir),
    cache_dir=str(cache_dir / "tiny_weight_fixture"),
)
tiny_weight_status = tiny_weight_model.runtime_status()
assert tiny_weight_status["state"] == "compiled_engine_ready", tiny_weight_status
assert tiny_weight_status["weight_tensor_count"] == 1, tiny_weight_status
assert tiny_weight_status["weight_tensor_bytes"] == 16, tiny_weight_status
weight_probe = tiny_weight_model.debug_probe("weight", "weight")
assert weight_probe["status"] == "ok", weight_probe
assert not bool(weight_probe["cache_hit"]), weight_probe
weight_cpu = garnet.tensor_to_cpu(weight_probe["tensor"])
np.testing.assert_allclose(
    np.asarray(weight_cpu.tolist()).reshape(1, 4),
    np.arange(4, dtype=np.float32).reshape(1, 4),
    rtol=0.0,
    atol=0.0,
)
cached_weight_probe = tiny_weight_model.debug_probe("weight", "weight")
assert cached_weight_probe["status"] == "ok", cached_weight_probe
assert bool(cached_weight_probe["cache_hit"]), cached_weight_probe
loaded_weight_status = tiny_weight_model.runtime_status()
assert loaded_weight_status["loaded_weight_count"] == 1, loaded_weight_status
assert loaded_weight_status["loaded_weight_bytes"] == 16, loaded_weight_status
print("native safetensors metadata and lazy GPU range loading passed")

named_weight_model = garnet.load_model(
    str(SCRIPT_DIR / "compiled_named_weight_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[1, 4]],
    weights=str(tiny_weights_dir),
    cache_dir=str(cache_dir / "named_weight_fixture"),
)
named_weight_status = named_weight_model.runtime_status()
assert named_weight_status["state"] == "compiled_engine_ready", named_weight_status
named_weight_engine = Path(named_weight_status["engine_path"])
assert named_weight_engine.stat().st_size < 1_000_000, named_weight_engine.stat().st_size
named_weight_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
named_weight_result = named_weight_model.forward({"inputs": [named_weight_input]})
assert named_weight_result["status"] == "ok", named_weight_result
named_weight_output = garnet.tensor_to_cpu(named_weight_result["output"])
np.testing.assert_allclose(
    np.asarray(named_weight_output.tolist()).reshape(1, 1),
    np.array([[20.0]], dtype=np.float32),
    rtol=1e-6,
    atol=1e-6,
)
print("stripped TensorRT plan refit directly from mapped safetensors passed")

fixture_model = garnet.load_model(
    str(SCRIPT_DIR / "compiled_add_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[1, 4]],
    cache_dir=str(cache_dir / "add_fixture"),
)
fixture_status = fixture_model.runtime_status()
assert fixture_status["state"] == "compiled_engine_ready", fixture_status
assert bool(fixture_status["ready"]), fixture_status
assert fixture_status["entry_function"] == "Model", fixture_status
assert "add" in fixture_status["graph_summary"].lower(), fixture_status
assert "mul" in fixture_status["graph_summary"].lower(), fixture_status
assert "minus" in fixture_status["graph_summary"].lower(), fixture_status
fixture_counters = fixture_status["forbidden_path_counters"]
assert fixture_counters["root_x_executions"] == 1, fixture_counters
assert fixture_counters["hardcoded_qwen_runner_calls"] == 0, fixture_counters

fixture_input = np.array([[1.0, -2.0, 3.5, 4.0]], dtype=np.float32)
fixture_result = fixture_model.forward({"inputs": [fixture_input]})
assert fixture_result["status"] == "ok", fixture_result
fixture_output = garnet.tensor_to_cpu(fixture_result["output"])
np.testing.assert_allclose(
    np.asarray(fixture_output.tolist()).reshape(fixture_input.shape),
    (fixture_input + fixture_input) * fixture_input - fixture_input,
    rtol=1e-6,
    atol=1e-6,
)

print("compiled fixture captured an op chain, built TensorRT, and returned correct GPU output")

repeat_model = garnet.load_model(
    str(SCRIPT_DIR / "compiled_repeat_function_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[1, 4]],
    cache_dir=str(cache_dir / "repeat_function_fixture"),
)
repeat_status = repeat_model.runtime_status()
assert repeat_status["state"] == "compiled_engine_ready", repeat_status
assert repeat_status["graph_summary"].lower().count("add") == 5, repeat_status
repeat_result = repeat_model.forward({"inputs": [fixture_input]})
assert repeat_result["status"] == "ok", repeat_result
repeat_output = garnet.tensor_to_cpu(repeat_result["output"])
np.testing.assert_allclose(
    np.asarray(repeat_output.tolist()).reshape(fixture_input.shape),
    fixture_input * 32.0,
    rtol=1e-6,
    atol=1e-6,
)
print("nested function and five-iteration xmodel loop compiled and executed correctly")

matmul_model = garnet.load_model(
    str(SCRIPT_DIR / "compiled_matmul_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[2, 3], [3, 2]],
    cache_dir=str(cache_dir / "matmul_fixture"),
)
matmul_status = matmul_model.runtime_status()
assert matmul_status["state"] == "compiled_engine_ready", matmul_status
left = np.arange(6, dtype=np.float32).reshape(2, 3)
right = (np.arange(6, dtype=np.float32).reshape(3, 2) + 1.0) / 4.0
matmul_result = matmul_model.forward({"inputs": [left, right]})
assert matmul_result["status"] == "ok", matmul_result
matmul_output = garnet.tensor_to_cpu(matmul_result["output"])
np.testing.assert_allclose(
    np.asarray(matmul_output.tolist()).reshape(2, 2),
    left @ right,
    rtol=1e-6,
    atol=1e-6,
)
print("custom binary_op matmul compiled and executed correctly")

branch_model = garnet.load_model(
    str(SCRIPT_DIR / "compiled_static_branch_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[1, 4]],
    cache_dir=str(cache_dir / "static_branch_fixture"),
)
branch_status = branch_model.runtime_status()
assert branch_status["state"] == "compiled_engine_ready", branch_status
branch_result = branch_model.forward({"inputs": [fixture_input]})
assert branch_result["status"] == "ok", branch_result
branch_output = garnet.tensor_to_cpu(branch_result["output"])
np.testing.assert_allclose(
    np.asarray(branch_output.tolist()).reshape(fixture_input.shape),
    fixture_input * 2.0,
    rtol=1e-6,
    atol=1e-6,
)
print("static xmodel branch compiled and executed correctly")

unary_model = garnet.load_model(
    str(SCRIPT_DIR / "compiled_unary_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[1, 4]],
    cache_dir=str(cache_dir / "unary_fixture"),
)
unary_status = unary_model.runtime_status()
assert unary_status["state"] == "compiled_engine_ready", unary_status
unary_result = unary_model.forward({"inputs": [fixture_input]})
assert unary_result["status"] == "ok", unary_result
unary_output = garnet.tensor_to_cpu(unary_result["output"])
np.testing.assert_allclose(
    np.asarray(unary_output.tolist()).reshape(fixture_input.shape),
    np.maximum(fixture_input, 0.0),
    rtol=1e-6,
    atol=1e-6,
)
print("custom unary_op activation compiled and executed correctly")

linear_model = garnet.load_model(
    str(SCRIPT_DIR / "compiled_linear_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[2, 3], [4, 3]],
    cache_dir=str(cache_dir / "linear_fixture"),
)
linear_status = linear_model.runtime_status()
assert linear_status["state"] == "compiled_engine_ready", linear_status
linear_input = np.arange(6, dtype=np.float32).reshape(2, 3) / 3.0
linear_weight = np.arange(12, dtype=np.float32).reshape(4, 3) / 5.0
linear_result = linear_model.forward({"inputs": [linear_input, linear_weight]})
assert linear_result["status"] == "ok", linear_result
linear_output = garnet.tensor_to_cpu(linear_result["output"])
np.testing.assert_allclose(
    np.asarray(linear_output.tolist()).reshape(2, 4),
    linear_input @ linear_weight.T,
    rtol=1e-6,
    atol=1e-6,
)
print("Qwen-style linear projection compiled and executed correctly")

bf16_model = garnet.load_model(
    str(SCRIPT_DIR / "compiled_add_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[1, 4]],
    input_dtypes=["bfloat16"],
    cache_dir=str(cache_dir / "bf16_fixture"),
)
bf16_status = bf16_model.runtime_status()
assert bf16_status["state"] == "compiled_engine_ready", bf16_status
bf16_bits = (fixture_input.view(np.uint32) >> 16).astype(np.uint16)
bf16_input = garnet.tensor_from_bfloat16_bits(bf16_bits)
bf16_result = bf16_model.forward({"inputs": [bf16_input]})
assert bf16_result["status"] == "ok", bf16_result
bf16_output = garnet.tensor_to_cpu(bf16_result["output"])
np.testing.assert_allclose(
    np.asarray(bf16_output.tolist()).reshape(fixture_input.shape),
    (fixture_input + fixture_input) * fixture_input - fixture_input,
    rtol=2e-2,
    atol=2e-2,
)
print("BF16 symbolic inputs, TensorRT execution, and explicit observation conversion passed")

paged_decode_model = garnet.load_model(
    str(SCRIPT_DIR / "compiled_paged_kv_decode_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[1, 64], [2, 4, 2, 8], [2, 4, 2, 8], [2], [1], [1]],
    input_dtypes=["bfloat16", "bfloat16", "bfloat16", "int32", "int32", "int32"],
    cache_dir=str(cache_dir / "paged_kv_decode_fixture"),
)
paged_decode_status = paged_decode_model.runtime_status()
assert paged_decode_status["state"] == "compiled_engine_ready", paged_decode_status
rng = np.random.default_rng(23)
paged_qkv_source = rng.normal(0.0, 0.2, size=(1, 64)).astype(np.float32)
paged_qkv_bits = (paged_qkv_source.view(np.uint32) >> 16).astype(np.uint16)
paged_qkv = garnet.tensor_from_bfloat16_bits(paged_qkv_bits)
paged_zero_bits = np.zeros((2, 4, 2, 8), dtype=np.uint16)
paged_keys = garnet.tensor_from_bfloat16_bits(paged_zero_bits)
paged_values = garnet.tensor_from_bfloat16_bits(paged_zero_bits.copy())
paged_result = paged_decode_model.forward({"inputs": [
    paged_qkv,
    paged_keys,
    paged_values,
    garnet.tensor_from_host([1, 0], dtype="int32"),
    garnet.tensor_from_host([1], dtype="int32"),
    garnet.tensor_from_host([0], dtype="int32"),
]})
assert paged_result["status"] == "ok", paged_result
paged_output = garnet.tensor_to_cpu(paged_result["output"])
paged_output_array = np.asarray(paged_output.tolist()).reshape(1, 32).astype(np.float32)
paged_qkv_bf16 = (paged_qkv_bits.astype(np.uint32) << 16).view(np.float32)
expected_value = paged_qkv_bf16[:, 48:64]
expected = np.concatenate([
    expected_value[:, :8], expected_value[:, :8],
    expected_value[:, 8:], expected_value[:, 8:],
], axis=1)
np.testing.assert_allclose(paged_output_array, expected, rtol=0, atol=0)
assert Path(paged_decode_status["engine_path"]).stat().st_size > 0

cached_paged_decode_model = garnet.load_model(
    str(SCRIPT_DIR / "compiled_paged_kv_decode_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[1, 64], [2, 4, 2, 8], [2, 4, 2, 8], [2], [1], [1]],
    input_dtypes=["bfloat16", "bfloat16", "bfloat16", "int32", "int32", "int32"],
    cache_dir=str(cache_dir / "paged_kv_decode_fixture"),
)
cached_paged_status = cached_paged_decode_model.runtime_status()
assert cached_paged_status["state"] == "engine_cache_loaded", cached_paged_status
assert cached_paged_status["forbidden_path_counters"]["root_x_executions"] == 0
second_qkv_source = rng.normal(0.0, 0.2, size=(1, 64)).astype(np.float32)
second_qkv_bits = (second_qkv_source.view(np.uint32) >> 16).astype(np.uint16)
second_qkv = garnet.tensor_from_bfloat16_bits(second_qkv_bits)
second_result = cached_paged_decode_model.forward({"inputs": [
    second_qkv,
    paged_keys,
    paged_values,
    garnet.tensor_from_host([1, 0], dtype="int32"),
    garnet.tensor_from_host([2], dtype="int32"),
    garnet.tensor_from_host([1], dtype="int32"),
]})
assert second_result["status"] == "ok", second_result
second_output = np.asarray(
    garnet.tensor_to_cpu(second_result["output"]).tolist()
).reshape(4, 8).astype(np.float32)
second_qkv_bf16 = (second_qkv_bits.astype(np.uint32) << 16).view(np.float32)
logical_keys = np.stack([
    paged_qkv_bf16[0, 32:48].reshape(2, 8),
    second_qkv_bf16[0, 32:48].reshape(2, 8),
])
logical_values = np.stack([
    paged_qkv_bf16[0, 48:64].reshape(2, 8),
    second_qkv_bf16[0, 48:64].reshape(2, 8),
])
expected_second = np.empty((4, 8), dtype=np.float32)
for q_head in range(4):
    kv_head = q_head // 2
    query = second_qkv_bf16[0, q_head * 8:(q_head + 1) * 8]
    scores = logical_keys[:, kv_head] @ query / np.sqrt(8)
    probabilities = np.exp(scores - scores.max())
    probabilities /= probabilities.sum()
    expected_second[q_head] = probabilities @ logical_values[:, kv_head]
expected_second_bits = (expected_second.view(np.uint32) >> 16).astype(np.uint16)
expected_second_bf16 = (expected_second_bits.astype(np.uint32) << 16).view(np.float32)
np.testing.assert_allclose(second_output, expected_second_bf16, rtol=0, atol=0.002)
print("explicit .x BF16 paged-KV plugin compiled, cached, and decoded two tokens correctly")

prefill_model = garnet.load_model(
    str(SCRIPT_DIR / "compiled_paged_kv_prefill_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[5, 64], [2, 4, 2, 8], [2, 4, 2, 8], [2], [1]],
    input_dtypes=["bfloat16", "bfloat16", "bfloat16", "int32", "int32"],
    cache_dir=str(cache_dir / "paged_kv_prefill_fixture"),
)
prefill_status = prefill_model.runtime_status()
assert prefill_status["state"] == "compiled_engine_ready", prefill_status
prefill_source = rng.normal(0.0, 0.2, size=(5, 64)).astype(np.float32)
prefill_bits = (prefill_source.view(np.uint32) >> 16).astype(np.uint16)
prefill_bf16 = (prefill_bits.astype(np.uint32) << 16).view(np.float32)
prefill_qkv = garnet.tensor_from_bfloat16_bits(prefill_bits)
prefill_keys = garnet.tensor_from_bfloat16_bits(np.zeros((2, 4, 2, 8), dtype=np.uint16))
prefill_values = garnet.tensor_from_bfloat16_bits(np.zeros((2, 4, 2, 8), dtype=np.uint16))
swapped_pages = garnet.tensor_from_host([1, 0], dtype="int32")
prefill_result = prefill_model.forward({"inputs": [
    prefill_qkv,
    prefill_keys,
    prefill_values,
    swapped_pages,
    garnet.tensor_from_host([0], dtype="int32"),
]})
assert prefill_result["status"] == "ok", prefill_result
prefill_passthrough = np.asarray(
    garnet.tensor_to_cpu(prefill_result["output"]).tolist()
).reshape(5, 64).astype(np.float32)
np.testing.assert_array_equal(prefill_passthrough, prefill_bf16)

after_prefill_source = rng.normal(0.0, 0.2, size=(1, 64)).astype(np.float32)
after_prefill_bits = (after_prefill_source.view(np.uint32) >> 16).astype(np.uint16)
after_prefill_bf16 = (after_prefill_bits.astype(np.uint32) << 16).view(np.float32)
after_prefill_result = cached_paged_decode_model.forward({"inputs": [
    garnet.tensor_from_bfloat16_bits(after_prefill_bits),
    prefill_keys,
    prefill_values,
    swapped_pages,
    garnet.tensor_from_host([6], dtype="int32"),
    garnet.tensor_from_host([5], dtype="int32"),
]})
assert after_prefill_result["status"] == "ok", after_prefill_result
after_prefill_output = np.asarray(
    garnet.tensor_to_cpu(after_prefill_result["output"]).tolist()
).reshape(4, 8).astype(np.float32)
all_keys = np.concatenate([
    prefill_bf16[:, 32:48].reshape(5, 2, 8),
    after_prefill_bf16[:, 32:48].reshape(1, 2, 8),
], axis=0)
all_values = np.concatenate([
    prefill_bf16[:, 48:64].reshape(5, 2, 8),
    after_prefill_bf16[:, 48:64].reshape(1, 2, 8),
], axis=0)
expected_after_prefill = np.empty((4, 8), dtype=np.float32)
for q_head in range(4):
    kv_head = q_head // 2
    query = after_prefill_bf16[0, q_head * 8:(q_head + 1) * 8]
    scores = all_keys[:, kv_head] @ query / np.sqrt(8)
    probabilities = np.exp(scores - scores.max())
    probabilities /= probabilities.sum()
    expected_after_prefill[q_head] = probabilities @ all_values[:, kv_head]
expected_after_prefill_bits = (expected_after_prefill.view(np.uint32) >> 16).astype(np.uint16)
expected_after_prefill_bf16 = (
    expected_after_prefill_bits.astype(np.uint32) << 16
).view(np.float32)
np.testing.assert_allclose(
    after_prefill_output, expected_after_prefill_bf16, rtol=0, atol=0.002
)
print("explicit .x BF16 prefill wrote five tokens across swapped pages and decode consumed them")

masked_batch = 4
masked_q_heads = 16
masked_kv_heads = 8
masked_head_dim = 128
masked_q_width = masked_q_heads * masked_head_dim
masked_kv_width = masked_kv_heads * masked_head_dim
masked_qkv_width = masked_q_width + 2 * masked_kv_width
masked_page_count = 16
masked_page_size = 16
masked_logical_pages = 4
masked_model = garnet.load_model(
    str(SCRIPT_DIR / "compiled_paged_kv_decode_masked_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[
        [masked_batch, 1, masked_qkv_width],
        [masked_page_count, masked_page_size, masked_kv_heads, masked_head_dim],
        [masked_page_count, masked_page_size, masked_kv_heads, masked_head_dim],
        [masked_batch, masked_logical_pages],
        [masked_batch],
        [masked_batch],
        [masked_batch],
    ],
    input_dtypes=[
        "bfloat16", "bfloat16", "bfloat16", "int32",
        "int32", "int32", "int32",
    ],
    cache_dir=str(cache_dir / "paged_kv_masked_batch_fixture"),
)
masked_status = masked_model.runtime_status()
assert masked_status["state"] == "compiled_engine_ready", masked_status
masked_source = rng.normal(
    0.0, 0.1, size=(masked_batch, 1, masked_qkv_width)
).astype(np.float32)
masked_bits = (masked_source.view(np.uint32) >> 16).astype(np.uint16)
masked_bf16 = (masked_bits.astype(np.uint32) << 16).view(np.float32)
masked_page_shape = (
    masked_page_count, masked_page_size, masked_kv_heads, masked_head_dim
)
masked_keys = garnet.tensor_from_bfloat16_bits(
    np.zeros(masked_page_shape, dtype=np.uint16)
)
masked_values = garnet.tensor_from_bfloat16_bits(
    np.zeros(masked_page_shape, dtype=np.uint16)
)
masked_page_table = np.arange(
    masked_batch * masked_logical_pages, dtype=np.int32
).reshape(masked_batch, masked_logical_pages)
active_mask = np.array([1, 0, 1, 0], dtype=np.int32)
masked_result = masked_model.forward({"inputs": [
    garnet.tensor_from_bfloat16_bits(masked_bits),
    masked_keys,
    masked_values,
    garnet.tensor_from_host(
        masked_page_table.reshape(-1).tolist(),
        dtype="int32",
        shape=[masked_batch, masked_logical_pages],
    ),
    garnet.tensor_from_host(
        np.ones(masked_batch, dtype=np.int32).tolist(),
        dtype="int32",
        shape=[masked_batch],
    ),
    garnet.tensor_from_host(
        np.zeros(masked_batch, dtype=np.int32).tolist(),
        dtype="int32",
        shape=[masked_batch],
    ),
    garnet.tensor_from_host(
        active_mask.tolist(), dtype="int32", shape=[masked_batch]
    ),
]})
assert masked_result["status"] == "ok", masked_result
masked_output = np.asarray(
    garnet.tensor_to_cpu(masked_result["output"]).tolist()
).reshape(masked_batch, masked_q_heads, masked_head_dim).astype(np.float32)
for batch_index in range(masked_batch):
    if active_mask[batch_index] == 0:
        np.testing.assert_array_equal(masked_output[batch_index], 0.0)
        continue
    values = masked_bf16[
        batch_index, 0, masked_q_width + masked_kv_width:
    ].reshape(masked_kv_heads, masked_head_dim)
    expected = np.repeat(values, masked_q_heads // masked_kv_heads, axis=0)
    np.testing.assert_allclose(masked_output[batch_index], expected, rtol=0, atol=0)
masked_keys_cpu = np.asarray(
    garnet.tensor_to_cpu(masked_keys).tolist()
).reshape(masked_page_shape)
for inactive_index in [1, 3]:
    np.testing.assert_array_equal(
        masked_keys_cpu[masked_page_table[inactive_index, 0], 0],
        0.0,
    )
masked_sample = masked_model.forward({
    "inputs": [
        garnet.tensor_from_bfloat16_bits(masked_bits),
        masked_keys,
        masked_values,
        garnet.tensor_from_host(
            masked_page_table.reshape(-1).tolist(),
            dtype="int32",
            shape=[masked_batch, masked_logical_pages],
        ),
        garnet.tensor_from_host(
            np.ones(masked_batch, dtype=np.int32).tolist(),
            dtype="int32",
            shape=[masked_batch],
        ),
        garnet.tensor_from_host(
            np.zeros(masked_batch, dtype=np.int32).tolist(),
            dtype="int32",
            shape=[masked_batch],
        ),
        garnet.tensor_from_host(
            active_mask.tolist(), dtype="int32", shape=[masked_batch]
        ),
    ],
    "sample": "greedy_batch",
})
assert masked_sample["status"] == "ok", masked_sample
assert list(masked_sample["token_ids"]) == [
    int(np.argmax(masked_output[index])) for index in range(masked_batch)
]
print("masked B4 paged decode skipped inactive rows and preserved their KV pages")

long_logical_pages = 257
long_page_count = long_logical_pages
long_model = garnet.load_model(
    str(SCRIPT_DIR / "compiled_paged_kv_decode_masked_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[
        [1, 1, masked_qkv_width],
        [long_page_count, masked_page_size, masked_kv_heads, masked_head_dim],
        [long_page_count, masked_page_size, masked_kv_heads, masked_head_dim],
        [1, long_logical_pages],
        [1],
        [1],
        [1],
    ],
    input_dtypes=[
        "bfloat16", "bfloat16", "bfloat16", "int32",
        "int32", "int32", "int32",
    ],
    cache_dir=str(cache_dir / "paged_kv_long_context_fixture"),
)
long_status = long_model.runtime_status()
assert bool(long_status["ready"]), long_status
long_page_shape = (
    long_page_count, masked_page_size, masked_kv_heads, masked_head_dim
)
long_result = long_model.forward({"inputs": [
    garnet.tensor_from_bfloat16_bits(masked_bits[:1]),
    garnet.tensor_from_bfloat16_bits(
        np.zeros(long_page_shape, dtype=np.uint16)
    ),
    garnet.tensor_from_bfloat16_bits(
        np.zeros(long_page_shape, dtype=np.uint16)
    ),
    garnet.tensor_from_host(
        list(range(long_logical_pages)),
        dtype="int32",
        shape=[1, long_logical_pages],
    ),
    garnet.tensor_from_host([4097], dtype="int32", shape=[1]),
    garnet.tensor_from_host([4096], dtype="int32", shape=[1]),
    garnet.tensor_from_host([1], dtype="int32", shape=[1]),
]})
assert long_result["status"] == "ok", long_result
long_output = np.asarray(
    garnet.tensor_to_cpu(long_result["output"]).tolist()
).astype(np.float32)
assert np.isfinite(long_output).all()
assert np.any(long_output != 0.0)
print("masked paged decode supports context lengths beyond the old 4096-token cap")

unsupported_model = garnet.load_model(
    str(SCRIPT_DIR / "compiled_unsupported_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[1, 4]],
    cache_dir=str(cache_dir / "unsupported_fixture"),
)
unsupported_status = unsupported_model.runtime_status()
assert unsupported_status["state"] == "failed", unsupported_status
assert unsupported_status["error_code"] == "generic_lowering_failed", unsupported_status
assert "unsupported_probe" in unsupported_status["error_message"], unsupported_status
assert not Path(unsupported_status["engine_path"]).exists(), unsupported_status
print("unsupported operation failed explicitly without publishing an engine")

packed_model = garnet.load_model(
    str(SCRIPT_DIR / "compiled_packed_state_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[1, 4]],
    cache_dir=str(cache_dir / "packed_state_fixture"),
)
packed_status = packed_model.runtime_status()
assert packed_status["state"] == "failed", packed_status
assert packed_status["error_code"] == "generic_lowering_failed", packed_status
assert "packed_state_fixture" in packed_status["error_message"], packed_status
assert not Path(packed_status["engine_path"]).exists(), packed_status
print("packed-state expression captured safely and failed at explicit lowering boundary")

cached_fixture_model = garnet.load_model(
    str(SCRIPT_DIR / "compiled_add_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[1, 4]],
    cache_dir=str(cache_dir / "add_fixture"),
)
cached_fixture_status = cached_fixture_model.runtime_status()
assert cached_fixture_status["state"] == "engine_cache_loaded", cached_fixture_status
cached_counters = cached_fixture_status["forbidden_path_counters"]
assert cached_counters["root_x_executions"] == 0, cached_counters
assert cached_counters["graph_cache_hits"] == 1, cached_counters
cached_fixture_result = cached_fixture_model.forward({"inputs": [fixture_input]})
assert cached_fixture_result["status"] == "ok", cached_fixture_result
cached_fixture_output = garnet.tensor_to_cpu(cached_fixture_result["output"])
np.testing.assert_allclose(
    np.asarray(cached_fixture_output.tolist()).reshape(fixture_input.shape),
    (fixture_input + fixture_input) * fixture_input - fixture_input,
    rtol=1e-6,
    atol=1e-6,
)

graph_cache = cache_dir / "add_fixture" / "runtime_graph.cache"
graph_cache.write_text("corrupt", encoding="utf-8")
recaptured_fixture_model = garnet.load_model(
    str(SCRIPT_DIR / "compiled_add_model.x"),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[1, 4]],
    cache_dir=str(cache_dir / "add_fixture"),
)
recaptured_fixture_status = recaptured_fixture_model.runtime_status()
assert recaptured_fixture_status["state"] == "compiled_engine_ready", recaptured_fixture_status
recaptured_counters = recaptured_fixture_status["forbidden_path_counters"]
assert recaptured_counters["root_x_executions"] == 1, recaptured_counters
assert recaptured_counters["graph_cache_misses"] == 1, recaptured_counters

dependency_dir = cache_dir / "dependency_fixture"
dependency_dir.mkdir(parents=True)
dependency = dependency_dir / "compiled_dependency.x"
dependency.write_text("def AddTwice(x):\n    return x + x\n", encoding="utf-8")
dependency_root = dependency_dir / "model.x"
dependency_root.write_text(
    "from garnet import garnet\n"
    "# garnet-dependency: compiled_dependency.x\n\n"
    "T = garnet.tensor()\n"
    "@T.fusion()\n"
    "def Model(x):\n"
    "    return x + x\n",
    encoding="utf-8",
)
dependency_cache = dependency_dir / "graph_cache"
dependency_model = garnet.load_model(
    str(dependency_root),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[1, 4]],
    cache_dir=str(dependency_cache),
)
assert dependency_model.runtime_status()["state"] == "compiled_engine_ready"
dependency.write_text("# changed dependency\ndef AddTwice(x):\n    return x + x\n", encoding="utf-8")
changed_dependency_model = garnet.load_model(
    str(dependency_root),
    runtime_mode="compiled_xmodel",
    entry_function="Model",
    input_shapes=[[1, 4]],
    cache_dir=str(dependency_cache),
)
changed_dependency_status = changed_dependency_model.runtime_status()
assert changed_dependency_status["state"] == "compiled_engine_ready", changed_dependency_status
assert changed_dependency_status["forbidden_path_counters"]["root_x_executions"] == 1

print("validated graph cache hit, corruption recovery, and imported .x invalidation")
print("Phase 24 compiled xmodel production-path guard passed.")
