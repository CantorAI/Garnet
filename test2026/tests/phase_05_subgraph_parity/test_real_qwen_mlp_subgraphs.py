import os
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
ARTIFACT_DIR = REPO_ROOT / "test2026" / "artifacts" / "qwen_vl_subgraphs"


def env_flag(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


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


def load_weight_map(model_dir):
    try:
        from safetensors import safe_open
    except Exception as exc:
        skip(f"safetensors is not available: {exc}")

    model_dir = Path(model_dir)
    files = list(model_dir.glob("*.safetensors"))
    if not files:
        skip(f"no safetensors files found in {model_dir}")

    key_to_file = {}
    for file_path in files:
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                normalized = key[6:] if key.startswith("model.") else key
                key_to_file[normalized] = file_path
    return key_to_file


def load_tensor(key_to_file, key):
    from safetensors import safe_open

    if key not in key_to_file:
        raise AssertionError(f"missing real Qwen weight: {key}")
    with safe_open(str(key_to_file[key]), framework="pt", device="cpu") as handle:
        raw_key = key if key in handle.keys() else "model." + key
        tensor = handle.get_tensor(raw_key).float().contiguous()
        return tensor.numpy().astype(np.float32, copy=False)


def silu(x):
    return x / (1.0 + np.exp(-x))


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def to_numpy(value):
    to_numpy_fn = getattr(value, "numpy", None)
    to_list_fn = getattr(value, "tolist", None)
    if callable(to_numpy_fn):
        return to_numpy_fn()
    if callable(to_list_fn):
        return np.array(to_list_fn(), dtype=np.float32)
    raise AssertionError("Garnet output has no numpy/tolist conversion")


def import_garnet():
    try:
        import xlang
    except Exception as exc:
        skip(f"xlang Python module not available: {exc}")

    garnet_dll = Path(os.environ.get(
        "GARNET_DLL_PATH",
        REPO_ROOT / "out" / "build" / "x64-Debug" / "bin" / "garnet.dll",
    ))
    if not garnet_dll.exists():
        skip(f"garnet.dll not found: {garnet_dll}")

    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for dll_dir in [
            garnet_dll.parent,
            REPO_ROOT.parent / "xlang" / "out" / "build" / "x64-Debug" / "bin",
        ]:
            if dll_dir.exists():
                os.add_dll_directory(str(dll_dir))

    try:
        return xlang.importModule("garnet", fromPath=str(garnet_dll)), garnet_dll
    except Exception as exc:
        skip(f"failed to import Garnet from {garnet_dll}: {exc}")


def run_rms_norm(garnet, key_to_file):
    key = "language_model.layers.0.input_layernorm.weight"
    weight = load_tensor(key_to_file, key)
    rng = np.random.default_rng(20260708)
    x = rng.normal(0.0, 0.02, size=(7, weight.shape[0])).astype(np.float32)
    denom = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + 1.0e-6)
    expected = (x / denom) * weight

    engine = garnet.load_model(
        str(SCRIPT_DIR / "rms_norm_trt.x"),
        weights={key: weight},
        cache_dir=str(SCRIPT_DIR / "cache_real"),
        input_shapes=[list(x.shape)],
        subgraph="rms_norm",
    )
    if engine is None:
        raise AssertionError("Garnet load_model returned None for real RMSNorm")

    actual = to_numpy(engine.forward(x)).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    return {
        "input_shape": list(x.shape),
        "weight_shape": list(weight.shape),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
    }


def run_layer_norm(garnet, key_to_file):
    weight_key = "visual.blocks.0.norm1.weight"
    bias_key = "visual.blocks.0.norm1.bias"
    weight = load_tensor(key_to_file, weight_key)
    bias = load_tensor(key_to_file, bias_key)
    rng = np.random.default_rng(20260709)
    x = rng.normal(0.0, 0.02, size=(9, weight.shape[0])).astype(np.float32)
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(np.square(x - mean), axis=-1, keepdims=True)
    expected = ((x - mean) / np.sqrt(variance + 1.0e-6)) * weight + bias

    engine = garnet.load_model(
        str(SCRIPT_DIR / "layer_norm_trt.x"),
        weights={weight_key: weight, bias_key: bias},
        cache_dir=str(SCRIPT_DIR / "cache_real"),
        input_shapes=[list(x.shape)],
        subgraph="layer_norm",
    )
    if engine is None:
        raise AssertionError("Garnet load_model returned None for real LayerNorm")

    actual = to_numpy(engine.forward(x)).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    return {
        "input_shape": list(x.shape),
        "weight_shape": list(weight.shape),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
    }


def run_text_qkv_proj(garnet, key_to_file):
    prefix = "language_model.layers.0.self_attn"
    weights = {
        prefix + ".q_proj.weight": load_tensor(key_to_file, prefix + ".q_proj.weight"),
        prefix + ".k_proj.weight": load_tensor(key_to_file, prefix + ".k_proj.weight"),
        prefix + ".v_proj.weight": load_tensor(key_to_file, prefix + ".v_proj.weight"),
    }
    rng = np.random.default_rng(20260710)
    x = rng.normal(0.0, 0.02, size=(4, weights[prefix + ".q_proj.weight"].shape[1])).astype(np.float32)
    q = x @ weights[prefix + ".q_proj.weight"].T
    k = x @ weights[prefix + ".k_proj.weight"].T
    v = x @ weights[prefix + ".v_proj.weight"].T
    expected = np.concatenate([q, k, v], axis=-1)

    engine = garnet.load_model(
        str(SCRIPT_DIR / "text_qkv_proj_trt.x"),
        weights=weights,
        cache_dir=str(SCRIPT_DIR / "cache_real"),
        input_shapes=[list(x.shape)],
        subgraph="text_qkv_proj",
    )
    if engine is None:
        raise AssertionError("Garnet load_model returned None for real TextQKV projection")

    actual = to_numpy(engine.forward(x)).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=7e-3, atol=7e-3)
    return {
        "input_shape": list(x.shape),
        "q_shape": list(weights[prefix + ".q_proj.weight"].shape),
        "k_shape": list(weights[prefix + ".k_proj.weight"].shape),
        "v_shape": list(weights[prefix + ".v_proj.weight"].shape),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
    }


def run_text_qkv_head_norm(garnet, key_to_file):
    prefix = "language_model.layers.0.self_attn"
    weights = {
        prefix + ".q_proj.weight": load_tensor(key_to_file, prefix + ".q_proj.weight"),
        prefix + ".k_proj.weight": load_tensor(key_to_file, prefix + ".k_proj.weight"),
        prefix + ".v_proj.weight": load_tensor(key_to_file, prefix + ".v_proj.weight"),
        prefix + ".q_norm.weight": load_tensor(key_to_file, prefix + ".q_norm.weight"),
        prefix + ".k_norm.weight": load_tensor(key_to_file, prefix + ".k_norm.weight"),
    }
    head_dim = weights[prefix + ".q_norm.weight"].shape[0]
    q_out = weights[prefix + ".q_proj.weight"].shape[0]
    k_out = weights[prefix + ".k_proj.weight"].shape[0]
    q_heads = q_out // head_dim
    kv_heads = k_out // head_dim

    rng = np.random.default_rng(20260711)
    x = rng.normal(0.0, 0.02, size=(4, weights[prefix + ".q_proj.weight"].shape[1])).astype(np.float32)
    q = x @ weights[prefix + ".q_proj.weight"].T
    k = x @ weights[prefix + ".k_proj.weight"].T
    v = x @ weights[prefix + ".v_proj.weight"].T

    q_heads_tensor = q.reshape(x.shape[0], q_heads, head_dim)
    k_heads_tensor = k.reshape(x.shape[0], kv_heads, head_dim)
    q_norm = q_heads_tensor / np.sqrt(np.mean(np.square(q_heads_tensor), axis=-1, keepdims=True) + 1.0e-6)
    k_norm = k_heads_tensor / np.sqrt(np.mean(np.square(k_heads_tensor), axis=-1, keepdims=True) + 1.0e-6)
    q_norm = (q_norm * weights[prefix + ".q_norm.weight"]).reshape(x.shape[0], q_out)
    k_norm = (k_norm * weights[prefix + ".k_norm.weight"]).reshape(x.shape[0], k_out)
    expected = np.concatenate([q_norm, k_norm, v], axis=-1)

    engine = garnet.load_model(
        str(SCRIPT_DIR / "text_qkv_head_norm_trt.x"),
        weights=weights,
        cache_dir=str(SCRIPT_DIR / "cache_real"),
        input_shapes=[list(x.shape)],
        subgraph="text_qkv_head_norm",
    )
    if engine is None:
        raise AssertionError("Garnet load_model returned None for real TextQKV head norm")

    actual = to_numpy(engine.forward(x)).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=7e-3, atol=7e-3)
    q_actual = actual[:, :q_out]
    k_actual = actual[:, q_out:q_out + k_out]
    v_actual = actual[:, q_out + k_out:]
    return {
        "input_shape": list(x.shape),
        "q_heads": int(q_heads),
        "kv_heads": int(kv_heads),
        "head_dim": int(head_dim),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
        "q_max_error": float(np.max(np.abs(q_actual - q_norm))),
        "k_max_error": float(np.max(np.abs(k_actual - k_norm))),
        "v_max_error": float(np.max(np.abs(v_actual - v))),
    }


def run_text_o_proj(garnet, key_to_file):
    key = "language_model.layers.0.self_attn.o_proj.weight"
    weight = load_tensor(key_to_file, key)
    rng = np.random.default_rng(20260712)
    x = rng.normal(0.0, 0.02, size=(4, weight.shape[1])).astype(np.float32)
    expected = x @ weight.T

    engine = garnet.load_model(
        str(SCRIPT_DIR / "text_o_proj_trt.x"),
        weights={key: weight},
        cache_dir=str(SCRIPT_DIR / "cache_real"),
        input_shapes=[list(x.shape)],
        subgraph="text_o_proj",
    )
    if engine is None:
        raise AssertionError("Garnet load_model returned None for real text o_proj")

    actual = to_numpy(engine.forward(x)).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=7e-3, atol=7e-3)
    return {
        "input_shape": list(x.shape),
        "weight_shape": list(weight.shape),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
    }


def run_text_mlp(garnet, key_to_file):
    prefix = "language_model.layers.0.mlp"
    weights = {
        prefix + ".gate_proj.weight": load_tensor(key_to_file, prefix + ".gate_proj.weight"),
        prefix + ".up_proj.weight": load_tensor(key_to_file, prefix + ".up_proj.weight"),
        prefix + ".down_proj.weight": load_tensor(key_to_file, prefix + ".down_proj.weight"),
    }

    rng = np.random.default_rng(20260706)
    x = rng.normal(0.0, 0.02, size=(3, weights[prefix + ".gate_proj.weight"].shape[1])).astype(np.float32)
    gate = x @ weights[prefix + ".gate_proj.weight"].T
    up = x @ weights[prefix + ".up_proj.weight"].T
    expected = (silu(gate) * up) @ weights[prefix + ".down_proj.weight"].T

    engine = garnet.load_model(
        str(SCRIPT_DIR / "text_mlp_trt.x"),
        weights=weights,
        cache_dir=str(SCRIPT_DIR / "cache_real"),
        input_shapes=[list(x.shape)],
        subgraph="qwen3_text_mlp",
    )
    if engine is None:
        raise AssertionError("Garnet load_model returned None for real Qwen3TextMLP")

    actual = to_numpy(engine.forward(x)).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=7e-3, atol=7e-3)
    return {
        "input_shape": list(x.shape),
        "gate_shape": list(weights[prefix + ".gate_proj.weight"].shape),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
    }


def run_vision_mlp(garnet, key_to_file):
    prefix = "visual.blocks.0.mlp"
    weights = {
        prefix + ".linear_fc1.weight": load_tensor(key_to_file, prefix + ".linear_fc1.weight"),
        prefix + ".linear_fc1.bias": load_tensor(key_to_file, prefix + ".linear_fc1.bias"),
        prefix + ".linear_fc2.weight": load_tensor(key_to_file, prefix + ".linear_fc2.weight"),
        prefix + ".linear_fc2.bias": load_tensor(key_to_file, prefix + ".linear_fc2.bias"),
    }

    rng = np.random.default_rng(20260707)
    x = rng.normal(0.0, 0.02, size=(5, weights[prefix + ".linear_fc1.weight"].shape[1])).astype(np.float32)
    hidden = gelu_tanh(x @ weights[prefix + ".linear_fc1.weight"].T + weights[prefix + ".linear_fc1.bias"])
    expected = hidden @ weights[prefix + ".linear_fc2.weight"].T + weights[prefix + ".linear_fc2.bias"]

    engine = garnet.load_model(
        str(SCRIPT_DIR / "vision_mlp_trt.x"),
        weights=weights,
        cache_dir=str(SCRIPT_DIR / "cache_real"),
        input_shapes=[list(x.shape)],
        subgraph="vision_mlp",
    )
    if engine is None:
        raise AssertionError("Garnet load_model returned None for real VisionMLP")

    actual = to_numpy(engine.forward(x)).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=7e-3, atol=7e-3)
    return {
        "input_shape": list(x.shape),
        "fc1_shape": list(weights[prefix + ".linear_fc1.weight"].shape),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
    }


print("Phase 05: real Qwen3-VL MLP subgraph parity")

if not env_flag("RUN_GARNET_REAL_QWEN_MLP_PARITY"):
    skip("set RUN_GARNET_REAL_QWEN_MLP_PARITY=1 to run real Qwen3-VL MLP subgraph parity")

model_dir = default_model_dir()
if model_dir is None or not Path(model_dir).exists():
    skip("Qwen3-VL model snapshot not found; set HF_QWEN_VL_MODEL_DIR")

garnet, garnet_dll = import_garnet()
key_to_file = load_weight_map(model_dir)

text_norm_result = run_rms_norm(garnet, key_to_file)
vision_norm_result = run_layer_norm(garnet, key_to_file)
text_qkv_result = run_text_qkv_proj(garnet, key_to_file)
text_qkv_head_norm_result = run_text_qkv_head_norm(garnet, key_to_file)
text_o_proj_result = run_text_o_proj(garnet, key_to_file)
text_result = run_text_mlp(garnet, key_to_file)
vision_result = run_vision_mlp(garnet, key_to_file)

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
summary_path = ARTIFACT_DIR / "real_qwen_mlp_subgraph_parity.json"
summary_path.write_text(
    __import__("json").dumps(
        {
            "model_dir": str(model_dir),
            "garnet": str(garnet_dll),
            "text_rms_norm": text_norm_result,
            "vision_layer_norm": vision_norm_result,
            "text_qkv_proj": text_qkv_result,
            "text_qkv_head_norm": text_qkv_head_norm_result,
            "text_o_proj": text_o_proj_result,
            "text_mlp": text_result,
            "vision_mlp": vision_result,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(f"model_dir={model_dir}")
print(f"garnet={garnet_dll}")
print(f"text_rms_norm={text_norm_result}")
print(f"vision_layer_norm={vision_norm_result}")
print(f"text_qkv_proj={text_qkv_result}")
print(f"text_qkv_head_norm={text_qkv_head_norm_result}")
print(f"text_o_proj={text_o_proj_result}")
print(f"text_mlp={text_result}")
print(f"vision_mlp={vision_result}")
print(f"summary={summary_path}")
print("Phase 05 real Qwen3-VL MLP subgraph parity passed.")
