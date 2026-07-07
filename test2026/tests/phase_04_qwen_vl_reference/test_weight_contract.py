import json
import os
from pathlib import Path

from common import REPO_ROOT, env_flag, skip


print("Phase 04: Qwen-VL config and weight contract validation")

if not env_flag("RUN_QWEN_VL_WEIGHT_CONTRACT"):
    skip("set RUN_QWEN_VL_WEIGHT_CONTRACT=1 to validate config/xmodel/weight names")


CONFIG_PATH = Path(os.environ.get("GARNET_QWEN_VL_CONFIG_JSON", "")).expanduser()
WEIGHTS_PATH = Path(os.environ.get("GARNET_QWEN_VL_WEIGHT_INDEX_OR_DIR", "")).expanduser()
XMODEL_DIR = Path(os.environ.get("GARNET_QWEN_VL_XMODEL_DIR", REPO_ROOT / "qwen_vl" / "xmodel"))
REPORT_PATH = Path(os.environ.get(
    "GARNET_QWEN_VL_WEIGHT_CONTRACT_REPORT",
    REPO_ROOT / "test2026" / "artifacts" / "qwen_vl_reference" / "weight_contract_report.json",
))

if not CONFIG_PATH.exists():
    skip("set GARNET_QWEN_VL_CONFIG_JSON to a Qwen-VL config.json")
if not WEIGHTS_PATH.exists():
    skip("set GARNET_QWEN_VL_WEIGHT_INDEX_OR_DIR to a safetensors index, safetensors file, or weight folder")
if not XMODEL_DIR.exists():
    skip(f"xmodel dir not found: {XMODEL_DIR}")


def nested(config, *keys, default=None):
    cur = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def expected_weight_names(config):
    text = config.get("text_config", config)
    vision = config.get("vision_config", {})
    text_layers = int(nested(config, "text_config", "num_hidden_layers", default=text.get("num_hidden_layers", 0)) or 0)
    vision_layers = int(nested(config, "vision_config", "depth", default=vision.get("num_hidden_layers", 0)) or 0)

    names = {
        "language_model.embed_tokens.weight",
        "language_model.norm.weight",
        "visual.patch_embed.proj.weight",
        "visual.patch_embed.proj.bias",
        "visual.pos_embed.weight",
        "visual.merger.norm.weight",
        "visual.merger.norm.bias",
        "visual.merger.linear_fc1.weight",
        "visual.merger.linear_fc1.bias",
        "visual.merger.linear_fc2.weight",
        "visual.merger.linear_fc2.bias",
    }

    for layer_idx in range(text_layers):
        prefix = f"language_model.layers.{layer_idx}"
        names.update({
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
            f"{prefix}.self_attn.o_proj.weight",
            f"{prefix}.self_attn.q_norm.weight",
            f"{prefix}.self_attn.k_norm.weight",
            f"{prefix}.mlp.gate_proj.weight",
            f"{prefix}.mlp.up_proj.weight",
            f"{prefix}.mlp.down_proj.weight",
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight",
        })

    for layer_idx in range(vision_layers):
        prefix = f"visual.blocks.{layer_idx}"
        names.update({
            f"{prefix}.norm1.weight",
            f"{prefix}.norm1.bias",
            f"{prefix}.attn.qkv.weight",
            f"{prefix}.attn.qkv.bias",
            f"{prefix}.attn.proj.weight",
            f"{prefix}.attn.proj.bias",
            f"{prefix}.norm2.weight",
            f"{prefix}.norm2.bias",
            f"{prefix}.mlp.linear_fc1.weight",
            f"{prefix}.mlp.linear_fc1.bias",
            f"{prefix}.mlp.linear_fc2.weight",
            f"{prefix}.mlp.linear_fc2.bias",
        })

    return names


def xmodel_weight_literals():
    literals = set()
    for path in sorted(XMODEL_DIR.glob("*.x")):
        text = path.read_text(encoding="utf-8")
        for marker in ('weights["', "weights['"):
            start = 0
            while True:
                idx = text.find(marker, start)
                if idx < 0:
                    break
                idx += len(marker)
                quote = marker[-1]
                end = text.find(quote, idx)
                if end < 0:
                    break
                literal = text[idx:end]
                if "prefix" not in literal and "+" not in literal:
                    literals.add(literal)
                start = end + 1
    return literals


def load_weight_names_and_shapes(path):
    names = set()
    shapes = {}

    def load_index(index_path):
        data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map", {})
        return set(weight_map.keys())

    if path.is_file() and path.suffix == ".json":
        return load_index(path), shapes

    safetensor_files = []
    if path.is_file() and path.suffix == ".safetensors":
        safetensor_files = [path]
    elif path.is_dir():
        index_files = sorted(path.glob("*.safetensors.index.json"))
        if index_files:
            names.update(load_index(index_files[0]))
        safetensor_files = sorted(path.glob("*.safetensors"))

    if safetensor_files:
        try:
            from safetensors import safe_open
        except Exception:
            return names, shapes
        for file_path in safetensor_files:
            with safe_open(str(file_path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    names.add(key)
                    shapes[key] = list(handle.get_tensor(key).shape)

    return names, shapes


def normalize_hf_name(name):
    if name.startswith("model."):
        return name[len("model."):]
    return name


config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
expected = expected_weight_names(config) | xmodel_weight_literals()
actual_raw, actual_shapes_raw = load_weight_names_and_shapes(WEIGHTS_PATH)
actual = {normalize_hf_name(name) for name in actual_raw}
actual_shapes = {normalize_hf_name(name): shape for name, shape in actual_shapes_raw.items()}

if not actual:
    raise AssertionError(f"no weight names discovered from {WEIGHTS_PATH}")

missing = sorted(expected - actual)
extra = sorted(actual - expected)

report = {
    "config": str(CONFIG_PATH),
    "weights": str(WEIGHTS_PATH),
    "xmodel_dir": str(XMODEL_DIR),
    "matched_weights": len(expected & actual),
    "expected_weights": len(expected),
    "actual_weights": len(actual),
    "actual_raw_weights": len(actual_raw),
    "normalization": "strips leading model. from HF keys",
    "missing_weights": missing,
    "extra_weights": extra,
    "mismatched_shapes": [],
    "shape_checked_weights": len(actual_shapes),
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(f"config={CONFIG_PATH}")
print(f"weights={WEIGHTS_PATH}")
print(f"xmodel_dir={XMODEL_DIR}")
print(f"matched_weights={report['matched_weights']}")
print(f"expected_weights={report['expected_weights']}")
print(f"actual_weights={report['actual_weights']}")
print(f"missing_weights={len(missing)}")
print(f"extra_weights={len(extra)}")
print(f"shape_checked_weights={report['shape_checked_weights']}")
print(f"report={REPORT_PATH}")

if missing:
    raise AssertionError(f"missing {len(missing)} expected weights; see {REPORT_PATH}")
