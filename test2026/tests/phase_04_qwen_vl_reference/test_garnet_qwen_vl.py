# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import json
from pathlib import Path

from common import (
    REPO_ROOT,
    add_windows_dll_dirs,
    discover_garnet_dll,
    env_flag,
    first_dataset_sample,
    qwen_prompt,
    skip,
)


print("Phase 04: Garnet Qwen-VL parity scaffold")

if not env_flag("RUN_GARNET_QWEN_VL"):
    skip("set RUN_GARNET_QWEN_VL=1 once Garnet Qwen-VL runtime is ready")

try:
    import numpy as np
except Exception as exc:
    skip(f"numpy is required for enabled Garnet Qwen-VL parity: {exc}")

garnet_dll = discover_garnet_dll()
if garnet_dll is None:
    skip("garnet.dll not found; build Garnet or set GARNET_DLL_PATH")

_dll_dir_handles = add_windows_dll_dirs(garnet_dll)

try:
    import xlang3
except Exception as exc:
    skip(f"xlang Python module not available: {exc}")

try:
    garnet = xlang3.importModule("garnet", fromPath=str(garnet_dll))
except Exception as exc:
    skip(f"failed to import Garnet from {garnet_dll}: {exc}")

image_path, metadata = first_dataset_sample()
prompt = qwen_prompt(metadata)
xmodel_path = REPO_ROOT / "xModel" / "qwen3" / "vl_2b_instruct" / "qwen_vl_model.py"
weights_path = os.environ.get("GARNET_QWEN_VL_WEIGHTS", "").strip()
cache_dir = Path(os.environ.get("GARNET_QWEN_VL_CACHE_DIR", Path(__file__).with_name("engine_cache")))
artifact_dir = REPO_ROOT / "test2026" / "artifacts" / "qwen_vl_reference"


def newest_processor_dump():
    explicit_json = os.environ.get("GARNET_QWEN_VL_PROCESSOR_DUMP_JSON", "").strip()
    explicit_npz = os.environ.get("GARNET_QWEN_VL_PROCESSOR_DUMP_NPZ", "").strip()
    if explicit_json and explicit_npz:
        return Path(explicit_json), Path(explicit_npz)

    candidates = sorted(
        artifact_dir.glob("processor_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for json_path in candidates:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        npz_path = Path(metadata.get("npz", ""))
        if npz_path.exists():
            return json_path, npz_path
        sibling = json_path.with_suffix(".npz")
        if sibling.exists():
            return json_path, sibling
    return None, None


def load_processor_inputs():
    json_path, npz_path = newest_processor_dump()
    if json_path is None or npz_path is None:
        skip(
            "processor dump not found; run test_processor_contract.py with "
            "RUN_HF_QWEN_VL_PROCESSOR=1 or set GARNET_QWEN_VL_PROCESSOR_DUMP_JSON/NPZ"
        )

    dump_metadata = json.loads(json_path.read_text(encoding="utf-8"))
    arrays = np.load(npz_path)
    inputs = {key: arrays[key] for key in arrays.files}
    if "input_ids" not in inputs:
        raise AssertionError(f"processor dump missing input_ids: {npz_path}")
    if "attention_mask" in inputs and inputs["attention_mask"].shape != inputs["input_ids"].shape:
        raise AssertionError(
            f"attention_mask shape {inputs['attention_mask'].shape} != input_ids shape {inputs['input_ids'].shape}"
        )
    return dump_metadata, inputs, json_path, npz_path

print(f"garnet={garnet_dll}")
print(f"xmodel={xmodel_path}")
print(f"image={image_path}")
print(f"prompt={prompt}")

try:
    dump_metadata, processor_inputs, dump_json_path, dump_npz_path = load_processor_inputs()
    print(f"processor_dump_json={dump_json_path}")
    print(f"processor_dump_npz={dump_npz_path}")
    print(f"processor_prompt_id={dump_metadata.get('prompt_id')}")
    for key, value in processor_inputs.items():
        print(f"processor_input {key}: shape={list(value.shape)} dtype={value.dtype}")

    if not weights_path:
        skip("processor dump loaded; set GARNET_QWEN_VL_WEIGHTS to run Garnet model loading/forward")

    weights = garnet.load_weights(weights_path) if hasattr(garnet, "load_weights") else weights_path
    engine = garnet.load_model(
        str(xmodel_path),
        weights=weights,
        cache_dir=str(cache_dir),
    )

    output = engine.forward(*[processor_inputs[key] for key in sorted(processor_inputs)])

    assert output is not None, "Garnet Qwen-VL output is null"
    print("Garnet Qwen-VL scaffold executed with HF processor dump tensors.")
except Exception as exc:
    print(f"Garnet Qwen-VL scaffold failed: {exc}")
    raise
