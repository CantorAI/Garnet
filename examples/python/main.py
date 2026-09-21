# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_dll_handles = []


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).resolve() if value else None


runtime_root = env_path("GARNET_RUNTIME_ROOT")
if runtime_root is None:
    raise RuntimeError("Set GARNET_RUNTIME_ROOT to the directory containing garnet.dll")

for candidate in (runtime_root, runtime_root / "cpython_bridge", runtime_root / "modules"):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

if os.name == "nt" and hasattr(os, "add_dll_directory"):
    _dll_handles.append(os.add_dll_directory(str(runtime_root)))

import xlang3

garnet = xlang3.importModule("garnet", fromPath=str(runtime_root))

acceleration_root = env_path("GARNET_ACCELERATION_ROOT")
if acceleration_root:
    activated = json.loads(garnet.activate_acceleration_path_json(str(acceleration_root)))
    if activated.get("status") != "ready":
        raise RuntimeError(activated)

print(json.dumps(json.loads(garnet.detect_acceleration_json()), indent=2))

model_root = env_path("GARNET_MODEL_ROOT")
if model_root:
    xmodel_root = env_path("GARNET_XMODEL_ROOT") or model_root / "xmodel"
    cache_root = env_path("GARNET_CACHE_ROOT") or model_root / "compiled_cache"
    loaded = json.loads(garnet.serve_model(
        str(model_root), str(xmodel_root), str(cache_root), "{}", "Qwen3-1.7B"
    ))
    if not loaded.get("ready"):
        raise RuntimeError(loaded)
    try:
        prompt = os.environ.get("GARNET_PROMPT", "Explain Garnet in one sentence.")
        result = json.loads(garnet.infer_json(prompt, "", 64))
        if result.get("status") != "ok":
            raise RuntimeError(result)
        print(result["text"])
    finally:
        garnet.stop_serving()
else:
    print("Set GARNET_MODEL_ROOT to run real Qwen3-1.7B inference.")
