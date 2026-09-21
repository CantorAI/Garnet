# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import garnet
import json
import os


def env(name, fallback=""):
    value = os.environ.get(name)
    if value is None or len(value) == 0:
        return fallback
    return value


acceleration_root = env("GARNET_ACCELERATION_ROOT")
if len(acceleration_root) > 0:
    activated = json.loads(garnet.activate_acceleration_path_json(acceleration_root))
    if activated["status"] != "ready":
        raise RuntimeError(str(activated))

hardware = json.loads(garnet.detect_acceleration_json())
print(json.dumps(hardware))

model_root = env("GARNET_MODEL_ROOT")
if len(model_root) == 0:
    raise RuntimeError("Set GARNET_MODEL_ROOT to the installed Qwen3-1.7B model")

xmodel_root = env("GARNET_XMODEL_ROOT", model_root + "/xmodel")
cache_root = env("GARNET_CACHE_ROOT", model_root + "/compiled_cache")
loaded_json = garnet.serve_model(model_root, xmodel_root, cache_root, "{}", "Qwen3-1.7B")
loaded = json.loads(loaded_json)
if not loaded["ready"]:
    raise RuntimeError(str(loaded))

try:
    prompt = env("GARNET_PROMPT", "Explain Garnet in one sentence.")
    result = json.loads(garnet.infer_json(prompt, "", 64))
    if result["status"] != "ok":
        raise RuntimeError(str(result))
    print(result["text"])
finally:
    garnet.stop_serving()
