import os
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
    import xlang
except Exception as exc:
    skip(f"xlang Python module not available: {exc}")

try:
    garnet = xlang.importModule("garnet", fromPath=str(garnet_dll))
except Exception as exc:
    skip(f"failed to import Garnet from {garnet_dll}: {exc}")

image_path, metadata = first_dataset_sample()
prompt = qwen_prompt(metadata)
xmodel_path = REPO_ROOT / "qwen_vl" / "xmodel" / "qwen_vl_model.x"
weights_path = os.environ.get("GARNET_QWEN_VL_WEIGHTS", "").strip()
cache_dir = Path(os.environ.get("GARNET_QWEN_VL_CACHE_DIR", Path(__file__).with_name("engine_cache")))

if not weights_path:
    skip("set GARNET_QWEN_VL_WEIGHTS to safetensors/bin weights when loader is ready")

print(f"garnet={garnet_dll}")
print(f"xmodel={xmodel_path}")
print(f"image={image_path}")
print(f"prompt={prompt}")

try:
    weights = garnet.load_weights(weights_path) if hasattr(garnet, "load_weights") else weights_path
    engine = garnet.load_model(
        str(xmodel_path),
        weights=weights,
        cache_dir=str(cache_dir),
    )

    # Temporary input contract until real Qwen processor parity is wired:
    # image tensor is NHWC float32, prompt bytes are uint8. Replace with the
    # same processor/tokenizer tensors used by the HF reference test.
    image_tensor = garnet.tensor(np.zeros((1, 1, 1, 3), dtype=np.float32))
    prompt_tensor = garnet.tensor(np.frombuffer(prompt.encode("utf-8"), dtype=np.uint8))
    output = engine.forward(image_tensor, prompt_tensor)

    assert output is not None, "Garnet Qwen-VL output is null"
    print("Garnet Qwen-VL scaffold executed. Next step: replace temporary inputs with processor parity tensors.")
except Exception as exc:
    print(f"Garnet Qwen-VL scaffold failed: {exc}")
    raise
