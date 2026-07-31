import json
import os
import time
from pathlib import Path

import openvino as ov


SCRIPT_DIR = Path(__file__).resolve().parent
default_repo_root = (
    SCRIPT_DIR.parents[2] if len(SCRIPT_DIR.parents) > 2 else SCRIPT_DIR
)
REPO_ROOT = Path(
    os.environ.get("GARNET_REPO_ROOT", str(default_repo_root))
).resolve()
GARNET_DLL = Path(
    os.environ.get(
        "GARNET_DLL",
        str(REPO_ROOT / "out" / "build" / "x64-Release" / "bin" / "garnet.dll"),
    )
).resolve()

handles = []
openvino_libs = Path(ov.__file__).resolve().parent / "libs"
for directory in [
    GARNET_DLL.parent,
    REPO_ROOT.parent / "xlang" / "out" / "build" / "x64-Release" / "bin",
    openvino_libs,
]:
    if directory.exists() and hasattr(os, "add_dll_directory"):
        handles.append(os.add_dll_directory(str(directory)))

import xlang


snapshot_root = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3-1.7B"
    / "snapshots"
)
snapshots = sorted(snapshot_root.glob("*"))
weights_path = os.environ.get("GARNET_QWEN3_1_7B_WEIGHTS")
if weights_path:
    weights = Path(weights_path).resolve()
else:
    assert snapshots, "local Qwen3-1.7B snapshot is required"
    weights = snapshots[-1]

layers = 28
max_prompt_tokens = 64
pages = 8
page_size = 16
device = os.environ.get("GARNET_OPENVINO_DEVICE", "CPU")
precision = os.environ.get("GARNET_OPENVINO_PRECISION", "bf16").lower()
assert precision in {"bf16", "fp16", "int4_fp16"}, precision
cache_device = "".join(
    character.lower() if character.isalnum() else "_"
    for character in device
)

garnet = xlang.importModule("garnet", fromPath=str(GARNET_DLL))
assert callable(garnet.load_model), (
    f"Garnet load_model is unavailable; xlang={xlang.__file__}"
)
load_start = time.perf_counter()
model = garnet.load_model(
    str(REPO_ROOT / "xModel" / "qwen3" / "text_1_7b" / "prefill.x"),
    runtime_mode="compiled_xmodel",
    backend="openvino",
    precision=precision,
    entry_function="Qwen3Prefill",
    frontend="qwen3_text",
    weights=str(weights),
    input_shapes=[
        [1, max_prompt_tokens],
        [1, 1, max_prompt_tokens],
        [1, max_prompt_tokens],
        [layers, pages, page_size, 8, 128],
        [layers, pages, page_size, 8, 128],
        [pages],
        [1],
    ],
    input_dtypes=[
        "int64",
        "int64",
        "int64",
        "bfloat16",
        "bfloat16",
        "int32",
        "int32",
    ],
    cache_dir=str(
        SCRIPT_DIR
        / f"qwen3_1_7b_openvino_generate_64_{cache_device}_{precision}_cache"
    ),
)
status = model.runtime_status()
assert bool(status["ready"]), status
assert str(status["precision"]) == precision, status
assert bool(status["frontend_prepared"]), status
assert bool(status["engines_prepared"]), status
load_ms = (time.perf_counter() - load_start) * 1000.0

request = {
    "prompt": os.environ.get("GARNET_TEST_PROMPT", "Say: Garnet works."),
    "enable_thinking": 0,
    "max_new_tokens": int(os.environ.get("GARNET_TEST_MAX_NEW_TOKENS", "12")),
}
start = time.perf_counter()
result = model.forward(request)
elapsed_ms = (time.perf_counter() - start) * 1000.0
assert result["status"] == "ok", result
text = str(result["text"])
token_ids = [int(token) for token in result["token_ids"]]
if request["prompt"] == "Say: Garnet works.":
    assert int(result["prompt_token_count"]) == 18, result
else:
    assert int(result["prompt_token_count"]) > 0, result
assert 0 < int(result["generated_token_count"]) <= request["max_new_tokens"], result
if precision == "int4_fp16" and request["prompt"] == "Say: Garnet works.":
    # Quantized logits are not expected to be token-identical to BF16, but
    # this fixed instruction must remain coherent and follow the prompt.
    assert (
        text.startswith("Garnet works.")
        or text.startswith("Sure!")
    ), result
elif request["prompt"] == "Say: Garnet works.":
    # Independent PyTorch Qwen3 reference has token 39814 ("Sure") as top-1.
    assert token_ids[0] == 39814, result
    assert text.startswith("Sure!"), result
result_json_path = os.environ.get("GARNET_TEST_RESULT_JSON")
if result_json_path:
    Path(result_json_path).write_text(
        json.dumps(
            {
                "text": text,
                "token_ids": token_ids,
                "prompt_token_count": int(result["prompt_token_count"]),
                "generated_token_count": int(
                    result["generated_token_count"]
                ),
                "decode_tokens_per_second": float(
                    result["decode_tokens_per_second"]
                ),
                "time_to_first_token_ms": float(
                    result["time_to_first_token_ms"]
                ),
                "load_ms": load_ms,
                "request_ms": elapsed_ms,
            }
        ),
        encoding="utf-8",
    )
print(
    "Qwen3-1.7B OpenVINO real-prompt generation passed: "
    f"device={device}, precision={precision}, load_ms={load_ms:.2f}, "
    f"request_ms={elapsed_ms:.2f}, "
    f"ttft_ms={result['time_to_first_token_ms']:.2f}, "
    f"prompt_tokens={result['prompt_token_count']}, "
    f"decode_tokens_per_second={result['decode_tokens_per_second']:.2f}, "
    f"token_ids={token_ids}, text={text!r}"
)
