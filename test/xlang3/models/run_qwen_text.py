"""Run with xlang3.exe: script backend weights-directory cache-directory."""

import sys
import time
from pathlib import Path

import garnet


assert len(sys.argv) == 4, "expected backend, weights directory, cache directory"
backend = sys.argv[1]
assert backend in ("openvino", "tensorrt"), backend
weights = Path(sys.argv[2]).resolve()
cache = Path(sys.argv[3]).resolve()
root = Path(__file__).resolve().parents[3]
assert (weights / "model.safetensors.index.json").is_file(), weights
assert (weights / "tokenizer.json").is_file(), weights

print("Loading Qwen3-1.7B:", backend, str(weights), flush=True)
started = time.perf_counter()
model = garnet.load_model(
    str(root / "xModel" / "qwen3" / "text_1_7b" / "prefill.py"),
    runtime_mode="compiled_xmodel",
    backend=backend,
    precision="bf16",
    entry_function="Qwen3Prefill",
    frontend="qwen3_text",
    weights=str(weights),
    input_shapes=[
        [1, 64], [1, 1, 64], [1, 64],
        [28, 8, 16, 8, 128], [28, 8, 16, 8, 128], [8], [1],
    ],
    input_dtypes=[
        "int64", "int64", "int64", "bfloat16", "bfloat16", "int32", "int32",
    ],
    cache_dir=str(cache),
)
status = model.runtime_status()
print("Runtime status:", status["state"], "backend=", status["backend"],
      "precision=", status["precision"], flush=True)
assert status["ready"], status
assert status["frontend_prepared"], status
assert status["engines_prepared"], status
assert status["precision"] == "bf16", status
load_seconds = time.perf_counter() - started
started = time.perf_counter()
result = model.forward({
    "prompt": "Say: Garnet works.", "enable_thinking": 0, "max_new_tokens": 12,
})
print("Generation result:", result, flush=True)
assert result["status"] == "ok", result
assert int(result["prompt_token_count"]) == 18, result
assert 0 < int(result["generated_token_count"]) <= 12, result
# Existing independent PyTorch BF16 reference: first token is 'Sure'.
assert int(result["token_ids"][0]) == 39814, result
assert str(result["text"]).startswith("Sure!"), result
request_seconds = time.perf_counter() - started
# A second request must start from fresh KV state, not the previous generation.
repeated = model.forward({
    "prompt": "Say: Garnet works.", "enable_thinking": 0, "max_new_tokens": 12,
})
assert repeated["status"] == "ok", repeated
assert repeated["token_ids"] == result["token_ids"], repeated
assert repeated["text"] == result["text"], repeated
assert int(repeated["prompt_token_count"]) == 18, repeated
print("garnet-qwen3-real-generation-passed:", backend,
      "load_seconds=", load_seconds,
      "request_seconds=", request_seconds, "repeated_request=passed", flush=True)
