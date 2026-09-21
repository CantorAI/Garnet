# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TOKEN_COUNT = int(os.environ.get("GARNET_BENCH_TOKENS", "64"))
PROMPT = os.environ.get("GARNET_BENCH_PROMPT", "Say: Garnet works.")
OLLAMA_MODEL = os.environ.get("GARNET_OLLAMA_MODEL", "qwen3:1.7b")
OLLAMA_URL = os.environ.get(
    "GARNET_OLLAMA_URL", "http://127.0.0.1:11434/api/generate"
)

# Run the real Garnet frontend, tokenizer, prefill, stateful decode, and Top-1
# path in an isolated process. Exiting that process releases the checkpoint and
# OpenVINO engine before Ollama is loaded, which avoids cross-runtime memory
# pressure on machines with limited RAM.
garnet_environment = os.environ.copy()
garnet_environment["GARNET_OPENVINO_DEVICE"] = "CPU"
garnet_environment["GARNET_OPENVINO_PRECISION"] = "int4_fp16"
garnet_environment["GARNET_TEST_MAX_NEW_TOKENS"] = str(TOKEN_COUNT)
garnet_environment["GARNET_TEST_PROMPT"] = PROMPT
with tempfile.TemporaryDirectory(prefix="garnet_ollama_bench_") as temp_dir:
    result_path = Path(temp_dir) / "garnet-result.json"
    garnet_environment["GARNET_TEST_RESULT_JSON"] = str(result_path)
    garnet_process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "test_openvino_qwen3_1_7b_generate.py"),
        ],
        cwd=str(SCRIPT_DIR),
        env=garnet_environment,
        text=True,
        capture_output=True,
        timeout=900,
        check=True,
    )
    if garnet_process.stdout:
        print(garnet_process.stdout, end="")
    if garnet_process.stderr:
        print(garnet_process.stderr, file=sys.stderr, end="")
    garnet_result = json.loads(result_path.read_text(encoding="utf-8"))
garnet_text = str(garnet_result["text"])
garnet_tps = float(garnet_result["decode_tokens_per_second"])

payload = {
    "model": OLLAMA_MODEL,
    "prompt": PROMPT,
    "stream": False,
    "think": False,
    "keep_alive": "10m",
    "options": {
        "temperature": 0,
        "seed": 0,
        "num_predict": TOKEN_COUNT,
    },
}
request = urllib.request.Request(
    OLLAMA_URL,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
ollama_start = time.perf_counter()
with urllib.request.urlopen(request, timeout=600) as response:
    ollama_result = json.loads(response.read().decode("utf-8"))
ollama_wall_seconds = time.perf_counter() - ollama_start
ollama_count = int(ollama_result["eval_count"])
ollama_seconds = int(ollama_result["eval_duration"]) / 1.0e9
ollama_tps = ollama_count / ollama_seconds
ollama_text = str(ollama_result.get("response", ""))

summary = {
    "prompt": PROMPT,
    "requested_tokens": TOKEN_COUNT,
    "garnet": {
        "backend": "openvino",
        "decode_tokens_per_second": garnet_tps,
        "generated_tokens": int(garnet_result["generated_token_count"]),
        "time_to_first_token_ms": float(
            garnet_result["time_to_first_token_ms"]
        ),
        "text": garnet_text,
    },
    "ollama": {
        "model": OLLAMA_MODEL,
        "decode_tokens_per_second": ollama_tps,
        "generated_tokens": ollama_count,
        "wall_seconds": ollama_wall_seconds,
        "text": ollama_text,
    },
    "garnet_over_ollama": garnet_tps / ollama_tps,
}
print(json.dumps(summary, indent=2, ensure_ascii=False))

assert garnet_text.strip(), summary
assert ollama_text.strip(), summary
assert garnet_tps >= ollama_tps, summary
