import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL = os.environ.get("GARNET_OLLAMA_MODEL", "qwen3:1.7b")
TOKEN_CAP = int(os.environ.get("GARNET_BENCH_TOKENS", "64"))
OLLAMA_GENERATE = "http://127.0.0.1:11434/api/generate"
OLLAMA_PS = "http://127.0.0.1:11434/api/ps"
PROMPTS = [
    "Write three short sentences explaining why Garnet works.",
    "In two sentences, compare CPU and GPU inference.",
    "Explain paged KV cache in three simple sentences.",
    "Give three concise tips for testing an LLM runtime.",
]


def ollama_loaded() -> bool:
    try:
        with urllib.request.urlopen(OLLAMA_PS, timeout=10) as response:
            state = json.loads(response.read().decode("utf-8"))
        return any(
            str(model.get("name", "")).split(":")[0]
            == MODEL.split(":")[0]
            for model in state.get("models", [])
        )
    except Exception:
        return False


def unload_ollama() -> None:
    subprocess.run(
        ["ollama", "stop", MODEL],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    deadline = time.monotonic() + 30
    while ollama_loaded() and time.monotonic() < deadline:
        time.sleep(0.25)
    if ollama_loaded():
        raise RuntimeError(f"Ollama model did not unload: {MODEL}")


def run_garnet(prompt: str) -> dict:
    environment = os.environ.copy()
    environment["GARNET_TEST_PROMPT"] = prompt
    environment["GARNET_TEST_MAX_NEW_TOKENS"] = str(TOKEN_CAP)
    with tempfile.TemporaryDirectory(prefix="garnet-cold-bench-") as temp:
        result_path = Path(temp) / "result.json"
        environment["GARNET_TEST_RESULT_JSON"] = str(result_path)
        wall_start = time.perf_counter()
        process = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "test_openvino_qwen3_1_7b_generate.py")],
            cwd=str(SCRIPT_DIR),
            env=environment,
            text=True,
            capture_output=True,
            timeout=900,
            check=False,
        )
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        if process.returncode != 0 or not result_path.is_file():
            raise RuntimeError(
                "Garnet benchmark failed:\n" + process.stdout + process.stderr
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "load_ms": float(result["load_ms"]),
        "wall_ms": wall_ms,
        "ttft_ms": float(result["time_to_first_token_ms"]),
        "decode_tps": float(result["decode_tokens_per_second"]),
        "tokens": int(result["generated_token_count"]),
        "text": str(result["text"]),
    }


def run_ollama(prompt: str) -> dict:
    unload_ollama()
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": 0,
        "options": {
            "temperature": 0,
            "seed": 0,
            "num_predict": TOKEN_CAP,
        },
    }
    request = urllib.request.Request(
        OLLAMA_GENERATE,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    wall_start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.loads(response.read().decode("utf-8"))
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    count = int(result["eval_count"])
    decode_seconds = int(result["eval_duration"]) / 1.0e9
    output = {
        "load_ms": int(result.get("load_duration", 0)) / 1.0e6,
        "wall_ms": wall_ms,
        "prompt_eval_ms": int(result.get("prompt_eval_duration", 0)) / 1.0e6,
        "decode_tps": count / decode_seconds,
        "tokens": count,
        "text": str(result.get("response", "")),
    }
    deadline = time.monotonic() + 30
    while ollama_loaded() and time.monotonic() < deadline:
        time.sleep(0.25)
    output["released"] = not ollama_loaded()
    if not output["released"]:
        raise RuntimeError("Ollama did not release the model after keep_alive=0")
    return output


records = []
unload_ollama()
for index, prompt in enumerate(PROMPTS):
    if index % 2 == 0:
        garnet = run_garnet(prompt)
        time.sleep(2)
        ollama = run_ollama(prompt)
    else:
        ollama = run_ollama(prompt)
        time.sleep(2)
        garnet = run_garnet(prompt)
    record = {
        "prompt": prompt,
        "order": "garnet_first" if index % 2 == 0 else "ollama_first",
        "garnet": garnet,
        "ollama": ollama,
        "garnet_over_ollama": garnet["decode_tps"] / ollama["decode_tps"],
    }
    records.append(record)
    print(json.dumps(record, ensure_ascii=True), flush=True)
    time.sleep(2)

garnet_speeds = [record["garnet"]["decode_tps"] for record in records]
ollama_speeds = [record["ollama"]["decode_tps"] for record in records]
summary = {
    "token_cap": TOKEN_CAP,
    "prompts": len(records),
    "garnet": {
        "mean_decode_tps": statistics.fmean(garnet_speeds),
        "median_decode_tps": statistics.median(garnet_speeds),
        "mean_load_ms": statistics.fmean(
            record["garnet"]["load_ms"] for record in records
        ),
    },
    "ollama": {
        "mean_decode_tps": statistics.fmean(ollama_speeds),
        "median_decode_tps": statistics.median(ollama_speeds),
        "mean_load_ms": statistics.fmean(
            record["ollama"]["load_ms"] for record in records
        ),
        "released_after_every_prompt": all(
            record["ollama"]["released"] for record in records
        ),
    },
    "garnet_over_ollama_mean": (
        statistics.fmean(garnet_speeds) / statistics.fmean(ollama_speeds)
    ),
    "records": records,
}
print("COLD_MULTI_PROMPT_SUMMARY=" + json.dumps(summary, ensure_ascii=True))
