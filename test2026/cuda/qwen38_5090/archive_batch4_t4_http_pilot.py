"""Retain the original full-model B2 ID gate, including any numerical failure."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

root = Path("D:/CantorAI/5090-study-20261023")
results = root / "results"
folder = results / "batch4-t4-http-pilot-attempt1"
target = results / "batch4-t4-http-pilot-attempt1-evidence.zip"
assert not target.exists()
workflow = json.loads((folder / "workflow.json").read_text())
for p in (folder / "ledger/commands.jsonl",):
    events = [json.loads(l) for l in p.read_text().splitlines()]
    assert events[-1]["event"] == "finished"
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
build_path = results / "native-full-attention-fp8-build-attempt1/manifest.json"
build = json.loads(build_path.read_text())
library = hashlib.sha256((root / "build/bin/garnet.dll").read_bytes()).hexdigest()
assert library == build["library_sha256"] == workflow["library_sha256"]
files = [p for p in folder.rglob("*") if p.is_file()]
files += [root / n for n in ("Garnet/test2026/cuda/qwen38_5090/run_batch4_t4_http_pilot.py", "test-batch4-t4-http-pilot.bat")]
files += [results / "unicode-transport-repair2-verified.json", results / "batch-http-serialization-regression.json", results / "batch-http-pilot-attempt1-failure-verified.json", results / "batch4-t4-stream-protocol-attempt1-verified.json", root / "models/Qwen3.8-27B-NVFP4/config.json", root / "serve-native-batch4-t4-shared-worker.bat", results / "shared-full-b4-t4-attempt1-verified.json"]
files += [root / "Garnet" / n for n in list(build["source_sha256"]) + [
    "test2026/cuda/qwen38_5090/probe_native_batch.py", "test2026/cuda/qwen38_5090/probe_shared_full_b4_t4.py", "test2026/cuda/qwen38_5090/native_batch4_t4_stream_worker.py", "test2026/cuda/qwen38_5090/probe_batch4_t4_http_pilot.py", "test2026/cuda/qwen38_5090/native_worker_io.py", "test2026/cuda/qwen38_5090/native_batch_http_bridge.py", "test2026/cuda/qwen38_5090/native_http_bridge.py", "test2026/cuda/qwen38_5090/benchmark_stream.py", "test2026/cuda/qwen38_5090/compare_http_runs.py", "test2026/cuda/qwen38_5090/metrics.py", "test2026/cuda/qwen38_5090/sample_vram.py", "test2026/cuda/qwen38_5090/native_prompt.py", "test2026/cuda/qwen38_5090/run_logged.py",
    "test2026/cuda/qwen38_5090/monitor_native_probe.py", "xModel/qwen38/vl_27b/model.py",
    "xModel/qwen35/vl_0_8b/cuda_packed_decoder.py", "xModel/qwen35/vl_0_8b/cuda_packed_program.py"]]
files += [results / "native-mtp3-http" / (key + suffix) for key in (
    "s3-text-c1-r0-greedy", "s3-image-c1-r0-greedy", "red", "green") for suffix in ("-request.json", "-response.json")]
files += [results / "full-attention-policy-full-b2-attempt1-verified.json", build_path, results / "full-attention-policy-candidate-stages-attempt1-verified.json", Path(__file__)]
files += [root / n for n in ("capture-full-b2-wait-dump.py", "inspect-full-b2-file-wait.py") if (root / n).exists()]
raw = {p.relative_to(root).as_posix(): p.read_bytes() for p in files}
manifest = {"library_sha256": library, "files": {n: hashlib.sha256(b).hexdigest() for n, b in raw.items()},
    "linked_archives": {n: hashlib.sha256((results / n).read_bytes()).hexdigest() for n in (
        "unicode-transport-repair2-evidence.zip", "batch-http-pilot-attempt1-evidence.zip", "batch4-t4-stream-protocol-attempt1-evidence.zip", "vllm-graph-linkfix-evidence.zip", "sglang-corrected-ipv4-evidence.zip", "shared-full-b4-t4-attempt1-evidence.zip", "full-attention-policy-full-b2-attempt1-evidence.zip", "full-attention-policy-candidate-stages-attempt1-evidence.zip", "native-batch2-initial-failure-evidence.zip")},
    "scope": "B2 HTTP quality and concurrency pilot, all successes/failures/raw SSE/worker events/dispatch groups retained. Disk/source provenance retained; no independent live loaded-module receipt. Not HTTP, MTP, broad quality or throughput admission."}
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
    for n, b in raw.items(): z.writestr(n, b)
    z.writestr("evidence-manifest.json", json.dumps(manifest, indent=2))
print(json.dumps({"bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "files": len(raw)}))
