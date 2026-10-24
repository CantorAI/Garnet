"""Retain the original full-model B2 ID gate, including any numerical failure."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

root = Path("D:/CantorAI/5090-study-20261023")
results = root / "results"
folder = results / "shared-mtp3-full-b2-attempt2"
target = results / "shared-mtp3-full-b2-attempt2-evidence.zip"
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
files += [results / "batch-http-pilot-attempt2-verified.json", results / "shared-full-b2-attempt1-verified.json"]
files += [root / n for n in ("Garnet/test2026/cuda/qwen38_5090/run_shared_mtp3_full_b2_workspace.py", "test-shared-mtp3-full-b2-workspace.bat")]
files += [root / "Garnet" / n for n in list(build["source_sha256"]) + [
    "test2026/cuda/qwen38_5090/probe_native_batch.py", "test2026/cuda/qwen38_5090/probe_shared_mtp3_full_b2_workspace.py", "test2026/cuda/qwen38_5090/run_logged.py",
    "test2026/cuda/qwen38_5090/monitor_native_probe.py", "xModel/qwen38/vl_27b/model.py",
    "xModel/qwen35/vl_0_8b/cuda_packed_decoder.py", "xModel/qwen35/vl_0_8b/cuda_packed_program.py", "xModel/qwen35/vl_0_8b/cuda_speculative.py", "xModel/qwen35/vl_0_8b/cuda_packed_replay.py", "xModel/qwen35/vl_0_8b/cuda_mtp.py", "xModel/qwen35/vl_0_8b/cuda_verify.py"]]
files += [results / "native-mtp3-http" / (key + suffix) for key in (
    "s3-text-c1-r0-greedy", "s3-image-c1-r0-greedy", "red", "green") for suffix in ("-request.json", "-response.json")]
files += [results / "full-attention-policy-full-b2-attempt1-verified.json", build_path, results / "full-attention-policy-candidate-stages-attempt1-verified.json", Path(__file__)]
files += [root / n for n in ("capture-full-b2-wait-dump.py", "inspect-full-b2-file-wait.py") if (root / n).exists()]
raw = {p.relative_to(root).as_posix(): p.read_bytes() for p in files}
manifest = {"library_sha256": library, "files": {n: hashlib.sha256(b).hexdigest() for n, b in raw.items()},
    "linked_archives": {n: hashlib.sha256((results / n).read_bytes()).hexdigest() for n in (
        "batch-http-pilot-attempt2-evidence.zip", "shared-full-b2-attempt1-evidence.zip", "full-attention-policy-full-b2-attempt1-evidence.zip", "full-attention-policy-candidate-stages-attempt1-evidence.zip", "native-batch2-initial-failure-evidence.zip")},
    "scope": "Closed B2 MTP3 retained-ID and complete ordinary-state hash witness, success or failure retained. Disk/source provenance retained; no independent live loaded-module receipt. Not HTTP, MTP, broad quality or throughput admission."}
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
    for n, b in raw.items(): z.writestr(n, b)
    z.writestr("evidence-manifest.json", json.dumps(manifest, indent=2))
print(json.dumps({"bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "files": len(raw)}))
