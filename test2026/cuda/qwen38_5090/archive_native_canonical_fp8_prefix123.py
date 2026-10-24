"""Retain all closed deeper-prefix observations using lossless SHA objects."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

root = Path("D:/CantorAI/5090-study-20261023")
results = root / "results"
folder = results / "native-canonical-fp8-prefix123-attempt1"
target = results / "native-canonical-fp8-prefix123-attempt1-evidence.zip"
assert not target.exists()
workflow = json.loads((folder / "workflow.json").read_text())
outer = results / "native-canonical-fp8-prefix123-controller-ledger"
for path in (folder / "ledger/commands.jsonl", outer / "commands.jsonl"):
    events = [json.loads(l) for l in path.read_text().splitlines()]
    assert events[-1]["event"] == "finished"
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
build_path = results / "native-canonical-fp8-build-attempt2/manifest.json"
build = json.loads(build_path.read_text())
library = hashlib.sha256((root / "build/bin/garnet.dll").read_bytes()).hexdigest()
assert library == build["library_sha256"] == workflow["library_sha256"]
files = [p for p in folder.rglob("*") if p.is_file()]
for name in (outer.name, "native-canonical-fp8-build-attempt2", "native-canonical-fp8-family-scope-install", "native-canonical-fp8-family-install-ledger"):
    files += [p for p in (results / name).rglob("*") if p.is_file()]
files += [root / n for n in ("run-native-canonical-fp8-prefix123.py", "test-native-canonical-fp8-prefix123.bat",
    "run-native-canonical-fp8-build.py", "install-fp8-gdn-family-scope.py", "build-layer0-stages.bat")]
files += [root / "Garnet" / n for n in list(build["source_sha256"]) + [
    "test2026/cuda/qwen38_5090/probe_native_prefix_batch.py", "test2026/cuda/qwen38_5090/run_logged.py",
    "test2026/cuda/qwen38_5090/monitor_native_probe.py", "src/backends/tensorrt/qwen35_lowering.h",
    "src/backends/tensorrt/trt_builder.cpp", "xModel/qwen35/vl_0_8b/cuda_packed_decoder.py",
    "xModel/qwen35/vl_0_8b/cuda_packed_program.py", "xModel/qwen38/vl_27b/model.py"]]
files += [results / "native-canonical-fp8-stages-attempt1-verified.json", Path(__file__)]
observations = {}; objects = {}
for p in files:
    raw = p.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    observations[p.relative_to(root).as_posix()] = digest
    if digest in objects: assert objects[digest] == raw
    else: objects[digest] = raw
manifest = {"storage": "sha256-objects-v1", "files": observations,
    "objects": {h: len(b) for h, b in objects.items()}, "library_sha256": library,
    "linked_archives": {n: hashlib.sha256((results / n).read_bytes()).hexdigest() for n in (
        "native-prefix12-internals-evidence.zip", "native-prefix23-batch-evidence.zip",
        "native-canonical-fp8-stages-attempt1-evidence.zip")},
    "scope": "All actual prefix1/2/3 observations, repeats and raw hidden/state bytes preserved by content deduplication. No full64-layer ID, quality or performance admission. Optional module-path query missed the already-exited worker; no loaded-module observation claimed."}
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
    for h, b in objects.items(): z.writestr("objects/" + h, b)
    z.writestr("evidence-manifest.json", json.dumps(manifest, indent=2))
print(json.dumps({"bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    "file_observations": len(observations), "unique_objects": len(objects)}))
