"""Losslessly deduplicate closed stage observations by raw content hash.

Every original file path and its bytes remain represented. Repeated tensors
share an object, reducing redundant disk/transfer use without dropping trials.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

root = Path("D:/CantorAI/5090-study-20261023")
results = root / "results"
folder = results / "native-canonical-fp8-stages-attempt1"
target = results / "native-canonical-fp8-stages-attempt1-evidence.zip"
assert not target.exists()
workflow = json.loads((folder / "workflow.json").read_text())
for path in folder.glob("*/ledger/commands.jsonl"):
    events = [json.loads(l) for l in path.read_text().splitlines()]
    assert events[-1]["event"] == "finished"
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
build_path = results / "native-canonical-fp8-build-attempt1/manifest.json"
build = json.loads(build_path.read_text())
library = hashlib.sha256((root / "build/bin/garnet.dll").read_bytes()).hexdigest()
assert library == build["library_sha256"] == workflow["library_sha256"]
files = [p for p in folder.rglob("*") if p.is_file()]
for name in ("native-canonical-fp8-build-attempt1", "native-canonical-fp8-source-install", "native-canonical-fp8-install-ledger"):
    files += [p for p in (results / name).rglob("*") if p.is_file()]
files += [root / n for n in ("run-native-canonical-fp8-stages.py", "test-native-canonical-fp8-stages.bat",
    "run-native-canonical-fp8-build.py", "install-native-canonical-fp8.py", "build-layer0-stages.bat")]
files += [root / "Garnet" / n for n in list(build["source_sha256"]) + [
    "test2026/cuda/qwen38_5090/probe_layer0_real_stages.py", "test2026/cuda/qwen38_5090/run_logged.py",
    "test2026/cuda/qwen38_5090/monitor_native_probe.py", "src/backends/tensorrt/qwen35_lowering.h",
    "src/backends/tensorrt/trt_builder.cpp", "xModel/qwen35/vl_0_8b/cuda_packed_decoder.py"]]
files += [results / "native-canonical-fp8-attempt2-verified.json", Path(__file__)]
observations = {}; objects = {}
for p in files:
    raw = p.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    observations[p.relative_to(root).as_posix()] = digest
    if digest in objects: assert objects[digest] == raw
    else: objects[digest] = raw
manifest = {"storage": "sha256-objects-v1", "files": observations,
    "objects": {h: len(b) for h, b in objects.items()}, "library_sha256": library,
    "linked_archives": {n: hashlib.sha256((results / n).read_bytes()).hexdigest() for n in (
        "layer0-real-stage-attempt2-evidence.zip", "native-canonical-fp8-attempt2-evidence.zip",
        "layer0-build-full-control-attempt1-evidence.zip")},
    "scope": "All closed actual-stage raw file observations preserved through lossless content deduplication, including repeats/failures. No full64-layer batching or speed admission. DLL bytes retained remotely under manifest hash."}
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
    for h, b in objects.items(): z.writestr("objects/" + h, b)
    z.writestr("evidence-manifest.json", json.dumps(manifest, indent=2))
print(json.dumps({"bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    "file_observations": len(observations), "unique_objects": len(objects)}))
