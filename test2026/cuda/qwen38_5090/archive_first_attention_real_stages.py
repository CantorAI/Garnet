"""Retain all closed stage observations and native-build provenance losslessly."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

root = Path("D:/CantorAI/5090-study-20261023")
results = root / "results"
folder = results / "first-attention-real-stages-attempt1"
outer = results / "first-attention-real-stages-controller-ledger"
build = results / "native-attention-stage-build-attempt1"
target = results / "first-attention-real-stages-attempt1-evidence.zip"
assert not target.exists()
for directory in (folder / "ledger", outer, build / "build-ledger"):
    events = [json.loads(v) for v in (directory / "commands.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "finished"
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
sha = lambda raw: hashlib.sha256(raw).hexdigest()
manifest = json.loads((build / "manifest.json").read_text())
assert sha((root / "build/bin/garnet.dll").read_bytes()) == manifest["library_sha256"]
files = []
for directory in (folder, outer, build, results / "native-attention-stage-build-controller-ledger",
        results / "first-attention-stage-witness", results / "first-attention-stage-witness-ledger"):
    files.extend(p for p in directory.rglob("*") if p.is_file())
files.extend(root / name for name in ("run-first-attention-real-stages.py", "test-first-attention-real-stages.bat",
    "prepare-first-attention-stage-witness.py", "run-native-attention-stage-build.py", "build-layer0-stages.bat"))
files.extend(results / name for name in ("native-attention-stage-source.zip", "native-attention-stage-original-snapshot.zip",
    "native-canonical-fp8-prefix4-hidden-geometry-attempt1-verified.json"))
for name, digest in manifest["source_sha256"].items():
    p = root / "Garnet" / name
    assert sha(p.read_bytes()) == digest
    files.append(p)
files.extend(root / "Garnet" / name for name in ("test2026/cuda/qwen38_5090/probe_first_attention_real_stages.py",
    "test2026/cuda/qwen38_5090/run_logged.py", "test2026/cuda/qwen38_5090/monitor_native_probe.py",
    "src/backends/tensorrt/modelopt_cublaslt.cpp", "src/backends/tensorrt/modelopt_cublaslt.h",
    "src/backends/tensorrt/modelopt_fp8_plugin.cpp", "src/backends/tensorrt/modelopt_fp8_plugin.h"))
files.extend([Path(__file__), root / "models/Qwen3.8-27B-NVFP4/config.json"])
observations = {}; objects = {}
for p in files:
    raw = p.read_bytes(); digest = sha(raw)
    observations[p.relative_to(root).as_posix()] = digest
    if digest in objects: assert objects[digest] == raw
    else: objects[digest] = raw
evidence = {"storage": "sha256-objects-v1", "files": observations,
    "objects": {h: len(raw) for h, raw in objects.items()}, "library_sha256": manifest["library_sha256"],
    "linked_archives": {name: sha((results / name).read_bytes()) for name in (
        "native-canonical-fp8-prefix123-attempt1-evidence.zip", "native-canonical-fp8-prefix4-hidden-attempt1-evidence.zip",
        "native-canonical-fp8-prefix4-hidden-geometry-attempt1-evidence.zip")},
    "scope": "All ordinary/diagnostic primary hidden+KV, stage arrays, repeats, failures, sources and build provenance. Additional exports require primary parity before attribution. No serving/performance admission."}
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
    for digest, raw in objects.items(): z.writestr("objects/" + digest, raw)
    z.writestr("evidence-manifest.json", json.dumps(evidence, indent=2))
print(json.dumps({"bytes": target.stat().st_size, "sha256": sha(target.read_bytes()),
    "file_observations": len(observations), "unique_objects": len(objects)}))
