"""Archive the closed full-depth duplicate-text prefill diagnostic."""
from pathlib import Path
import hashlib
import argparse
import json
import subprocess
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", choices=("prefix64", "prefix8-16", "prefix4-hidden", "prefix4-hidden-geometry"), default="prefix64")
experiment = parser.parse_args().experiment
geometry = experiment == "prefix4-hidden-geometry"
stem = "native-canonical-fp8-" + experiment
root = Path("D:/CantorAI/5090-study-20261023")
results = root / "results"
folder = results / (stem + "-attempt1")
outer = results / (stem + "-controller-ledger")
target = results / (stem + "-attempt1-evidence.zip")
assert not target.exists()
workflow = json.loads((folder / "workflow.json").read_text())
events = [json.loads(v) for v in (outer / "commands.jsonl").read_text().splitlines()]
assert events[-1]["event"] == "finished"
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
sha = lambda raw: hashlib.sha256(raw).hexdigest()
library = sha((root / "build/bin/garnet.dll").read_bytes())
assert library == "1ed48e36d31c1cd7cca8fe6825049331fc0b584f54215fef0d940d4aa624d767"
if geometry:
    assert workflow["restored_library_sha256"] == library
    assert workflow["library_sha256"] == "8568b9192a06067352636bfd94c97439d5524c3c0a0e7af744b19963cc0abe51"
else: assert workflow["library_sha256"] == library
files = []
for directory in (folder, outer, results / "study-ssd-cache-setup-ledger",
        results / "study-ssd-engine-cache", results / "native-canonical-fp8-build-attempt2"):
    files.extend(p for p in directory.rglob("*") if p.is_file())
if geometry:
    files.extend(p for p in (results / "native-canonical-fp8-build-attempt1").rglob("*") if p.is_file())
    files.append(results / "native-canonical-fp8-prefix4-hidden-attempt1-verified.json")
    original_lowerer = results / "native-canonical-fp8-family-scope-install/original-lowering.cpp"
    original_build = json.loads((results / "native-canonical-fp8-build-attempt1/manifest.json").read_text())
    assert sha(original_lowerer.read_bytes()) == original_build["source_sha256"]["src/backends/tensorrt/modelopt_fp8_lowering.cpp"]
    files.append(original_lowerer)
files.extend(root / n for n in ("run-" + stem + ".py",
    ("test-native-canonical-fp8-prefix4-hidden.bat" if geometry else "test-" + stem + ".bat"), "prepare-ssd-engine-cache.py"))
files.extend(root / "Garnet" / n for n in (
    "test2026/cuda/qwen38_5090/" + ("probe_native_prefix_hidden_chunks.py" if experiment.startswith("prefix4-hidden") else "probe_native_deeper_prefix_batch.py"),
    "test2026/cuda/qwen38_5090/run_logged.py",
    "test2026/cuda/qwen38_5090/monitor_native_probe.py",
    "src/backends/tensorrt/qwen35_lowering.h",
    "xModel/qwen35/vl_0_8b/cuda_packed_decoder.py",
    "xModel/qwen35/vl_0_8b/cuda_packed_program.py"))
build = json.loads((results / "native-canonical-fp8-build-attempt2/manifest.json").read_text())
for name, digest in build["source_sha256"].items():
    p = root / "Garnet" / name
    assert sha(p.read_bytes()) == digest
    files.append(p)
files.extend([Path(__file__), results / "native-canonical-fp8-full-b2-attempt1-verified.json"])
observations = {}; objects = {}
for p in files:
    raw = p.read_bytes(); digest = sha(raw)
    observations[p.relative_to(root).as_posix()] = digest
    if digest in objects: assert objects[digest] == raw
    else: objects[digest] = raw
manifest = {"storage": "sha256-objects-v1", "files": observations,
    "objects": {h: len(b) for h, b in objects.items()}, "library_sha256": workflow["library_sha256"], "restored_library_sha256": library,
    "linked_archives": {n: sha((results / n).read_bytes()) for n in (
        "native-canonical-fp8-prefix4-attempt1-evidence.zip", "native-canonical-fp8-full-b2-attempt1-evidence.zip") +
        (("native-canonical-fp8-prefix64-attempt1-evidence.zip",) if experiment != "prefix64" else ()) +
        (("native-prefix-batch-evidence.zip", "native-canonical-fp8-prefix8-16-attempt1-evidence.zip") if experiment.startswith("prefix4-hidden") else ()) +
        (("native-canonical-fp8-prefix4-hidden-attempt1-evidence.zip",) if geometry else ())},
    "scope": "Deeper-prefix duplicate-text prefill outputs, raw arrays, source/build provenance and resources. " + ("Geometry intervention uses preserved build1 DLL while source checkout remains build2; both manifests retained. " if geometry else "") + "No ragged text/image generation, quality or performance admission."}
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
    for h, raw in objects.items(): z.writestr("objects/" + h, raw)
    z.writestr("evidence-manifest.json", json.dumps(manifest, indent=2))
print(json.dumps({"bytes": target.stat().st_size, "sha256": sha(target.read_bytes()),
    "file_observations": len(observations), "unique_objects": len(objects)}))
