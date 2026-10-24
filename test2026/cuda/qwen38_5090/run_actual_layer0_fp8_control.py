"""Build and run the bounded real projection control, retaining failures."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

root = Path("D:/CantorAI/5090-study-20261023")
sys.path.insert(0, str(root / "Garnet/test2026/cuda/qwen38_5090"))
from run_logged import run

out = root / "results/actual-layer0-fp8-control-attempt1"
assert not out.exists()
assert int(subprocess.check_output([
    "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"
], text=True).strip()) < 1024
fixture = root / "results/actual-layer0-fp8-fixture"
manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf8"))
assert manifest["status"] == "real_projection_fixture_extracted_unqualified"
for name, item in manifest["files"].items():
    raw = (fixture / name).read_bytes()
    assert len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"]
library = root / "build/bin/garnet.dll"
assert hashlib.sha256(library.read_bytes()).hexdigest() == manifest["library_sha256"]
out.mkdir()
sources = [root / "actual-layer0-fp8-control.cu", root / "build-actual-layer0-fp8-control.bat",
           root / "prepare-actual-layer0-fp8-fixture.py", Path(__file__)]
sources += [root / "Garnet" / p for p in (
    "src/backends/tensorrt/modelopt_cublaslt.cpp", "src/backends/tensorrt/modelopt_cublaslt.h",
    "src/cuda/modelopt_fp8.cu", "src/cuda/modelopt_fp8.cuh")]
hashes = {}
for p in sources:
    name = p.relative_to(root).as_posix()
    raw = p.read_bytes()
    target = out / "sources" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    hashes[name] = hashlib.sha256(raw).hexdigest()
code = run(["cmd.exe", "/d", "/c", str(root / "build-actual-layer0-fp8-control.bat")],
           out / "build-ledger", "build", "real-fp8-projection-control", sources)
result = {"status": "build_failed", "build_exit_code": code, "source_sha256": hashes,
          "fixture_manifest_sha256": hashlib.sha256((fixture / "manifest.json").read_bytes()).hexdigest()}
if code == 0:
    exe = root / "build/actual-layer0-fp8-control.exe"
    result["executable_sha256"] = hashlib.sha256(exe.read_bytes()).hexdigest()
    outputs = out / "native"
    outputs.mkdir()
    os.environ["PATH"] = str(root / "tools/cuda13/nvidia/cu13/bin/x86_64") + os.pathsep + os.environ["PATH"]
    code = run([str(exe), str(fixture), str(outputs)], out / "native-ledger",
               "correctness", "both-real-m16-m32-projection-outputs", sources + [exe, fixture / "manifest.json"])
    result.update({"status": "control_failed" if code else "control_finished_needs_independent_replay",
                   "native_exit_code": code})
result["scope"] = "Standalone real projection reproduction; not full-model correctness, batch admission or performance."
(out / "workflow.json").write_text(json.dumps(result, indent=2), encoding="utf8")
print(json.dumps(result, indent=2))
raise SystemExit(code)
