"""Actual TensorRT stage control/candidate; full arrays precede batch attribution."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys
import zipfile

root = Path("D:/CantorAI/5090-study-20261023")
methods = root / "Garnet/test2026/cuda/qwen38_5090"
sys.path.insert(0, str(methods))
from run_logged import run
out = root / "results/native-canonical-fp8-stages-attempt1"
assert not out.exists()
build_path = root / "results/native-canonical-fp8-build-attempt1/manifest.json"
build = json.loads(build_path.read_text())
assert build["status"] == "native_build_finished_unqualified" and build["exit_code"] == 0
library = root / "build/bin/garnet.dll"
assert hashlib.sha256(library.read_bytes()).hexdigest() == build["library_sha256"]
for n, h in build["source_sha256"].items(): assert hashlib.sha256((root / "Garnet" / n).read_bytes()).hexdigest() == h
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
old_archive = root / "results/layer0-real-stage-attempt2-evidence.zip"
assert hashlib.sha256(old_archive.read_bytes()).hexdigest() == "d05aa555feac3183d54dd7c1975741fc3bb525739eccff26a98166b5db761a99"
old = zipfile.ZipFile(old_archive)
probe = methods / "probe_layer0_real_stages.py"
assert probe.read_bytes() == old.read("Garnet/test2026/cuda/qwen38_5090/probe_layer0_real_stages.py")
for n in ("src/backends/tensorrt/qwen35_lowering.h", "src/backends/tensorrt/trt_builder.cpp"):
    assert (root / "Garnet" / n).read_bytes() == old.read("Garnet/" + n)
out.mkdir()
launcher = root / "test-native-canonical-fp8-stages.bat"
workflow = {"status": "running", "library_sha256": build["library_sha256"], "legs": [],
    "scope": "Prefix1 actual TensorRT B1/B2 control/candidate. All81 raw files must preserve old control or match canonical B1. Not full64-layer batch or speed admission."}
code = 1
for label, policy in (("control", "0"), ("candidate", "1")):
    leg = out / label
    leg.mkdir()
    code = run([sys.executable, str(methods / "monitor_native_probe.py"), "--study", str(root),
        "--output", str(leg / "resources"), "--", "cmd.exe", "/d", "/c", str(launcher), policy,
        str(leg / "native")], leg / "ledger", "correctness", label + "-actual-native-fp8-stages",
        [launcher, probe, library, build_path, Path(__file__)] + [root / "Garnet" / n for n in build["source_sha256"]])
    mismatches = []
    if code == 0:
        report = json.loads((leg / "native/report.json").read_text())
        assert report["status"] == "witness_passed_needs_independent_array_replay" and len(report["files"]) == 81
        for item in report["files"]:
            name = item["name"]
            expected_name = name
            if label == "candidate":
                match = re.fullmatch(r"b([12])-r(-?\d+)-(.+)-row([01])\.f32", name)
                assert match, name
                expected_name = f"b1-r{match[2]}-{match[3]}-row0.f32"
            expected = old.read("results/layer0-real-stage-attempt2/native/" + expected_name)
            if (leg / "native" / name).read_bytes() != expected: mismatches.append(name)
        if mismatches: code = 1
    workflow["legs"].append({"label": label, "policy": policy, "exit_code": code, "mismatched_files": mismatches})
    workflow["status"] = label + "_failed" if code else label + "_passed_needs_independent_replay"
    (out / "workflow.json").write_text(json.dumps(workflow, indent=2), encoding="utf8")
    if code: break
old.close()
print(json.dumps(workflow, indent=2))
raise SystemExit(code)
