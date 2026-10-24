"""Build isolated candidate sources; preserve application source and DLL."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile

root = Path("D:/CantorAI/5090-study-20261023")
sys.path.insert(0, str(root / "Garnet/test2026/cuda/qwen38_5090"))
from run_logged import run

parser = argparse.ArgumentParser(); parser.add_argument("--attempt", type=int, choices=(1, 2), default=1); args = parser.parse_args()
out = root / f"results/native-canonical-fp8-attempt{args.attempt}"
source = root / "results/native-canonical-fp8-source"
assert not out.exists()
if args.attempt == 1: assert not source.exists()
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
library = root / "build/bin/garnet.dll"
library_hash = hashlib.sha256(library.read_bytes()).hexdigest()
assert library_hash == "34cb221458064a4b7264586412be5b3c7acca888cd9e74681ac2a41d259413da"
verified = root / "results/actual-layer0-fp8-shared-algorithm-attempt1-verified.json"
assert json.loads(verified.read_text())["status"] == "shared_fp8_algorithm_real_projection_verified"
package = root / ("native-canonical-fp8-source.zip" if args.attempt == 1 else "native-canonical-fp8-source-attempt2.zip")
fix = None
if args.attempt == 2:
    previous = root / "results/native-canonical-fp8-attempt1"
    assert json.loads((previous / "workflow.json").read_text())["status"] == "build_failed"
    failure = root / "results/native-canonical-fp8-attempt1-evidence.zip"
    with zipfile.ZipFile(failure) as failed:
        assert failed.testzip() is None
        for n, h in json.loads(failed.read("evidence-manifest.json"))["files"].items():
            assert hashlib.sha256(failed.read(n)).hexdigest() == h
    with zipfile.ZipFile(root / "native-canonical-fp8-source.zip") as old:
        for n, h in json.loads(old.read("manifest.json"))["files"].items():
            assert hashlib.sha256((source / n).read_bytes()).hexdigest() == h
    fix = {"failure_archive_sha256": hashlib.sha256(failure.read_bytes()).hexdigest(),
           "old_source_files": {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()},
           "reason": "Test cloned TensorRT plugin has protected destructor; use its destroy() lifecycle. Native math/policy unchanged."}
with zipfile.ZipFile(package) as z:
    assert z.testzip() is None
    manifest = json.loads(z.read("manifest.json"))
    for n, h in manifest["files"].items():
        assert not Path(n).is_absolute() and ".." not in Path(n).parts
        raw = z.read(n)
        assert hashlib.sha256(raw).hexdigest() == h
        p = source / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
out.mkdir()
if fix:
    fix["new_source_files"] = manifest["files"]
    (out / "source-fix-receipt.json").write_text(json.dumps(fix, indent=2), encoding="utf8")
fixture = root / "results/actual-layer0-fp8-fixture"
fmanifest = json.loads((fixture / "manifest.json").read_text())
for n, v in fmanifest["files"].items():
    raw = (fixture / n).read_bytes()
    assert len(raw) == v["bytes"] and hashlib.sha256(raw).hexdigest() == v["sha256"]
artifacts = [package, verified, Path(__file__), library, fixture / "manifest.json"]
code = run(["cmd.exe", "/d", "/c", str(source / "test2026/cuda/qwen38_5090/build_native_canonical_fp8.bat")],
    out / "build-ledger", "build", "isolated-native-canonical-fp8", artifacts)
report = {"status": "build_failed", "build_exit_code": code, "library_sha256": library_hash,
          "source_package_sha256": hashlib.sha256(package.read_bytes()).hexdigest(), "source_files": manifest["files"]}
if code == 0:
    os.environ["PATH"] = os.pathsep.join([str(root / "tools/TensorRT/bin"), str(root / "tools/cuda13/nvidia/cu13/bin/x86_64"), os.environ["PATH"]])
    plugin = root / "build/native-canonical-fp8-plugin-test.exe"
    code = run([str(plugin), "--formats-only"], out / "policy-ledger", "correctness", "default-wire-and-candidate-policy", artifacts + [plugin])
    report["policy_exit_code"] = code
    report["status"] = "policy_failed" if code else "policy_finished"
    if code == 0:
        exe = root / "build/native-canonical-fp8.exe"
        report["executable_sha256"] = hashlib.sha256(exe.read_bytes()).hexdigest()
        for label in ("control", "candidate"):
            native = out / label
            native.mkdir()
            command = [str(exe), str(fixture), str(native)]
            if label == "candidate": command.append("--native-canonical")
            code = run(command, out / (label + "-ledger"), "correctness", label + "-native-real-projection", artifacts + [exe])
            report[label + "_exit_code"] = code
            report["status"] = label + "_failed" if code else label + "_finished_needs_independent_replay"
            if code: break
assert hashlib.sha256(library.read_bytes()).hexdigest() == library_hash
report["scope"] = "Isolated native helper/plugin policy source build and actual projection gates. Application source and DLL unchanged. Not integration, full-model batching or performance proof."
(out / "workflow.json").write_text(json.dumps(report, indent=2), encoding="utf8")
print(json.dumps(report, indent=2))
raise SystemExit(code)
