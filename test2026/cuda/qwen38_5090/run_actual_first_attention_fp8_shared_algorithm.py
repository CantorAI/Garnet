"""Same-executable control then checked M16-algorithm intervention on real M32."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

root = Path("D:/CantorAI/5090-study-20261023")
sys.path.insert(0, str(root / "Garnet/test2026/cuda/qwen38_5090"))
from run_logged import run

out = root / "results/actual-first-attention-fp8-shared-algorithm-attempt1"
assert not out.exists()
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
verified_path = root / "results/actual-first-attention-fp8-control-attempt1-verified.json"
verified = json.loads(verified_path.read_text())
assert verified["status"] == "real_fp8_projection_control_verified"
fixture = root / "results/actual-first-attention-fp8-fixture"
manifest = json.loads((fixture / "manifest.json").read_text())
for n, item in manifest["files"].items():
    raw = (fixture / n).read_bytes()
    assert len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"]
assert hashlib.sha256((root / "build/bin/garnet.dll").read_bytes()).hexdigest() == manifest["library_sha256"]
out.mkdir()
sources = [root / "actual-first-attention-fp8-control.cu", root / "build-actual-first-attention-fp8-control.bat", Path(__file__)]
sources += [root / "Garnet" / n for n in ("src/backends/tensorrt/modelopt_cublaslt.cpp",
    "src/backends/tensorrt/modelopt_cublaslt.h", "src/cuda/modelopt_fp8.cu", "src/cuda/modelopt_fp8.cuh")]
hashes = {}
for p in sources:
    name = p.relative_to(root).as_posix()
    raw = p.read_bytes()
    target = out / "sources" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    hashes[name] = hashlib.sha256(raw).hexdigest()
code = run(["cmd.exe", "/d", "/c", str(root / "build-actual-first-attention-fp8-control.bat")],
    out / "build-ledger", "build", "real-fp8-checked-shared-algorithm", sources)
report = {"status": "build_failed", "build_exit_code": code, "source_sha256": hashes,
          "control_verification_sha256": hashlib.sha256(verified_path.read_bytes()).hexdigest()}
if code == 0:
    exe = root / "build/actual-first-attention-fp8-control.exe"
    report["executable_sha256"] = hashlib.sha256(exe.read_bytes()).hexdigest()
    os.environ["PATH"] = str(root / "tools/cuda13/nvidia/cu13/bin/x86_64") + os.pathsep + os.environ["PATH"]
    for label in ("control", "shared"):
        native = out / label
        native.mkdir()
        command = [str(exe), str(fixture), str(native)]
        if label == "shared": command.append("--same-algorithm")
        code = run(command, out / (label + "-ledger"), "correctness", label + "-real-m16-m32",
                   sources + [exe, fixture / "manifest.json", verified_path])
        report[label + "_exit_code"] = code
        report["status"] = label + "_failed" if code else label + "_finished_needs_independent_replay"
        if code: break
        canonical = (fixture / "expected-m16-n12288.bf16").read_bytes()
        for m in (16, 32):
            expected = (fixture / f"expected-m{m}-n12288.bf16").read_bytes() if label == "control" else canonical * (m // 16)
            for repeat in (0, 1): assert (native / f"actual-m{m}-r{repeat}.bf16").read_bytes() == expected
report["scope"] = "Untimed standalone real-layer projection intervention only. AlgoCheck required; no application math/default change, full-model batching admission or speed proof."
(out / "workflow.json").write_text(json.dumps(report, indent=2), encoding="utf8")
print(json.dumps(report, indent=2))
raise SystemExit(code)
