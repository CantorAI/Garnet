"""Repair only test interception, reuse valid plugin/GDN evidence, rerun query."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import zipfile

root = Path("D:/CantorAI/5090-study-20261023")
sys.path.insert(0, str(root / "Garnet/test2026/cuda/qwen38_5090"))
from run_logged import run
out = root / "results/native-attention-fp8-query-repair-attempt1"
assert not out.exists()
sha = lambda raw: hashlib.sha256(raw).hexdigest()
proof_path = root / "results/native-attention-fp8-policy-attempt1-verified.json"
proof = json.loads(proof_path.read_text())
assert proof["status"] == "isolated_native_attention_fp8_policy_incomplete" and not proof["native_qualification_passed"]
assert not proof["matrices"]["query"]["native_algorithm_selection_verified"]
assert proof["matrices"]["gdn"]["native_algorithm_selection_verified"]
prior_archive = root / "results/native-attention-fp8-policy-attempt1-evidence.zip"
assert sha(prior_archive.read_bytes()) == proof["archive_sha256"]
fixture = root / "results/actual-first-attention-fp8-fixture"
fixture_raw = (fixture / "manifest.json").read_bytes()
with zipfile.ZipFile(prior_archive) as previous:
    assert fixture_raw == previous.read("results/actual-first-attention-fp8-fixture/manifest.json")
for filename, item in json.loads(fixture_raw)["files"].items():
    raw = (fixture / filename).read_bytes()
    assert sha(raw) == item["sha256"] and len(raw) == item["bytes"]
library = root / "build/bin/garnet.dll"
assert sha(library.read_bytes()) == proof["application_library_sha256"]
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
source = root / "results/native-attention-fp8-policy-source"
for name, digest in proof["source_files"].items(): assert sha((source / name).read_bytes()) == digest
name = "test2026/cuda/qwen38_5090/native_attention_fp8_control.cu"
old = (source / name).read_bytes()
fixed = (root / "results/native-attention-fp8-query-control-fixed.cu").read_bytes()
before = b"if ((sameAlgorithm || nativeCanonical) && m == 32)"
after = b"if (sameAlgorithm && m == 32)"
assert old.count(before) == 2 and fixed == old.replace(before, after, 1)
assert b"nativeCanonical" not in fixed[fixed.index(b"static cublasStatus_t ObserveMatmul"):fixed.index(b"#define cublasLtMatmul")]
out.mkdir(); (out / "original-query-control.cu").write_bytes(old)
(source / name).write_bytes(fixed)
updated = dict(proof["source_files"]); updated[name] = sha(fixed)
receipt = {"reason": "Native-candidate mode must observe helper selection without the shared-mode interceptor override",
    "prior_archive_sha256": proof["archive_sha256"], "prior_proof_sha256": sha(proof_path.read_bytes()),
    "changed_source": name, "old_sha256": sha(old), "new_sha256": sha(fixed),
    "native_math_source_unchanged": True, "source_files": updated}
(out / "repair-receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf8")
# Reuse the same source/build slot. Rebuild only the changed query test executable.
lines = (source / "test2026/cuda/qwen38_5090/build_native_attention_fp8_policy.bat").read_bytes().splitlines(keepends=True)
index = next(i for i, line in enumerate(lines) if b"native-attention-fp8-query.exe" in line)
launcher = out / "rebuild-query.bat"
launcher.write_bytes(b"".join(lines[:index + 2]) + b"exit /b %errorlevel%\r\n")
artifacts = [proof_path, prior_archive, library, Path(__file__), source / name, out / "repair-receipt.json"]
code = run(["cmd.exe", "/d", "/c", str(launcher)], out / "build-ledger", "build", "query-observer-only-repair", artifacts + [launcher])
report = {"status": "build_failed", "build_exit_code": code, "source_files": updated,
    "application_library_sha256": proof["application_library_sha256"], "prior_archive_sha256": proof["archive_sha256"]}
if code == 0:
    exe = root / "build/native-attention-fp8-query.exe"; report["executable_sha256"] = sha(exe.read_bytes())
    os.environ["PATH"] = str(root / "tools/cuda13/nvidia/cu13/bin/x86_64") + os.pathsep + os.environ["PATH"]
    for mode in ("control", "candidate"):
        destination = out / mode; destination.mkdir()
        command = [str(exe), str(fixture), str(destination)]
        if mode == "candidate": command.append("--native-canonical")
        code = run(command, out / (mode + "-ledger"), "correctness", mode + "-unintercepted-native-query", artifacts + [exe, fixture / "manifest.json"])
        report[mode + "_exit_code"] = code
        report["status"] = mode + "_failed" if code else mode + "_finished_needs_independent_replay"
        (out / "workflow.json").write_text(json.dumps(report, indent=2), encoding="utf8")
        if code: break
assert sha(library.read_bytes()) == proof["application_library_sha256"]
for n, digest in updated.items(): assert sha((source / n).read_bytes()) == digest
report["scope"] = "Observer-only repair; native helper/plugin/lowering sources unchanged. Fresh same-executable default/native query checks; valid prior GDN/wire/plugin evidence retained. No app/full-model/performance admission."
(out / "workflow.json").write_text(json.dumps(report, indent=2), encoding="utf8")
print(json.dumps(report, indent=2))
raise SystemExit(code)
