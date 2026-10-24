"""Build isolated research policy and check old/new real matrices before install."""
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
out = root / "results/native-attention-fp8-policy-attempt1"
source = root / "results/native-attention-fp8-policy-source"
assert not out.exists() and not source.exists()
sha = lambda raw: hashlib.sha256(raw).hexdigest()
library = root / "build/bin/garnet.dll"
proof = root / "results/actual-first-attention-fp8-shared-algorithm-attempt1-verified.json"
verified = json.loads(proof.read_text())
assert verified["status"] == "shared_fp8_algorithm_real_projection_verified"
assert verified["both_m32_rows_match_m16_canonical_both_repeats"]
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
package = root / "results/native-attention-fp8-policy-source.zip"
with zipfile.ZipFile(package) as z:
    assert z.testzip() is None
    manifest = json.loads(z.read("manifest.json"))
    assert sha(library.read_bytes()) == manifest["application_library_sha256"]
    files = {}
    for name, digest in manifest["files"].items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        raw = z.read(name); assert sha(raw) == digest; files[name] = raw
source.mkdir(); out.mkdir()
for name, raw in files.items():
    p = source / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
fixtures = {}
for name in ("query", "gdn"):
    folder = root / ("results/actual-first-attention-fp8-fixture" if name == "query" else "results/actual-layer0-fp8-fixture")
    fixture = json.loads((folder / "manifest.json").read_text())
    for filename, item in fixture["files"].items():
        raw = (folder / filename).read_bytes()
        assert sha(raw) == item["sha256"] and len(raw) == item["bytes"]
    fixtures[name] = folder
artifacts = [package, proof, Path(__file__), library]
launcher = source / "test2026/cuda/qwen38_5090/build_native_attention_fp8_policy.bat"
code = run(["cmd.exe", "/d", "/c", str(launcher)], out / "build-ledger", "build",
    "isolated-full-attention-fp8-policy", artifacts + [launcher])
report = {"status": "build_failed", "build_exit_code": code, "source_files": manifest["files"],
    "application_library_sha256": manifest["application_library_sha256"], "source_package_sha256": sha(package.read_bytes()),
    "prior_verification_sha256": sha(proof.read_bytes()), "executables": {}, "checks": []}
os.environ["PATH"] = os.pathsep.join([str(root / "tools/TensorRT/bin"),
    str(root / "tools/cuda13/nvidia/cu13/bin/x86_64"), os.environ["PATH"]])
if code == 0:
    commands = [("formats", [str(root / "build/native-attention-fp8-plugin-test.exe"), "--formats-only"]),
        ("plugin-default", [str(root / "build/native-attention-fp8-plugin-test.exe")])]
    for name in ("query", "gdn"):
        for mode in ("control", "candidate"):
            destination = out / (name + "-" + mode); destination.mkdir()
            command = [str(root / ("build/native-attention-fp8-" + name + ".exe")), str(fixtures[name]), str(destination)]
            if mode == "candidate": command.append("--native-canonical")
            commands.append((name + "-" + mode, command))
    for label, command in commands:
        exe = Path(command[0]); report["executables"][exe.name] = sha(exe.read_bytes())
        code = run(command, out / (label + "-ledger"), "correctness", label + "-native-policy", artifacts + [exe])
        report["checks"].append({"label": label, "exit_code": code})
        report["status"] = label + "_failed" if code else label + "_finished_needs_independent_replay"
        (out / "workflow.json").write_text(json.dumps(report, indent=2), encoding="utf8")
        if code: break
assert sha(library.read_bytes()) == manifest["application_library_sha256"]
report["scope"] = "Isolated candidate only. Default wire/clone/profile and synthetic plugin regression, real query and GDN controls/candidates precede app installation. No full-model or performance admission."
(out / "workflow.json").write_text(json.dumps(report, indent=2), encoding="utf8")
print(json.dumps(report, indent=2))
raise SystemExit(code)
