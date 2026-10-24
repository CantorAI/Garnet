"""Guarded native diagnostic build in the existing study build slot."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import zipfile

root = Path("D:/CantorAI/5090-study-20261023")
sys.path.insert(0, str(root / "Garnet/test2026/cuda/qwen38_5090"))
from run_logged import run
out = root / "results/native-attention-stage-build-attempt1"
assert not out.exists()
sha = lambda raw: hashlib.sha256(raw).hexdigest()
library = root / "build/bin/garnet.dll"
old_library = "1ed48e36d31c1cd7cca8fe6825049331fc0b584f54215fef0d940d4aa624d767"
assert sha(library.read_bytes()) == old_library
retained = root / "build/bin/garnet-native-canonical-fp8-attempt2.dll"
assert sha(retained.read_bytes()) == old_library
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
proof = root / "results/native-canonical-fp8-prefix4-hidden-geometry-attempt1-verified.json"
verified = json.loads(proof.read_text())
assert verified["status"] == "canonical_fp8_deeper_prefix_evidence_verified"
assert any(not v["exact"] for v in verified["chunk_comparisons"])
payload = root / "results/native-attention-stage-source.zip"
snapshot = root / "results/native-attention-stage-original-snapshot.zip"
expected = {"src/backends/tensorrt/" + name for name in (
    "modelopt_fp8_lowering.cpp", "modelopt_nvfp4_lowering.cpp", "trt_builder.h",
    "qwen35_lowering.h", "trt_builder.cpp")}
with zipfile.ZipFile(payload) as z, zipfile.ZipFile(snapshot) as original:
    assert z.testzip() is None and original.testzip() is None
    manifest = json.loads(z.read("manifest.json"))
    assert set(manifest["source_sha256"]) == set(manifest["new_source_sha256"]) == expected
    assert manifest["original_snapshot_sha256"] == sha(snapshot.read_bytes())
    assert manifest["library_sha256"] == old_library
    old = {}; new = {}
    for name in sorted(expected):
        old[name] = (root / "Garnet" / name).read_bytes()
        new[name] = z.read(name)
        assert old[name] == original.read(name)
        assert sha(old[name]) == manifest["source_sha256"][name]
        assert sha(new[name]) == manifest["new_source_sha256"][name]
out.mkdir()
for name, raw in old.items():
    p = out / "original" / name
    p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
for name, raw in new.items():
    (root / "Garnet" / name).write_bytes(raw)
    assert (root / "Garnet" / name).read_bytes() == raw
receipt = {"status": "diagnostic_sources_installed_unbuilt", "old_library_sha256": old_library,
    "original_source_sha256": manifest["source_sha256"], "new_source_sha256": manifest["new_source_sha256"],
    "source_bundle_sha256": sha(payload.read_bytes()), "prior_evidence_sha256": sha(proof.read_bytes())}
(out / "source-receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf8")
launcher = root / "build-layer0-stages.bat"
code = run(["cmd.exe", "/d", "/c", str(launcher)], out / "build-ledger", "build",
    "fp8-logical-identity-and-default-off-attention-stages", [payload, snapshot, proof, launcher,
    library, Path(__file__)] + [root / "Garnet" / name for name in sorted(expected)])
report = {"status": "build_failed" if code else "native_diagnostic_build_finished_unqualified",
    "exit_code": code, "original_library_sha256": old_library,
    "source_sha256": {name: sha((root / "Garnet" / name).read_bytes()) for name in sorted(expected)},
    "scope": "Native build only. Diagnostic primary parity, every stage and negative profiles must pass before attribution; no serving/performance admission."}
if code == 0:
    report["library_sha256"] = sha(library.read_bytes())
    saved = root / "build/bin/garnet-native-attention-stage-attempt1.dll"
    assert not saved.exists(); saved.write_bytes(library.read_bytes())
else:
    report["failed_output_library_sha256"] = sha(library.read_bytes()) if library.exists() else None
    library.write_bytes(retained.read_bytes())
    assert sha(library.read_bytes()) == old_library
    report["restored_library_sha256"] = old_library
(out / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf8")
print(json.dumps(report, indent=2))
raise SystemExit(code)
