"""Controlled existing-binary GDN-only versus geometry-wide prefix4 witness.

The latter binary also selects canonical FP8 for full-attention o_proj. Neither
leg is a serving or speed admission. Restore the current DLL after closure.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

root = Path("D:/CantorAI/5090-study-20261023")
methods = root / "Garnet/test2026/cuda/qwen38_5090"
sys.path.insert(0, str(methods))
from run_logged import run
out = root / "results/native-canonical-fp8-prefix4-attempt1"
assert not out.exists()
cache_receipt = root / "results/study-ssd-engine-cache/receipt.json"
scratch = json.loads(cache_receipt.read_text())
cache = Path(scratch["cache_root"])
assert scratch["status"] == "ssd_engine_scratch_prepared" and cache.resolve().drive.upper() == "C:"
assert json.loads((cache / "study-owner.json").read_text())["study"] == str(root)
proof = root / "results/native-canonical-fp8-full-b2-attempt1-verified.json"
assert json.loads(proof.read_text())["model_gate_passed"] is False
library = root / "build/bin/garnet.dll"
variants = [("gdn", "1ed48e36d31c1cd7cca8fe6825049331fc0b584f54215fef0d940d4aa624d767", root / "build/bin/garnet-native-canonical-fp8-attempt2.dll"),
    ("geometry", "8568b9192a06067352636bfd94c97439d5524c3c0a0e7af744b19963cc0abe51", root / "build/bin/garnet-native-canonical-fp8-attempt1.dll")]
for label, digest, path in variants: assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
assert hashlib.sha256(library.read_bytes()).hexdigest() == variants[0][1]
old4 = root / "results/native-prefix-batch-evidence.zip"
assert hashlib.sha256(old4.read_bytes()).hexdigest() == "7093c0aa054b4d63533073c8995434441acc0f4af40367c8f02401b596f940e3"
probe = methods / "probe_native_prefix_batch.py"
assert hashlib.sha256(probe.read_bytes()).hexdigest() == "a5892d0342f333e4108210caa22de2063d9c1542ef3696ef5e8379eeba72873e"
launcher = root / "test-native-canonical-fp8-prefix4.bat"
def idle():
    assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()) < 1024
def retire_new_cache_directories(before, started):
    idle()
    assert json.loads((cache / "study-owner.json").read_text())["study"] == str(root)
    resolved_root = cache.resolve(strict=True)
    retired = []
    for path in cache.iterdir():
        if path.name in before or not path.name.startswith("garnet-cuda-graph-"): continue
        st = path.lstat()
        assert not (getattr(st, "st_file_attributes", 0) & 0x400)
        target = path.resolve(strict=True)
        assert target.parent == resolved_root and st.st_ctime >= started - 2
        entries = list(target.iterdir())
        assert all(p.name in ("model.engine", "model.engine.tmp") for p in entries)
        metadata = []
        for p in entries:
            ps = p.lstat()
            assert not (getattr(ps, "st_file_attributes", 0) & 0x400) and p.is_file()
            assert p.resolve(strict=True).parent == target
            h = hashlib.sha256()
            with p.open("rb") as f:
                for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
            metadata.append({"path": str(p), "bytes": ps.st_size, "sha256": h.hexdigest()})
        # Only this just-closed leg's new private engine files; no recursive
        # deletion and no source/model/result/deployment asset is touched.
        for p in entries: p.unlink()
        target.rmdir()
        retired.extend(metadata)
    return retired
idle(); out.mkdir()
workflow = {"status": "running", "legs": [], "scope": "Two preserved binaries differ in lowering family selection (GDN-only vs same-shaped outputs, including attention). Current source checkout remains GDN-only; older geometry DLL is linked to its original build manifest. Actual prefix4 first-attention-layer witness only; no fullmodel/speed admission."}
code = 1
try:
    for label, digest, path in variants:
        idle()
        library.write_bytes(path.read_bytes())
        assert hashlib.sha256(library.read_bytes()).hexdigest() == digest
        leg = out / label; leg.mkdir()
        before = {p.name for p in cache.iterdir()}; started = time.time()
        code = run([sys.executable, str(methods / "monitor_native_probe.py"), "--study", str(root),
            "--output", str(leg / "resources"), "--", "cmd.exe", "/d", "/c", str(launcher),
            str(leg / "native"), str(cache)], leg / "ledger", "correctness", label + "-prefix4-existing-binary",
            [library, path, launcher, probe, old4, proof, cache_receipt, Path(__file__),
             root / "results/native-canonical-fp8-build-attempt1/manifest.json", root / "results/native-canonical-fp8-build-attempt2/manifest.json"])
        record = {"label": label, "library_sha256": digest, "probe_exit_code": code}
        if code == 0:
            report = json.loads((leg / "native/report.json").read_text())
            assert report["status"] == "diagnostic_completed" and report["prefixes"] == [4] and len(report["cases"]) == 2
            assert all(v["repeat_exact"] and v["duplicate_rows_exact"] for v in report["cases"])
            with zipfile.ZipFile(old4) as old:
                for repeat in (0, 1):
                    name = f"prefix-4-batch-1-repeat-{repeat}-hidden.f32"
                    assert (leg / "native" / name).read_bytes() == old.read("results/native-prefix-batch-diagnostic/" + name)
            record["cross_batch_hidden_exact"] = report["b1_b2_comparisons"][0]["hidden_exact"]
        record["retired_private_engine_files"] = retire_new_cache_directories(before, started)
        workflow["legs"].append(record)
        workflow["status"] = label + "_process_failed" if code else label + "_completed_needs_independent_replay"
        (out / "workflow.json").write_text(json.dumps(workflow, indent=2), encoding="utf8")
        if code: break
except Exception as error:
    workflow["status"] = "controller_failed"
    workflow["error"] = repr(error)
    raise
finally:
    idle()
    library.write_bytes(variants[0][2].read_bytes())
    assert hashlib.sha256(library.read_bytes()).hexdigest() == variants[0][1]
    workflow["restored_library_sha256"] = variants[0][1]
    (out / "workflow.json").write_text(json.dumps(workflow, indent=2), encoding="utf8")
print(json.dumps(workflow, indent=2))
raise SystemExit(code)
