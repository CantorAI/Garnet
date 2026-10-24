"""Capture every hidden token through prefix4 after verified deeper divergence."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import time

root = Path("D:/CantorAI/5090-study-20261023")
methods = root / "Garnet/test2026/cuda/qwen38_5090"
sys.path.insert(0, str(methods))
from run_logged import run
out = root / "results/native-canonical-fp8-prefix4-hidden-attempt1"
assert not out.exists()
library = root / "build/bin/garnet.dll"
digest = "1ed48e36d31c1cd7cca8fe6825049331fc0b584f54215fef0d940d4aa624d767"
assert hashlib.sha256(library.read_bytes()).hexdigest() == digest
proof = root / "results/native-canonical-fp8-prefix8-16-attempt1-verified.json"
prior = json.loads(proof.read_text())
assert prior["status"] == "canonical_fp8_deeper_prefix_evidence_verified"
assert prior["prefixes"]["8"]["cross_batch_hidden_exact"] is False
receipt = root / "results/study-ssd-engine-cache/receipt.json"
scratch = json.loads(receipt.read_text()); cache = Path(scratch["cache_root"])
assert scratch["status"] == "ssd_engine_scratch_prepared" and cache.resolve().drive.upper() == "C:"
assert json.loads((cache / "study-owner.json").read_text())["study"] == str(root)
assert shutil.disk_usage(cache).free >= 96 * 1024**3
def idle():
    assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip()) < 1024
idle(); out.mkdir()
before = {p.name for p in cache.iterdir()}; started = time.time()
launcher = root / "test-native-canonical-fp8-prefix4-hidden.bat"
probe = methods / "probe_native_prefix_hidden_chunks.py"
workflow = {"status": "running", "library_sha256": digest,
    "probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
    "scope": "Every-token hidden-chunk prefix4 identical text-row prefill at B1/B2, resets/repeats and graph replay. Does not exercise ragged rows, vision, token generation, head, MTP, serving or performance. Full B2 generation gate remains failed."}
(out / "workflow.json").write_text(json.dumps(workflow, indent=2), encoding="utf8")
code = run([sys.executable, str(methods / "monitor_native_probe.py"), "--study", str(root),
    "--output", str(out / "resources"), "--", "cmd.exe", "/d", "/c", str(launcher),
    str(out / "native"), str(cache)], out / "ledger", "correctness", "canonical-fp8-prefix4-every-hidden-token",
    [library, launcher, probe, proof, receipt, Path(__file__),
     root / "results/native-canonical-fp8-build-attempt2/manifest.json"])
workflow["exit_code"] = code
workflow["status"] = "process_failed" if code else "completed_needs_independent_replay"
try:
    idle()
    assert hashlib.sha256(library.read_bytes()).hexdigest() == digest
    retired = []; resolved = cache.resolve(strict=True)
    assert json.loads((cache / "study-owner.json").read_text())["study"] == str(root)
    for path in cache.iterdir():
        if path.name in before or not path.name.startswith("garnet-cuda-graph-"): continue
        st = path.lstat(); target = path.resolve(strict=True)
        assert not (getattr(st, "st_file_attributes", 0) & 0x400)
        assert target.parent == resolved and st.st_ctime >= started - 2
        entries = list(target.iterdir())
        assert all(p.name in ("model.engine", "model.engine.tmp") for p in entries)
        for p in entries:
            ps = p.lstat()
            assert p.is_file() and not (getattr(ps, "st_file_attributes", 0) & 0x400)
            assert p.resolve(strict=True).parent == target
            h = hashlib.sha256()
            with p.open("rb") as f:
                for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
            retired.append({"path": str(p), "bytes": ps.st_size, "sha256": h.hexdigest()})
        # Guarded, nonrecursive deletion of this closed run's private engine cache.
        for p in entries: p.unlink()
        target.rmdir()
    workflow["retired_private_engine_files"] = retired
finally:
    (out / "workflow.json").write_text(json.dumps(workflow, indent=2), encoding="utf8")
print(json.dumps(workflow, indent=2))
raise SystemExit(code)
