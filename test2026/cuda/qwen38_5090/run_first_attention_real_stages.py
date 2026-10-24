"""Run primary-gated first-attention stage witness on the guarded diagnostic build."""
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
out = root / "results/first-attention-real-stages-attempt1"
assert not out.exists()
library = root / "build/bin/garnet.dll"
build_path = root / "results/native-attention-stage-build-attempt1/manifest.json"
build = json.loads(build_path.read_text())
assert build["status"] == "native_diagnostic_build_finished_unqualified" and build["exit_code"] == 0
digest = build["library_sha256"]
for name, expected in build["source_sha256"].items(): assert hashlib.sha256((root / "Garnet" / name).read_bytes()).hexdigest() == expected
assert hashlib.sha256(library.read_bytes()).hexdigest() == digest
proof = root / "results/native-canonical-fp8-prefix4-hidden-geometry-attempt1-verified.json"
prior = json.loads(proof.read_text())
assert prior["status"] == "canonical_fp8_deeper_prefix_evidence_verified"
assert any(not v["exact"] for v in prior["chunk_comparisons"])
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
launcher = root / "test-first-attention-real-stages.bat"
probe = methods / "probe_first_attention_real_stages.py"
workflow = {"status": "running", "library_sha256": digest,
    "probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
    "scope": "Actual first-attention ten-stage B1/B2 witness, complete ordinary/diagnostic hidden+KV and historical hidden required. Additional exports may change fusion; no attribution before independent replay. No serving/performance admission."}
(out / "workflow.json").write_text(json.dumps(workflow, indent=2), encoding="utf8")
code = run([sys.executable, str(methods / "monitor_native_probe.py"), "--study", str(root),
    "--output", str(out / "resources"), "--", "cmd.exe", "/d", "/c", str(launcher),
    str(out / "native"), str(cache)], out / "ledger", "correctness", "first-attention-real-stage-primary-gate",
    [library, launcher, probe, proof, receipt, Path(__file__),
     build_path, root / "results/first-attention-stage-witness/manifest.json"])
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
