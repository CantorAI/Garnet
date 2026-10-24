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
out = root / "results/shared-mtp3-full-b2-attempt1"
assert not out.exists()
library = root / "build/bin/garnet.dll"
build_path = root / "results/native-full-attention-fp8-build-attempt1/manifest.json"
build = json.loads(build_path.read_text())
assert build["status"] == "native_full_attention_policy_build_finished_unqualified" and build["exit_code"] == 0
digest = build["library_sha256"]
http_terminal = root / 'results/batch-http-pilot-attempt2/ledger/commands.jsonl'
http_events = [json.loads(line) for line in http_terminal.read_text().splitlines()]
assert http_events[-1]['event'] == 'finished', 'Do not overlap the original live HTTP GPU worker'
http_workflow = json.loads((root / 'results/batch-http-pilot-attempt2/workflow.json').read_text())
assert http_workflow['exit_code'] == 0, 'Inspect any preceding HTTP failure before launching another experiment'
http_proof_path=root/'results/batch-http-pilot-attempt2-verified.json'
http_proof=json.loads(http_proof_path.read_text())
assert http_proof['status']=='batch_http_pilot_independently_verified' and http_proof['serialization_regression_captures']==4
assert http_proof['library_sha256']==digest
shared_proof_path = root / 'results/shared-full-b2-attempt1-verified.json'
shared_proof = json.loads(shared_proof_path.read_text())
assert shared_proof['status'] == 'shared_full_b2_id_gate_verified' and shared_proof['model_gate_passed']
assert shared_proof['library_sha256'] == digest
control_path = root / "results/full-attention-policy-control-stages-attempt1-verified.json"
control = json.loads(control_path.read_text())
assert control["status"] == "full_attention_policy_default_control_verified"
assert control["all_171_arrays_preserve_old_build"] and control["library_sha256"] == digest
for name, expected in build["source_sha256"].items(): assert hashlib.sha256((root / "Garnet" / name).read_bytes()).hexdigest() == expected
assert hashlib.sha256(library.read_bytes()).hexdigest() == digest
stage_path = root / "results/full-attention-policy-candidate-stages-attempt1-verified.json"
full_path = root / "results/full-attention-policy-full-b2-attempt1-verified.json"
full = json.loads(full_path.read_text())
assert full['status'] == 'full_b2_original_id_gate_verified' and full['model_gate_passed']
assert full['library_sha256'] == digest
stage = json.loads(stage_path.read_text())
assert stage["status"] == "full_attention_policy_candidate_stages_verified"
assert stage["all_171_arrays_match_retained_canonical_b1"] and stage["library_sha256"] == digest
import zipfile
failure = root / "results/native-batch2-initial-failure-evidence.zip"
assert hashlib.sha256(failure.read_bytes()).hexdigest() == "0d79e863108f3efa9ffb1af90b777ec1f1b71d37c6ecf3dad654bf1eb89abca4"
with zipfile.ZipFile(failure) as z:
    assert (methods / "probe_native_batch.py").read_bytes() == z.read("Garnet/test2026/cuda/qwen38_5090/probe_native_batch.py")
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
launcher = root / "test-shared-mtp3-full-b2.bat"
probe = methods / "probe_shared_mtp3_full_b2.py"
workflow = {"status": "running", "library_sha256": digest,
    "probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
    "scope": "B2 MTP3 target T16/draft T4: complete ordinary target state and retained original B1 IDs, two resets; no HTTP/broad quality/performance admission."}
(out / "workflow.json").write_text(json.dumps(workflow, indent=2), encoding="utf8")
code = run([sys.executable, str(methods / "monitor_native_probe.py"), "--study", str(root),
    "--output", str(out / "resources"), "--", "cmd.exe", "/d", "/c", str(launcher),
    str(out / "native"), str(cache)], out / "ledger", "correctness", "first-attention-real-stage-primary-gate",
    [library, launcher, probe, proof, receipt, control_path, stage_path, full_path, shared_proof_path, http_terminal, http_proof_path, root / "Garnet/xModel/qwen35/vl_0_8b/cuda_speculative.py", root / "Garnet/xModel/qwen35/vl_0_8b/cuda_packed_replay.py", root / "Garnet/xModel/qwen35/vl_0_8b/cuda_mtp.py", root / "Garnet/xModel/qwen35/vl_0_8b/cuda_verify.py", root / "Garnet/xModel/qwen35/vl_0_8b/cuda_packed_decoder.py", failure, Path(__file__),
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
