"""Run actual B2 worker protocol gates after shared full-model ID qualification."""
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
out = root / "results/batch-mtp-http-pilot-attempt1"
assert not out.exists()
library = root / "build/bin/garnet.dll"
build_path = root / "results/native-full-attention-fp8-build-attempt1/manifest.json"
build = json.loads(build_path.read_text())
assert build["status"] == "native_full_attention_policy_build_finished_unqualified" and build["exit_code"] == 0
digest = build["library_sha256"]
mtp_path = root / 'results/shared-mtp3-full-b2-attempt2-verified.json'
mtp = json.loads(mtp_path.read_text())
assert mtp['status'] == 'shared_mtp3_full_b2_id_state_hash_gate_verified'
assert mtp['model_gate_passed'] and mtp['strict_target_state_hash_gate_passed']
assert mtp['library_sha256'] == digest
closed = [json.loads(line) for line in (root / 'results/batch-mtp-stream-protocol-attempt1/ledger/commands.jsonl').read_text().splitlines()]
assert closed[-1]['event'] == 'finished' and closed[-1]['exit_code'] == 0
workspace = json.loads((root / 'results/shared-mtp3-full-b2-attempt2/workflow.json').read_text())
for name, want in workspace['workspace_source_sha256'].items():
    assert hashlib.sha256((root / 'Garnet' / name).read_bytes()).hexdigest() == want

repair_path=root/'results/unicode-transport-repair2-verified.json'
repair=json.loads(repair_path.read_text())
assert repair['status']=='canonical_unicode_transport_repair_independently_verified'
assert repair['all_repaired_bytes_match_cpython'] and repair['runtime_sha256']['garnet.dll']==digest
assert hashlib.sha256((methods/'native_worker_io.py').read_bytes()).hexdigest()==repair['repair_source_sha256']
negative_path=root/'results/batch-http-pilot-attempt1-failure-verified.json'
negative=json.loads(negative_path.read_text())
assert negative['status']=='batch_http_failed_pilot_evidence_verified' and not negative['full_run_passed']
regression=json.loads((root/'results/batch-http-serialization-regression.json').read_text())
assert regression['source_archive_sha256']==negative['archive_sha256'] and len(regression['cases'])==2
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
shared_path = root / 'results/shared-full-b2-attempt1-verified.json'
protocol_path = root / 'results/batch-mtp-stream-protocol-attempt1-verified.json'
protocol = json.loads(protocol_path.read_text())
assert protocol['status'] == 'batch_mtp_stream_protocol_independently_verified' and protocol['library_sha256'] == digest
shared = json.loads(shared_path.read_text())
assert shared['status'] == 'shared_full_b2_id_gate_verified' and shared['model_gate_passed']
assert shared['library_sha256'] == digest
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
launcher = root / "test-batch-mtp-http-pilot.bat"
probe = methods / "probe_batch_mtp_http_pilot.py"
workflow = {"status": "running", "library_sha256": digest,
    "probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
    "scope": "B2 same-worker HTTP ID/stream gates then exploratory C1/2/4/8 timing. No 2x admission before independent raw SSE/resource/workload replay."}
(out / "workflow.json").write_text(json.dumps(workflow, indent=2), encoding="utf8")
code = run([sys.executable, str(methods / "monitor_native_probe.py"), "--study", str(root),
    "--output", str(out / "resources"), "--", "cmd.exe", "/d", "/c", str(launcher),
    str(out / "native"), str(cache)], out / "ledger", "correctness", "first-attention-real-stage-primary-gate",
    [library, launcher, probe, mtp_path, root / "Garnet/xModel/qwen35/vl_0_8b/cuda_mtp.py", root / "Garnet/xModel/qwen35/vl_0_8b/cuda_speculative.py", repair_path, negative_path, methods / "native_worker_io.py", root / "results/batch-http-serialization-regression.json", methods / "native_batch_mtp_stream_worker.py", root / "serve-native-batch2-mtp-shared-worker.bat", proof, receipt, control_path, stage_path, full_path, shared_path, protocol_path, methods / "native_batch_http_bridge.py", methods / "native_http_bridge.py", methods / "benchmark_stream.py", failure, Path(__file__),
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
