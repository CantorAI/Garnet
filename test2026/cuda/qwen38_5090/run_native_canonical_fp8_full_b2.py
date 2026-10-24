"""Original full64-layer ragged text/image B2 ID gate after GDN correction."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import zipfile

root = Path("D:/CantorAI/5090-study-20261023")
methods = root / "Garnet/test2026/cuda/qwen38_5090"
sys.path.insert(0, str(methods))
from run_logged import run
out = root / "results/native-canonical-fp8-full-b2-attempt1"
assert not out.exists()
proof_path = root / "results/native-canonical-fp8-prefix123-attempt1-verified.json"
proof = json.loads(proof_path.read_text())
assert proof["status"] == "canonical_fp8_prefix123_real_states_verified"
prefix_workflow = json.loads((root / "results/native-canonical-fp8-prefix123-attempt1/workflow.json").read_text())
assert prefix_workflow["status"] == "prefix123_passed_needs_independent_replay" and prefix_workflow["exit_code"] == 0
build_path = root / "results/native-canonical-fp8-build-attempt2/manifest.json"
build = json.loads(build_path.read_text())
library = root / "build/bin/garnet.dll"
assert hashlib.sha256(library.read_bytes()).hexdigest() == build["library_sha256"] == proof["library_sha256"]
for n, h in build["source_sha256"].items(): assert hashlib.sha256((root / "Garnet" / n).read_bytes()).hexdigest() == h
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
failure = root / "results/native-batch2-initial-failure-evidence.zip"
assert hashlib.sha256(failure.read_bytes()).hexdigest() == "0d79e863108f3efa9ffb1af90b777ec1f1b71d37c6ecf3dad654bf1eb89abca4"
probe = methods / "probe_native_batch.py"
with zipfile.ZipFile(failure) as z:
    assert probe.read_bytes() == z.read("Garnet/test2026/cuda/qwen38_5090/probe_native_batch.py")
out.mkdir()
launcher = root / "test-native-canonical-fp8-full-b2.bat"
sources = [root / "Garnet" / n for n in ("xModel/qwen38/vl_27b/model.py",
    "xModel/qwen35/vl_0_8b/cuda_packed_decoder.py", "xModel/qwen35/vl_0_8b/cuda_packed_program.py")]
code = run([sys.executable, str(methods / "monitor_native_probe.py"), "--study", str(root),
    "--output", str(out / "resources"), "--", "cmd.exe", "/d", "/c", str(launcher)],
    out / "ledger", "correctness", "original-full64-layer-ragged-b2-greedy-id-gate",
    [launcher, probe, library, build_path, failure, proof_path, Path(__file__)] + sources)
workflow = {"status": "process_failed" if code else "full_b2_finished_needs_independent_id_replay",
    "exit_code": code, "library_sha256": build["library_sha256"],
    "scope": "Original full64-layer B2 ragged text/image/earlyEOS, two resets against retained fullB1IDs. Not HTTP, MTP, broad quality, model speed or concurrency throughput admission."}
if code == 0:
    report = json.loads((out / "native/report.json").read_text())
    assert report["status"] == "backend_smoke_completed" and len(report["cases"]) == 4
    assert all(v["exact_b1_ids_match"] for v in report["cases"])
(out / "workflow.json").write_text(json.dumps(workflow, indent=2), encoding="utf8")
print(json.dumps(workflow, indent=2))
raise SystemExit(code)
