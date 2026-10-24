"""Requalify default full27B behavior after the optional native stage-export build."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import zipfile

root = Path("D:/CantorAI/5090-study-20261023")
sys.path.insert(0, str(root / "Garnet/test2026/cuda/qwen38_5090"))
from run_logged import run

out = root / "results/layer0-build-full-control-attempt1"
assert not out.exists()
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
library = root / "build/bin/garnet.dll"
assert hashlib.sha256(library.read_bytes()).hexdigest() == "34cb221458064a4b7264586412be5b3c7acca888cd9e74681ac2a41d259413da"
probe = root / "probe-warm-mtp-trace-startup.py"
old = root / "results/warm-mtp-trace-unprofiled-control-evidence.zip"
assert hashlib.sha256(old.read_bytes()).hexdigest() == "9b55712ca377e6499ca06463c03ad30fbfb46f3661b695fe83b9bf1f312d39ae"
with zipfile.ZipFile(old) as z:
    assert z.testzip() is None
    assert probe.read_bytes() == z.read("probe-warm-mtp-trace-startup.py")
launcher = root / "test-layer0-build-full-control.bat"
out.mkdir()
code = run([sys.executable, str(root / "Garnet/test2026/cuda/qwen38_5090/monitor_native_probe.py"),
            "--study", str(root), "--output", str(out / "resources"), "--",
            "cmd.exe", "/d", "/c", str(launcher)], out / "ledger", "correctness",
           "default-full27b-14case-after-stage-export-build", [launcher, probe, library, old, Path(__file__)])
report = {"status": "process_failed" if code else "process_finished_needs_independent_full_output_replay",
          "exit_code": code, "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
          "scope": "Default full27B untimed14-case regression control; separate full old-ID/HTTP/SSE and batch gates still required before performance."}
if code == 0:
    assert json.loads((out / "native/report.json").read_text())["status"] == "passed"
(out / "workflow.json").write_text(json.dumps(report, indent=2), encoding="utf8")
print(json.dumps(report, indent=2))
raise SystemExit(code)
