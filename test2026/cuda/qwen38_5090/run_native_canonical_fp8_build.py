"""Build installed research candidate in the existing native build slot."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

root = Path("D:/CantorAI/5090-study-20261023")
sys.path.insert(0, str(root / "Garnet/test2026/cuda/qwen38_5090"))
from run_logged import run
out = root / "results/native-canonical-fp8-build-attempt1"
assert not out.exists()
receipt_path = root / "results/native-canonical-fp8-source-install/receipt.json"
receipt = json.loads(receipt_path.read_text())
assert receipt["status"] == "native_candidate_sources_installed_unbuilt"
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
library = root / "build/bin/garnet.dll"
assert hashlib.sha256(library.read_bytes()).hexdigest() == receipt["library_sha256"]
sources = [root / "Garnet" / n for n in receipt["new_source_sha256"]]
for p in sources: assert hashlib.sha256(p.read_bytes()).hexdigest() == receipt["new_source_sha256"][p.relative_to(root / "Garnet").as_posix()]
out.mkdir()
launcher = root / "build-layer0-stages.bat"
code = run(["cmd.exe", "/d", "/c", str(launcher)], out / "build-ledger", "build",
    "default-off-native-canonical-fp8-integration", sources + [launcher, library, receipt_path, Path(__file__)])
report = {"status": "build_failed" if code else "native_build_finished_unqualified", "exit_code": code,
    "source_install_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    "source_sha256": receipt["new_source_sha256"],
    "scope": "Integrated native build only. Default/candidate actual TensorRT stages, full27B/default oldIDs, full64-layer B2 and HTTP/SSE gates required; no batch or performance admission."}
if code == 0:
    raw = library.read_bytes()
    report["library_sha256"] = hashlib.sha256(raw).hexdigest()
    backup = root / "build/bin/garnet-native-canonical-fp8-attempt1.dll"
    assert not backup.exists()
    backup.write_bytes(raw)
(out / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf8")
print(json.dumps(report, indent=2))
raise SystemExit(code)
