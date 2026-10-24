"""Archive the closed real-matrix algorithm intervention, successful or failed."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

root = Path("D:/CantorAI/5090-study-20261023")
results = root / "results"
folder = results / "actual-first-attention-fp8-shared-algorithm-attempt1"
target = results / "actual-first-attention-fp8-shared-algorithm-attempt1-evidence.zip"
assert not target.exists()
workflow = json.loads((folder / "workflow.json").read_text())
for name in ("build", "control", "shared"):
    path = folder / (name + "-ledger/commands.jsonl")
    if path.exists():
        events = [json.loads(l) for l in path.read_text().splitlines()]
        assert events[-1]["event"] == "finished"
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
files = [p for p in folder.rglob("*") if p.is_file()]
files += [p for p in (results / "actual-first-attention-fp8-shared-algorithm-controller-ledger").rglob("*") if p.is_file()]
files += [results / "actual-first-attention-fp8-control-attempt1-verified.json", Path(__file__)]
exe = root / "build/actual-first-attention-fp8-control.exe"
if "executable_sha256" in workflow:
    assert hashlib.sha256(exe.read_bytes()).hexdigest() == workflow["executable_sha256"]
    files.append(exe)
raw = {p.relative_to(root).as_posix(): p.read_bytes() for p in files}
manifest = {"files": {n: hashlib.sha256(b).hexdigest() for n, b in raw.items()},
    "linked_archives": {n: hashlib.sha256((results / n).read_bytes()).hexdigest() for n in (
        "actual-first-attention-fp8-control-attempt1-evidence.zip", "first-attention-real-stages-attempt1-evidence.zip")},
    "scope": "Untimed real projection same-executable control/intervention; all arrays retained. Check terminal statuses and independently compare canonical outputs before any candidate admission. No app default/model/batch/speed claim."}
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
    for n, b in raw.items(): z.writestr(n, b)
    z.writestr("evidence-manifest.json", json.dumps(manifest, indent=2))
print(json.dumps({"bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "files": len(raw)}))
