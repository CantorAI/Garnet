"""Retain closed isolated policy attempts, including failures, without weights."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

root = Path("D:/CantorAI/5090-study-20261023")
results = root / "results"
folder = results / "native-attention-fp8-policy-attempt1"
outer = results / "native-attention-fp8-policy-controller-ledger"
target = results / "native-attention-fp8-policy-attempt1-evidence.zip"
assert not target.exists()
workflow = json.loads((folder / "workflow.json").read_text())
for path in list(folder.glob("*-ledger/commands.jsonl")) + [outer / "commands.jsonl"]:
    events = [json.loads(v) for v in path.read_text().splitlines()]
    assert events[-1]["event"] == "finished"
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
sha = lambda raw: hashlib.sha256(raw).hexdigest()
assert sha((root / "build/bin/garnet.dll").read_bytes()) == workflow["application_library_sha256"]
files = []
for directory in (folder, outer, results / "native-attention-fp8-policy-source"):
    files.extend(p for p in directory.rglob("*") if p.is_file())
files.extend(p for p in (results / "native-attention-fp8-plugin-wait-ledger").rglob("*") if p.is_file())
files.append(root / "capture-native-policy-wait-dump.py")
files.extend([results / "native-attention-fp8-policy-source.zip", root / "run-native-attention-fp8-policy.py", Path(__file__)])
files.extend(results / name for name in ("actual-first-attention-fp8-shared-algorithm-attempt1-verified.json",
    "actual-layer0-fp8-shared-algorithm-attempt1-verified.json", "actual-first-attention-fp8-fixture/manifest.json",
    "actual-layer0-fp8-fixture/manifest.json"))
for name, digest in workflow.get("executables", {}).items():
    exe = root / "build" / name; assert sha(exe.read_bytes()) == digest; files.append(exe)
raw = {p.relative_to(root).as_posix(): p.read_bytes() for p in files}
evidence = {"files": {name: sha(b) for name, b in raw.items()},
    "application_library_sha256": workflow["application_library_sha256"],
    "linked_archives": {name: sha((results / name).read_bytes()) for name in (
        "actual-first-attention-fp8-control-attempt1-evidence.zip", "actual-first-attention-fp8-shared-algorithm-attempt1-evidence.zip",
        "actual-layer0-fp8-control-attempt1-evidence.zip", "actual-layer0-fp8-shared-algorithm-attempt1-evidence.zip")},
    "scope": "Isolated policy/source build, wire/profile/default plugin regression and actual query/GDN matrices. All actual arrays/failures retained. Lowering source packaged but not compiled here; app source/DLL unchanged. No model/performance admission."}
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
    for name, b in raw.items(): z.writestr(name, b)
    z.writestr("evidence-manifest.json", json.dumps(evidence, indent=2))
print(json.dumps({"bytes": target.stat().st_size, "sha256": sha(target.read_bytes()), "files": len(raw)}))
