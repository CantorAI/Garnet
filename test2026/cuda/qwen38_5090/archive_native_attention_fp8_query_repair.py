"""Archive observer-only repair and fresh native query outputs after closure."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

root = Path("D:/CantorAI/5090-study-20261023"); results = root / "results"
folder = results / "native-attention-fp8-query-repair-attempt1"
outer = results / "native-attention-fp8-query-repair-controller-ledger"
target = results / "native-attention-fp8-query-repair-attempt1-evidence.zip"
assert not target.exists()
workflow = json.loads((folder / "workflow.json").read_text())
for path in list(folder.glob("*-ledger/commands.jsonl")) + [outer / "commands.jsonl"]:
    events = [json.loads(v) for v in path.read_text().splitlines()]; assert events[-1]["event"] == "finished"
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()) < 1024
sha = lambda raw: hashlib.sha256(raw).hexdigest()
assert sha((root / "build/bin/garnet.dll").read_bytes()) == workflow["application_library_sha256"]
source = results / "native-attention-fp8-policy-source"
for name, digest in workflow["source_files"].items(): assert sha((source / name).read_bytes()) == digest
files = [p for directory in (folder, outer) for p in directory.rglob("*") if p.is_file()]
files.extend([source / "test2026/cuda/qwen38_5090/native_attention_fp8_control.cu",
    results / "native-attention-fp8-policy-attempt1-verified.json", root / "run-native-attention-fp8-query-repair.py", Path(__file__)])
exe = root / "build/native-attention-fp8-query.exe"
if "executable_sha256" in workflow:
    assert sha(exe.read_bytes()) == workflow["executable_sha256"]; files.append(exe)
raw = {p.relative_to(root).as_posix(): p.read_bytes() for p in files}
evidence = {"files": {name: sha(b) for name, b in raw.items()},
    "linked_archives": {name: sha((results / name).read_bytes()) for name in (
        "native-attention-fp8-policy-attempt1-evidence.zip", "actual-first-attention-fp8-control-attempt1-evidence.zip")},
    "scope": "Observer-only repair; native source unchanged, prior valid plugin/GDN regressions reused by source identity. Fresh unintercepted query control/candidate arrays retained. No app/model/performance admission."}
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
    for name, b in raw.items(): z.writestr(name, b)
    z.writestr("evidence-manifest.json", json.dumps(evidence, indent=2))
print(json.dumps({"bytes": target.stat().st_size, "sha256": sha(target.read_bytes()), "files": len(raw)}))
