"""Archive each closed isolated candidate attempt, including partial builds."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import zipfile

p = argparse.ArgumentParser(); p.add_argument("--attempt", type=int, choices=(1, 2), default=1); args = p.parse_args()
root = Path("D:/CantorAI/5090-study-20261023")
results = root / "results"
folder = results / f"native-canonical-fp8-attempt{args.attempt}"
target = results / (folder.name + "-evidence.zip")
assert not target.exists()
workflow = json.loads((folder / "workflow.json").read_text())
for path in folder.glob("*-ledger/commands.jsonl"):
    events = [json.loads(l) for l in path.read_text().splitlines()]
    assert events[-1]["event"] == "finished"
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
files = [p for p in folder.rglob("*") if p.is_file()]
source = results / "native-canonical-fp8-source"
files += [p for p in source.rglob("*") if p.is_file()]
suffix = "" if args.attempt == 1 else "-attempt2"
files += [root / f"native-canonical-fp8-source{suffix}.zip", root / "run-native-canonical-fp8.py", Path(__file__),
    results / "actual-layer0-fp8-shared-algorithm-attempt1-verified.json", results / "actual-layer0-fp8-fixture/manifest.json"]
files += [p for p in (root / "build").glob("native-canonical-fp8*.exe") if p.is_file()]
raw = {p.relative_to(root).as_posix(): p.read_bytes() for p in files}
links = ["actual-layer0-fp8-shared-algorithm-attempt1-evidence.zip", "actual-layer0-fp8-control-attempt1-evidence.zip"]
if args.attempt == 2: links.append("native-canonical-fp8-attempt1-evidence.zip")
manifest = {"files": {n: hashlib.sha256(b).hexdigest() for n, b in raw.items()},
    "linked_archives": {n: hashlib.sha256((results / n).read_bytes()).hexdigest() for n in links},
    "library_sha256": hashlib.sha256((root / "build/bin/garnet.dll").read_bytes()).hexdigest(),
    "scope": "Isolated native candidate build/policy/matrix attempt, including failures. App source/DLL not installed or changed. Require independent replay; lowering source is packaged but not compiled by this component build."}
assert manifest["library_sha256"] == workflow["library_sha256"]
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
    for n, b in raw.items(): z.writestr(n, b)
    z.writestr("evidence-manifest.json", json.dumps(manifest, indent=2))
print(json.dumps({"bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "files": len(raw)}))
