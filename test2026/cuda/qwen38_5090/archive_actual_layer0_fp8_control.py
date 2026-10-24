"""Archive closed projection control, keeping the bounded weight fixture in place."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

root = Path("D:/CantorAI/5090-study-20261023")
results = root / "results"
out = results / "actual-layer0-fp8-control-attempt1-evidence.zip"
assert not out.exists()
control = results / "actual-layer0-fp8-control-attempt1"
workflow = json.loads((control / "workflow.json").read_text(encoding="utf8"))
assert workflow["status"] == "control_finished_needs_independent_replay"
assert workflow["build_exit_code"] == workflow["native_exit_code"] == 0
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
for name in ("build-ledger", "native-ledger"):
    events = [json.loads(l) for l in (control / name / "commands.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "finished" and events[-1]["exit_code"] == 0
files = [p for p in control.rglob("*") if p.is_file()]
fixture = results / "actual-layer0-fp8-fixture"
files += [p for p in fixture.iterdir() if p.is_file() and p.name != "weight-n5120-k6144.e4m3"]
files += [results / "layer0-real-stage-attempt2-verified.json", root / "build/actual-layer0-fp8-control.exe", Path(__file__)]
raw = {p.relative_to(root).as_posix(): p.read_bytes() for p in files}
manifest = {
    "scope": "Closed real projection control. Weight bytes remain in fixed remote fixture, linked by safetensors header/offset and SHA256 in fixture manifest; not embedded in this archive. All cast, quantized, expected and actual arrays included. No full-model batch/speed admission.",
    "files": {n: hashlib.sha256(b).hexdigest() for n, b in raw.items()},
    "linked_archives": {"layer0-real-stage-attempt2-evidence.zip": hashlib.sha256((results / "layer0-real-stage-attempt2-evidence.zip").read_bytes()).hexdigest()},
}
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for n, b in raw.items(): z.writestr(n, b)
    z.writestr("evidence-manifest.json", json.dumps(manifest, indent=2))
print(json.dumps({"bytes": out.stat().st_size, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "files": len(raw)}))
