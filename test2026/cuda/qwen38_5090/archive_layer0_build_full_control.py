"""Retain closed full-model default-path regression, including raw failures."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

root = Path("D:/CantorAI/5090-study-20261023")
results = root / "results"
folder = results / "layer0-build-full-control-attempt1"
target = results / "layer0-build-full-control-attempt1-evidence.zip"
assert not target.exists()
events = [json.loads(l) for l in (folder / "ledger/commands.jsonl").read_text().splitlines()]
assert events[-1]["event"] == "finished"
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
files = [p for p in folder.rglob("*") if p.is_file()]
files += [root / n for n in ("run-layer0-build-full-control.py", "test-layer0-build-full-control.bat",
    "probe-warm-mtp-trace-startup.py", "results/layer0-stage-build/manifest.json")]
files += [root / "Garnet" / n for n in (
    "test2026/cuda/qwen38_5090/run_logged.py", "test2026/cuda/qwen38_5090/monitor_native_probe.py",
    "src/backends/tensorrt/trt_builder.cpp", "src/backends/tensorrt/qwen35_lowering.h",
    "src/backends/tensorrt/modelopt_cublaslt.cpp", "src/cuda/modelopt_fp8.cu",
    "xModel/qwen35/vl_0_8b/cuda_packed_decoder.py", "xModel/qwen35/vl_0_8b/cuda_speculative.py",
    "xModel/qwen35/vl_0_8b/cuda_packed_replay.py", "xModel/qwen35/vl_0_8b/cuda_packed_program.py",
    "xModel/qwen35/vl_0_8b/cuda_mtp.py")]
files.append(Path(__file__))
raw = {p.relative_to(root).as_posix(): p.read_bytes() for p in files}
manifest = {"terminal": events[-1],
    "library_sha256": hashlib.sha256((root / "build/bin/garnet.dll").read_bytes()).hexdigest(),
    "files": {n: hashlib.sha256(b).hexdigest() for n, b in raw.items()},
    "linked_archives": {n: hashlib.sha256((results / n).read_bytes()).hexdigest() for n in (
        "warm-mtp-trace-unprofiled-control-evidence.zip", "layer0-real-stage-attempt2-evidence.zip")},
    "scope": "Default-path full27B untimed14-case postbuild regression. Old full IDs and source linkage require independent replay. Not HTTP, broad quality, full batch or performance admission."}
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
    for n, b in raw.items(): z.writestr(n, b)
    z.writestr("evidence-manifest.json", json.dumps(manifest, indent=2))
print(json.dumps({"bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "files": len(raw)}))
