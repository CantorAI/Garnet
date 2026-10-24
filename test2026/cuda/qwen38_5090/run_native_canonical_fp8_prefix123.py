"""Check all chunks/states of actual GDN prefixes before full-model B2 admission."""
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
out = root / "results/native-canonical-fp8-prefix123-attempt1"
assert not out.exists()
build_path = root / "results/native-canonical-fp8-build-attempt2/manifest.json"
build = json.loads(build_path.read_text())
assert build["status"] == "native_build_finished_unqualified" and build["exit_code"] == 0
library = root / "build/bin/garnet.dll"
assert hashlib.sha256(library.read_bytes()).hexdigest() == build["library_sha256"]
for n, h in build["source_sha256"].items(): assert hashlib.sha256((root / "Garnet" / n).read_bytes()).hexdigest() == h
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
old12 = root / "results/native-prefix12-internals-evidence.zip"
old23 = root / "results/native-prefix23-batch-evidence.zip"
assert hashlib.sha256(old12.read_bytes()).hexdigest() == "32586aea53c0fd77b1d96935f700abe0061bd24f2cc1915400454929aac10564"
assert hashlib.sha256(old23.read_bytes()).hexdigest() == "b2a21c2c96e797502f56b8229de341dc0373706fac0027a4d2444f43527d1253"
probe = methods / "probe_native_prefix_batch.py"
with zipfile.ZipFile(old12) as z:
    assert probe.read_bytes() == z.read("Garnet/test2026/cuda/qwen38_5090/probe_native_prefix_batch.py")
out.mkdir()
launcher = root / "test-native-canonical-fp8-prefix123.bat"
code = run([sys.executable, str(methods / "monitor_native_probe.py"), "--study", str(root),
    "--output", str(out / "resources"), "--", "cmd.exe", "/d", "/c", str(launcher)],
    out / "ledger", "correctness", "gdn-prefix123-all41tokens-all-chunks-states",
    [launcher, probe, library, build_path, old12, old23, Path(__file__)] + [root / "Garnet" / n for n in build["source_sha256"]])
workflow = {"status": "process_failed", "exit_code": code, "probe_exit_code": code, "library_sha256": build["library_sha256"],
    "scope": "Actual prefixes1/2/3,B1/B2,two resets,all41tokens/3chunks/hidden+every GDN state. Old B1 witnesses and cross-profile state comparisons precede full64batch and any speed claim."}
if code == 0:
    report = json.loads((out / "native/report.json").read_text())
    assert report["status"] == "diagnostic_completed" and report["prefixes"] == [1, 2, 3]
    assert len(report["input_prompt_ids"]) == 41 and report["capture_internals"]
    assert len(report["cases"]) == 6 and len(report["internal_files"]) == 162
    assert len(report["b1_b2_comparisons"]) == 3 and len(report["internal_comparisons"]) == 27
    failures = []
    for v in report["b1_b2_comparisons"]:
        if not v["hidden_exact"]: failures.append({"prefix": v["prefix"], "part": "last_hidden"})
    for v in report["internal_comparisons"]:
        if not (v["same_profile_repeat_and_duplicate_rows_exact"] and v["b1_b2_exact"]): failures.append(v)
    with zipfile.ZipFile(old12) as z12, zipfile.ZipFile(old23) as z23:
        for item in report["internal_files"]:
            name = item["name"]
            if "-batch-1-" in name and (name.startswith("prefix-1-") or name.startswith("prefix-2-")):
                if (out / "native" / name).read_bytes() != z12.read("results/native-prefix12-internals/" + name): failures.append({"old_b1_file": name})
        for prefix in (1, 2, 3):
            for repeat in (0, 1):
                name = f"prefix-{prefix}-batch-1-repeat-{repeat}-hidden.f32"
                old = z12 if prefix == 1 else z23
                base = "results/native-prefix12-internals/" if prefix == 1 else "results/native-prefix23-batch-diagnostic/"
                if (out / "native" / name).read_bytes() != old.read(base + name): failures.append({"old_b1_file": name})
    code = 1 if failures else 0
    workflow.update({"status": "array_gate_failed" if code else "prefix123_passed_needs_independent_replay",
        "exit_code": code, "failures": failures})
(out / "workflow.json").write_text(json.dumps(workflow, indent=2), encoding="utf8")
print(json.dumps(workflow, indent=2))
raise SystemExit(code)
