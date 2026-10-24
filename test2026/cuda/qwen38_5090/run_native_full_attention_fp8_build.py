"""Install exactly five verified policy sources after closure, then rebuild."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import zipfile

root = Path("D:/CantorAI/5090-study-20261023"); results = root / "results"
sys.path.insert(0, str(root / "Garnet/test2026/cuda/qwen38_5090"))
from run_logged import run
out = results / "native-full-attention-fp8-build-attempt1"
assert not out.exists()
sha = lambda raw: hashlib.sha256(raw).hexdigest()
proof_path = results / "native-attention-fp8-query-repair-attempt1-verified.json"
proof = json.loads(proof_path.read_text())
assert proof["status"] == "isolated_native_attention_fp8_policy_verified" and proof["native_qualification_passed"]
assert proof["observer_override_absent"] and proof["query_default_preserved_and_candidate_canonical"]
assert sha((results / "native-attention-fp8-query-repair-attempt1-evidence.zip").read_bytes()) == proof["archive_sha256"]
old_proof_path = results / "first-attention-real-stages-attempt1-verified.json"
old_proof = json.loads(old_proof_path.read_text())
assert old_proof["status"] == "first_attention_real_stages_independently_verified" and old_proof["complete_primary_parity"]
library = root / "build/bin/garnet.dll"; old_library = sha(library.read_bytes())
assert old_library == proof["application_library_sha256"] == old_proof["library_sha256"]
retained = root / "build/bin/garnet-native-attention-stage-attempt1.dll"
assert sha(retained.read_bytes()) == old_library
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
prior = results / "first-attention-real-stages-attempt1-evidence.zip"
assert sha(prior.read_bytes()) == old_proof["archive_sha256"]
names = ["src/backends/tensorrt/" + name for name in (
    "modelopt_cublaslt.cpp", "modelopt_cublaslt.h", "modelopt_fp8_plugin.cpp",
    "modelopt_fp8_plugin.h", "modelopt_fp8_lowering.cpp")]
source = results / "native-attention-fp8-policy-source"
old = {}; new = {}
with zipfile.ZipFile(prior) as z:
    assert z.testzip() is None; evidence = json.loads(z.read("evidence-manifest.json"))
    for name in names:
        old[name] = (root / "Garnet" / name).read_bytes(); new[name] = (source / name).read_bytes()
        expected = evidence["files"]["Garnet/" + name]
        assert sha(old[name]) == expected and old[name] == z.read("objects/" + expected)
        assert sha(new[name]) == proof["source_files"][name]
out.mkdir()
for name, raw in old.items():
    p = out / "original" / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
for name, raw in new.items():
    (root / "Garnet" / name).write_bytes(raw)
    assert (root / "Garnet" / name).read_bytes() == raw
receipt = {"status": "verified_policy_sources_installed_unbuilt", "old_library_sha256": old_library,
    "old_source_sha256": {name: sha(raw) for name, raw in old.items()},
    "new_source_sha256": {name: sha(raw) for name, raw in new.items()}, "qualification_receipt_sha256": sha(proof_path.read_bytes()),
    "scope": "Five policy source files only. Original source/DLL retained; both policies default OFF. Native app build and actual model gates still required."}
(out / "source-receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf8")
launcher = root / "build-layer0-stages.bat"
code = run(["cmd.exe", "/d", "/c", str(launcher)], out / "build-ledger", "build",
    "integrate-default-off-full-attention-fp8-policy", [proof_path, old_proof_path, prior, library, launcher, Path(__file__)] + [root / "Garnet" / name for name in names])
report = {"status": "build_failed" if code else "native_full_attention_policy_build_finished_unqualified",
    "exit_code": code, "original_library_sha256": old_library, "source_sha256": receipt["new_source_sha256"],
    "scope": "App build only. Actual same-DLL default/control/candidate primary/stages and full64 ragged B2 IDs required before serving/performance admission."}
if code == 0:
    report["library_sha256"] = sha(library.read_bytes())
    saved = root / "build/bin/garnet-native-full-attention-fp8-attempt1.dll"
    assert not saved.exists(); saved.write_bytes(library.read_bytes())
else:
    report["failed_output_library_sha256"] = sha(library.read_bytes()) if library.exists() else None
    library.write_bytes(retained.read_bytes()); assert sha(library.read_bytes()) == old_library
    report["restored_library_sha256"] = old_library
(out / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf8")
print(json.dumps(report, indent=2))
raise SystemExit(code)
