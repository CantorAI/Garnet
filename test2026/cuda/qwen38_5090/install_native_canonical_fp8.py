"""Guarded five-file application source install after isolated candidate replay."""
from pathlib import Path
import hashlib
import json
import subprocess

root = Path("D:/CantorAI/5090-study-20261023")
out = root / "results/native-canonical-fp8-source-install"
assert not out.exists()
source = root / "results/native-canonical-fp8-source"
proof = root / "results/native-canonical-fp8-attempt2-verified.json"
verified = json.loads(proof.read_text())
assert verified["status"] == "isolated_native_canonical_fp8_verified"
assert hashlib.sha256((root / "results/native-canonical-fp8-attempt2-evidence.zip").read_bytes()).hexdigest() == verified["archive_sha256"]
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
old_library = "34cb221458064a4b7264586412be5b3c7acca888cd9e74681ac2a41d259413da"
assert hashlib.sha256((root / "build/bin/garnet.dll").read_bytes()).hexdigest() == old_library
backup_library = root / "build/bin/garnet-layer0-stage-attempt1.dll"
assert hashlib.sha256(backup_library.read_bytes()).hexdigest() == old_library
expected = {
    "src/backends/tensorrt/modelopt_cublaslt.cpp": "b550107b7ce3b63542eaceb0b5f3c8556f2404fc8c3237a1028de6cba5171611",
    "src/backends/tensorrt/modelopt_cublaslt.h": "7f7decc32821115ed59a2905004fd94def81f3a7b6d769b97c339c443c3f6fcb",
    "src/backends/tensorrt/modelopt_fp8_lowering.cpp": "1a3ae3c3bf03a2d702dc53a12669e7aa1bfcf2b21b9b399ed16cbe39548f09b9",
    "src/backends/tensorrt/modelopt_fp8_plugin.cpp": "36f289b04d753584bc2704d6c0dbdaf54e8c2512fb5046268b8cd97e005270cd",
    "src/backends/tensorrt/modelopt_fp8_plugin.h": "508ad759bb32a79c5691c188d0ca64052905c897dfc9bdc26ae476e1c50f1162",
}
old = {}; new = {}
for n, h in expected.items():
    old[n] = (root / "Garnet" / n).read_bytes()
    assert hashlib.sha256(old[n]).hexdigest() == h, n
    new[n] = (source / n).read_bytes()
    assert hashlib.sha256(new[n]).hexdigest() == verified["source_files"][n], n
out.mkdir()
for n, raw in old.items():
    target = out / "original" / n
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
for n, raw in new.items():
    (root / "Garnet" / n).write_bytes(raw)
    assert (root / "Garnet" / n).read_bytes() == raw
receipt = {"status": "native_candidate_sources_installed_unbuilt", "old_source_sha256": expected,
    "new_source_sha256": {n: hashlib.sha256(b).hexdigest() for n, b in new.items()},
    "library_sha256": old_library, "retained_library": str(backup_library),
    "independent_verification_sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
    "candidate_archive_sha256": verified["archive_sha256"],
    "scope": "Five native sources only, originals retained; current DLL unchanged. DefaultOFF. Build and actual graph/full-model gates required."}
(out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf8")
print(json.dumps(receipt, indent=2))
