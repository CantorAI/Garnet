"""Restrict the opt-in to named GDN weights after archiving the original witness."""
from pathlib import Path
import hashlib
import json
import subprocess

root = Path("D:/CantorAI/5090-study-20261023")
results = root / "results"
out = results / "native-canonical-fp8-family-scope-install"
assert not out.exists()
proof_path = results / "native-canonical-fp8-stages-attempt1-verified.json"
proof = json.loads(proof_path.read_text())
assert proof["status"] == "actual_tensorrt_canonical_fp8_stages_verified"
assert hashlib.sha256((results / "native-canonical-fp8-stages-attempt1-evidence.zip").read_bytes()).hexdigest() == proof["archive_sha256"]
assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip()) < 1024
build = json.loads((results / "native-canonical-fp8-build-attempt1/manifest.json").read_text())
library = root / "build/bin/garnet.dll"
assert hashlib.sha256(library.read_bytes()).hexdigest() == proof["library_sha256"] == build["library_sha256"]
assert hashlib.sha256((root / "build/bin/garnet-native-canonical-fp8-attempt1.dll").read_bytes()).hexdigest() == build["library_sha256"]
for n, h in build["source_sha256"].items(): assert hashlib.sha256((root / "Garnet" / n).read_bytes()).hexdigest() == h
name = "src/backends/tensorrt/modelopt_fp8_lowering.cpp"
target = root / "Garnet" / name
old = target.read_bytes()
assert hashlib.sha256(old).hexdigest() == "e3675c53b75022d20f5cfd5d73412cdc6e9f52c53f97d69a7526e2f9b1dd7ba6"
raw = (root / "modelopt-fp8-lowering-family.cpp").read_bytes()
digest = hashlib.sha256(raw).hexdigest()
assert digest == "1c0dc3fc32af61c4545d18ae33c47ce58103ce6a774a7ea02058356762acbae2"
out.mkdir()
(out / "original-lowering.cpp").write_bytes(old)
(out / "new-lowering.cpp").write_bytes(raw)
target.write_bytes(raw)
assert target.read_bytes() == raw
sources = dict(build["source_sha256"]); sources[name] = digest
receipt = {"status": "native_candidate_family_scope_installed_unbuilt", "library_sha256": build["library_sha256"],
    "new_source_sha256": sources, "old_lowering_sha256": hashlib.sha256(old).hexdigest(),
    "stage_verification_sha256": hashlib.sha256(proof_path.read_bytes()).hexdigest(), "stage_archive_sha256": proof["archive_sha256"],
    "scope": "One lowering-file change restricts opt-in to .linear_attn.out_proj.weight names. Full-attention o_proj shares geometry but remains unqualified and excluded. Native helper math and default path unchanged. Rebuild and deeper actual graph/batch gates required."}
(out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf8")
print(json.dumps(receipt, indent=2))
