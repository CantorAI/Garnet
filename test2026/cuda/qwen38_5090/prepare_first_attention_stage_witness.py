"""Extract independently verified prefix3 input and prefix4 output witnesses."""
from pathlib import Path
import hashlib
import json
import zipfile

root = Path("D:/CantorAI/5090-study-20261023")
results = root / "results"
out = results / "first-attention-stage-witness"
assert not out.exists()
sha = lambda raw: hashlib.sha256(raw).hexdigest()
proof3_path = results / "native-canonical-fp8-prefix123-attempt1-verified.json"
proof4_path = results / "native-canonical-fp8-prefix4-hidden-attempt1-verified.json"
proof3 = json.loads(proof3_path.read_text()); proof4 = json.loads(proof4_path.read_text())
assert proof3["status"] == "canonical_fp8_prefix123_real_states_verified"
assert proof4["status"] == "canonical_fp8_deeper_prefix_evidence_verified"
archives = [results / name for name in ("native-canonical-fp8-prefix123-attempt1-evidence.zip",
    "native-canonical-fp8-prefix4-hidden-attempt1-evidence.zip")]
payloads = []; reports = []
for archive, proof, experiment, prefix in zip(archives, (proof3, proof4), ("prefix123", "prefix4-hidden"), (3, 4)):
    assert sha(archive.read_bytes()) == proof["archive_sha256"]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        evidence = json.loads(z.read("evidence-manifest.json"))
        def read(name):
            digest = evidence["files"][name]; raw = z.read("objects/" + digest)
            assert sha(raw) == digest
            return raw
        base = "results/native-canonical-fp8-" + experiment + "-attempt1/native/"
        reports.append(json.loads(read(base + "report.json")))
        frames = {}
        for batch in (1, 2):
            for repeat in (0, 1):
                for row in range(batch):
                    raw = read(base + f"prefix-{prefix}-batch-{batch}-repeat-{repeat}-chunk-0-hidden-row-{row}.f32")
                    assert len(raw) == 81920 * 4
                    frames[(batch, repeat, row)] = raw
            assert all(frames[(batch, repeat, row)] == frames[(batch, 0, 0)] for repeat in (0, 1) for row in range(batch))
        if prefix == 3: assert frames[(1, 0, 0)] == frames[(2, 0, 0)]
        payloads.append(frames)
assert reports[0]["config_sha256"] == reports[1]["config_sha256"]
assert reports[0]["input_prompt_ids"] == reports[1]["input_prompt_ids"]
assert sha((root / "models/Qwen3.8-27B-NVFP4/config.json").read_bytes()) == reports[0]["config_sha256"]
files = {"input-hidden.f32": payloads[0][(1, 0, 0)],
    "expected-b1-hidden.f32": payloads[1][(1, 0, 0)], "expected-b2-hidden.f32": payloads[1][(2, 0, 0)]}
out.mkdir()
for name, raw in files.items(): (out / name).write_bytes(raw)
manifest = {"status": "first_attention_stage_witness_prepared", "config_sha256": reports[0]["config_sha256"],
    "input_ids": reports[0]["input_prompt_ids"][:16], "files": {name: sha(raw) for name, raw in files.items()},
    "linked_archives": {p.name: sha(p.read_bytes()) for p in archives},
    "verification_receipts": {p.name: sha(p.read_bytes()) for p in (proof3_path, proof4_path)},
    "scope": "Actual canonical prefix3 first16-token F32 output and retained per-profile prefix4 first-chunk hidden. Initial first-attention KV is zero. No generated-token or performance admission."}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
print(json.dumps(manifest, indent=2))
