"""Independently replay exact baseline and shared-algorithm matrix intervention."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

out = Path(__file__).parent
sha = lambda b: hashlib.sha256(b).hexdigest()
archive = out / "actual-layer0-fp8-shared-algorithm-attempt1-evidence.zip"
with zipfile.ZipFile(archive) as z, zipfile.ZipFile(out / "actual-layer0-fp8-control-attempt1-evidence.zip") as old:
    assert z.testzip() is None
    evidence = json.loads(z.read("evidence-manifest.json"))
    for n, h in evidence["files"].items(): assert sha(z.read(n)) == h, n
    for n, h in evidence["linked_archives"].items(): assert sha((out / n).read_bytes()) == h, n
    base = "results/actual-layer0-fp8-shared-algorithm-attempt1/"
    prior = "results/actual-layer0-fp8-control-attempt1/"
    workflow = json.loads(z.read(base + "workflow.json"))
    assert workflow["status"] == "shared_finished_needs_independent_replay"
    assert workflow["build_exit_code"] == workflow["control_exit_code"] == workflow["shared_exit_code"] == 0
    for n, h in workflow["source_sha256"].items(): assert sha(z.read(base + "sources/" + n)) == h
    assert sha(z.read("build/actual-layer0-fp8-control.exe")) == workflow["executable_sha256"]
    for name in ("build", "control", "shared"):
        events = [json.loads(l) for l in z.read(base + name + "-ledger/commands.jsonl").decode().splitlines()]
        assert events[-1]["event"] == "finished" and events[-1]["exit_code"] == 0
    canonical = old.read(prior + "native/actual-m16-r0.bf16")
    for m in (16, 32):
        for label in ("control", "shared"):
            for suffix in (".bf16", ".e4m3"):
                kind = "cast" if suffix == ".bf16" else "quantized"
                assert z.read(base + f"{label}/{kind}-m{m}{suffix}") == old.read(prior + f"native/{kind}-m{m}{suffix}")
        for repeat in (0, 1):
            assert z.read(base + f"control/actual-m{m}-r{repeat}.bf16") == old.read(prior + f"native/actual-m{m}-r{repeat}.bf16")
            assert z.read(base + f"shared/actual-m{m}-r{repeat}.bf16") == canonical * (m // 16)
    logs = {name: z.read(base + name + "-ledger/commands.log").decode() for name in ("control", "shared")}
    pattern = r"ALGORITHM status=0 id_query=0 id=(\d+) workspace=4194304 opaque=([0-9a-f]+)"
    control = re.findall(pattern, logs["control"])
    shared = re.findall(pattern, logs["shared"])
    assert [int(v[0]) for v in control] == [35, 35, 67, 67]
    assert len(shared) == 4 and all(v == shared[0] for v in shared)
    assert shared[0] == control[0]
    checks = re.findall(r"SHARED_ALGORITHM_CHECK status=(\d+) state=(\d+) workspace=(\d+)", logs["shared"])
    assert len(checks) == 2 and all(a == b == "0" and int(c) <= 4 << 20 for a, b, c in checks)
    assert "SHARED_ALGORITHM_CANDIDATE_PASSED" in logs["shared"]
    result = {"status": "shared_fp8_algorithm_real_projection_verified", "archive_sha256": sha(archive.read_bytes()),
        "files_verified": len(evidence["files"]), "control_matches_both_archived_profiles": True,
        "shared_algorithm_id": 35, "both_m32_rows_match_m16_canonical_both_repeats": True,
        "cast_and_quantization_unchanged": True,
        "scope": "Successful checked algorithm intervention on one real layer0 projection fixture. Does not establish correctness for other matrices, full64-layer batches, application integration, performance or2x target."}
(out / "actual-layer0-fp8-shared-algorithm-attempt1-verified.json").write_text(json.dumps(result, indent=2), encoding="utf8")
print(json.dumps(result, indent=2))
