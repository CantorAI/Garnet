"""Replay isolated native-helper candidate plus policy and retained build failure."""
from pathlib import Path
import hashlib
import io
import json
import re
import zipfile

out = Path(__file__).parent
sha = lambda b: hashlib.sha256(b).hexdigest()
archive = out / "native-canonical-fp8-attempt2-evidence.zip"
with zipfile.ZipFile(archive) as z, zipfile.ZipFile(out / "native-canonical-fp8-attempt1-evidence.zip") as failed, zipfile.ZipFile(out / "actual-layer0-fp8-control-attempt1-evidence.zip") as old:
    for source in (z, failed):
        assert source.testzip() is None
        evidence = json.loads(source.read("evidence-manifest.json"))
        for n, h in evidence["files"].items(): assert sha(source.read(n)) == h, n
        for n, h in evidence["linked_archives"].items(): assert sha((out / n).read_bytes()) == h, n
    evidence = json.loads(z.read("evidence-manifest.json"))
    base = "results/native-canonical-fp8-attempt2/"
    prior = "results/actual-layer0-fp8-control-attempt1/"
    workflow = json.loads(z.read(base + "workflow.json"))
    assert workflow["status"] == "candidate_finished_needs_independent_replay"
    assert workflow["build_exit_code"] == workflow["policy_exit_code"] == workflow["control_exit_code"] == workflow["candidate_exit_code"] == 0
    assert workflow["library_sha256"] == evidence["library_sha256"] == "34cb221458064a4b7264586412be5b3c7acca888cd9e74681ac2a41d259413da"
    assert sha(z.read("build/native-canonical-fp8.exe")) == workflow["executable_sha256"]
    package_raw = z.read("native-canonical-fp8-source-attempt2.zip")
    assert sha(package_raw) == workflow["source_package_sha256"]
    with zipfile.ZipFile(io.BytesIO(package_raw)) as package:
        assert package.testzip() is None
        manifest = json.loads(package.read("manifest.json"))
        assert manifest["files"] == workflow["source_files"]
        for n, h in manifest["files"].items():
            raw = z.read("results/native-canonical-fp8-source/" + n)
            assert raw == package.read(n) and sha(raw) == h
    fix = json.loads(z.read(base + "source-fix-receipt.json"))
    assert fix["failure_archive_sha256"] == sha((out / "native-canonical-fp8-attempt1-evidence.zip").read_bytes())
    assert set(fix["old_source_files"]) == set(fix["new_source_files"])
    changed = {n for n, h in fix["old_source_files"].items() if h != fix["new_source_files"][n]}
    assert changed == {"test2026/cuda/qwen38_5090/modelopt_fp8_plugin_test.cpp", "test2026/cuda/qwen38_5090/run_native_canonical_fp8.py"}
    for n, h in fix["old_source_files"].items(): assert sha(failed.read("results/native-canonical-fp8-source/" + n)) == h
    assert json.loads(failed.read("results/native-canonical-fp8-attempt1/workflow.json"))["status"] == "build_failed"
    assert b"error C2248" in failed.read("results/native-canonical-fp8-attempt1/build-ledger/commands.log")
    logs = {}
    for name in ("build", "policy", "control", "candidate"):
        events = [json.loads(l) for l in z.read(base + name + "-ledger/commands.jsonl").decode().splitlines()]
        assert events[-1]["event"] == "finished" and events[-1]["exit_code"] == 0
        logs[name] = z.read(base + name + "-ledger/commands.log")
    assert b"COMPRESSED_STORAGE_FORMAT_AND_CANONICAL_POLICY_PASS" in logs["policy"]
    canonical = old.read(prior + "native/actual-m16-r0.bf16")
    for m in (16, 32):
        for label in ("control", "candidate"):
            for kind, suffix in (("cast", "bf16"), ("quantized", "e4m3")):
                assert z.read(base + f"{label}/{kind}-m{m}.{suffix}") == old.read(prior + f"native/{kind}-m{m}.{suffix}")
        for repeat in (0, 1):
            assert z.read(base + f"control/actual-m{m}-r{repeat}.bf16") == old.read(prior + f"native/actual-m{m}-r{repeat}.bf16")
            assert z.read(base + f"candidate/actual-m{m}-r{repeat}.bf16") == canonical * (m // 16)
    pattern = rb"ALGORITHM status=0 id_query=0 id=(\d+) workspace=4194304 opaque=([0-9a-f]+)"
    control = re.findall(pattern, logs["control"]); candidate = re.findall(pattern, logs["candidate"])
    assert [int(v[0]) for v in control] == [35, 35, 67, 67]
    assert len(candidate) == 4 and all(v == control[0] for v in candidate)
    assert b"SHARED_ALGORITHM_CHECK" not in logs["candidate"]  # No wrapper intervention in native mode.
    assert b"NATIVE_CANONICAL_CANDIDATE_PASSED" in logs["candidate"]
    result = {"status": "isolated_native_canonical_fp8_verified", "archive_sha256": sha(archive.read_bytes()),
        "files_verified": len(json.loads(z.read("evidence-manifest.json"))["files"]),
        "source_files": workflow["source_files"], "source_package_sha256": workflow["source_package_sha256"],
        "policy_and_wire_gates_passed": True, "both_control_profiles_match_old_outputs": True,
        "native_m32_outputs_match_m16_canonical_both_rows_repeats": True,
        "application_library_sha256": workflow["library_sha256"],
        "scope": "Isolated native helper and plugin-policy tests only. App source/DLL not installed. Lowering packaged but not compiled here. Full TensorRT graph, full-model batching and performance remain unqualified."}
(out / "native-canonical-fp8-attempt2-verified.json").write_text(json.dumps(result, indent=2), encoding="utf8")
print(json.dumps(result, indent=2))
