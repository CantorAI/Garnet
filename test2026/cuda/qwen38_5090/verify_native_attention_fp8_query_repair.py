"""Verify fresh direct native selection and exact unchanged-source reuse."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

out = Path(__file__).parent; sha = lambda raw: hashlib.sha256(raw).hexdigest()
archive = out / "native-attention-fp8-query-repair-attempt1-evidence.zip"
with zipfile.ZipFile(archive) as z, zipfile.ZipFile(out / "native-attention-fp8-policy-attempt1-evidence.zip") as first, zipfile.ZipFile(out / "actual-first-attention-fp8-control-attempt1-evidence.zip") as old:
    assert all(v.testzip() is None for v in (z, first, old))
    evidence = json.loads(z.read("evidence-manifest.json"))
    for name, digest in evidence["files"].items(): assert sha(z.read(name)) == digest
    for name, digest in evidence["linked_archives"].items(): assert sha((out / name).read_bytes()) == digest
    base = "results/native-attention-fp8-query-repair-attempt1/"
    report = json.loads(z.read(base + "workflow.json")); receipt = json.loads(z.read(base + "repair-receipt.json"))
    prior_raw = z.read("results/native-attention-fp8-policy-attempt1-verified.json"); prior = json.loads(prior_raw)
    assert prior["status"] == "isolated_native_attention_fp8_policy_incomplete" and not prior["native_qualification_passed"]
    assert receipt["prior_proof_sha256"] == sha(prior_raw)
    assert receipt["prior_archive_sha256"] == report["prior_archive_sha256"] == prior["archive_sha256"] == sha((out / "native-attention-fp8-policy-attempt1-evidence.zip").read_bytes())
    assert report["status"] == "candidate_finished_needs_independent_replay"
    assert report["build_exit_code"] == report["control_exit_code"] == report["candidate_exit_code"] == 0
    name = receipt["changed_source"]
    assert name == "test2026/cuda/qwen38_5090/native_attention_fp8_control.cu"
    original = first.read("results/native-attention-fp8-policy-source/" + name)
    fixed = z.read("results/native-attention-fp8-policy-source/" + name)
    assert original == z.read(base + "original-query-control.cu")
    before = b"if ((sameAlgorithm || nativeCanonical) && m == 32)"; after = b"if (sameAlgorithm && m == 32)"
    assert original.count(before) == 2 and fixed == original.replace(before, after, 1)
    assert b"nativeCanonical" not in fixed[fixed.index(b"static cublasStatus_t ObserveMatmul"):fixed.index(b"#define cublasLtMatmul")]
    assert b"nullptr,nativeCanonical&&m==32" in fixed
    assert sha(original) == receipt["old_sha256"] and sha(fixed) == receipt["new_sha256"] == report["source_files"][name]
    changed = [key for key in prior["source_files"] if report["source_files"][key] != prior["source_files"][key]]
    assert set(report["source_files"]) == set(prior["source_files"]) and changed == [name]
    assert report["application_library_sha256"] == prior["application_library_sha256"]
    assert sha(z.read("build/native-attention-fp8-query.exe")) == report["executable_sha256"]
    for label in ("build", "control", "candidate"):
        events = [json.loads(v) for v in z.read(base + label + "-ledger/commands.jsonl").decode().splitlines()]
        assert events[-1]["event"] == "finished" and events[-1]["exit_code"] == 0
        if label != "build":
            assert any(k.endswith("native-attention-fp8-query.exe") and digest == report["executable_sha256"] for k, digest in events[0]["artifact_sha256"].items())
    events = [json.loads(v) for v in z.read("results/native-attention-fp8-query-repair-controller-ledger/commands.jsonl").decode().splitlines()]
    assert events[-1]["event"] == "finished" and events[-1]["exit_code"] == 0
    previous = "results/actual-first-attention-fp8-control-attempt1/native/"
    canonical = old.read(previous + "actual-m16-r0.bf16")
    selections = {}
    for mode in ("control", "candidate"):
        log = z.read(base + mode + "-ledger/commands.log").decode()
        assert "SHARED_ALGORITHM_CHECK" not in log and "CUBLASLT_VERSION 130400" in log
        assert ("REAL_PROJECTION_CONTROL_PASSED" if mode == "control" else "NATIVE_CANONICAL_CANDIDATE_PASSED") in log
        selections[mode] = re.findall(r"ALGORITHM status=0 id_query=0 id=(\d+) workspace=4194304 opaque=([0-9a-f]+)", log)
        assert len(selections[mode]) == 4
        for m in (16, 32):
            for kind, extension in (("cast", "bf16"), ("quantized", "e4m3")):
                filename = f"{kind}-m{m}.{extension}"
                assert z.read(base + mode + "/" + filename) == old.read(previous + filename)
            for repeat in (0, 1):
                expected = old.read(previous + f"actual-m{m}-r{repeat}.bf16") if mode == "control" else canonical * (m // 16)
                assert z.read(base + mode + f"/actual-m{m}-r{repeat}.bf16") == expected
    assert all(v == selections["control"][0] for v in selections["candidate"])
    assert selections["candidate"][0][0] == "67"
    assert prior["both_geometry_wire_clone_profile_checks_passed"] and prior["native_default_serialized_graph_oracle_cases_passed"] == 6
    assert prior["matrices"]["gdn"]["native_algorithm_selection_verified"]
    result = {"status": "isolated_native_attention_fp8_policy_verified", "native_qualification_passed": True,
        "archive_sha256": sha(archive.read_bytes()), "prior_incomplete_archive_sha256": prior["archive_sha256"],
        "application_library_sha256": report["application_library_sha256"], "source_files": report["source_files"],
        "observer_override_absent": True, "query_default_preserved_and_candidate_canonical": True,
        "native_source_unchanged_from_prior": True, "reused_default_plugin_oracle_cases": 6,
        "reused_gdn_and_both_geometry_wire_checks": True, "performance_admitted": False,
        "scope": "Combined isolated helper/plugin evidence: fresh unintercepted query checks plus unchanged-source prior wire/default/GDN checks. Lowering still not compiled here; no app/full-model/batch/performance admission."}
(out / "native-attention-fp8-query-repair-attempt1-verified.json").write_text(json.dumps(result, indent=2), encoding="utf8")
print(json.dumps(result, indent=2))
