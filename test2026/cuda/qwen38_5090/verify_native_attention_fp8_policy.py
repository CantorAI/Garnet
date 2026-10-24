"""Replay isolated policy matrices and provenance before any application install."""
from pathlib import Path
import hashlib
import io
import json
import re
import zipfile

out = Path(__file__).parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()
archive = out / "native-attention-fp8-policy-attempt1-evidence.zip"
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    evidence = json.loads(z.read("evidence-manifest.json"))
    for name, digest in evidence["files"].items(): assert sha(z.read(name)) == digest
    for name, digest in evidence["linked_archives"].items(): assert sha((out / name).read_bytes()) == digest
    base = "results/native-attention-fp8-policy-attempt1/"
    workflow = json.loads(z.read(base + "workflow.json"))
    assert workflow["status"] == "gdn-candidate_finished_needs_independent_replay" and workflow["build_exit_code"] == 0
    labels = ["formats", "plugin-default", "query-control", "query-candidate", "gdn-control", "gdn-candidate"]
    assert workflow["checks"] == [{"label": label, "exit_code": 0} for label in labels]
    assert workflow["application_library_sha256"] == evidence["application_library_sha256"]
    package = z.read("results/native-attention-fp8-policy-source.zip")
    assert sha(package) == workflow["source_package_sha256"]
    with zipfile.ZipFile(io.BytesIO(package)) as source:
        assert source.testzip() is None
        manifest = json.loads(source.read("manifest.json"))
        assert manifest["files"] == workflow["source_files"]
        for name, digest in manifest["files"].items():
            raw = z.read("results/native-attention-fp8-policy-source/" + name)
            assert raw == source.read(name) and sha(raw) == digest
    for name, digest in workflow["executables"].items(): assert sha(z.read("build/" + name)) == digest
    for label in ["build"] + labels:
        events = [json.loads(line) for line in z.read(base + label + "-ledger/commands.jsonl").decode().splitlines()]
        assert events[-1]["event"] == "finished" and events[-1]["exit_code"] == 0
        if label != "build":
            exe = next(name for name in workflow["executables"] if any(k.endswith(name) for k in events[0]["artifact_sha256"]))
            assert any(k.endswith(exe) and value == workflow["executables"][exe] for k, value in events[0]["artifact_sha256"].items())
    outer = [json.loads(line) for line in z.read("results/native-attention-fp8-policy-controller-ledger/commands.jsonl").decode().splitlines()]
    assert outer[-1]["event"] == "finished" and outer[-1]["exit_code"] == 0
    formats = z.read(base + "formats-ledger/commands.log").decode()
    for n, k in ((5120, 6144), (12288, 5120)):
        assert f"CANONICAL_GEOMETRY_WIRE_AND_PROFILE_PASS n={n} k={k}" in formats
    assert "COMPRESSED_STORAGE_FORMAT_AND_CANONICAL_POLICY_PASS" in formats
    plugin = z.read(base + "plugin-default-ledger/commands.log").decode()
    cases = re.findall(r"FP8_PLUGIN_SERIALIZATION_AND_GRAPH_PASS: external_int32=(\d+), batch=(\d+)", plugin)
    assert len(cases) == 6 and {(int(a), int(b)) for a, b in cases} == {(external, batch) for external in (0, 1) for batch in (1, 16, 4096)}
    matrices = {}; query_intercepted = False
    for name in ("query", "gdn"):
        previous_name = "actual-first-attention-fp8-control-attempt1" if name == "query" else "actual-layer0-fp8-control-attempt1"
        with zipfile.ZipFile(out / (previous_name + "-evidence.zip")) as old:
            assert old.testzip() is None
            previous_base = "results/" + previous_name + "/native/"
            canonical = old.read(previous_base + "actual-m16-r0.bf16")
            parsed = {}
            for mode in ("control", "candidate"):
                directory = base + name + "-" + mode + "/"
                log = z.read(base + name + "-" + mode + "-ledger/commands.log").decode()
                assert "CUBLASLT_VERSION 130400" in log
                assert ("REAL_PROJECTION_CONTROL_PASSED" if mode == "control" else "NATIVE_CANONICAL_CANDIDATE_PASSED") in log
                if mode == "candidate":
                    if name == "query": query_intercepted = "SHARED_ALGORITHM_CHECK" in log
                    else: assert "SHARED_ALGORITHM_CHECK" not in log
                parsed[mode] = re.findall(r"ALGORITHM status=0 id_query=0 id=(\d+) workspace=4194304 opaque=([0-9a-f]+)", log)
                assert len(parsed[mode]) == 4
                for m in (16, 32):
                    for kind, extension in (("cast", "bf16"), ("quantized", "e4m3")):
                        filename = f"{kind}-m{m}.{extension}"
                        assert z.read(directory + filename) == old.read(previous_base + filename)
                    for repeat in (0, 1):
                        raw = z.read(directory + f"actual-m{m}-r{repeat}.bf16")
                        expected = old.read(previous_base + f"actual-m{m}-r{repeat}.bf16") if mode == "control" else canonical * (m // 16)
                        assert raw == expected
            assert all(item == parsed["control"][0] for item in parsed["candidate"])
            assert int(parsed["candidate"][0][0]) == (67 if name == "query" else 35)
            matrices[name] = {"default_outputs_preserved": True, "candidate_all_rows_canonical_both_repeats": True,
                "native_algorithm_selection_verified": not (name == "query" and query_intercepted),
                "cast_quantization_unchanged": True, "canonical_algorithm_id": int(parsed["candidate"][0][0]),
                "canonical_algorithm_opaque": parsed["candidate"][0][1]}
    result = {"status": ("isolated_native_attention_fp8_policy_incomplete" if query_intercepted else "isolated_native_attention_fp8_policy_verified"), "archive_sha256": sha(archive.read_bytes()),
        "native_qualification_passed": not query_intercepted,
        "qualification_failure": "Query test interceptor overrode native candidate selection; rerun required without that override" if query_intercepted else None,
        "files_verified": len(evidence["files"]), "application_library_sha256": evidence["application_library_sha256"],
        "source_files": workflow["source_files"], "both_geometry_wire_clone_profile_checks_passed": True,
        "native_default_serialized_graph_oracle_cases_passed": 6, "matrices": matrices,
        "performance_admitted": False,
        "scope": "Isolated helper/plugin candidate only. Default outputs preserved and real query/GDN candidates canonical; lowering source not compiled here, no application/full-model/batch/performance admission."}
(out / "native-attention-fp8-policy-attempt1-verified.json").write_text(json.dumps(result, indent=2), encoding="utf8")
print(json.dumps(result, indent=2))
