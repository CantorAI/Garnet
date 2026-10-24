"""Replay full-model B2 IDs independently; verified failures never count as passes."""
from pathlib import Path, PureWindowsPath
import hashlib
import json
import zipfile

out = Path(__file__).parent
sha = lambda b: hashlib.sha256(b).hexdigest()
archive = out / "native-canonical-fp8-full-b2-attempt1-evidence.zip"
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    evidence = json.loads(z.read("evidence-manifest.json"))
    for n, h in evidence["files"].items(): assert sha(z.read(n)) == h, n
    for n, h in evidence["linked_archives"].items(): assert sha((out / n).read_bytes()) == h, n
    base = "results/native-canonical-fp8-full-b2-attempt1/"
    workflow = json.loads(z.read(base + "workflow.json"))
    report = json.loads(z.read(base + "native/report.json"))
    build = json.loads(z.read("results/native-canonical-fp8-build-attempt2/manifest.json"))
    assert evidence["library_sha256"] == workflow["library_sha256"] == report["garnet_dll_sha256"] == build["library_sha256"]
    for n, h in build["source_sha256"].items(): assert sha(z.read("Garnet/" + n)) == h
    assert report["batch"] == 2 and report["capacity"] == 4096 and report["chunk"] == 16
    keys = ["s3-text-c1-r0-greedy", "s3-image-c1-r0-greedy", "red", "green"]
    assert len(report["oracles"]) == 4 and {v["key"] for v in report["oracles"]} == set(keys)
    for oracle in report["oracles"]:
        for path, digest in oracle["oracle_sha256"].items():
            name = PureWindowsPath(path).relative_to(PureWindowsPath("D:/CantorAI/5090-study-20261023")).as_posix()
            assert sha(z.read(name)) == digest
    teacher = {}
    for key in keys:
        prefix = "results/native-mtp3-http/" + key
        choice = json.loads(z.read(prefix + "-response.json"))["choices"][0]
        request = json.loads(z.read(prefix + "-request.json"))
        teacher[key] = (choice, request)
    comparisons = []
    for case in report["cases"]:
        expected_keys = keys[:2] if case["group"] == 0 else keys[2:]
        assert case["keys"] == expected_keys and case["repeat"] in (0, 1)
        expected = [teacher[k][0]["response_token_ids"] for k in expected_keys]
        assert case["expected_ids"] == expected
        assert case["prompt_lengths"] == [len(teacher[k][0]["prompt_token_ids"]) for k in expected_keys]
        assert case["limits"] == [teacher[k][1]["max_tokens"] for k in expected_keys]
        assert len(case["generated_ids"]) == 2
        bodies = [g for g in case["decoder_memory"]["graphs"] if g["native_nvfp4_projection_weights"] == 192 and g["native_fp8_projection_weights"] == 208]
        assert len(bodies) == 1 and bodies[0]["checkpoint_weight_precision"] == "modelopt_mixed"
        assert bodies[0]["graph_capture_failures"] == 0 and bodies[0]["graph_captures"] >= 1 and bodies[0]["graph_replays"] >= 1
        exact = case["generated_ids"] == expected
        assert case["exact_b1_ids_match"] == exact
        for key, actual, reference in zip(expected_keys, case["generated_ids"], expected):
            assert 0 < len(actual) <= teacher[key][1]["max_tokens"]
            first = next((i for i, (a, b) in enumerate(zip(actual, reference)) if a != b), None)
            if first is None and len(actual) != len(reference): first = min(len(actual), len(reference))
            comparisons.append({"key": key, "repeat": case["repeat"], "exact": actual == reference,
                "generated_count": len(actual), "expected_count": len(reference), "first_differing_index": first})
    for directory in (base + "ledger/", "results/native-canonical-fp8-full-b2-controller-ledger/"):
        events = [json.loads(l) for l in z.read(directory + "commands.jsonl").decode().splitlines()]
        assert events[-1]["event"] == "finished" and events[-1]["exit_code"] == workflow["exit_code"]
    monitor = json.loads(z.read(base + "resources/resource-monitor.json"))
    samples = [json.loads(l) for l in z.read(base + "resources/resource-vram.jsonl").decode().splitlines()]
    assert monitor["child_exit_code"] == workflow["exit_code"]
    assert monitor["sampled_memory"]["samples"] == len(samples)
    assert monitor["sampled_memory"]["sampled_peak_adapter_used_bytes"] == max(v["used_bytes"] for v in samples)
    module = json.loads(z.read(base + "loaded-native-modules.json"))
    assert module["disk_sha256"] == evidence["library_sha256"] and module["pid"] > 0
    assert PureWindowsPath(module["path"]) == PureWindowsPath("D:/CantorAI/5090-study-20261023/build/bin/garnet.dll")
    passed = workflow["exit_code"] == 0
    if passed:
        assert report["status"] == "backend_smoke_completed" and len(report["cases"]) == 4
        assert {(v["group"], v["repeat"]) for v in report["cases"]} == {(0, 0), (0, 1), (1, 0), (1, 1)}
        assert len(comparisons) == 8 and all(v["exact"] for v in comparisons)
    else: assert report["status"] == "failed"
    result = {"status": "full_b2_original_id_gate_verified" if passed else "full_b2_failed_gate_evidence_verified",
        "model_gate_passed": passed, "archive_sha256": sha(archive.read_bytes()), "files_verified": len(evidence["files"]),
        "library_sha256": evidence["library_sha256"], "comparisons": comparisons,
        "loaded_module_path_observed": True, "module_disk_hash_matches_build": True,
        "performance_admitted": False,
        "scope": "Original full64 B2 ragged text/image/earlyEOS pilot ID gate. Failure is preserved as failure. No HTTP, MTP, broad quality, decode speed, true concurrency throughput or2x claim."}
(out / "native-canonical-fp8-full-b2-attempt1-verified.json").write_text(json.dumps(result, indent=2), encoding="utf8")
print(json.dumps(result, indent=2))
