"""Independently replay full primary parity, cache layout and stage differences."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
import numpy as np

out = Path(__file__).parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()
archive = out / "full-attention-policy-control-stages-attempt1-evidence.zip"
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    evidence = json.loads(z.read("evidence-manifest.json")); assert evidence["storage"] == "sha256-objects-v1"
    objects = {h: z.read("objects/" + h) for h in evidence["objects"]}
    for digest, raw in objects.items(): assert sha(raw) == digest and len(raw) == evidence["objects"][digest]
    assert set(evidence["files"].values()) == set(objects)
    for name, digest in evidence["linked_archives"].items(): assert sha((out / name).read_bytes()) == digest
    def read(name): return objects[evidence["files"][name]]
    def doc(name): return json.loads(read(name))
    base = "results/full-attention-policy-control-stages-attempt1/"
    report = doc(base + "native/report.json"); workflow = doc(base + "workflow.json")
    assert report["status"] == "witness_passed_needs_independent_array_replay"
    assert workflow["status"] == "completed_needs_independent_replay" and workflow["exit_code"] == 0
    build = doc("results/native-full-attention-fp8-build-attempt1/manifest.json")
    assert build["exit_code"] == 0
    assert build["library_sha256"] == report["library_sha256"] == workflow["library_sha256"] == evidence["library_sha256"]
    for name, digest in build["source_sha256"].items(): assert sha(read("Garnet/" + name)) == digest
    assert sha(read("Garnet/test2026/cuda/qwen38_5090/probe_first_attention_real_stages.py")) == workflow["probe_sha256"]
    witness_base = "results/first-attention-stage-witness/"
    witness = doc(witness_base + "manifest.json")
    assert report["witness_manifest_sha256"] == sha(read(witness_base + "manifest.json"))
    assert witness["config_sha256"] == sha(read("models/Qwen3.8-27B-NVFP4/config.json"))
    for name, digest in witness["files"].items(): assert sha(read(witness_base + name)) == digest
    launcher = read("test-full-attention-policy-control-stages.bat").decode()
    assert "set GARNET_FP8_FULL_ATTN_M32_CANONICAL=0" in launcher
    assert "set GARNET_FP8_GDN_M32_CANONICAL=1" in launcher
    normal_width = 81920 + 2 * 4 * 4096 * 256
    parts = [(f"primary-part-{i}", min(1048576, normal_width - start), start)
        for i, start in enumerate(range(0, normal_width, 1048576))]
    stages = [("masked_input", 81920), ("input_norm", 81920), ("q_gate", 196608),
        ("projected_k", 16384), ("projected_v", 16384), ("attention_core", 98304),
        ("attention_output", 81920), ("residual", 81920), ("mlp_input_norm", 81920), ("output_hidden", 81920)]
    assert report["primary_parts"] == [list(v) for v in parts] and report["stages"] == [list(v) for v in stages]
    widths = {mode + "-" + name: width for mode in ("ordinary", "diagnostic") for name, width, _ in parts}
    widths.update(dict(stages)); widths["fresh_input"] = 81920
    frames = {}
    for item in report["files"]:
        raw = read(base + "native/" + item["name"])
        match = re.fullmatch(r"b([12])-r(-1|[01])-(.+)-row([01])\.f32", item["name"]); assert match
        batch, repeat, label, row = match.groups(); key = (int(batch), int(repeat), label, int(row))
        assert key not in frames and len(raw) == item["bytes"] == widths[label] * 4 and sha(raw) == item["sha256"]
        assert np.isfinite(np.frombuffer(raw, dtype="<f4")).all()
        frames[key] = raw
    expected_keys = {(batch, repeat, label, row) for batch in (1, 2) for repeat in (0, 1)
        for row in range(batch) for label in widths if label != "fresh_input"}
    expected_keys |= {(batch, -1, "fresh_input", row) for batch in (1, 2) for row in range(batch)}
    assert set(frames) == expected_keys and len(frames) == 171
    assert len(report["cases"]) == 2
    for batch in (1, 2):
        case = next(v for v in report["cases"] if v["batch"] == batch)
        assert case["complete_primary_matches_ordinary"] and case["historical_hidden_exact"] and case["repeats_and_duplicate_rows_exact"]
        for row in range(batch):
            assert frames[(batch, -1, "fresh_input", row)] == read(witness_base + "input-hidden.f32")
            assert frames[(batch, 0, "masked_input", row)] == read(witness_base + "input-hidden.f32")
            for label in widths:
                if label == "fresh_input": continue
                assert frames[(batch, 0, label, row)] == frames[(batch, 1, label, row)] == frames[(batch, 0, label, 0)]
            for repeat in (0, 1):
                for name, _, _ in parts:
                    assert frames[(batch, repeat, "ordinary-" + name, row)] == frames[(batch, repeat, "diagnostic-" + name, row)]
                primary = b"".join(frames[(batch, repeat, "ordinary-" + name, row)] for name, _, _ in parts)
                assert len(primary) == normal_width * 4
                hidden = primary[:81920 * 4]
                assert hidden == read(witness_base + f"expected-b{batch}-hidden.f32") == frames[(batch, repeat, "output_hidden", row)]
                cache = np.frombuffer(primary[81920 * 4:], dtype="<f4").reshape(2, 4, 4096, 256)
                assert np.count_nonzero(cache[:, :, 16:, :]) == 0
                projected_v = np.frombuffer(frames[(batch, repeat, "projected_v", row)], dtype="<f4").reshape(16, 4, 256).transpose(1, 0, 2)
                assert np.array_equal(cache[1, :, :16, :], projected_v)
        for label, extra in (("ordinary", 0), ("diagnostic", sum(width for _, width in stages))):
            s = case["stats"][label]
            assert s["native_nvfp4_projection_weights"] == 3 and s["native_fp8_projection_weights"] == 4 and s["native_int4_g128_projection_weights"] == 0
            assert s["checkpoint_weight_precision"] == "modelopt_mixed" and s["device_output_bytes"] == batch * (normal_width + extra) * 4
            assert s["graph_capture_failures"] == 0 and s["graph_captures"] == 1 and s["graph_replays"] >= 1
    negatives = report["invalid_profiles"]
    assert len(negatives) == 6 and all(v["rejected"] and v["error"] for v in negatives)
    assert {(v["flag"], v["batch"], v["time"], v["capacity"], v["layer"]) for v in negatives} == {
        (2, 1, 16, 4096, 3), (-1, 1, 16, 4096, 3), (1, 4, 16, 4096, 3),
        (1, 1, 8, 4096, 3), (1, 1, 16, 32, 3), (1, 1, 16, 4096, 2)}
    with zipfile.ZipFile(out / "first-attention-real-stages-attempt1-evidence.zip") as old:
        old_manifest = json.loads(old.read("evidence-manifest.json"))
        for item in report["files"]:
            old_name = "results/first-attention-real-stages-attempt1/native/" + item["name"]
            old_raw = old.read("objects/" + old_manifest["files"][old_name])
            assert sha(old_raw) == old_manifest["files"][old_name]
            assert read(base + "native/" + item["name"]) == old_raw, "Default-off output changed: " + item["name"]
    comparisons = {}
    for label, width in stages:
        left = frames[(1, 0, label, 0)]; right = frames[(2, 0, label, 0)]
        a = np.frombuffer(left, dtype="<f4").astype(np.float64); b = np.frombuffer(right, dtype="<f4").astype(np.float64)
        different = np.frombuffer(left, dtype="<u4") != np.frombuffer(right, dtype="<u4")
        assert report["cross_batch_stage_exact"][label] == (left == right)
        comparisons[label] = {"exact": left == right, "changed_words": int(np.count_nonzero(different)),
            "changed_tokens": np.flatnonzero(different.reshape(16, width // 16).any(axis=1)).tolist(),
            "max_abs": float(np.max(np.abs(a - b)))}
    for directory in (base + "ledger/",
            "results/native-full-attention-fp8-build-attempt1/build-ledger/"):
        events = [json.loads(v) for v in read(directory + "commands.jsonl").decode().splitlines()]
        assert events[-1]["event"] == "finished" and events[-1]["exit_code"] == 0
    monitor = doc(base + "resources/resource-monitor.json")
    samples = [json.loads(v) for v in read(base + "resources/resource-vram.jsonl").decode().splitlines()]
    assert monitor["child_exit_code"] == 0 and monitor["sampled_memory"]["samples"] == len(samples)
    peak = max(v["used_bytes"] for v in samples)
    assert monitor["sampled_memory"]["sampled_peak_adapter_used_bytes"] == peak
    result = {"status": "full_attention_policy_default_control_verified", "all_171_arrays_preserve_old_build": True, "archive_sha256": sha(archive.read_bytes()),
        "library_sha256": evidence["library_sha256"], "config_sha256": witness["config_sha256"],
        "file_observations_verified": len(evidence["files"]), "raw_arrays_verified": len(frames),
        "complete_primary_parity": True, "historical_hidden_exact": True, "value_cache_layout_verified": True,
        "negative_profiles_rejected": 6, "first_different_stage": next((name for name, _ in stages if not comparisons[name]["exact"]), None),
        "stages": comparisons, "sampled_peak_adapter_used_bytes": peak, "performance_admitted": False,
        "scope": "Actual layer3 first16 prompt tokens, zero initial KV. Primary/stage witness only; no unique algorithm cause, full-model generation, quality or performance admission."}
(out / "full-attention-policy-control-stages-attempt1-verified.json").write_text(json.dumps(result, indent=2), encoding="utf8")
print(json.dumps(result, indent=2))
