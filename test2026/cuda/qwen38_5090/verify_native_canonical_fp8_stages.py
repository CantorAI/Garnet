"""Replay every deduplicated TensorRT stage observation and its original bytes."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
import numpy as np

out = Path(__file__).parent
sha = lambda b: hashlib.sha256(b).hexdigest()
archive = out / "native-canonical-fp8-stages-attempt1-evidence.zip"
with zipfile.ZipFile(archive) as z, zipfile.ZipFile(out / "layer0-real-stage-attempt2-evidence.zip") as old:
    assert z.testzip() is None
    evidence = json.loads(z.read("evidence-manifest.json"))
    assert evidence["storage"] == "sha256-objects-v1"
    objects = {h: z.read("objects/" + h) for h in evidence["objects"]}
    for h, raw in objects.items(): assert sha(raw) == h and len(raw) == evidence["objects"][h]
    assert set(evidence["files"].values()) == set(objects)
    for n, h in evidence["linked_archives"].items(): assert sha((out / n).read_bytes()) == h, n
    def read(n): return objects[evidence["files"][n]]
    base = "results/native-canonical-fp8-stages-attempt1/"
    workflow = json.loads(read(base + "workflow.json"))
    build = json.loads(read("results/native-canonical-fp8-build-attempt1/manifest.json"))
    assert workflow["status"] == "candidate_passed_needs_independent_replay"
    assert [(v["label"], v["policy"], v["exit_code"], v["mismatched_files"]) for v in workflow["legs"]] == [("control", "0", 0, []), ("candidate", "1", 0, [])]
    assert workflow["library_sha256"] == build["library_sha256"] == evidence["library_sha256"]
    assert build["status"] == "native_build_finished_unqualified" and build["exit_code"] == 0
    for n, h in build["source_sha256"].items(): assert sha(read("Garnet/" + n)) == h
    probe = "Garnet/test2026/cuda/qwen38_5090/probe_layer0_real_stages.py"
    assert read(probe) == old.read(probe)
    old_base = "results/layer0-real-stage-attempt2/native/"
    reference = json.loads(old.read(old_base + "report.json"))
    arrays = 0
    for label in ("control", "candidate"):
        prefix = base + label + "/"
        report = json.loads(read(prefix + "native/report.json"))
        assert report["status"] == "witness_passed_needs_independent_array_replay"
        assert report["input_ids"] == reference["input_ids"] and report["stages"] == reference["stages"]
        assert report["config_sha256"] == reference["config_sha256"]
        assert report["library_sha256"] == evidence["library_sha256"] and len(report["files"]) == 81
        for item in report["files"]:
            name = item["name"]; raw = read(prefix + "native/" + name)
            assert len(raw) == item["bytes"] and sha(raw) == item["sha256"]
            assert np.isfinite(np.frombuffer(raw, dtype="<f4")).all()
            expected_name = name
            if label == "candidate":
                match = re.fullmatch(r"b([12])-r(-?\d+)-(.+)-row([01])\.f32", name)
                assert match
                expected_name = f"b1-r{match[2]}-{match[3]}-row0.f32"
            assert raw == old.read(old_base + expected_name), (label, name)
            arrays += 1
        for case in report["cases"]:
            assert case["duplicate_rows_exact"]
            for row in case["rows"]:
                assert all(row[k] for k in ("primary_matches_ordinary", "masked_input_matches_prompt", "ordinary_repeat_exact", "stage_repeats_exact"))
        for entry in report["native_projection_counts"]:
            b = entry["batch"]; stats = entry["stats"]
            assert stats["native_nvfp4_projection_weights"] == stats["native_fp8_projection_weights"] == 3
            assert stats["native_int4_g128_projection_weights"] == 0
            assert stats["graph_capture_failures"] == 0 and stats["graph_captures"] == 1 and stats["graph_replays"] >= 1
            assert stats["device_output_bytes"] == b * (909312 + sum(v[1] for v in reference["stages"])) * 4
        assert {(v["flag"], v["batch"], v["time"]) for v in report["invalid_profiles"]} == {(2, 1, 16), (-1, 1, 16), (1, 4, 16), (1, 1, 4)}
        assert all(v["rejected"] for v in report["invalid_profiles"])
        events = [json.loads(l) for l in read(prefix + "ledger/commands.jsonl").decode().splitlines()]
        assert events[-1]["event"] == "finished" and events[-1]["exit_code"] == 0
        resource = json.loads(read(prefix + "resources/resource-monitor.json"))
        samples = [json.loads(l) for l in read(prefix + "resources/resource-vram.jsonl").decode().splitlines()]
        assert resource["child_exit_code"] == 0 and resource["sampled_memory"]["samples"] == len(samples)
        assert resource["sampled_memory"]["sampled_peak_adapter_used_bytes"] == max(v["used_bytes"] for v in samples)
        if label == "candidate": assert all(report["cross_batch_stage_exact"].values())
    result = {"status": "actual_tensorrt_canonical_fp8_stages_verified", "archive_sha256": sha(archive.read_bytes()),
        "file_observations_verified": len(evidence["files"]), "unique_objects_verified": len(objects), "raw_arrays_verified": arrays,
        "library_sha256": evidence["library_sha256"], "default_control_preserves_old_profiles": True,
        "candidate_all_rows_repeats_stages_and_primary_match_old_b1": True,
        "scope": "Actual layer0 first16 prompt-token witness, zero initial state, B1/B2, two retained repeats. Not deeper layers, later chunks, full-model batching, quality or speed admission. Geometry-only candidate also covers same-shaped full-attention projections; restrict family before deeper integration."}
(out / "native-canonical-fp8-stages-attempt1-verified.json").write_text(json.dumps(result, indent=2), encoding="utf8")
print(json.dumps(result, indent=2))
