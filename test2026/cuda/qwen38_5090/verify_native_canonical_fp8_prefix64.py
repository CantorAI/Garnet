"""Independent full-depth prefill row/repeat replay; never a serving gate."""
from pathlib import Path
import hashlib
import argparse
import json
import math
import zipfile
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", choices=("prefix64", "prefix8-16", "prefix4-hidden", "prefix4-hidden-geometry"), default="prefix64")
experiment = parser.parse_args().experiment
geometry = experiment == "prefix4-hidden-geometry"
prefixes = {"prefix64": [64], "prefix8-16": [8, 16], "prefix4-hidden": [4], "prefix4-hidden-geometry": [4]}[experiment]
stem = "native-canonical-fp8-" + experiment
out = Path(__file__).parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()
archive = out / (stem + "-attempt1-evidence.zip")
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    evidence = json.loads(z.read("evidence-manifest.json"))
    assert evidence["storage"] == "sha256-objects-v1"
    objects = {h: z.read("objects/" + h) for h in evidence["objects"]}
    for h, raw in objects.items(): assert sha(raw) == h and len(raw) == evidence["objects"][h]
    assert set(evidence["files"].values()) == set(objects)
    for name, digest in evidence["linked_archives"].items(): assert sha((out / name).read_bytes()) == digest
    def read(name): return objects[evidence["files"][name]]
    def document(name): return json.loads(read(name))
    base = "results/" + stem + "-attempt1/"
    workflow = document(base + "workflow.json")
    assert workflow["status"] == "completed_needs_independent_replay" and workflow["exit_code"] == 0
    build = document("results/native-canonical-fp8-build-attempt" + ("1" if geometry else "2") + "/manifest.json")
    source_build = document("results/native-canonical-fp8-build-attempt2/manifest.json")
    if geometry: assert workflow["restored_library_sha256"] == source_build["library_sha256"] == evidence["restored_library_sha256"]
    assert workflow["library_sha256"] == build["library_sha256"] == evidence["library_sha256"]
    for name, digest in source_build["source_sha256"].items(): assert sha(read("Garnet/" + name)) == digest
    if geometry:
        old_lowerer = "results/native-canonical-fp8-family-scope-install/original-lowering.cpp"
        assert sha(read(old_lowerer)) == build["source_sha256"]["src/backends/tensorrt/modelopt_fp8_lowering.cpp"]
    assert sha(read("Garnet/test2026/cuda/qwen38_5090/" + ("probe_native_prefix_hidden_chunks.py" if experiment.startswith("prefix4-hidden") else "probe_native_deeper_prefix_batch.py"))) == workflow["probe_sha256"]
    report = document(base + "native/report.json")
    assert report["status"] == "diagnostic_completed" and report["prefixes"] == prefixes
    assert report["capacity"] == 4096 and report["chunk"] == 16 and not report["capture_internals"]
    assert len(report["input_prompt_ids"]) == 41 and len(report["cases"]) == 2 * len(prefixes)
    comparisons = {}
    assert len(report["b1_b2_comparisons"]) == len(prefixes)
    for prefix in prefixes:
        frames = {}
        for batch in (1, 2):
            for repeat in (0, 1):
                raw = read(base + f"native/prefix-{prefix}-batch-{batch}-repeat-{repeat}-hidden.f32")
                assert len(raw) == batch * 5120 * 4 and np.isfinite(np.frombuffer(raw, dtype="<f4")).all()
                frames[(batch, repeat)] = raw
            assert frames[(batch, 0)] == frames[(batch, 1)]
            if batch == 2: assert frames[(batch, 0)][:20480] == frames[(batch, 0)][20480:]
            case = next(v for v in report["cases"] if v["batch"] == batch and v["prefix"] == prefix)
            assert case["prefix"] == prefix and case["repeat_exact"] and case["duplicate_rows_exact"]
            assert case["hidden_sha256"] == sha(frames[(batch, 0)])
            bodies = [g for g in case["memory"]["graphs"] if g["native_nvfp4_projection_weights"] == 3 * prefix and g["native_fp8_projection_weights"] == 3 * prefix + prefix // 4]
            assert len(bodies) == 1
            g = bodies[0]
            assert g["checkpoint_weight_precision"] == "modelopt_mixed" and g["graph_capture_failures"] == 0
            assert g["graph_captures"] == 1 and g["graph_replays"] >= 1
        left = np.frombuffer(frames[(1, 0)], dtype="<f4").astype(np.float64)
        right = np.frombuffer(frames[(2, 0)][:20480], dtype="<f4").astype(np.float64)
        exact = frames[(2, 0)] == frames[(1, 0)] * 2
        max_abs = float(np.max(np.abs(left - right)))
        relative = math.sqrt(float(np.sum((left - right)**2)) / max(float(np.sum(right**2)), 1e-30))
        comparison = next(v for v in report["b1_b2_comparisons"] if v["prefix"] == prefix)
        assert comparison["hidden_exact"] == exact and comparison["max_abs"] == max_abs
        assert math.isclose(comparison["relative_l2"], relative, rel_tol=1e-12, abs_tol=1e-18)
        comparisons[str(prefix)] = {"cross_batch_hidden_exact": exact, "max_abs": max_abs, "relative_l2": relative, "changed_elements": int(np.count_nonzero(left != right))}
    chunk_comparisons = []
    if experiment.startswith("prefix4-hidden"):
        assert report["capture_hidden_chunks"] is True and len(report["internal_files"]) == 18
        chunk_frames = {}
        for item in report["internal_files"]:
            raw = read(base + "native/" + item["name"])
            assert sha(raw) == item["sha256"] and len(raw) == item["bytes"] == 16 * 5120 * 4
            assert np.isfinite(np.frombuffer(raw, dtype="<f4")).all()
            assert item["name"] not in chunk_frames
            chunk_frames[item["name"]] = raw
        for start in (0, 16, 32):
            canonical = chunk_frames[f"prefix-4-batch-1-repeat-0-chunk-{start}-hidden-row-0.f32"]
            other = chunk_frames[f"prefix-4-batch-2-repeat-0-chunk-{start}-hidden-row-0.f32"]
            for batch in (1, 2):
                expected = canonical if batch == 1 else other
                for repeat in (0, 1):
                    for row in range(batch):
                        assert chunk_frames[f"prefix-4-batch-{batch}-repeat-{repeat}-chunk-{start}-hidden-row-{row}.f32"] == expected
            left = np.frombuffer(canonical, dtype="<f4").astype(np.float64).reshape(16, 5120)
            right = np.frombuffer(other, dtype="<f4").astype(np.float64).reshape(16, 5120)
            chunk_comparisons.append({"start": start, "exact": canonical == other,
                "changed_elements": int(np.count_nonzero(left != right)),
                "changed_valid_prompt_tokens": [start + i for i in range(min(16, 41 - start)) if np.any(left[i] != right[i])],
                "max_abs": float(np.max(np.abs(left - right)))})
        with zipfile.ZipFile(out / "native-prefix-batch-evidence.zip") as old:
            assert old.testzip() is None
            for repeat in (0, 1):
                name = f"prefix-4-batch-1-repeat-{repeat}-hidden.f32"
                assert read(base + "native/" + name) == old.read("results/native-prefix-batch-diagnostic/" + name)
    intervention = []
    if geometry:
        with zipfile.ZipFile(out / "native-canonical-fp8-prefix4-hidden-attempt1-evidence.zip") as prior:
            assert prior.testzip() is None
            previous = json.loads(prior.read("evidence-manifest.json"))
            def previous_read(name):
                digest = previous["files"][name]
                raw = prior.read("objects/" + digest)
                assert sha(raw) == digest
                return raw
            old_base = "results/native-canonical-fp8-prefix4-hidden-attempt1/native/"
            for start in (0, 16, 32):
                for repeat in (0, 1):
                    name = f"prefix-4-batch-1-repeat-{repeat}-chunk-{start}-hidden-row-0.f32"
                    assert read(base + "native/" + name) == previous_read(old_base + name)
                name = f"prefix-4-batch-2-repeat-0-chunk-{start}-hidden-row-0.f32"
                a = np.frombuffer(read(base + "native/" + name), dtype="<f4")
                b = np.frombuffer(previous_read(old_base + name), dtype="<f4")
                intervention.append({"start": start, "b1_preserved": True,
                    "changed_b2_words_vs_gdn_only": int(np.count_nonzero(a != b))})
    for directory in (base + "ledger/", "results/" + stem + "-controller-ledger/"):
        events = [json.loads(v) for v in read(directory + "commands.jsonl").decode().splitlines()]
        assert events[-1]["event"] == "finished" and events[-1]["exit_code"] == 0
    monitor = document(base + "resources/resource-monitor.json")
    samples = [json.loads(v) for v in read(base + "resources/resource-vram.jsonl").decode().splitlines()]
    assert monitor["child_exit_code"] == 0 and monitor["sampled_memory"]["samples"] == len(samples)
    peak = max(v["used_bytes"] for v in samples)
    assert monitor["sampled_memory"]["sampled_peak_adapter_used_bytes"] == peak
    result = {"status": ("canonical_fp8_full_depth_prefill_evidence_verified" if experiment == "prefix64" else "canonical_fp8_deeper_prefix_evidence_verified"), "archive_sha256": sha(archive.read_bytes()),
        "file_observations_verified": len(evidence["files"]), "raw_arrays_verified": 4 * len(prefixes) + (18 if experiment.startswith("prefix4-hidden") else 0), "prefixes": comparisons,
        "chunk_comparisons": chunk_comparisons, "intervention": intervention,
        **(comparisons["64"] if experiment == "prefix64" else {}), "sampled_peak_adapter_used_bytes": peak,
        "performance_admitted": False,
        "scope": "Repeated identical41-token text rows, selected deeper prefixes, B1/B2 prefill. No independent B1 historical deeper-hidden witness; no ragged VLM/generation/quality or speed admission."}
(out / (stem + "-attempt1-verified.json")).write_text(json.dumps(result, indent=2), encoding="utf8")
print(json.dumps(result, indent=2))
