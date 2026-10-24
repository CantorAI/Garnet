"""Replay saved raw prefix4 rows independently of the experiment controller."""
from pathlib import Path
import hashlib
import json
import math
import zipfile
import numpy as np

out = Path(__file__).parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()
archive = out / "native-canonical-fp8-prefix4-attempt1-evidence.zip"
digests = {"gdn": "1ed48e36d31c1cd7cca8fe6825049331fc0b584f54215fef0d940d4aa624d767",
    "geometry": "8568b9192a06067352636bfd94c97439d5524c3c0a0e7af744b19963cc0abe51"}
with zipfile.ZipFile(archive) as z, zipfile.ZipFile(out / "native-prefix-batch-evidence.zip") as old:
    assert z.testzip() is None and old.testzip() is None
    evidence = json.loads(z.read("evidence-manifest.json"))
    assert evidence["storage"] == "sha256-objects-v1"
    objects = {h: z.read("objects/" + h) for h in evidence["objects"]}
    for h, raw in objects.items(): assert sha(raw) == h and len(raw) == evidence["objects"][h]
    assert set(evidence["files"].values()) == set(objects)
    for name, digest in evidence["linked_archives"].items(): assert sha((out / name).read_bytes()) == digest
    def read(name): return objects[evidence["files"][name]]
    def document(name): return json.loads(read(name))
    base = "results/native-canonical-fp8-prefix4-attempt1/"
    workflow = document(base + "workflow.json")
    assert workflow["status"] == "geometry_completed_needs_independent_replay"
    assert workflow["restored_library_sha256"] == evidence["restored_library_sha256"] == digests["gdn"]
    assert [v["label"] for v in workflow["legs"]] == ["gdn", "geometry"]
    prior = json.loads(old.read("results/native-prefix-batch-diagnostic/report.json"))
    probe = "Garnet/test2026/cuda/qwen38_5090/probe_native_prefix_batch.py"
    assert sha(read(probe)) == "a5892d0342f333e4108210caa22de2063d9c1542ef3696ef5e8379eeba72873e"
    results = {}; arrays = 0
    for label, build_id in (("gdn", 2), ("geometry", 1)):
        leg = next(v for v in workflow["legs"] if v["label"] == label)
        assert leg["library_sha256"] == digests[label] and leg["probe_exit_code"] == 0
        build = document(f"results/native-canonical-fp8-build-attempt{build_id}/manifest.json")
        assert build["library_sha256"] == digests[label]
        report = document(base + label + "/native/report.json")
        assert report["status"] == "diagnostic_completed" and report["prefixes"] == [4]
        assert report["capacity"] == 4096 and report["chunk"] == 16 and not report["capture_internals"]
        for key in ("input_prompt_ids", "config_sha256", "oracle_sha256"): assert report[key] == prior[key]
        assert len(report["input_prompt_ids"]) == 41 and len(report["cases"]) == 2
        frames = {}
        for batch in (1, 2):
            for repeat in (0, 1):
                name = f"prefix-4-batch-{batch}-repeat-{repeat}-hidden.f32"
                raw = read(base + label + "/native/" + name)
                assert len(raw) == batch * 5120 * 4
                assert np.isfinite(np.frombuffer(raw, dtype="<f4")).all()
                frames[(batch, repeat)] = raw; arrays += 1
                if batch == 1: assert raw == old.read("results/native-prefix-batch-diagnostic/" + name)
            assert frames[(batch, 0)] == frames[(batch, 1)]
            if batch == 2: assert frames[(batch, 0)][:20480] == frames[(batch, 0)][20480:]
            case = next(v for v in report["cases"] if v["batch"] == batch)
            assert case["prefix"] == 4 and case["repeat_exact"] and case["duplicate_rows_exact"]
            assert case["hidden_sha256"] == sha(frames[(batch, 0)])
            body = [g for g in case["memory"]["graphs"] if g["native_nvfp4_projection_weights"] == 12 and g["native_fp8_projection_weights"] == 13]
            assert len(body) == 1
            assert body[0]["checkpoint_weight_precision"] == "modelopt_mixed"
            assert body[0]["graph_capture_failures"] == 0 and body[0]["graph_captures"] == 1 and body[0]["graph_replays"] >= 1
        left = np.frombuffer(frames[(1, 0)], dtype="<f4").astype(np.float64)
        right = np.frombuffer(frames[(2, 0)][:20480], dtype="<f4").astype(np.float64)
        exact = frames[(2, 0)] == frames[(1, 0)] * 2
        max_abs = float(np.max(np.abs(left - right)))
        relative = math.sqrt(float(np.sum((left - right)**2)) / max(float(np.sum(right**2)), 1e-30))
        comparison = report["b1_b2_comparisons"][0]
        assert comparison["hidden_exact"] == leg["cross_batch_hidden_exact"] == exact
        assert comparison["max_abs"] == max_abs
        assert math.isclose(comparison["relative_l2"], relative, rel_tol=1e-12, abs_tol=1e-18)
        events = [json.loads(v) for v in read(base + label + "/ledger/commands.jsonl").decode().splitlines()]
        assert events[-1]["event"] == "finished" and events[-1]["exit_code"] == 0
        monitor = document(base + label + "/resources/resource-monitor.json")
        samples = [json.loads(v) for v in read(base + label + "/resources/resource-vram.jsonl").decode().splitlines()]
        assert monitor["child_exit_code"] == 0 and monitor["sampled_memory"]["samples"] == len(samples)
        assert monitor["sampled_memory"]["sampled_peak_adapter_used_bytes"] == max(v["used_bytes"] for v in samples)
        results[label] = {"cross_batch_hidden_exact": exact, "max_abs": max_abs, "relative_l2": relative,
            "changed_elements": int(np.count_nonzero(left != right)), "library_sha256": digests[label]}
    events = [json.loads(v) for v in read("results/native-canonical-fp8-prefix4-queue-ledger/commands.jsonl").decode().splitlines()]
    assert events[-1]["event"] == "finished" and events[-1]["exit_code"] == 0
    result = {"status": "canonical_fp8_prefix4_scope_evidence_verified", "archive_sha256": sha(archive.read_bytes()),
        "file_observations_verified": len(evidence["files"]), "raw_arrays_verified": arrays,
        "legs": results, "performance_admitted": False,
        "scope": "Repeated identical 41-token text rows through three GDN layers and the first full-attention layer. Preserved B1 witness verified. Does not qualify full64 B2/VLM/quality or timing."}
(out / "native-canonical-fp8-prefix4-attempt1-verified.json").write_text(json.dumps(result, indent=2), encoding="utf8")
print(json.dumps(result, indent=2))
