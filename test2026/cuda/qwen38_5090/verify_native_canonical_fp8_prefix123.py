"""Independently replay all three-prefix chunks/states and retained B1 witnesses."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
import numpy as np

out = Path(__file__).parent
sha = lambda b: hashlib.sha256(b).hexdigest()
archive = out / "native-canonical-fp8-prefix123-attempt1-evidence.zip"
with zipfile.ZipFile(archive) as z, zipfile.ZipFile(out / "native-prefix12-internals-evidence.zip") as z12, zipfile.ZipFile(out / "native-prefix23-batch-evidence.zip") as z23:
    assert z.testzip() is None
    evidence = json.loads(z.read("evidence-manifest.json"))
    assert evidence["storage"] == "sha256-objects-v1"
    objects = {h: z.read("objects/" + h) for h in evidence["objects"]}
    for h, raw in objects.items(): assert sha(raw) == h and len(raw) == evidence["objects"][h]
    assert set(evidence["files"].values()) == set(objects)
    for n, h in evidence["linked_archives"].items(): assert sha((out / n).read_bytes()) == h, n
    def read(n): return objects[evidence["files"][n]]
    base = "results/native-canonical-fp8-prefix123-attempt1/"
    workflow = json.loads(read(base + "workflow.json"))
    report = json.loads(read(base + "native/report.json"))
    assert workflow["status"] == "prefix123_passed_needs_independent_replay" and workflow["exit_code"] == workflow["probe_exit_code"] == 0
    assert workflow["library_sha256"] == evidence["library_sha256"]
    assert report["status"] == "diagnostic_completed" and report["prefixes"] == [1, 2, 3]
    assert report["capacity"] == 4096 and report["chunk"] == 16 and report["capture_internals"]
    prior = json.loads(z12.read("results/native-prefix12-internals/report.json"))
    assert report["input_prompt_ids"] == prior["input_prompt_ids"] and len(report["input_prompt_ids"]) == 41
    assert report["config_sha256"] == prior["config_sha256"] and report["oracle_sha256"] == prior["oracle_sha256"]
    probe = "Garnet/test2026/cuda/qwen38_5090/probe_native_prefix_batch.py"
    assert read(probe) == z12.read(probe)
    assert len(report["cases"]) == 6 and len(report["internal_files"]) == 162
    assert len(report["internal_comparisons"]) == 27 and len(report["b1_b2_comparisons"]) == 3
    frames = {}; preserved_b1 = 0
    for item in report["internal_files"]:
        name = item["name"]; raw = read(base + "native/" + name)
        assert len(raw) == item["bytes"] and sha(raw) == item["sha256"]
        assert np.isfinite(np.frombuffer(raw, dtype="<f4")).all()
        match = re.fullmatch(r"prefix-([123])-batch-([12])-repeat-([01])-chunk-(0|16|32)-(.+)-row-([01])\.f32", name)
        assert match
        p, b, repeat, chunk, part, row = match.groups()
        key = (int(p), int(b), int(repeat), int(chunk), part, int(row))
        assert key not in frames
        frames[key] = raw
        assert len(raw) == (81920 if part == "hidden" else 827392) * 4
        if b == "1" and p in ("1", "2"):
            assert raw == z12.read("results/native-prefix12-internals/" + name)
            preserved_b1 += 1
    comparisons = 0
    for p in (1, 2, 3):
        for chunk in (0, 16, 32):
            for part in ["hidden"] + [f"layer-{i}-state" for i in range(p)]:
                canonical = frames[(p, 1, 0, chunk, part, 0)]
                for b in (1, 2):
                    for repeat in (0, 1):
                        for row in range(b): assert frames[(p, b, repeat, chunk, part, row)] == canonical
                comparisons += 1
        canonical_hidden = None
        for b in (1, 2):
            for repeat in (0, 1):
                name = f"prefix-{p}-batch-{b}-repeat-{repeat}-hidden.f32"
                raw = read(base + "native/" + name)
                assert len(raw) == b * 5120 * 4
                assert np.isfinite(np.frombuffer(raw, dtype="<f4")).all()
                if canonical_hidden is None: canonical_hidden = raw
                assert raw == canonical_hidden * b
                if b == 1:
                    old = z12 if p == 1 else z23
                    old_base = "results/native-prefix12-internals/" if p == 1 else "results/native-prefix23-batch-diagnostic/"
                    assert raw == old.read(old_base + name)
                    preserved_b1 += 1
    assert comparisons == 27 and len(frames) == 162
    for case in report["cases"]:
        assert case["repeat_exact"] and case["duplicate_rows_exact"]
        p = case["prefix"]
        bodies = [g for g in case["memory"]["graphs"] if g["native_nvfp4_projection_weights"] == 3 * p and g["native_fp8_projection_weights"] == 3 * p]
        assert len(bodies) == 1
        g = bodies[0]
        assert g["checkpoint_weight_precision"] == "modelopt_mixed" and g["graph_capture_failures"] == 0
        assert g["graph_captures"] == 1 and g["graph_replays"] >= 1
    for directory in (base + "ledger/", "results/native-canonical-fp8-prefix123-controller-ledger/"):
        events = [json.loads(l) for l in read(directory + "commands.jsonl").decode().splitlines()]
        assert events[-1]["event"] == "finished" and events[-1]["exit_code"] == 0
    monitor = json.loads(read(base + "resources/resource-monitor.json"))
    samples = [json.loads(l) for l in read(base + "resources/resource-vram.jsonl").decode().splitlines()]
    assert monitor["child_exit_code"] == 0 and monitor["sampled_memory"]["samples"] == len(samples)
    assert monitor["sampled_memory"]["sampled_peak_adapter_used_bytes"] == max(v["used_bytes"] for v in samples)
    result = {"status": "canonical_fp8_prefix123_real_states_verified", "archive_sha256": sha(archive.read_bytes()),
        "file_observations_verified": len(evidence["files"]), "unique_objects_verified": len(objects),
        "internal_arrays_verified": len(frames), "final_hidden_arrays_verified": 12,
        "cross_batch_chunk_state_comparisons": comparisons, "old_b1_arrays_preserved": preserved_b1,
        "library_sha256": evidence["library_sha256"],
        "scope": "Actual GDN prefixes1/2/3,all41text tokens/3chunks,both profiles,all rows/two resets. Full-attention/full64 model, text/image ID, HTTP, MTP and performance gates remain open. No loaded-module observation or speed claim."}
(out / "native-canonical-fp8-prefix123-attempt1-verified.json").write_text(json.dumps(result, indent=2), encoding="utf8")
print(json.dumps(result, indent=2))
