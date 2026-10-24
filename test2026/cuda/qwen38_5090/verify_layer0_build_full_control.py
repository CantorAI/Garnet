"""Independently replay all retained full27B postbuild output IDs and run evidence."""
from pathlib import Path
import hashlib
import json
import zipfile

out = Path(__file__).parent
archive = out / "layer0-build-full-control-attempt1-evidence.zip"
sha = lambda b: hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z, zipfile.ZipFile(out / "warm-mtp-trace-unprofiled-control-evidence.zip") as old:
    assert z.testzip() is None
    evidence = json.loads(z.read("evidence-manifest.json"))
    for n, h in evidence["files"].items(): assert sha(z.read(n)) == h, n
    for n, h in evidence["linked_archives"].items(): assert sha((out / n).read_bytes()) == h, n
    assert evidence["terminal"]["event"] == "finished" and evidence["terminal"]["exit_code"] == 0
    assert z.read("probe-warm-mtp-trace-startup.py") == old.read("probe-warm-mtp-trace-startup.py")
    base = "results/layer0-build-full-control-attempt1/"
    report = json.loads(z.read(base + "native/report.json"))
    prior = json.loads(old.read("results/warm-mtp-trace-unprofiled-control/native/report.json"))
    assert report["status"] == prior["status"] == "passed"
    assert report["library_sha256"] == evidence["library_sha256"] == "34cb221458064a4b7264586412be5b3c7acca888cd9e74681ac2a41d259413da"
    assert len(report["cases"]) == len(prior["cases"]) == 14
    for actual, expected in zip(report["cases"], prior["cases"]):
        assert set(actual) == set(expected)
        for field in ("key", "reuse_verified_head", "generated_ids", "request_sha256", "teacher_sha256", "phase"):
            assert actual[field] == expected[field], (actual["key"], field)
        if "repeat" in actual: assert actual["repeat"] == expected["repeat"]
        assert len(actual["generated_ids"]) == (128 if actual["key"].startswith("s3-") else 2)
    assert [v["phase"] for v in report["cases"]] == ["warmup"] * 12 + ["capture"] * 2
    monitor = json.loads(z.read(base + "resources/resource-monitor.json"))
    samples = [json.loads(l) for l in z.read(base + "resources/resource-vram.jsonl").decode().splitlines()]
    assert monitor["child_exit_code"] == 0
    assert monitor["sampled_memory"]["samples"] == len(samples)
    assert monitor["sampled_memory"]["sampled_peak_adapter_used_bytes"] == max(s["used_bytes"] for s in samples)
    workflow = json.loads(z.read(base + "workflow.json"))
    assert workflow["exit_code"] == 0
    result = {"status": "default_full27b_postbuild_14cases_verified", "archive_sha256": sha(archive.read_bytes()),
        "files_verified": len(evidence["files"]), "cases": len(report["cases"]), "old_full_output_ids_exact": True,
        "library_sha256": evidence["library_sha256"],
        "scope": "Byte-identical old14-case producer, matching requests/teacher hashes/all output IDs: six128-token cases and eight2-token semantic cases. Untimed regression only; broader20oldIDs/HTTP/SSE/fullB2 and performance gates remain open."}
(out / "layer0-build-full-control-attempt1-verified.json").write_text(json.dumps(result, indent=2), encoding="utf8")
print(json.dumps(result, indent=2))
