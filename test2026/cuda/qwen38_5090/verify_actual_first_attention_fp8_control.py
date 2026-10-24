"""Replay real projection outputs and independently compute BF16/FP8 input casts."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
import numpy as np

out = Path(__file__).parent
archive = out / "actual-first-attention-fp8-control-attempt1-evidence.zip"
sha = lambda b: hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z, zipfile.ZipFile(out / "first-attention-real-stages-attempt1-evidence.zip") as stage:
    assert z.testzip() is None
    assert stage.testzip() is None
    stage_manifest = json.loads(stage.read("evidence-manifest.json"))
    def stage_read(name):
        digest = stage_manifest["files"][name]; raw = stage.read("objects/" + digest)
        assert sha(raw) == digest
        return raw
    evidence = json.loads(z.read("evidence-manifest.json"))
    for n, h in evidence["files"].items(): assert sha(z.read(n)) == h, n
    for n, h in evidence["linked_archives"].items(): assert sha((out / n).read_bytes()) == h, n
    base = "results/actual-first-attention-fp8-control-attempt1/"
    fixture = "results/actual-first-attention-fp8-fixture/"
    workflow = json.loads(z.read(base + "workflow.json"))
    manifest_raw = z.read(fixture + "manifest.json")
    manifest = json.loads(manifest_raw)
    assert sha(manifest_raw) == workflow["fixture_manifest_sha256"]
    assert sha(z.read("build/actual-first-attention-fp8-control.exe")) == workflow["executable_sha256"]
    assert workflow["status"] == "control_finished_needs_independent_replay"
    assert workflow["build_exit_code"] == workflow["native_exit_code"] == 0
    for n, h in workflow["source_sha256"].items(): assert sha(z.read(base + "sources/" + n)) == h
    for name in ("build-ledger", "native-ledger"):
        events = [json.loads(l) for l in z.read(base + name + "/commands.jsonl").decode().splitlines()]
        assert events[-1]["event"] == "finished" and events[-1]["exit_code"] == 0
    for n, item in manifest["files"].items():
        if n == "weight-n12288-k5120.e4m3": continue  # Explicit external checkpoint-derived matrix.
        raw = z.read(fixture + n)
        assert sha(raw) == item["sha256"] and len(raw) == item["bytes"]
    log = z.read(base + "native-ledger/commands.log").decode()
    assert "CUBLASLT_VERSION 130400" in log and "REAL_PROJECTION_CONTROL_PASSED" in log
    selected = re.findall(r"ALGORITHM status=(\d+) id_query=(\d+) id=(\d+) workspace=(\d+) opaque=([0-9a-f]+)", log)
    assert len(selected) == 4 and all(v[:2] == ("0", "0") and int(v[3]) == 4 << 20 for v in selected)
    assert selected[0] == selected[1] and selected[2] == selected[3]
    scale = np.frombuffer(z.read(fixture + "input-scale.f32"), dtype="<f4")[0]
    codes = np.arange(127, dtype=np.uint8)
    exponent = (codes.astype(np.int32) >> 3) & 15
    mantissa = codes & 7
    table = np.where(exponent == 0, np.ldexp(mantissa.astype(np.float32), -9),
                     np.ldexp(1 + mantissa.astype(np.float32) / 8, exponent - 7)).astype(np.float32)
    saved = {}
    for b, m in ((1, 16), (2, 32)):
        raw = z.read(fixture + f"input-m{m}-k5120.f32")
        stage_prefix = "results/first-attention-real-stages-attempt1/native/"
        assert raw == b"".join(stage_read(stage_prefix + f"b{b}-r0-input_norm-row{row}.f32") for row in range(b))
        words = np.frombuffer(raw, dtype="<u4")
        cast = ((words + np.uint32(0x7fff) + ((words >> 16) & 1)) >> 16).astype("<u2")
        assert cast.tobytes() == z.read(base + f"native/cast-m{m}.bf16")
        bf32 = (cast.astype("<u4") << 16).view("<f4")
        scaled = bf32 / scale
        values = np.minimum(np.abs(scaled), np.float32(448))
        upper = np.minimum(np.searchsorted(table, values), 126)
        lower = np.maximum(upper - 1, 0)
        dl, du = values - table[lower], table[upper] - values
        chosen = np.where((du < dl) | ((du == dl) & ((upper & 1) == 0)), upper, lower).astype(np.uint8)
        chosen |= np.signbit(scaled).astype(np.uint8) << 7
        assert chosen.tobytes() == z.read(base + f"native/quantized-m{m}.e4m3")
        expected = z.read(fixture + f"expected-m{m}-n12288.bf16")
        reference = b"".join(stage_read(stage_prefix + f"b{b}-r0-q_gate-row{row}.f32") for row in range(b))
        assert expected == (np.frombuffer(reference, dtype="<u4") >> 16).astype("<u2").tobytes()
        for repeat in (0, 1): assert z.read(base + f"native/actual-m{m}-r{repeat}.bf16") == expected
        saved[m] = {"cast": cast, "quantized": chosen, "output": np.frombuffer(expected, dtype="<u2")}
    for field in ("cast", "quantized"):
        assert np.array_equal(saved[16][field], saved[32][field][:16 * 5120])
        assert np.array_equal(saved[16][field], saved[32][field][16 * 5120:])
    changed = np.flatnonzero(saved[16]["output"] != saved[32]["output"][:16 * 12288])
    assert len(changed) == 16
    assert selected[0][2] == selected[2][2] and selected[0][4] != selected[2][4]
    result = {"status": "real_fp8_projection_control_verified", "archive_sha256": sha(archive.read_bytes()),
              "files_verified": len(evidence["files"]), "algorithm_ids_m16_m32": [int(selected[0][2]), int(selected[2][2])],
              "algorithm_opaque_m16_m32": [selected[0][4], selected[2][4]],
              "native_outputs_match_archived_stages_both_repeats": True,
              "bf16_cast_and_fp8_quantization_independently_replayed": True,
              "cast_and_quantized_duplicate_rows_cross_profile_exact": True,
              "cross_profile_output_changed_words_per_row": len(changed),
              "first_changed_indices": changed.tolist(),
              "scope": "Actual matrix control reproduces both archived profiles; cast/quantized inputs identical per duplicate row. Same algorithm ID with different opaque configurations observed. No alternate algorithm experiment, unique-cause, model/batch or speed admission. Weight bytes external, hash/offset provenance retained."}
(out / "actual-first-attention-fp8-control-attempt1-verified.json").write_text(json.dumps(result, indent=2), encoding="utf8")
print(json.dumps(result, indent=2))
