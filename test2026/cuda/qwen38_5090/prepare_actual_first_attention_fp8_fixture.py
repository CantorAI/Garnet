"""CPU-only extraction of the verified, real layer3 query/gate FP8 projection witness.

This creates a bounded matrix fixture, not another checkpoint. A standalone
native control must reproduce BOTH archived outputs before testing candidates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zipfile

import numpy as np


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    root = args.study
    results = root / "results"
    verified_path = results / "first-attention-real-stages-attempt1-verified.json"
    verified = json.loads(verified_path.read_text(encoding="utf8"))
    assert verified["status"] == "first_attention_real_stages_independently_verified"
    assert verified["complete_primary_parity"] and verified["historical_hidden_exact"]
    assert verified["first_different_stage"] == "q_gate"
    archive = results / "first-attention-real-stages-attempt1-evidence.zip"
    assert sha(archive.read_bytes()) == verified["archive_sha256"]
    output = results / "actual-first-attention-fp8-fixture"
    assert not output.exists(), "Preserve previous extraction, including failures"
    output.mkdir()
    files = {}

    def save(name, raw):
        (output / name).write_bytes(raw)
        files[name] = {"bytes": len(raw), "sha256": sha(raw)}

    prefix = "results/first-attention-real-stages-attempt1/native/"
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        evidence = json.loads(z.read("evidence-manifest.json"))

        def array(b, label, row=0):
            name = prefix + f"b{b}-r0-{label}-row{row}.f32"
            raw = z.read("objects/" + evidence["files"][name])
            assert sha(raw) == evidence["files"][name]
            assert np.isfinite(np.frombuffer(raw, dtype="<f4")).all()
            return raw

        first = array(1, "input_norm")
        assert len(first) == 16 * 5120 * 4
        assert first == array(2, "input_norm", 0)
        assert first == array(2, "input_norm", 1)
        save("input-m16-k5120.f32", first)
        save("input-m32-k5120.f32", first + first)
        for b in (1, 2):
            raw = b"".join(array(b, "q_gate", row) for row in range(b))
            words = np.frombuffer(raw, dtype="<u4")
            assert len(raw) == b * 16 * 12288 * 4
            assert np.all((words & 65535) == 0), "Expected BF16-restored F32 output"
            save(f"expected-m{b * 16}-n12288.bf16", (words >> 16).astype("<u2").tobytes())
        report = json.loads(z.read("objects/" + evidence["files"][prefix + "report.json"]))

    checkpoint = root / "models/Qwen3.8-27B-NVFP4"
    config = (checkpoint / "config.json").read_bytes()
    assert sha(config) == verified["config_sha256"]
    index_raw = (checkpoint / "model.safetensors.index.json").read_bytes()
    index = json.loads(index_raw)
    tensor_prefix = "model.language_model.layers.3.self_attn.q_proj."
    tensors = {}
    for suffix, dtype, shape, filename in (
        ("weight", "F8_E4M3", [12288, 5120], "weight-n12288-k5120.e4m3"),
        ("weight_scale", "F32", [], "weight-scale.f32"),
        ("input_scale", "F32", [], "input-scale.f32"),
    ):
        key = tensor_prefix + suffix
        shard = checkpoint / index["weight_map"][key]
        with shard.open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            header = f.read(length)
            info = json.loads(header)[key]
            assert info["dtype"] in (("F8_E4M3", "F8_E4M3FN") if suffix == "weight" else (dtype,)) and info["shape"] == shape, (key, info)
            begin, end = info["data_offsets"]
            f.seek(8 + length + begin)
            raw = f.read(end - begin)
            assert len(raw) == end - begin
        if suffix == "weight":
            assert len(raw) == 12288 * 5120
            assert not np.any((np.frombuffer(raw, dtype="u1") & 127) == 127)
        else:
            assert len(raw) == 4
            value = struct.unpack("<f", raw)[0]
            assert np.isfinite(value) and value > 0
        save(filename, raw)
        tensors[key] = {"shard": shard.name, "header_sha256": sha(header), **info,
                        "file": filename, "value": value if suffix != "weight" else None}
    manifest = {
        "status": "real_projection_fixture_extracted_unqualified",
        "source_archive_sha256": verified["archive_sha256"],
        "verification_sha256": sha(verified_path.read_bytes()),
        "config_sha256": sha(config), "index_sha256": sha(index_raw),
        "library_sha256": verified["library_sha256"],
        "dimensions": {"m": [16, 32], "n": 12288, "k": 5120},
        "tensors": tensors, "files": files,
        "gate": "Reproduce all archived M16/M32 BF16 output words with the deployed native FP8 helper and real F32-to-BF16 input cast before candidates or timing. No model, batch, speed or unique-cause admission.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
