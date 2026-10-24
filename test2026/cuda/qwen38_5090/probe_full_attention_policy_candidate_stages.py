"""Actual layer3 stage witness; complete primary parity precedes attribution."""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

sys.path.insert(0, os.environ["GARNET_RUNTIME_ROOT"])
sys.path.insert(0, os.getcwd())
import garnet
import tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T
from test2026.cuda.qwen38_5090.run_logged import source_identity

root, out = map(Path, sys.argv[1:3])
assert os.environ.get("GARNET_FP8_FULL_ATTN_M32_CANONICAL") == "1"
assert not out.exists(); out.mkdir()
witness = root / "results/first-attention-stage-witness"
saved = json.loads((witness / "manifest.json").read_text())
assert saved["status"] == "first_attention_stage_witness_prepared"
sha = lambda raw: hashlib.sha256(raw).hexdigest()
for name, digest in saved["files"].items(): assert sha((witness / name).read_bytes()) == digest
checkpoint = root / "models/Qwen3.8-27B-NVFP4"
assert sha((checkpoint / "config.json").read_bytes()) == saved["config_sha256"]
normal_width = 81920 + 2 * 4 * 4096 * 256
parts = [(f"primary-part-{i}", min(1048576, normal_width - start), start)
    for i, start in enumerate(range(0, normal_width, 1048576))]
stages = [("masked_input", 81920), ("input_norm", 81920), ("q_gate", 196608),
    ("projected_k", 16384), ("projected_v", 16384), ("attention_core", 98304),
    ("attention_output", 81920), ("residual", 81920), ("mlp_input_norm", 81920), ("output_hidden", 81920)]
report = {"status": "starting", "source": source_identity(), "files": [], "cases": [],
    "witness_manifest_sha256": sha((witness / "manifest.json").read_bytes()), "stages": stages,
    "primary_parts": parts, "library_sha256": sha((root / "build/bin/garnet.dll").read_bytes()),
    "scope": "Actual prefix3 first16 hidden rows enter layer3 with zero first-attention KV. Full ordinary/diagnostic primary hidden+KV parity and canonical B1 hidden required. No full generation, serving or performance claim."}
frames = {}; graphs = []; io = None
def save(): (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
def capture(value, batch, repeat, label):
    for row in range(batch):
        raw = bytes(io.finite_f32_bytes(io.row(value, row)))
        name = f"b{batch}-r{repeat}-{label}-row{row}.f32"
        (out / name).write_bytes(raw); frames[(batch, repeat, label, row)] = raw
        report["files"].append({"name": name, "bytes": len(raw), "sha256": sha(raw)})
def forward(flag, layer="model.language_model.layers.3"):
    @T.fusion(name="qwen38_first_attention_real_stages", boundary="required")
    def apply(hidden, key, value, positions, slots, active):
        bound = hidden * T.binary_op("qwen35_attention_bind_key") * key
        bound = bound * T.binary_op("qwen35_attention_bind_value") * value
        bound = bound * T.binary_op("qwen35_attention_bind_positions") * positions
        bound = bound * T.binary_op("qwen35_attention_bind_slots") * slots
        bound = bound * T.binary_op("qwen35_attention_bind_active") * active
        return bound * T.unary_op("qwen35_full_decoder_chunk", layer_prefix=layer,
            num_heads=24, num_kv_heads=4, head_dim=256, rotary_dim=64, mrope_t=11, mrope_h=11, mrope_w=10,
            rope_theta=10000000.0, eps=0.000001, fused_attention=1, diagnostic_full_stages=flag)
    return apply
def symbols(batch, time=16, capacity=4096):
    shapes = ([batch, time, 5120], [batch, 4, capacity, 256], [batch, 4, capacity, 256],
        [batch, time, 3], [batch, time], [batch, time])
    return [tensor.input("attention_stage_" + str(i), shape=shape, dtype=tensor.float32) for i, shape in enumerate(shapes)]
save()
try:
    for batch in (1, 2):
        io = garnet.cuda_tensor_io(); io.open(str(checkpoint))
        raw_input = (witness / "input-hidden.f32").read_bytes()
        hidden = io.from_host(list(struct.unpack("<81920f", raw_input)) * batch, [batch, 16, 5120])
        key = io.zeros([batch, 4, 4096, 256]); value = io.zeros([batch, 4, 4096, 256])
        positions = io.from_host([float(t) for _ in range(batch) for t in range(16) for axis in range(3)], [batch, 16, 3])
        slots = io.from_host([float(t) for _ in range(batch) for t in range(16)], [batch, 16])
        active = io.from_host([1.0] * (batch * 16), [batch, 16]); boundary = io.zeros([batch, 1])
        capture(io.storage_view(hidden, [batch, 81920]), batch, -1, "fresh_input")
        if batch == 1:
            report["invalid_profiles"] = []
            for flag, bad_batch, time, capacity, layer in ((2, 1, 16, 4096, 3), (-1, 1, 16, 4096, 3),
                    (1, 4, 16, 4096, 3), (1, 1, 8, 4096, 3), (1, 1, 16, 32, 3), (1, 1, 16, 4096, 2)):
                invalid = garnet.compiled_graph(); rejected = False; reason = None
                syms = symbols(bad_batch, time, capacity)
                try: invalid.build(forward(flag, f"model.language_model.layers.{layer}")(*syms), syms, str(checkpoint))
                except Exception as error: rejected = True; reason = repr(error)
                finally: invalid.close()
                report["invalid_profiles"].append({"flag": flag, "batch": bad_batch, "time": time,
                    "capacity": capacity, "layer": layer, "rejected": rejected, "error": reason}); save()
                assert rejected, "Unsupported full-attention diagnostic accepted"
        stats = {}
        for flag, label in ((0, "ordinary"), (1, "diagnostic")):
            syms = symbols(batch); graph = garnet.compiled_graph(); graphs.append(graph)
            graph.set_builder_workspace_bytes(256 << 20)
            graph.build(forward(flag)(*syms), syms, str(checkpoint)); graph.enable_cuda_graph(True)
            for warmup in range(3):
                io.zero(key); io.zero(value)
                result = graph.run_device([hidden, key, value, positions, slots, active])
                io.slice_into(result, 0, boundary); bytes(io.finite_f32_bytes(boundary)); result = None
            for repeat in (0, 1):
                io.zero(key); io.zero(value)
                result = graph.run_device([hidden, key, value, positions, slots, active])
                for name, width, offset in parts:
                    part = io.zeros([batch, width]); io.slice_into(result, offset, part)
                    capture(part, batch, repeat, label + "-" + name)
                if flag:
                    offset = normal_width
                    for name, width in stages:
                        part = io.zeros([batch, width]); io.slice_into(result, offset, part)
                        capture(part, batch, repeat, name); offset += width
                result = None; part = None; save()
            stats[label] = json.loads(graph.stats_json())
            expected_width = normal_width + (sum(width for _, width in stages) if flag else 0)
            s = stats[label]
            assert s["native_nvfp4_projection_weights"] == 3 and s["native_fp8_projection_weights"] == 4
            assert s["checkpoint_weight_precision"] == "modelopt_mixed" and s["native_int4_g128_projection_weights"] == 0
            assert s["device_output_bytes"] == batch * expected_width * 4
            assert s["graph_capture_failures"] == 0 and s["graph_captures"] == 1 and s["graph_replays"] >= 1
            graph.close(); graphs.remove(graph)
        labels = [mode + "-" + name for mode in ("ordinary", "diagnostic") for name, _, _ in parts] + [name for name, _ in stages]
        for row in range(batch):
            for name, _, _ in parts:
                assert frames[(batch, 0, "ordinary-" + name, row)] == frames[(batch, 0, "diagnostic-" + name, row)], "Diagnostic primary differs"
            expected = (witness / "expected-b1-hidden.f32").read_bytes()
            assert frames[(batch, 0, "ordinary-primary-part-0", row)][:81920 * 4] == expected, "Candidate layer3 differs from canonical B1 prefix4 witness"
            assert frames[(batch, 0, "masked_input", row)] == raw_input
            for label in labels:
                assert frames[(batch, 0, label, row)] == frames[(batch, 1, label, row)], "Repeat changed"
                assert frames[(batch, 0, label, row)] == frames[(batch, 0, label, 0)], "Duplicate row changed"
        report["cases"].append({"batch": batch, "complete_primary_matches_ordinary": True,
            "canonical_b1_hidden_exact": True, "repeats_and_duplicate_rows_exact": True, "stats": stats}); save()
        io.close(); io = None
    for label in [mode + "-" + name for mode in ("ordinary", "diagnostic") for name, _, _ in parts] + [name for name, _ in stages]:
        for row in (0, 1):
            assert frames[(2, 0, label, row)] == frames[(1, 0, label, 0)], "Candidate cross-batch primary/stage differs: " + label
    report["complete_cross_batch_primary_and_stages_exact"] = True
    report["cross_batch_stage_exact"] = {name: frames[(1, 0, name, 0)] == frames[(2, 0, name, 0)] for name, _ in stages}
    report["status"] = "witness_passed_needs_independent_array_replay"; save()
    print("FIRST_ATTENTION_STAGE_WITNESS", json.dumps(report["cross_batch_stage_exact"]), flush=True)
except Exception as error:
    report["status"] = "failed"; report["error"] = repr(error); save(); raise
finally:
    for graph in graphs: graph.close()
    if io is not None: io.close()
