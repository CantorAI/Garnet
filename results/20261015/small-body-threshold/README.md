# Smaller lossless body admission: rejected after confirmation

PID46404 exited0 in24.8044s. Independent audit verifies50 files,24 exact-output/FP64 checks,90 tail/fallback/prefill/ownership/abort/recovery profiles and384 timed submissions. Candidate adds524288-value matrices; control retains the current coalesced large-body route at1048576 values. The exact same shaders, fixed64-thread policy, FP32 order and BF16 values remain. All29 original fixture matrices are present.

| Model / batch | GPU fraction | Wall fraction |
| --- | ---: | ---: |
| Qwen3-0.6B / 1 | 1.0513 | 1.0657 |
| Qwen3-0.6B / 2 | 1.0698 | 1.0359 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | 0.9788 | 0.9735 |
| Qwen2.5-Coder-0.5B-Instruct / 2 | 0.9765 | 0.9799 |
| Qwen3.5-0.8B / 1 | 0.9882 | 1.0116 |
| Qwen3.5-0.8B / 2 | 1.0017 | 0.9972 |

CPU admission checks checkpoint slices and exact representation. It adds the two896x896 coder q/o projections and two512x1024 VL full-attention k/v fixtures; text adds none. Runtime compact-block counts match that independent report. There are6/5/13 admitted text/coder/VL matrices in the fixture, with current large-body control still enabled for the original6/3/11.

Coder wall time improves2.0–2.7% in this initial component. Text is a mathematical no-op and varies3.6–6.6%; that variation is larger than the coder signal. VL is mixed. No broad gain or full-model benefit is established. Confirmation PID47024 exited0 in24.1627s. The combined independent audit verifies97 files,48 exact-output/FP64 checks,180 safety profiles and768 timed submissions. The identity join confirms matching sources, versions, hardware, assets and configuration.

Both policies retain extra smaller buffers prepared outside timing. These are not process-memory or initialization measurements. Each set uses one submission, two ABBA rounds and eight samples/position after common warmup. Full-model integration requires repeat evidence and original trained-model validation. Production sources and selected serving library remain unchanged.

Initial archive SHA256: `71c28df155f64894a06f8d4df60d939f6c505148303cfb49adfb65ae0dcc5234`.

```text
python3 launch_small_body_threshold.py ROOT ROOT/logs/small-body-threshold-20261015 component
python3 archive_small_body_threshold.py ROOT ARCHIVE
python -X utf8 test2026/metal/verify_small_body_threshold.py RESULTS RESULTS/verification.json
python3 launch_confirm_small_body_threshold.py ROOT ROOT/logs/small-body-threshold-confirm-20261015 checks
```

## Repeated component result

| Model / batch | Initial wall fraction | Repeat wall fraction |
| --- | ---: | ---: |
| Qwen3-0.6B / 1 | 1.0657 | 1.0401 |
| Qwen3-0.6B / 2 | 1.0359 | 0.9912 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | 0.9735 | 1.0216 |
| Qwen2.5-Coder-0.5B-Instruct / 2 | 0.9799 | 1.0133 |
| Qwen3.5-0.8B / 1 | 1.0116 | 1.0028 |
| Qwen3.5-0.8B / 2 | 0.9972 | 0.9964 |

Coder wall gains do not repeat: confirmation regresses1.3–2.2%. VL is mixed/tiny and text is a mathematical no-op with visible timing variation. Reject the smaller admission for native integration; do not rerun unchanged or promote the small initial signal. No whole-model gain, memory saving or fresh vLLM ratio is established.

Combined archive SHA256: `e8fbb3dc50dd7aa72e08c8986dfaa6d4966eda6ddb501204dbb4ea051de9bb68`.
`combined-verification.json`, `repeat-summary.json` and the separate `combined/` snapshot preserve both cohorts without replacing the original evidence.
