# Combined projection graphs: component gains with rejected batch-2 profiles

Owned PID6026 exited0 in71.6171s after its exact native dependencies were terminal. Independent audit verifies29 archived files,48 original-bound numerical checks,12 first-graph records and512 timed submissions. Four of12 policy/model/batch profiles fail the original native-comparison envelope and have no timed windows. A successful process exit does not mean every numerical profile passed.

Each graph computes all independent first-layer projections in one synchronous multi-result call. Policy1 supplies original FP32 values as weight placeholders; policy2 embeds them as constants. All29 matrices are included, preserving special native control routes. Graph inputs, weights, products and outputs are asserted FP32. BF16 learned values expand losslessly; inputs remain independent even when synthetic bytes coincide. This is not a complete dependency layer, full model, fused MLP or a TTFT result.

| Model / batch | Weight policy | Native wall ms | Graph wall ms | Candidate/control |
| --- | --- | ---: | ---: | ---: |
| Qwen3-0.6B / 1 | 1 | 1.8507 | 1.5448 | 0.8347 |
| Qwen3-0.6B / 1 | 2 | 1.7045 | 1.4035 | 0.8234 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | 1 | 1.5810 | 1.4508 | 0.9176 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | 2 | 1.5112 | 1.4273 | 0.9445 |
| Qwen3.5-0.8B / 1 | 1 | 5.9589 | 4.8627 | 0.8160 |
| Qwen3.5-0.8B / 1 | 2 | 6.3558 | 4.9077 | 0.7722 |
| Qwen3.5-0.8B / 2 | 1 | 9.3518 | 7.6942 | 0.8228 |
| Qwen3.5-0.8B / 2 | 2 | 8.9481 | 7.8400 | 0.8762 |

All admitted groups improve component wall latency about5.5–22.8% in this initial trial. Text B2 policies1/2 fail pattern1 with worst comparison ratio1.14625; coder B2 policies1/2 fail pattern1 with ratio1.04393. These graph results remain inside the independent FP64 envelope, but that does not excuse failure of the separate unchanged native-comparison bound. No tolerance is relaxed, and rejected profiles must not be integrated or timed as valid candidates.

Every admitted profile uses two ABBA rounds with eight samples per position, after common warmup. Timings include encoding/execution/synchronization, not weight preparation, graph creation or initial graph execution. First-graph time is recorded separately and spans7.22–109.81ms; no native GPU span substitutes for graph GPU time. Static constant copies and FP32 storage can increase memory/init costs; this harness does not measure peak RSS or isolate compile time from the first graph call. No full-model gain or fresh vLLM ratio follows from this private study.

Archive SHA256: `f4114845d10c54cba9d6759a2649216055e5c3a3c8a4a23bf8e5ee732abfc11e`.

Next investigate per-projection error locations before selecting a fallback policy or integration; repeat any selected policy without changing numerical bounds. Production prefill remains unchanged. The full2x decode / half-TTFT goal remains unmet.

```text
python3 launch_combined_graph_prefill.py ROOT ROOT/logs/combined-graph-prefill-20261015 component
python3 archive_combined_graph_prefill.py ROOT ARCHIVE
python -X utf8 test2026/metal/verify_combined_graph_prefill.py RESULTS RESULTS/verification.json
```
