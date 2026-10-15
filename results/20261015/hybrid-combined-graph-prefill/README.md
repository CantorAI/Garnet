# Hybrid native-key / combined graph prefill: numerically valid, mixed performance

PID35785 exited0 in74.0969s after exact confirmation PID34285. Independent audit verifies34 files,48 unchanged original numerical checks,12 first calls and768 timed submissions. All original matrices remain; special text/coder keys run on the native route and match original control bits. No tolerance or precision changes.

| Model / batch | Policy1 wall fraction | Policy2 wall fraction |
| --- | ---: | ---: |
| Qwen3-0.6B / 1 | 1.2317 | 1.3882 |
| Qwen3-0.6B / 2 | 1.1146 | 1.1064 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | 1.1687 | 1.2977 |
| Qwen2.5-Coder-0.5B-Instruct / 2 | 1.1178 | 1.1702 |
| Qwen3.5-0.8B / 1 | 0.7146 | 0.7522 |
| Qwen3.5-0.8B / 2 | 0.8645 | 0.8892 |

Fractions are candidate/control; above1 is slower. Text/coder regress10.6–38.8%. VL improves11.1–28.5%. These are independent early projection sets, including VL linear/full-attention fixture types, not complete transformer layers or model TTFT. Do not infer a new vLLM ratio.

Candidate clocks include the native key submission and its CPU wait before the synchronous graph call. This is a plausible source of extra latency, not a separately measured causal attribution. Native control uses one submission for the full set. Policy1 supplies FP32 placeholders; policy2 original FP32 constants. First call, uploads and graph preparation remain separate. Added storage/init costs are unmeasured by warm timings.

No broad native integration is justified. Next test asynchronous key submission using existing Flush, followed by graph execution and a final native wait. Both workloads must remain inside candidate wall time; hardware overlap is not assumed. All original bounds, workloads and key bit checks remain required.

Archive SHA256: `59cdb3fcd7155ad8c08197ce2a286da853527ca11d4c20719c0e227509fb1477`.

```text
python3 launch_hybrid_combined_graph_prefill.py ROOT ROOT/logs/hybrid-combined-graph-prefill-20261015 component
python3 archive_hybrid_combined_graph_prefill.py ROOT ARCHIVE
python -X utf8 test2026/metal/verify_hybrid_combined_graph_prefill.py RESULTS RESULTS/verification.json
```
