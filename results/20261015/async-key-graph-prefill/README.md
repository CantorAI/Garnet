# Asynchronous native keys plus combined graphs: repeated component signal

Initial PID47481 exited0 in72.8583s; confirmation PID48881 exited0 in72.9631s. Independent audit verifies67 combined files,96 original numerical checks,24 first calls and1536 timed submissions. The identity join checks identical source/config, versions, hardware, assets and generated code. Every original key output is bit-exact; actual native flush counts are1 for text/coder and0 for VL. Original numerical bounds are unchanged.

| Model / batch | Placeholder initial / repeat wall fraction | Constant initial / repeat wall fraction |
| --- | ---: | ---: |
| Qwen3-0.6B / 1 | 0.8394 / 0.9875 | 0.8435 / 0.9829 |
| Qwen3-0.6B / 2 | 0.9217 / 0.8555 | 0.8580 / 0.8547 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | 0.8908 / 0.9168 | 0.9693 / 1.0148 |
| Qwen2.5-Coder-0.5B-Instruct / 2 | 0.8900 / 0.8881 | 0.9045 / 0.8859 |
| Qwen3.5-0.8B / 1 | 0.7809 / 0.6972 | 0.7558 / 0.7636 |
| Qwen3.5-0.8B / 2 | 0.8673 / 0.8930 | 0.9368 / 0.9422 |

Fractions are candidate/control. Placeholder policy1 is faster in all six groups in both trials; text B1 shrinks to1.25% on repeat, so do not claim a stable16% improvement there or statistical significance. Constant policy2 coder B1 regresses1.48% on repeat. A fixed placeholder policy is the candidate for further dependency-aware work; no per-profile best-sample dispatch policy is promoted.

Candidate Begin/Encode/Flush submits independent native keys without an intervening CPU wait, then runs the combined graph and finally SubmitAndWait. All work is included in candidate wall time. Hardware overlap is not measured or assumed. All original matrices, input patterns, FP32 values and shapes remain; keys are not omitted or cached. Setup, first calls and larger weight-storage costs remain separate from warm timings.

These are independent early projection fixtures, including VL linear/full-attention types. They are not a complete dependency layer: actual q/k/v, attention/output and MLP stages have data dependencies. Do not port the entire independent fixture as one model operation or multiply component ratios into predicted TTFT. Native integration first requires dependent-subgraph tests, command-buffer/ownership/abort validation and original trained-model gates, followed by full-model repeats and a fresh matched comparator.

Initial archive SHA256: `a3c971345d64127a0869885cd6fd229f634a43f1331ad408540c3f2cbbecccad`.
Combined archive SHA256: `c9268d438ac46e0d588655f290198762b71869bca7bda433d2afeff9a1aee565`.
No full-model TTFT/decode or vLLM improvement is established by this component; the overall goal remains unmet.

```text
python3 launch_async_key_graph_prefill.py ROOT ROOT/logs/async-key-graph-prefill-20261015 component
python3 archive_async_key_graph_prefill.py ROOT ARCHIVE
python -X utf8 test2026/metal/verify_async_key_graph_prefill.py RESULTS RESULTS/verification.json
python3 launch_confirm_async_key_graph_prefill.py ROOT ROOT/logs/async-key-graph-prefill-confirm-20261015 checks
python3 archive_async_key_graph_prefill.py ROOT COMBINED_ARCHIVE async-key-graph-prefill-20261015 async-key-graph-prefill-confirm-20261015
python -X utf8 test2026/metal/verify_async_key_graph_prefill.py RESULTS/combined RESULTS/combined-verification.json
python -X utf8 test2026/metal/summarize_component_repeats.py async-key RESULTS
```
