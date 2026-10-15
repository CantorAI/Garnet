# Core-only graph compilation: original key failures remain

PID50246 exited0 in72.9034s. Independent audit verifies34 files,48 numerical records,12 first calls and512 timed submissions. Four profiles are rejected and have no timed windows: text/coder B2, both weight policies. Their original native-comparison failures remain; selecting core compilation does not resolve the key-route problem. No tolerance is relaxed and no broad integration is justified.

The study requests MPSGraphOptimizationLevel0 through an execution compilation descriptor, with waitForCompilationCompletion and waitUntilCompleted enabled. It uses runAsyncWithMTLCommandQueue with waitUntilCompleted=YES, so timed calls are synchronous. All original matrices, FP32 tensor values and bounds remain, including special keys; there is no native-key fallback. Candidate/API configuration changes are recorded together, not isolated as a causal compiler-pass claim.

| Model / batch | Policy | Candidate/control wall fraction |
| --- | ---: | ---: |
| Qwen3-0.6B / 1 | 1 | 0.8273 |
| Qwen3-0.6B / 1 | 2 | 0.8464 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | 1 | 0.9017 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | 2 | 0.9173 |
| Qwen3.5-0.8B / 1 | 1 | 0.7556 |
| Qwen3.5-0.8B / 1 | 2 | 0.7835 |
| Qwen3.5-0.8B / 2 | 1 | 0.9265 |
| Qwen3.5-0.8B / 2 | 2 | 0.9241 |

First graph calls/setup are separate. These admitted-profile component timings do not establish full-model or vLLM gains. Runtime tensor dtypes and original numerical envelopes are checked; internal hardware placement and per-kernel arithmetic are not inferred from a framework option. Do not repeat unchanged or excuse the failed profiles.

Archive SHA256: `1067101d4a71d6ec5063b1b28ad74f56e7bbb015d2a4a0dd7c82e21c37f98798`.

```text
python3 launch_core_graph_prefill.py ROOT ROOT/logs/core-graph-prefill-20261015 component
python3 archive_core_graph_prefill.py ROOT ARCHIVE
python -X utf8 test2026/metal/verify_core_graph_prefill.py RESULTS RESULTS/verification.json
```
