# Screened greedy head versus fresh vLLM-Metal

The full goal remains unmet. Actual PID 20154 exited 0 in 539.3471 s. An
independent audit checks 507 archived files, three actual GPU dtype audits,
raw worker logs, 72 complete matching outputs, screening counters, fresh
images, source/checkpoint identities and recomputed metrics.

Four profiles meet 2x decode in this cohort: all B2 profiles and VL B1. VL B1
is only 2.011x, so this is not a repeated threshold claim. Text/coder B1 remain
below 2x and no profile reaches half warm TTFT. A fresh whole-cohort comparator
repeat is still required before claiming a robust target win.

| Model | Batch | Garnet / vLLM TPS per row | Decode ratio | Garnet / vLLM TTFT ms | TTFT fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 64.5263 / 38.4982 | 1.6761 | 46.0189 / 69.5649 | 0.6615 |
| Qwen3-0.6B | 2 | 62.7674 / 27.7558 | 2.2614 | 80.2130 / 92.1822 | 0.8702 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 79.0111 / 46.8702 | 1.6857 | 31.8040 / 57.0936 | 0.5571 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 76.1789 / 28.3076 | 2.6911 | 49.0812 / 84.4986 | 0.5809 |
| Qwen3.5-0.8B VL | 1 | 52.8264 / 26.2671 | 2.0111 | 183.2694 / 256.3496 | 0.7149 |
| Qwen3.5-0.8B VL | 2 | 48.6438 / 17.0983 | 2.8449 | 342.6450 / 428.8551 | 0.7990 |

Four warm requests per runtime/profile come from two fresh sequential workers
in Garnet/vLLM/vLLM/Garnet order. Each worker's first request is excluded.
B2 aggregate decode throughput is twice the per-row rate. Decode excludes the
first prediction and includes terminal predictions; TTFT includes preparation
and prefill. Both runtimes use the same combined threaded image decode and
exact-coordinate striped resize, with fresh image encoding for every request.
The image/prompt/output limits, checkpoint values, stop semantics, physical
prefill profiles and complete normalized generated tokens remain frozen.
Prefix caching and speculation are disabled. Resource guards sample every
0.5 s; no competing inference runtime is intentionally run simultaneously.

This is an additional matched-FP32 study, not a stock BF16 comparison. Actual
audits confirm MLX GPU execution with FP32 parameters, activations and KV.
Native BF16/lossless scaled-half storage and comparator losslessly expanded
FP32 storage retain the same learned values but have different bandwidth
traffic. A derived eight-bit index only screens greedy-head candidates;
retained scores use original BF16 weights and FP32 refinement. Original
complete-logit graph execution remains available. The optional backend is
OFF by default. Its method, numerical limits and two preceding native trials
are recorded in [the native study](../screened-head-inference/README.md).

## Initialization and first request after initialization

| Model / batch | Garnet / vLLM init s | Garnet / vLLM first-request TTFT ms |
| --- | ---: | ---: |
| Text / 1 | 2.1680 / 16.7950 | 157.0820 / 67.2775 |
| Text / 2 | 2.1844 / 16.2852 | 177.4360 / 106.5049 |
| Coder / 1 | 1.8120 / 15.8830 | 126.9237 / 60.8366 |
| Coder / 2 | 1.8085 / 15.8472 | 146.8716 / 97.6449 |
| VL / 1 | 4.8310 / 33.4182 | 308.4134 / 260.6347 |
| VL / 2 | 4.8809 / 33.1259 | 525.3023 / 452.1150 |

Native initialization is faster, while its first-request TTFT after init is
slower in all six profiles. These are separate observations. OS/driver caches
were not flushed, and their sum is not claimed as an observed hardware-cold
end-to-end latency.

## Remaining work and preserved evidence

Text/coder B1 still require about 19.3% / 18.6% additional decode speed relative
to this cohort. All six warm TTFTs need another 10.2–42.5% reduction. Exact
thresholds, aggregate rates, init and first-use observations are in
`remaining-budgets.json`; native phase intervals must not be summed because
GPU and wait measurements overlap.

Candidate dylib SHA-256:
`2b44d93ab1e1980845be89ea59cbaedbebcb66ad385ba24df3d52af2b6bacc23`.
Archive SHA-256:
`eda639c520e9aea250d1169d327fb2a4bd3bc8292348cd3073b8ce883629dc74`.

Hardware is the same Apple M4 Mac mini, 10 CPU cores and 16 GiB RAM, running
macOS 15.6.1 arm64. Isolated comparator versions: vLLM 0.30.0+cpu,
vLLM-Metal 0.30.0, MLX 0.32.1, mlx-lm 0.32.0, mlx-vlm 0.6.17, Torch 2.13.0,
Transformers 5.19.0, NumPy 2.3.5 and Pillow 12.3.0. The actual dtype audit
establishes Metal GPU execution independently of the vLLM package suffix.

The next candidate combines screening, ordered prefill streaming and partial
cache reset without changing source weights or agreed workloads. Validation
PID 29706 subsequently passed all 26 stages. Native jobs 32026 and 56025 also
passed, with independent audits of 144 complete outputs. Latency improvements
repeat with a text decode cost; the measurements are recorded in
[the combined native study](../screened-prefill-reset/README.md). They do not
replace this cohort or establish new vLLM target ratios.
The selected build, original Mac checkout, GPT-OSS and Pi worktrees remain
preserved. No combined-candidate speed claim is made before actual inference.

```text
python3 launch_screened_head_vllm.py ROOT ROOT/logs/screened-head-vllm-20261015
python3 archive_screened_head_vllm.py ROOT ARCHIVE
python -X utf8 verify_screened_head_vllm.py RESULTS VERIFICATION_JSON
python -X utf8 summarize_screened_head_vllm.py RESULTS
python -X utf8 summarize_screened_head_phases.py RESULTS
python3 launch_screened_prefill_reset_checks.py ROOT ROOT/logs/screened-prefill-reset-checks-20261015 checks
python3 queue_screened_prefill_reset_profiles.py ROOT ROOT/logs/screened-prefill-reset-native-20261015
```
