# Coalesced lossless activation MMA for prefill: rejected

The actual owned component job PID 83698 exited 0 in 67.4566 seconds. The
independent audit verified 45 archived files, 104 matrix/row configurations,
60,606 validated executions, and 9,984 timed executions. Numerical checks pass,
but the candidate is much slower than the selected global-alpha MPS matrix
operation. No production integration or full-model speedup is claimed. The
2x decode / half-TTFT goal remains unmet.

This tests a distinct workload from the rejected single-token coalesced MMA:
8, 16, 32, and 80 prefill rows, actual first decoder-layer matrices from each
checkpoint, and two irregular synthetic weight fixtures per model. Inputs are
synthetic FP32 sinusoids at four signed/scaled variants, with negative zeros.
These are diagnostic row counts, not serving batch sizes or end-to-end prompts.

The candidate splits each FP32 input into three exactly reconstructing scaled
half components, runs a shared-memory 8x32 matrix tile with half operands and an
explicit FP32 SIMD-matrix accumulator, and recombines into FP32. Learned BF16
weight values restore bit exactly from a common scaled-half representation.
The control uses the selected global-alpha MPS primitive: FP32 input, the same
exact half weight values, FP32 output, and MPS alpha equal to the inverse common
power-of-two scale. It does not use the older per-column rescale baseline.
The primitive comparison does not establish the serving graph's route choice
for every first-layer matrix, including special key-prefill paths.

All original admission, reconstruction, output-padding, comparison, and
independent FP64 envelope gates remain. The maximum candidate FP64 bound ratio
is 0.04504; the maximum control-comparison ratio is 0.52233. Wide-exponent inputs
are rejected in all three models. Two VL matrices fail global exact-half
admission and are explicitly excluded; they require the existing fallback.
No bounds were widened and no weight values were quantized.

The following sums include trained matrices with more than 256 output columns.
They are unweighted component pools, not layer, inference, or runtime speedups.
Fractions above 1 mean slower candidate time divided by control time.

| Model | Prefill rows | GPU time fraction | Wall time fraction |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 8 | 2.452 | 1.411 |
| Qwen3-0.6B | 16 | 6.548 | 2.128 |
| Qwen3-0.6B | 32 | 10.177 | 3.254 |
| Qwen3-0.6B | 80 | 12.020 | 4.832 |
| Qwen2.5-Coder-0.5B-Instruct | 8 | 2.408 | 1.387 |
| Qwen2.5-Coder-0.5B-Instruct | 16 | 6.796 | 2.393 |
| Qwen2.5-Coder-0.5B-Instruct | 32 | 9.901 | 3.519 |
| Qwen2.5-Coder-0.5B-Instruct | 80 | 10.449 | 4.657 |
| Qwen3.5-0.8B | 8 | 2.541 | 1.504 |
| Qwen3.5-0.8B | 16 | 6.468 | 2.513 |
| Qwen3.5-0.8B | 32 | 11.057 | 4.031 |
| Qwen3.5-0.8B | 80 | 12.155 | 5.198 |

Each configuration has two ABBA rounds and 12 samples per window, following
at least 250 ms of checked warmup. Timed runs include split, product,
recombination, encoding, submission, and wait; the reported GPU interval is
separate. The 0.5-second inference guard remains active and only the owned
process group may be stopped. The previous retained-source inference repeat
was terminal before this probe launched. Protected GPT-OSS and Pi checkouts
are unchanged, and the selected serving library is unchanged.

Archive SHA-256:
`f0c0beaa413f3c01c15ae5d2731d91b8f0982a75463179018d55b8e46111abe8`.
Exact executed source, commands, versions, checkpoint/tensor identities,
monitors, and raw timing records are in `evidence/`. Checkpoint arrays and the
compiled binary are omitted; their hashes are retained. `verification.json`
recomputes every timing pool and count from the archived records.

```text
python3 launch_coalesced_activation_prefill.py ROOT ROOT/logs/coalesced-activation-prefill-20261015
python3 archive_coalesced_activation_prefill.py ROOT ARCHIVE
python -X utf8 verify_coalesced_activation_prefill.py RESULTS VERIFICATION_JSON
```
