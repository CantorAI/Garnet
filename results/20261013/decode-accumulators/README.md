# Independent decode accumulators: component candidate, not a serving gain

This experiment breaks a projection's FP32 accumulation dependency into two
or four independent per-lane sums, then combines them before the SIMD
reduction. BF16 learned values and FP32 inputs, accumulation and outputs
remain; summation order changes. The production source and selected native
libraries are unchanged. No quantization, prefix reuse or speculation is used.

An actual isolated M4 run tests four variants (two/four sums, 64/128 threads)
on real first-layer matrices and complete vocabulary heads for all three
checkpoints, batches 1/2 and irregular synthetic tails. All 272 candidate
checks pass an independent host FP64 dot-product reference. The fixed
forward-error bound is `gamma(n) * sum(abs(products)) + 1e-7`, where
`n = ceil(inner / 32) + 32` and `gamma(n) = n*float_epsilon / (1-n*float_epsilon)`.
The original control must pass that bound too. Output padding is preserved.
Worst error/bound ratio is 0.00997264. Archive audits verify finite learned
values and finite dot-product bounds for every asset. Candidate output bits
can differ from control; no existing full-layer tolerance has been changed.

Most body projections are slower or flat. The two-sum/64-thread variant has
a small text batch-1 vocabulary-head signal. A separate sequential repeat
therefore tests only that variant and the three complete heads at both
batches. All six repeat checks pass; worst error/bound ratio is 0.00343388.

| Actual vocabulary head | Batch | Initial GPU speedup | Repeat GPU speedup |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 1.073 | 1.086 |
| Qwen3-0.6B | 2 | 0.972 | 0.994 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 1.004 | 1.026 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 0.994 | 0.995 |
| Qwen3.5-0.8B | 1 | 1.052 | 1.019 |
| Qwen3.5-0.8B | 2 | 1.000 | 0.999 |

Each timing uses two ABBA rounds, three warm-ups and twelve samples per
position. These are warm isolated matrix timings with deterministic finite
sinusoidal inputs, not actual model hidden states or model throughput.
Text batch 1's head improves in both runs; the VL effect shrinks in the
repeat, coder remains marginal and batch 2 is flat. Small-output body
controls here use the general four-output kernel, whereas selected serving
uses its separate small-output kernel. These body results cannot justify
replacing that serving path.

Only a bounded text batch-1 head candidate merits a full-model investigation.
It has not been integrated or validated against real normalized hidden
states and complete generated outputs. Even an 8.6% head improvement would
be a smaller whole-model gain. No decode-TPS, TTFT or vLLM speedup is claimed.
The full three-model 2x decode / half-TTFT goal remains active and unmet.

## Actual commands and evidence

```text
python3 run_decode_accumulators_probe.py INVESTIGATION_ROOT OUTPUT_DIR
python3 run_decode_accumulators_head_probe.py INVESTIGATION_ROOT OUTPUT_DIR
```

Both durable jobs have authoritative exit code 0. The second waits for the
first's successful terminal status. Exact shader-generation, compiler and
GPU commands, source/checkpoint/matrix hashes, raw timing windows, device
limits, numerical checks and process snapshots are under `evidence/`.
The runner monitors every 0.5 seconds and terminates only its owned process
group on contention or deadline; unrelated jobs are not stopped. Sampling
cannot prove absence of arbitrarily short jobs between polls. No concurrent
inference was admitted. Existing WorldSense services remained untouched.

Raw matrix copies and compiled binaries are omitted from Git with individual
hashes and sizes retained in `evidence-proof.json`. All executed source
versions and generated shaders are retained. Archive SHA256:
`ee356339e9455bd2f3f26477f34b5b21b41547aa7acc65f3d72a61bb70797f86`.
