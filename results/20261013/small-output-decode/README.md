# Bounded small-output decode: modest repeated gain

`GARNET_METAL_SMALL_OUTPUT_DECODE=ON` uses two outputs per SIMD group and
64 threads per work group for one/two-row BF16 projections with at most
256 outputs. Larger projections keep the existing four-output / 128-thread
path. Input grouping, per-output FP32 accumulation and reduction order,
original weight bits and shared batch-2 weight reads are preserved. The
option defaults OFF. Prior component evidence established bit-identical
work-distribution variants; these are actual full-model measurements.

All eleven kernel/numerical/serving/head/vision checks pass. Six completed
checks from the first validation attempt are retained with exact command
and log-hash checks; the remaining five pass after the service installer
that interrupted the original preflight exits. No benchmark ran from the
failed validation dependency. Neither the guard nor external processes
were changed. Guard checks precede every new numerical test.

Two native ABBA runs preserve all 144 complete requests across text, coder
and VL, including prompt IDs, finished flags and identical image bytes.
Each policy/profile uses two fresh workers with three requests each; four
non-first requests form each warm mean. Dense uses chunk 32; VL uses the
retained chunk recurrence and its previous 64/32 profiles. Caps, stop rules,
precision and learned values remain unchanged, with sequential runtimes.

| Model | Batch | Initial decode ratio | Confirmation decode ratio | Confirmation TTFT fraction |
| --- | ---: | ---: | ---: | ---: |
| Text | 1 | 0.994 | 0.999 | 1.008 |
| Text | 2 | 1.002 | 0.996 | 0.989 |
| Coder | 1 | 1.012 | 1.011 | 0.994 |
| Coder | 2 | 1.016 | 1.014 | 1.000 |
| VL | 1 | 1.016 | 1.013 | 1.004 |
| VL | 2 | 1.004 | 1.010 | 0.993 |

The repeated coder/VL decode benefit is small, roughly 1%. Text does not
improve and no material TTFT improvement is claimed. This is a native
comparison, with no new vLLM or 2x/half-TTFT achievement claim.

Initialization is recorded separately and is worse for the first text
profile in both runs: 2.22/2.52 seconds versus 1.66/1.65 seconds. Confirmation
coder batch-1 initialization is 1.79 versus 1.40 seconds. Other profiles
are near their controls. These observations are not assigned an unverified
cause; caches were not flushed. First-use TTFT, RSS and phase counters are
preserved in the raw records. The option remains experimental/default OFF.

The retained control library is unchanged at
`e727c3c81e7af9f252bbd090ea506e6db6c2ffee021f2edf6e167e380d9ea5e8`.
Candidate library SHA256 is
`56bc06c693f3776c994d3f4078dd2e25093b575f93f6f09c4e2087a4917ee8d4`.
`completed/` includes the actual source, commands, terminal statuses,
validation reuse records, model outputs, process/resource samples and
interrupted dependency logs. Tensor/RGB data remain private with hashes
in `evidence-proof.json`. Archive SHA256:
`7e984412c2a3cf04ba1e37bdedee5f3db7cff16f888ebb61315f4ee89a47bee2`.

The full three-model 2x decode and half-TTFT goal remains active and unmet.
These measured short prompts/batches do not establish every possible workload.
