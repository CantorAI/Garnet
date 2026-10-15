# Captured-query greedy head screening

Two actual eight-bit screening trials preserve the sampled original GPU
logits/predictions and reduce head-component time across all three models at
B1/B2. The four-bit variant also passes sampled correctness but regresses
several timings, so retain the eight-bit candidate for further investigation.
An optional native integration is now under validation. No complete-generation
speedup or fresh vLLM comparison has been established for it. The full
optimization goal remains active and unmet.

## Method and precision

The original BF16 checkpoint remains the model. A derived integer index
approximates each vocabulary row using an exact power-of-two scale. CPU FP64
calculations form inflated, upward-stored residual/original/reconstructed L2
norms. Each request computes an exact 8192-row native seed and its actual
maximum, then FP32 query norms. Screening dots plus error allowances reject
rows below that seed threshold. Every retained row is recomputed from the
unchanged original BF16 bits with the original FP32 accumulation/reduction;
the native argmax runs over retained/seed values and finite private sentinels.

The public complete-logit API is unchanged. The component requires every raw
control logit to match the previously captured native GPU bits, every retained
logit to match those bits, every rejected actual logit to be strictly below
its actual seed threshold, and each selected token/nonfinite flag to match.
This establishes the sampled queries only. The inflated screening allowances
are not certified for all future input magnitudes, and no production safety
or full-model precision claim follows from these component checks.

Each model uses eight captured real native normalized states at B1, eight at
B2, and a B2 pair combining different actual states. Checkpoint/header/head
selection, original matrix bytes, captures and every input/reference digest
are checked. Heads follow runtime selection, including the coder tied embedding
and VL language embedding. No answer, prefix or image-encoder cache is used.

## Eight-bit initial and fresh-process confirmation

Numbers below average the paired per-query fractions for the eight ordinary
states per model/batch. Mixed-state correctness and timings remain in the raw
audit. Fractions below one are faster. These are complete head-component
intervals, including exact seed, bounds, screening/refinement and argmax;
normalization common to both policies is outside this component.

| Model / batch | GPU fraction initial / repeat | Wall fraction initial / repeat |
| --- | ---: | ---: |
| Text / 1 | 0.74775 / 0.76074 | 0.78266 / 0.79455 |
| Text / 2 | 0.74695 / 0.74779 | 0.78369 / 0.78493 |
| Coder / 1 | 0.73408 / 0.71047 | 0.76705 / 0.74801 |
| Coder / 2 | 0.76539 / 0.75085 | 0.80220 / 0.78923 |
| VL / 1 | 0.76687 / 0.77344 | 0.78759 / 0.79658 |
| VL / 2 | 0.76254 / 0.75894 | 0.78764 / 0.78487 |

The repeated GPU reduction is about 23–29%; wall reduction is about 20–25%.
These are head-only observations, not model decode TPS or TTFT. Each query
has at least 250 ms alternating warmup, two ABBA rounds, two per-window warmups
and eight raw timed submissions per window. Compilation, index preparation,
allocation, normalization and CPU validation/readback are outside timing.
No competing inference runtime or GPU probe intentionally runs simultaneously;
the existing 0.5 s guards and disk-space checks apply. Caches were not flushed.

The initial owned job 83447 exited 0 in 58.6883 s including preparation;
confirmation 84995 exited 0 in 40.1560 s, reusing only byte-verified immutable
weight/index/input assets. It still executes every GPU seed, screening,
refinement and argmax anew. Independent eight-bit audit covers 73 files,
102 query checks / 156 row-state checks and 6528 timed submissions. The earlier
queue 81590 exited 1 on the vision dependency and executed zero queries; that
failure is preserved separately from the successful trials.

Counted weight-byte fractions, including the index and seed/refinement original
weights, are about 0.527 for text/coder and 0.516 for VL. These are algorithmic
visit counts, excluding metadata, activations and cache/prefetch effects, not
measured DRAM bandwidth or total memory. The index adds one byte per original
head element; original weights remain allocated.

Initial archive SHA-256:
`99fa00cd6acb33b245fd325f423919d1596d9f0904823ec49a3b1cb206fac6fb`.
Two-trial eight-bit archive SHA-256:
`206ee8db14f84d8660f306a050994871029df5c82de6b0dbac1afc84c20f9a2b`.

## Four-bit trial: not selected

Owned PID 85879 terminated exit 0 in 54.0472 s. Its packed derived index uses
half a byte per original weight, but larger residual bounds retain many more
rows for original projection. Counted weight-byte fractions rise to 0.649–0.690.
All 51 queries / 78 row states pass the same complete-logit, every-rejection
and greedy checks; that compatibility does not make it a better optimization.

| Model / batch | GPU fraction | Wall fraction |
| --- | ---: | ---: |
| Text / 1 | 0.99032 | 1.01114 |
| Text / 2 | 1.02273 | 1.04624 |
| Coder / 1 | 0.92705 | 0.94578 |
| Coder / 2 | 0.96257 | 0.97669 |
| VL / 1 | 1.04333 | 1.05191 |
| VL / 2 | 1.07019 | 1.08451 |

The combined audit now verifies 112 archived files, 153 query checks / 234
row-state checks and 9792 timed submissions across the three successful trials.
Its scope separates index widths and preserves the unexecuted dependency job.
Combined archive SHA-256:
`617d07d759ee006b58ff00b3e43b7de3ba93cd7aff8c72a814bec0f2749bd67f`.

## Six-bit trial: not selected

Owned PID 87355 at `logs/screened-head-int6-20261015` terminated exit 0 after
63.9366 s, following all prior terminal trials and their independent audit. A
pre-timing GPU roundtrip reconstructed every derived index element exactly
against CPU rounding of the original weight/scale, covering packed-word
boundaries as well as the captured-query checks. All 51 queries pass again.

| Model / batch | GPU fraction | Wall fraction |
| --- | ---: | ---: |
| Text / 1 | 0.99977 | 1.03024 |
| Text / 2 | 1.01413 | 1.04459 |
| Coder / 1 | 0.99411 | 1.02326 |
| Coder / 2 | 1.01995 | 1.04732 |
| VL / 1 | 1.02168 | 1.04022 |
| VL / 2 | 1.03611 | 1.05730 |

Counted weight-byte fractions fall to 0.396–0.411, while wall time is 2.3–5.7%
higher than the original complete head. Reduced byte visits alone do not prove
a faster kernel. This six-bit layout is rejected for integration without
seeking a favorable repeat. The eight-bit route remains the repeated candidate.

The final audit verifies 151 archived files, four successful trials, 204 query
checks / 312 row-state checks and 13,056 timed submissions. Every complete
six-bit index reconstruction is recorded separately. All original/generated
sources and input/index/reference identities remain preserved by trial.
All-width archive SHA-256:
`6a2ce969d7ed032e0b28b74989285c50923bc82a852b7052ba7f4a0f878fe419`.

## Bounded strict-math eight-bit trial

Owned PID 89166 exited 0 after 40.9025 s. Screening now uses a separate
`fastMathEnabled=NO` Metal library; the original full/seed projection retains
its original math policy. Nonfinite inputs, absolute input values above 1e6,
and norms outside [1e-10, 1e10] force full original-row refinement. Metadata
must be finite and bounded. This limits screening admission; it does not
establish a universal numerical proof for all possible admitted queries.

All 51 captured queries pass retained-bit, every-rejection and greedy checks.
Another 33 cases check zero, tiny, large, NaN, infinity and mixed safe/unsafe
batch rows against the original full projection and its error flags.

| Model / batch | GPU fraction | Wall fraction |
| --- | ---: | ---: |
| Text / 1 | 0.74387 | 0.78045 |
| Text / 2 | 0.77047 | 0.81050 |
| Coder / 1 | 0.73468 | 0.77126 |
| Coder / 2 | 0.76899 | 0.80989 |
| VL / 1 | 0.77382 | 0.79698 |
| VL / 2 | 0.77270 | 0.79872 |

These are head-component intervals, not whole-model TPS or TTFT. Combined
component audit: 187 archived files, five successful trials, 255 query checks,
390 row states, 16,320 timed submissions and the 33 fallback checks.
Archive SHA-256:
`533e97172600b595df9f63f3ad02e6e6412d4f6ed944b2758f6dc5c6094261c0`.

## Optional native integration: validation in progress

`GARNET_METAL_SCREENED_HEAD` defaults OFF. Initialization builds the derived
index from the unchanged original head; unsupported shapes or metadata use
the existing head. Greedy B1/B2 session prediction uses the seed, screening,
original retained-row refinement and original argmax. Public full-logit graph
execution continues to compute all logits. The index increases resident
memory and initialization work; inference measurements must report these.

Owned validation PID 91274 exited 0 after 1219.3464 s. All 20 stages passed:
the original 19 model-validation stages plus all 51 trained-head queries
through the native context, including original full-logit bits, retained bits,
every rejected row and predictions. Queued PID 3415 verified that terminal
status and the pinned source manifest, then began trained session/recovery
checks before six-profile ABBA. It subsequently exited 0 with six recovery
cases and 72 complete matching requests. Initial whole-model decode gains
are 7.5–10.4%; fresh-process confirmation is running as PID 17037.
Those inference observations are recorded separately in
[the native study](../screened-head-inference/README.md).
Repeated native measurements and a fresh matched comparator study remain
necessary. The component timings alone do not establish serving performance.

```text
python3 launch_screened_head_after_rejection.py ROOT ROOT/logs/screened-head-after-vision-20261015 checks
python3 archive_screened_head.py ROOT ARCHIVE
python -X utf8 verify_screened_head.py RESULTS VERIFICATION_JSON
python3 launch_confirm_screened_head.py ROOT ROOT/logs/screened-head-confirm-20261015 checks
python3 queue_screened_head_int4.py ROOT ROOT/logs/screened-head-int4-20261015
python3 launch_screened_head_int6.py ROOT ROOT/logs/screened-head-int6-20261015 checks
```
