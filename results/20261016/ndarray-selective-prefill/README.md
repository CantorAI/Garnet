# Selective F32 NDArray: predeclared trained-component screen

Status: two actual Mac component cohorts complete, archived and independently
audited on Mac and Windows. Both pass the predeclared selective screen.
An isolated native validation candidate is prepared; there is no complete-model
speed result yet. Full goal remains unmet; retained serving library/defaults unchanged.

The [full-checkpoint CPU audit](../ndarray-weight-admission/README.md) identifies
one unsupported exact-half body matrix in Qwen3.5-0.8B: first-layer linear
attention QKV, 6144 by 1024. The other 185 VL matrices retain their common or
row-wise lossless half path. Text/coder have no unsupported body matrices.
Projected additional FP32 weight payload is 24 MiB, not measured RSS. The
[broad F32 native experiment](../../20261015/ndarray-native/README.md) added
1.854 GiB of logical weight storage and was rejected after complete inference.

Policies: 1 expands all eligible non-special body matrices to F32 NDArray;
2 expands only matrices lacking both common and row-wise exact half;
3 retains the original native path. Both new products use two-source FP32
unit alpha, beta zero, virtual transpose, stable input/output views, strong
owners and a 256-product cache. New views after the cap keep the original
fallback. The buffer contents pointer is read once per expansion; its
first-use effect is unmeasured. Policies 1/2 share that change, so this test
cannot isolate pointer-hoisting causality. Exact BF16 learned values and
FP32 activations/output/accumulation are preserved.

Use all original 29 trained first-layer fixtures, four activation patterns,
all original physical prefill row shapes: text 24/44, coder 26/52, VL 80/160.
Keep the original FP64 forward-error and original-control comparison bounds,
padding/finite checks, 2 ABBA rounds and 8 measured samples per window. Failed
numerical groups receive no timings. Zero-operation policies must be bitwise
identical to control; their timing changes cannot establish algorithmic gain.

Before measurements: consider selective policy 2 for isolated native
integration only if **both GPU and wall fractions <=0.95** for VL B1 and B2
in two complete fresh cohorts, with each corresponding null policy within
0.95 through 1.05, and all selective numerical checks passing. Full F32 is a
component reference, already rejected for broad native promotion. No favorable
repeat search if the initial selective screen fails. Native integration, if
admitted, requires original fresh full validation and complete generated
outputs, first/warm latency, decode, memory and independent archive audits.
Only fresh repeated matched vLLM cohorts can establish target progress.

```text
python3 launch_ndarray_selective_prefill.py ROOT OUTPUT component
python3 archive_ndarray_selective_prefill.py ROOT ARCHIVE TRIAL
```

These are projection components, not complete-model inference or a fresh
vLLM comparison. No competing inference runtimes may execute concurrently.

## Actual component results

All 24 selective and all 24 null numerical records pass per cohort. The full
F32 reference passes 21 of 24; pattern 1 fails the original-control comparison
for text B1 and coder B1/B2 in both runs, despite passing the independent FP64
bound. Those three groups receive no timings. No tolerance was relaxed.
Each cohort has 72 records and 960 admitted timed executions. Actual dispatches:
selective exactly one per VL projection set, zero per text/coder set; full
reference 15 VL and six non-special text/coder; null zero. Every zero-operation
record is bitwise identical to control. Zero-operation timing fluctuations
cannot establish an algorithmic gain.

| Cohort | VL batch | GPU ms control / selective | Wall ms control / selective | Selective GPU / control | Selective wall / control | Null GPU / control | Null wall / control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial | 1 | 4.413000 / 3.970904 | 5.246760 / 4.842983 | 0.899820 | 0.923043 | 1.032385 | 1.029010 |
| Initial | 2 | 7.551362 / 6.255000 | 8.459178 / 7.185711 | 0.828327 | 0.849457 | 1.015989 | 1.022998 |
| Repeat | 1 | 4.532659 / 3.698501 | 5.428117 / 4.629496 | 0.815967 | 0.852873 | 1.024950 | 1.015859 |
| Repeat | 2 | 7.320596 / 6.505921 | 8.173341 / 7.415698 | 0.888715 | 0.907303 | 1.036874 | 1.037128 |

These timings cover the original first-layer projection set in one submission,
not one isolated QKV operation and not a full decoder. Both cohorts have
identical captured sources/assets/generated context/header/harness/macros.
Initial PID19306 exits0 in77.300869250s; repeat PID20686 exits0 in77.263889209s.
Each archive has62 verified files. Full raw outputs, all passing/reference/null
timings and failed checks are in [initial evidence](initial/evidence/manifest.json)
and [repeat evidence](repeat/evidence/manifest.json). Independent audits:
[initial](initial/verification.json), [repeat](repeat/verification.json).

Archive SHA256 initial:
`f70a9eb265f99f768c0b23cbf033cadbffb758211febaaae1f5bc887638bf05f`.
Repeat:
`a4de900f6d79249ed4ef69c5ce6c5d51fc09fdcf50f7199ea1f570f4573a62b4`.

The [decision](decision-repeat.json) admits isolated native investigation,
not promotion. No new vLLM ratios or target passes. Native candidate details
and the next gates are recorded [separately](../ndarray-selective-native/README.md).
