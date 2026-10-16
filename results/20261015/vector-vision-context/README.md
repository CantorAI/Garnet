# Adjacent-dimension vision vectorization: bit-exact, initial calibration failed

Status: actual Mac compilation/execution complete and archived, with Mac/Windows
audits passing all original numerical gates. Initial same-pipeline B1 null
calibration fails the predeclared limit. No width qualifies, no unchanged
confirmation or native integration is admitted, and no model gain is established.
The earlier NDArray chain is terminal, archived/audited and its candidate rejected.

Hypothesis: sharing each probability across two/four adjacent value dimensions
may reduce load/scheduling work while retaining each output's original FMA order.
Use float2/float4 value loads and separate vector components; no reduction across
keys/dimensions and no partial-sum regrouping. Checkpoint, vision patch dimensions,
softmax, keys, probabilities and FP32 per-output key order remain unchanged.
Whether the compiler/hardware preserves exact outputs and improves speed must
be tested; vector source syntax or fewer logical loads does not prove either.

This differs from the [rejected parallel contraction](../parallel-vision-context/README.md),
which changed the key reduction order. It also differs from the prior
[scalar prefetch variants](../serial-vision-context/README.md), which only changed
prefetch width/threadgroup. The control comes directly from the selected native
`serial_vision_prefetch_shader.h` scalar eight-key kernel, compiled with fast
math disabled. Candidate components retain all eight FMAs in order and the
original tail loop. Vector widths require aligned hidden/head dimensions and
input/output row strides; only the original aligned component shapes are used.

Three policies: same-scalar/same-pipeline null, float2, float4;128 threads per
group for all. Null and control dispatch identical GPU work. Softmax stays
separate and unchanged in both arms. Dispatch count divides by vector width
only for vector candidates. Padding and typed vector alignment are checked.

Preserve all24 original shapes (B1/B2,4/24/216 keys,2/12 heads,8/64 dimensions),
four changed-input generations including large scores, full output/padding bit
identity and unchanged FP64 `5e-6+5e-5*abs(reference)` bound. Failed admission
gets no timings. Passing groups retain250ms alternating warmup, two ABBA rounds,
20 raw GPU/wall samples per position and all original guard/source identities.
These synthetic inputs are not the trained vision model or full inference.

Before any GPU timing, declare the screen: repeat the unchanged complete
three-policy cohort only if a vector policy passes every original numerical
gate and, at both serving-shaped batches (216 keys,12 heads,64 dimensions),
GPU and wall means are <=0.95 of control with own null GPU/wall within5%.
Confirmation must satisfy the same conditions. If both widths qualify, choose
the lower geometric-mean serving wall fraction across the two cohorts, with
float2 on an exact tie. This is a pragmatic screen, not a confidence interval.
No favorable-repeat search. Whole trained-grid bit equality, native validation,
full model and fresh matched vLLM evidence remain required for later promotion.

Prepared generators, runner, frozen original-gate auditor and terminal archiver
are under `test2026/metal/*vector_vision*`. The helper
`freeze_vector_vision_auditor.py` preserves the original assertions while adding
selected-control regeneration, current-host identity and exact null output.
The launcher requires terminal, archived and independently audited complete
native cohorts from the current NDArray chain before accepting this probe.
It also verifies the retained c186 library and the original quiet-process guard.
The analyzer enforces the above predeclared screen and identical sources,
variants/environment across confirmation. The new terminal archiver captures
only this synthetic experiment; it does not assume model fixture assets exist.
The launcher enforces those prerequisites before each dispatch. Do not run another inference/probe
while a confirmed live validation/native/comparator job remains active.

```text
python3 launch_vector_vision_context.py ROOT OUTPUT component
python3 archive_vector_vision_component.py ROOT ARCHIVE TRIAL
python -X utf8 test2026/metal/verify_vector_vision_context.py EXTRACTED_RESULTS AUDIT_JSON
python -X utf8 test2026/metal/analyze_vector_vision_context.py RESULT_ROOT initial
```

## Actual initial result: numerical pass, no selection

PID53012 exits 0 in 38.676133083s. Archive33 files, SHA256
`e2ec3e3531b6503e68c4d30b6be1eabe1df18e29ca1959346caf39c20dd4cbc9`.
Both-host regeneration/raw audits verify72 shape/variant records, zero numerical
failures and11,520 timed submissions. Every vector/control output and padding
bit matches across four changed-input generations; original FP64 bounds remain.
[Audit](initial/verification.json), [manifest](initial/evidence/manifest.json),
[decision](decision-initial.json).

| Vector width | Batch | GPU fraction | Wall fraction | Own-null GPU fraction | Own-null wall fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 0.678234 | 0.764214 | 1.118033 | 1.117906 |
| 2 | 2 | 0.719771 | 0.757169 | 0.986761 | 0.999652 |
| 4 | 1 | 0.669169 | 0.780027 | 1.118033 | 1.117906 |
| 4 | 2 | 0.660777 | 0.700194 | 0.986761 | 0.999652 |

Serving shape is216 keys/12 heads/64 dimensions. Both widths show apparent
component improvements, but B1 identical-kernel calibration drifts about12%,
outside5%. Therefore neither width qualifies under the original prospective
rule. Do not drop the bad null, rescale candidate timings by it, or present these
ratios as established model gains. B2 alone cannot satisfy the both-batch rule.

The [complete null-window analysis](null-drift-analysis.json) retains every
window: round0 GPU means1.578144/1.599190/1.416550/0.954771ms in ABBA order;
round1 0.798069/0.786873/0.800398/0.786079ms. The compiled source binds the same
scalar pipeline object and geometry in both null arms. This is nonstationary
timing; frequency/thermal causality was not instrumented and remains unknown.
No samples are removed or later-round-only gain substituted.

A [separate prospective calibration-only protocol](../../20261016/vector-vision-calibrated/README.md) investigates longer fixed
conditioning and shorter windows. It must pass its own independent null gates
before collecting new candidate gains; this failed initial cohort stays failed.
The full three-model 2x/half-TTFT goal remains unmet.
