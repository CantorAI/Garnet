# Adjacent-dimension vision vectorization: local preparation only

Status: generated locally; Python runner/auditor AST passes. Metal/Objective-C++
is uncompiled and unexecuted. Files have not been copied to the Mac. No numerical
compatibility, latency, throughput, memory or model gain is established.
The live NDArray validation/comparison chain retains exclusive Mac use.

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
No launcher is provided yet: finish/archive the current NDArray chain before
choosing the next exclusive GPU experiment. Do not run another inference/probe
while a confirmed live validation/native/comparator job remains active.
