# Paired query and key normalization and RoPE

Current status: all original-bit correctness prerequisites pass; the first
complete timing cohort has unstable own-nulls in 16 of 24 endpoint profiles.
Do not admit native integration, an unchanged favorable repeat, or a model
speed claim from this cohort. Preserve all results before a separate timing
calibration with longer GPU work in each command batch.

Combine Q and K normalization/RoPE into one dispatch with independent output
buffers. Retain the original per-head arithmetic body verbatim: same 32-lane
SIMD sum, rsqrt, FP32 operations and precise sine/cosine calls. Threadgroups
select query or key pointers and their original shape parameters. No weight
packing, joined output, scatter, input/prefix cache or speculative result is
introduced. The component context is generated privately from accepted d52;
serving code and native libraries remain unchanged.

## Executed numerical prerequisites

Primitive PID 46016 exits 0 in 10.590449334 s. All 60 changed-input cases and
3,710,720 original FP32 output-bit comparisons pass, including normalized and
unnormalized modes, offsets 0/1 and +/-0.25, partial rotary dimensions,
nonstandard head sizes, pending/foreign writer rejection, aliases,
abort/recovery and unsupported rotary fallback. The exact original arithmetic
body is independently regenerated. [Primitive audit](primitive/verification.json).

Learned PID 46457 exits 0 in 119.748470250 s. Actual current native Q/K
projections generate changed structured hidden-input fixtures; VL gated query
columns are selected as in the graph. Use checkpoint norm weights, including
all 68 learned Q/K norm matrices and coder's 48 all-ones placeholders.
All 58 separate full-attention layers are covered, including the two VL
triples excluded by the earlier half-union experiments. Four decode/prefill
row profiles per layer produce 232 profiles and 928 changed-input cases.
All 47,321,088 original FP32 output-bit comparisons pass. The timing stage
additionally validates 7,385,088 output bits at its 24 endpoint profiles.
[Complete learned source, numerical and timing audit](initial/verification.json).

Host fixtures are reused only for kernel-level alternating measurements.
This component experiment is not a complete model request benchmark, and it
cannot prove model accuracy on arbitrary inputs or supply a fresh vLLM ratio.
Full native graph/session/output validation remains mandatory.

## Complete first timing cohort

Fix first/last separate full-attention layers before execution: text 0/27,
coder 0/23, VL 3/23. Include decode rows 1/2 and both serving prefill row counts.
Keep 2 s alternating warmup, own-null then comparison, 10 ABBA rounds, 3 warm
calls and 5 measured samples per window. Retain all 9,600 samples in 1,920
windows. Every window checks complete output bits. Each sample submits one
normalization pair and waits for completion.

GPU kernel time, CPU encoding time and single-submit wall time measure
different costs. Report them separately; own-nulls exclude unstable profiles
and never normalize gains. Only 8 of 24 endpoint own-nulls stay within 5% for
both GPU and wall time. Short GPU intervals and observed drift make this
cohort insufficient for a stable across-profile gain claim.
[All means and fractions](initial/analysis.json).

Fractions below are candidate/control; smaller is better. Stable refers only
to the own-null test, not admission of a speedup.

| Model | Layer | Rows | GPU fraction | Wall fraction | Null GPU fraction | Null wall fraction | Own-null stable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-Coder-0.5B-Instruct | 0 | 1 | 0.801631 | 0.996582 | 1.304653 | 1.048570 | No |
| Qwen2.5-Coder-0.5B-Instruct | 0 | 2 | 0.588070 | 0.940138 | 1.299252 | 1.064812 | No |
| Qwen2.5-Coder-0.5B-Instruct | 0 | 26 | 0.833265 | 0.940768 | 0.901287 | 0.967863 | No |
| Qwen2.5-Coder-0.5B-Instruct | 0 | 52 | 0.933195 | 0.989962 | 0.984880 | 0.988945 | Yes |
| Qwen2.5-Coder-0.5B-Instruct | 23 | 1 | 0.619267 | 0.908474 | 0.948554 | 0.978314 | No |
| Qwen2.5-Coder-0.5B-Instruct | 23 | 2 | 0.588985 | 0.973435 | 0.765247 | 0.966664 | No |
| Qwen2.5-Coder-0.5B-Instruct | 23 | 26 | 0.780595 | 0.898862 | 0.989777 | 0.981374 | Yes |
| Qwen2.5-Coder-0.5B-Instruct | 23 | 52 | 1.001667 | 1.004995 | 1.071193 | 0.993546 | No |
| Qwen3-0.6B | 0 | 1 | 0.619575 | 0.872089 | 0.965952 | 1.015943 | Yes |
| Qwen3-0.6B | 0 | 2 | 0.629423 | 0.879730 | 1.189063 | 1.017450 | No |
| Qwen3-0.6B | 0 | 24 | 1.025093 | 1.007569 | 0.961938 | 0.969348 | Yes |
| Qwen3-0.6B | 0 | 44 | 0.963006 | 1.010317 | 0.956176 | 0.980118 | Yes |
| Qwen3-0.6B | 27 | 1 | 0.808200 | 0.877527 | 0.951579 | 0.923890 | No |
| Qwen3-0.6B | 27 | 2 | 0.731258 | 0.932634 | 0.955966 | 1.046334 | Yes |
| Qwen3-0.6B | 27 | 24 | 1.083051 | 1.011675 | 0.941979 | 0.974101 | No |
| Qwen3-0.6B | 27 | 44 | 0.823751 | 0.948572 | 0.936580 | 0.968936 | No |
| Qwen3.5-0.8B | 3 | 1 | 0.586166 | 0.933158 | 1.077639 | 1.029905 | No |
| Qwen3.5-0.8B | 3 | 2 | 0.545241 | 0.885117 | 1.160861 | 1.010303 | No |
| Qwen3.5-0.8B | 3 | 80 | 0.980356 | 1.028149 | 0.961741 | 0.979717 | Yes |
| Qwen3.5-0.8B | 3 | 160 | 1.047170 | 1.070584 | 1.025373 | 1.003743 | Yes |
| Qwen3.5-0.8B | 23 | 1 | 0.617895 | 0.885752 | 0.844794 | 0.967531 | No |
| Qwen3.5-0.8B | 23 | 2 | 0.625484 | 0.911534 | 1.061216 | 0.944369 | No |
| Qwen3.5-0.8B | 23 | 80 | 0.881376 | 0.955052 | 0.894731 | 0.953736 | No |
| Qwen3.5-0.8B | 23 | 160 | 0.925077 | 0.956625 | 0.934515 | 0.957129 | No |

## Costs and next measurement

The paired operation adds no weight or output storage to the original two
normalization operations. This component harness additionally allocates
projection and host fixtures solely for verification. Numeric-stage peak RSS
is 253,640,704 bytes and timing-stage peak RSS is 132,186,112 bytes; neither is
model memory or startup cost. Pipeline construction remains a setup cost.

A proposed calibration will execute a fixed 32-stage chain in one command
batch, with each stage consuming the preceding stage's outputs and retaining
independent output allocations. This changes timing granularity and workload;
it is not an unchanged confirmation of these measurements. Keep original
arithmetic/output-bit gates, all endpoints and samples, own-null exclusion,
separate clocks and an independent repeated cohort before native admission.
The calibration is planned, not executed in this report.

Frozen archives, independently extracted, regenerated and audited on Mac
and Windows:

- Primitive, 39 files: `61cd813008cb2431819dc34fa2f719ac7acefce9ba2178ac90a1375b370fe4eb`.
- Complete learned cohort, 44 files: `31d657a088e16ec3217c4db3ebb72f670b0e4e6fc301ff9fa3c8834d91a1acc2`.

No native integration, serving binary, complete-model timing or latest
matched vLLM result changes.
