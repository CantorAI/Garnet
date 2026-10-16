# Prepared heterogeneous Q/K/V: actual-context GPU investigation

Current final component status: both complete cohorts audited on Mac/Windows;
only Qwen3 text meets the repeated component selection rule. Subsequent private
native validation rejects the full-QKV implementation: it violates the original
trained key-cache bound while the accepted control passes the identical fixture.
This supersedes component admission for native use. No full-model performance
result is established. [Terminal native rejection](../prepared-qkv-native/README.md#final-rejection).

Private generated context only. Accepted vector serving code, independent
weights and original key policies remain unchanged. No model speed claim.

Prepare one exactly packed common-scale half union before Begin. Keep all
three original weights, including current FP32 text/coder key baselines. One
F32-input MPS product writes joined scratch; a Metal scatter copies all Q, K
and V elements into their original independent output allocations. This adds
union weights and output traffic, and may reduce two MPS product encodes; no
internal GPU kernel count or full-model gain is inferred. Preparation and lazy
first GPU use remain separate costs. Original numerical bounds are retained.

## Actual primitive completed

PID6834 exits0 in11.194792416 seconds including quiet interval and compilation.
Twenty-four exact dyadic changed-input cases cover all six physical row counts,
unequal output widths13/5/3, original FP32-key baseline, all three outputs and
joined intermediates. Pending/foreign ownership, alias, abort/recovery,
unsupported profiles, nonexact values and64-group/2-reject bounded strong-owner
controls pass. Original small union payload336 bytes is shared across shapes.
This checks logical outputs, not physical padding or captured model inputs.

Freeze39-file primitive archive SHA256
`11a891b16bb61f63f23fe536437a3ae9eb4750658fb0416cb262744e6d7e42e4`.
Mac audit regenerates the exact context/header and checks source/compile/guard
records and complete primitive output. Windows transfer completes0 before
extraction, and its independent regeneration/audit passes too.
[Both-host primitive audit](primitive/verification.json).

## Learned design, declared before learned GPU execution

First require all58 separate triples, both serving batches and four changed
structured dyadic input patterns (116 profiles/464 changed-input cases).
All56 CPU-admitted triples must use the candidate successfully. VL layers7/23
must reject it and retain unchanged original outputs/counters at both batches.
The18 already combined VL linear-attention layers retain their original route.

Control uses three actual original EncodeBF16MatVec calls with the original
retained_float_key flags. Check each side against independent FP64 dot/sumabs
using gamma(ceil(K/32)+32)*sumabs+1e-7 AND5e-6+5e-5*abs(reference). Compare every
candidate element against the original output with the latter original bound;
scatter output must also equal the joined scratch bits exactly. Changed inputs
are structured synthetic vectors, not captured model activations. No timing
stage is admitted before the complete all-layer numerical/fallback screen.

Before observing any learned GPU results, select first and last CPU-admitted
triples: text layers0/27, coder0/23 and VL3/19, each at both batches. All-layer
numeric coverage still includes the mandatory rejected triples and all interior
groups. Endpoint selection uses CPU eligibility only, not favorable GPU data.
Twelve timed profiles use original-versus-original null, then original-versus-
candidate, each with2 seconds alternating common warmup,10ABBA rounds,3 warm
calls and5 recorded samples/window. Keep every sample. Update inputs and check
complete outputs outside the recorded clocks. Report CPU encode, GPU and
submit/wait wall clocks separately; null is an exclusion gate, not a correction.

A model passes the prospective component screen only if both endpoints at
both batches reduce GPU AND wall means by>=5%, with each own-null fraction in
0.95..1.05 and every numerical/fallback gate passing. Only a complete passing
initial screen admits one unchanged complete confirmation. A repeated component
pass permits private graph work for those models, not a model or overall win.
Graph integration must retain all three outputs, validate live aliases/consumers,
and preserve unsupported fallback before original full native validation and
all-six-profile repeated comparisons. No old-vLLM/new-native ratio or relaxed
precision/output bound may establish the active overall goal.

## Actual all-layer numerical stage passes; timing running

Learned launcher7300 compiles successfully and its complete numeric stage
exits0:116 profiles,464 changed-input cases,188,276,736 independent projection
checks and59,482,112 exact scatter-bit checks. This includes52 original FP32
key matrices and16 required fallback cases for VL layers7/23 at both batches.
Worst original gamma ratio0.0213529, projection-comparison ratio0.0678973 and
control/candidate comparison ratio0.0507740; no bound is relaxed. Numeric
process peak RSS323,338,240 bytes, not full-model memory usage.

The fixed first/last CPU-admitted endpoint timing stage is now running after
that complete numerical gate. These observations require terminal archive and
independent raw audits before admission. No model performance claim follows.

## Complete repeated component result

Initial7300 exits0 in85.327812625 seconds; one unchanged full confirmation8868
exits0 in85.406805458 seconds. Freeze before auditing or changing measured
sources. Initial45-file archive SHA256
`890cb0e810115043d3cf066686a9b6b9b100e5892a0c1b90d7c5b88cee69d95a`;
repeat46-file archive SHA256
`ed28aa1ce94c8c006b51c66839052ad661aba5a4d9fa2a78488fe2421dfa5b32`.
Transfers exit0 before Windows extraction and all independent audits pass.
Both cohorts retain232 complete learned numerical profiles and9600 timing
samples. Each complete numeric cohort checks188,276,736 projection references
and59,482,112 exact scatter bits. Each timing stage additionally screens48
changed-input endpoint cases with38,332,416 references/12,777,472 scatter bits.
Inputs are structured synthetic, not captured model activations.

Fractions below are candidate/control means using every sample; lower is better.

| Endpoint model | Layer | Rows | Initial GPU | Initial wall | Repeat GPU | Repeat wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-Coder-0.5B-Instruct | 0 | 26 | 0.709651 | 0.865309 | 0.699805 | 0.867335 |
| Qwen2.5-Coder-0.5B-Instruct | 0 | 52 | 0.808673 | 0.965880 | 0.769475 | 0.847467 |
| Qwen2.5-Coder-0.5B-Instruct | 23 | 26 | 0.690671 | 0.881699 | 0.702425 | 0.888772 |
| Qwen2.5-Coder-0.5B-Instruct | 23 | 52 | 0.810419 | 0.893086 | 0.843344 | 0.907493 |
| Qwen3-0.6B | 0 | 24 | 0.697479 | 0.795798 | 0.719165 | 0.847805 |
| Qwen3-0.6B | 0 | 44 | 0.625216 | 0.725260 | 0.650452 | 0.795903 |
| Qwen3-0.6B | 27 | 24 | 0.717523 | 0.828654 | 0.698773 | 0.828562 |
| Qwen3-0.6B | 27 | 44 | 0.643725 | 0.774992 | 0.655579 | 0.809635 |
| Qwen3.5-0.8B | 3 | 80 | 1.241678 | 1.126969 | 1.179967 | 1.067352 |
| Qwen3.5-0.8B | 3 | 160 | 1.343516 | 1.213269 | 1.359875 | 1.214678 |
| Qwen3.5-0.8B | 19 | 80 | 1.209889 | 1.100259 | 1.207363 | 1.079056 |
| Qwen3.5-0.8B | 19 | 160 | 1.382492 | 1.234055 | 1.413402 | 1.233377 |

Text passes all four endpoint/batch gain and own-null gates in both runs.
Its repeat GPU/wall reductions are28.08%/15.22%,34.95%/20.41%,30.12%/17.14%
and34.44%/19.04%. Initial coder first-layer B2 wall reduction3.412% misses5%;
its last-layer B1 null GPU .943972 and wall .949777 are outside the declared
range. Its favorable repeat cannot rescue those initial failures. VL is slower
at every tested endpoint in both cohorts despite reduced CPU encode time;
do not substitute encode timing for GPU/wall or infer an unmeasured hardware
cause. No third favorable-repeat search or threshold change is admitted.

[Initial complete audit](initial/verification.json),
[initial means and own-nulls](initial/decision.json),
[repeat complete audit](repeat/verification.json),
[repeat means and own-nulls](repeat/decision.json),
[unchanged-source two-cohort decision](decision-repeat.json).

## Measured setup and component memory costs

Initial first-shape preparation ranges238.485..246.950ms per text triple,
58.855..60.251ms per coder triple and299.732..305.829ms per admitted VL triple.
Subsequent supported shapes reuse weights and take0.0116..0.0661ms. Rejected
VL preparation still costs261.938..285.650ms; cache rejection is not free.
Original uploads are separately recorded. Do not sum these component contexts
into measured model initialization or startup. Logical all-admitted duplicate
payload estimates remain224/47.25/40MiB; actual native cache/first-use/RSS needs
its own measurement. Initial numeric/timing process peaks are323,338,240/
146,833,408 bytes; repeat321,929,216/146,489,344 bytes. One triple is held at a
time with an anchor pipeline context; these are not simultaneous model peaks.
[Derived component cost ranges](cost-summary.json).

Compile retains the original two fastMathEnabled deprecation warnings. Sources,
flags, binary identities, original FP32-key controls, full outputs and guard
monitors remain in the frozen archives. The main serving branch is unchanged.
Only text-only private graph/session work is admitted; full native correctness,
all-six-profile native repeats and fresh matched-vLLM measurements remain.
