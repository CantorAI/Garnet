# Prepared exact union weights and activation-only output

Final status: two complete learned cohorts audited on Mac and Windows; no model
passes the declared repeat rule. Reject this version before native integration.

Private generated context only; no serving runtime integration or model speed
claim. Generate from the retained validated d52 vector-vision context. Preserve
the original independent weights. Before Begin, prepare an exact common-scale
joined weight buffer, MPS product and scratch for each selected row count.
Encode one joined product and one activation kernel, writing only the activation.
This removes an independent MPS product and avoids materializing separate gate
and up outputs; no claim about MPS's internal GPU kernel count follows.

Costs: the prototype duplicates the half weight payload, scans and bit-checks
every value during preparation, and allocates joined scratch. Lazy first GPU
compilation may still occur. Future graph integration must prove gate/up slots
are internal, unobserved and dead after the activation; unsupported cases retain
the original operation. No mutation of shared original half buffers is allowed.

## Primitive completed

Actual M4 execution PID914 exits0 in 11.2132 seconds including quiet interval,
compilation and execution. Twenty-four changed-input dyadic cases cover physical
rows 24/44, 26/52 and 80/160. Original output bits match, joined intermediates
match, and pending reads, foreign writers, alias, abort/recovery, unsupported
shapes, nonexact weights and bounded 128-pair ownership controls pass. The
initial small fixture allocates 512 extra weight bytes shared across shapes.
This is synthetic coverage, not learned-matrix or full-model validation.

The 38-file archive SHA256 is
`98db64683fec2df9d3f553ac9211f9031dae0a0365e136f8ddd85dc3324a85c6`.
Both-host audits regenerate the exact private context/header, check compile
defines, source closure, exclusivity monitors and complete primitive output.
[Primitive audit](primitive/verification.json),
[frozen source and commands](primitive/evidence/manifest.json).

## Learned component design, declared before learned GPU execution

Require every one of the 76 learned pairs at both serving physical row counts
and four changed-input patterns. All 75 CPU-admitted pairs must pass GPU checks;
coder layer8 must reject preparation and execute the original path unchanged.
Inputs are structured dyadic vectors with exact power-of-two row/variant
scales, not captured model activations. Independent FP64 dot/sumabs references
use the original gamma(ceil(K/32)+32)*sumabs+1e-7 bound AND the original
5e-6+5e-5*abs(reference) comparison. Check both projections, candidate joined
intermediates and stage-rounded FP32 activation references. No tolerance change.
Complete numerical screening precedes any learned timing admission.

Time first AND last layers of each model, selected before observing data, at
both batch shapes: twelve profiles. Do not select favorable interior layers.
For each profile, perform a same-original-pipeline A/B null comparison, followed
by original versus candidate. Each uses 2 seconds common alternating warmup,
10 ABBA rounds, 3 warm calls and 5 recorded samples per window. Retain every
sample; do not normalize results by the null. Update inputs outside the clocks;
record GPU, CPU encode and submit/wait wall clocks separately. Recheck complete
outputs outside recorded clocks after each window. Report preparation, uploads,
first executions and allocated extra union payload independently.

Prospective component admission: a model must achieve at least 5% GPU AND wall
reduction at both selected layers and both batches, every own-null GPU/wall
fraction must be within 0.95..1.05, and all numerical/fallback gates must pass.
Only a passing complete initial screen admits one unchanged confirmation.
Component success still requires full graph correctness, all-six-profile native
repeats and fresh matched-vLLM cohorts before claiming the overall goal.

## Actual all-layer numeric stage complete; timing running

Learned launcher PID1987 has compiled the private context and completed all
152 layer/batch profiles, 608 changed-input cases and 849,807,360 individual
projection/activation reference checks. All original bounds pass: worst gamma
ratio0.0176357, projection-comparison ratio0.0653276 and activation ratio0.0385515.
Learned control/candidate values agree for these structured inputs (worst
comparison ratio0). Eight changed-input fallback checks cover coder layer8 at
both batches. Numeric process peak RSS596,000,768 bytes; not a full-model RSS.

The predetermined endpoint timing process is running after this numerical gate.
These live observations are provisional until terminal archive and independent
audits. No timing result, model admission or full goal progress ratio is claimed.

## Two terminal learned cohorts: numerically valid, no repeated selection

Initial PID1987 exits0 in120.124820292 seconds; unchanged confirmation PID4097
exits0 in119.978341708 seconds. Freeze both before analysis/source changes.
Initial archive43 files, SHA256
`a3e4df6ecb8626299ec3c5cfeb3c9deec172a21801208035398d5c709293ed5e`;
repeat44 files, SHA256
`954c6ff4e22f70437a8156d6b7ec952a774fe8b4fe619f3bf6b7edf35f111a36`.
Transfers complete0 before Windows extraction. Both-host independent audits
regenerate context/header, verify all source/compile/asset identities and
exclusive guards, all304 layer/batch numeric profiles, and9600 timing samples.
Each full numerical cohort has608 changed-input cases and849,807,360 individual
checks. Additional timing-stage numeric screening has48 cases and69,525,504
checks per cohort. These are repeated structured inputs, not unique model
activations. The original tolerances remain unchanged.

The following fractions are candidate/control arithmetic means of every
recorded sample. Lower is better; required GPU AND wall fraction<=0.95 at
both endpoints and both batches, with every own-null within0.95..1.05.
Layers are zero-based; physical rows map to the declared B1/B2 profiles.

| Endpoint model | Layer | Rows | Initial GPU | Initial wall | Repeat GPU | Repeat wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-Coder-0.5B-Instruct | 0 | 26 | 0.838410 | 0.864298 | 0.824714 | 0.831429 |
| Qwen2.5-Coder-0.5B-Instruct | 0 | 52 | 0.916942 | 0.915905 | 0.933141 | 0.939184 |
| Qwen2.5-Coder-0.5B-Instruct | 23 | 26 | 0.809837 | 0.822815 | 0.835965 | 0.862516 |
| Qwen2.5-Coder-0.5B-Instruct | 23 | 52 | 0.947080 | 0.930774 | 0.976858 | 0.969217 |
| Qwen3-0.6B | 0 | 24 | 0.935938 | 0.923065 | 0.961168 | 0.949840 |
| Qwen3-0.6B | 0 | 44 | 0.842058 | 0.869012 | 0.829997 | 0.824799 |
| Qwen3-0.6B | 27 | 24 | 0.968933 | 0.952249 | 0.986149 | 0.966083 |
| Qwen3-0.6B | 27 | 44 | 0.840838 | 0.846905 | 0.854716 | 0.871630 |
| Qwen3.5-0.8B | 0 | 80 | 0.898458 | 0.902285 | 0.941691 | 0.932140 |
| Qwen3.5-0.8B | 0 | 160 | 0.973318 | 0.978288 | 0.956978 | 0.949684 |
| Qwen3.5-0.8B | 23 | 80 | 0.907381 | 0.902236 | 0.884475 | 0.881370 |
| Qwen3.5-0.8B | 23 | 160 | 0.995034 | 0.988676 | 0.943999 | 0.936526 |

Initial coder passes all four profiles, admitting exactly one unchanged full
confirmation. Text misses its last-layer B1 gain; VL misses B2 gains and has
an initial first-layer B1 own-null GPU fraction1.063222 (outside5%). All repeat
own-nulls pass. Repeat coder last-layer B2 gains only2.314%GPU/3.078%wall, below
5%; repeat text B1 also misses. Repeat VL first-layer B2 GPU gain4.302% misses.
Do not normalize by nulls, rescue initially failed models with repeat values,
select favorable layers, relax thresholds or seek a third unchanged repeat.
No hardware-cause attribution is established. No model is selected.

[Initial complete raw audit](initial/verification.json),
[initial complete means/nulls](initial/decision.json),
[repeat complete raw audit](repeat/verification.json),
[repeat complete means/nulls](repeat/decision.json),
[two-cohort unchanged-source decision](decision-repeat.json).

## Costs and limitations

Actual first-shape CPU preparation ranges in the initial full numeric cohort:
text342.700..350.692ms/pair, coder474.971..485.724ms/pair and
VL399.267..409.199ms/pair. A second shape reuses each packed buffer and takes
0.0176..0.1174ms; rejected coder layer8 preparation still costs52.3..52.5ms.
Original per-pair upload costs remain separately recorded. These per-pair
contexts are not a full-model initialization benchmark and the times must not
be added to prior startup measurements. Private numeric/timing processes peak
at596,000,768/206,815,232 bytes initially and584,056,832/199,229,440 bytes on
repeat. No full-model RSS or shared-memory-saving claim follows.

Logical extra full-model weight payloads, if all admitted pairs were retained,
are336 MiB text,382.375 MiB coder and336 MiB VL. The actual component holds one
pair at a time; it does not measure simultaneous full-model memory. Original
independent weights, all graph outputs and the accepted vector serving branch
remain unchanged. [Derived cost ranges and source identity](cost-summary.json).

Compile output retains one Objective-C auto/id deduction warning and two
existing deprecated fastMathEnabled warnings; compile and both execution
stages exit0. Raw warnings are preserved, not edited after measurement.

Executed commands (ROOT=`/Users/zsdx/garnet-metal-investigation`):

```text
python3 ROOT/source/test2026/metal/launch_prepared_union_mlp_learned.py ROOT ROOT/logs/prepared-union-mlp-learned-20261016 learned
python3 ROOT/source/test2026/metal/archive_prepared_union_mlp_learned.py ROOT ROOT/logs/prepared-union-mlp-learned-20261016 ROOT/prepared-union-mlp-learned-20261016.tar.gz
python3 ROOT/source/test2026/metal/verify_prepared_union_mlp_learned.py INITIAL_EXTRACT INITIAL_EXTRACT/verification.json
python3 ROOT/source/test2026/metal/analyze_prepared_union_mlp_learned.py INITIAL_EXTRACT INITIAL_EXTRACT/decision.json
python3 ROOT/source/test2026/metal/launch_prepared_union_mlp_learned.py ROOT ROOT/logs/prepared-union-mlp-learned-confirm-20261016 learned
python3 ROOT/source/test2026/metal/archive_prepared_union_mlp_learned.py ROOT ROOT/logs/prepared-union-mlp-learned-confirm-20261016 ROOT/prepared-union-mlp-learned-confirm-20261016.tar.gz
python3 ROOT/source/test2026/metal/verify_prepared_union_mlp_learned.py REPEAT_EXTRACT REPEAT_EXTRACT/verification.json
python3 ROOT/source/test2026/metal/analyze_prepared_union_mlp_learned.py REPEAT_EXTRACT REPEAT_EXTRACT/decision.json
python3 ROOT/source/test2026/metal/confirm_prepared_union_mlp_learned.py INITIAL_EXTRACT REPEAT_EXTRACT REPEAT_EXTRACT/decision-repeat.json
```

The repeat launch records initial archive/audit/decision hashes and verifies
every measured source unchanged before launching. No native graph integration,
new matched-vLLM ratio, target-count update or overall goal completion follows.
