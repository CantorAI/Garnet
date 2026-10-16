# Ordered float4 vision context: private native integration

Status: full validation and initial native cohort audited on both hosts; initial VL TTFT screen passes. Unchanged confirmation running; full goal unmet.

Motivation: full repeated matched FP32 comparison still misses half-TTFT for
Qwen3.5-0.8B B1/B2 by about 28.3% of Garnet's current TTFT. Decode already exceeds
2x vLLM-Metal for these profiles. This candidate attacks the vision context stage.

Selection: two complete independently calibrated component cohorts select
width4 by the predeclared geometric-mean serving wall fraction, 0.690318881,
versus width2 0.712403945. All original output/padding bits and FP64 bounds pass;
these are synthetic component results, not complete-model measurements.
See [selection](../vector-vision-calibrated/decision-repeat.json).

Copy the retained source into a private investigation root and append the exact
audited `vision_vector4` shader to the strict scalar-prefetch library. Preserve
the eight sequential FP32 FMAs and scalar tail for each output. Reuse existing
buffers; add no weight copies or input caching. Only serving tokens216,
hidden768, heads12, rows216/432 with aligned FP32 matrices take the vector route.
Unsupported layouts or a missing vector pipeline retain the scalar route.
Expose an actual dispatch counter through the graph and frame worker.
The retained c1869309 library is untouched.

Before any full-model timing, require fresh original28 numerical, trained
vision, full-head/layer, tail, KV/recovery and wide-prefill stages plus compilation
and execution of the new actual-library primitive: 96 changed-input shape
checks, eight serving and 88 fallback records; pending/abort/reset/recovery.
The public allocator supplies aligned layouts; this primitive does not claim
to exercise arbitrary user-created misaligned MPS views. Existing original
numerical bounds are unchanged.

Prospective retention rule, declared before native timing: in two complete
native cohorts, both VL batches must reduce warm TTFT by at least 2%, with no
decode reduction exceeding 2%; all original correctness gates must pass.
Report initialization, first-after-init, peak RSS and text/coder effects
separately. Full-model comparison includes all six model/batch profiles and
all original complete outputs. No hardware-cold or startup-sum claim.
Retention does not establish the full goal; fresh repeated matched vLLM-Metal
measurements remain necessary for the 2x/half-TTFT objective.

Commands, build identities, guard observations and full source closure will
be captured by `prepare_vector_vision_native.py` and
`launch_vector_vision_native_checks.py`; all evidence is preserved before any
candidate repair. No live-job source mutation.

## Build and initial actual validation

Private source preparation succeeds with1406 files and eight transformations;
[immutable preparation record](preparation.json). Candidate build and primitive
compile/execution exit0. Primitive covers96 bit-exact cases,8 serving and88
fallback, plus pending/abort/reset/recovery. All seven CTests pass. Trained full
and small vision-grid B1/B2 stages and trained real-head stage also exit0.
Remaining original decoder tail/layer/recovery stages are still running under
exact validation PID64007. This is partial validation, not a complete audit.

Finalizer PID66876 watches only64007, archives terminal success or failure and
audits success. Continuation PID67688 watches only66876; it admits a native
cohort only after the exact successful archive/audit. It preserves72 complete
outputs over all six profiles, archives and independently audits each cohort,
then applies the prospective rule. A failed initial screen stops confirmation;
only a passing screen admits one unchanged repeat. Retained baseline stays
unchanged. These controllers are running; no native performance is yet claimed.

Executed preparation/launch commands (ROOT is
`/Users/zsdx/garnet-metal-investigation`, CANDIDATE is
`ROOT/vector-vision-native-investigation-20261016`):

```text
python3 ROOT/source/test2026/metal/prepare_vector_vision_native.py ROOT CANDIDATE
python3 ROOT/source/test2026/metal/launch_vector_vision_native_checks.py CANDIDATE CANDIDATE/logs/vector-vision-native-checks-20261016 checks
python3 ROOT/source/test2026/metal/finalize_vector_vision_native_checks.py CANDIDATE ROOT/logs/vector-vision-native-finalize-20261016 64007
python3 ROOT/source/test2026/metal/continue_vector_vision_native.py CANDIDATE ROOT/logs/vector-vision-native-continuation-20261016 66876
```

## Retained-control clock breakdown for next optimization priorities

This is a derived description of the earlier frozen selective-NDArray cohort's
retained c186 control, not a current vector timing result. Verify18 raw files
against that archive manifest, four warm control requests/profile and all six
complete profiles. [Raw-source identities and phase means](retained-control-phase-analysis.json).
Source archive SHA256:
`1861996b1455f4a6aa7e81fb89bac30a2be2526d359eb332cc974d31aa6e520e`.

| Model | Batch | TTFT ms | Vision wall ms | Prefill wall ms | Prefill GPU ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| phase Qwen2.5-Coder-0.5B-Instruct | 1 | 26.806 | — | — | 22.932 |
| phase Qwen2.5-Coder-0.5B-Instruct | 2 | 43.772 | — | — | 39.010 |
| phase Qwen3-0.6B | 1 | 35.638 | — | — | 30.964 |
| phase Qwen3-0.6B | 2 | 55.890 | — | — | 49.873 |
| phase Qwen3.5-0.8B | 1 | 178.289 | 58.630 | 89.603 | 83.781 |
| phase Qwen3.5-0.8B | 2 | 296.004 | 113.156 | 151.072 | 144.813 |

GPU spans overlap CPU encode/wait and must never be added to those clocks.
Vision/assembly/prefill are worker intervals; TTFT also spans frontend/IPC.
Their residual does not isolate a causal frontend bottleneck. These observations
support prioritizing prefill GPU work alongside vision and decode; they cannot
predict a candidate's end-to-end gain. No existing timing or vLLM ratio changes.

## Full native validation passes; model timing running

Validation PID64007 is terminal0 after1437.991319083s. All30 stages are fresh;
no prior numerical stage is reused. Finalizer66876 is terminal0 and freezes a
1764-file archive at `ROOT/vector-vision-native-checks-20261016.tar.gz`, SHA256:
`239dbf64f81578b7616bc7728c542b0032ca32318ec30d9314c596a1fbbf72d6`.
The independent Mac raw/regeneration audit passes all30 stages,7 CTests,
96 vector primitive cases,80 retained coalesced primitive cases,51 trained-head
queries/78 rows,4 trained vision profiles,6+6 recovery cases and4 wide trained
profiles/12 carried steps. Worst original wide bound fraction0.5432836183473805.
All eight source changes and the exact owned-process full30 driver regenerate.
Windows archive/raw audit is still pending while performance work is running.

Candidate library SHA256:
`d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e`.
Retained library remains c1869309. Native initial full six-profile timing
PID89884 is live after the verified quiet interval. Do not copy heavy archives
or mutate source/driver while it runs. This proves compatibility only; no
warm-TTFT/decode/RSS benefit, confirmation or fresh-vLLM target pass yet.

## Initial model cohort passes after audit-path repair

Initial PID89884 exits0 in192.701290041s; all six profiles,72 complete outputs,
1319 serving source files and1104 external tool files are archived. Full archive
SHA256 `4391422a668b2e4b551c981635e252e00e5274f3ffa8eabd7259a325759bded5`,
5611 files. Mac/Windows raw audits and source regeneration now pass. Candidate
VL requests record12 vector dispatches/image; control/text/coder record0.
[Full-output audit](comparison/initial/verification.json),
[predeclared decision](comparison/decision-initial.json),
[full30 validation audit](validation/verification.json).

| Native model | Batch | Decode TPS control / candidate | Warm TTFT ms control / candidate | Decode fraction | TTFT fraction | Mean worker peak RSS delta MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native Qwen3-0.6B | 1 | 74.9157 / 74.5706 | 35.4968 / 36.0237 | 0.995394 | 1.014841 | 3.625 |
| Native Qwen3-0.6B | 2 | 69.7161 / 70.6258 | 56.8193 / 56.3546 | 1.013049 | 0.991822 | -7.992 |
| Native Qwen2.5-Coder-0.5B-Instruct | 1 | 88.2668 / 88.2007 | 27.1347 / 26.7988 | 0.999250 | 0.987622 | 3.453 |
| Native Qwen2.5-Coder-0.5B-Instruct | 2 | 84.0863 / 84.6414 | 44.1418 / 42.4584 | 1.006601 | 0.961865 | 0.133 |
| Native Qwen3.5-0.8B | 1 | 61.1809 / 61.2709 | 177.9961 / 172.5206 | 1.001471 | 0.969238 | -20.242 |
| Native Qwen3.5-0.8B | 2 | 55.1446 / 55.2624 | 294.5845 / 287.8972 | 1.002136 | 0.977299 | -6.570 |

VL B1/B2 warm TTFT improves3.076%/2.270%; decode changes+0.147%/+0.214%.
Both pass the prospective >=2% TTFT screen; every profile decode fraction>=.98.
Text/coder never dispatch this kernel; their observed variation is not a
claimed vision optimization benefit. Peak process RSS spans initialization and
inference; these small differences do not establish unique GPU or logical
weight-memory savings. No new weight allocation is introduced.

| First native model | Batch | Initialization s control / candidate | First-after-init TTFT ms control / candidate |
| --- | ---: | ---: | ---: |
| First native Qwen3-0.6B | 1 | 3.527615 / 3.510491 | 133.730 / 130.325 |
| First native Qwen3-0.6B | 2 | 3.536318 / 3.531673 | 147.680 / 142.876 |
| First native Qwen2.5-Coder-0.5B-Instruct | 1 | 2.818650 / 2.821043 | 107.003 / 107.692 |
| First native Qwen2.5-Coder-0.5B-Instruct | 2 | 2.813799 / 2.827994 | 122.130 / 127.949 |
| First native Qwen3.5-0.8B | 1 | 5.494078 / 5.445742 | 273.536 / 266.995 |
| First native Qwen3.5-0.8B | 2 | 5.458587 / 5.465656 | 394.966 / 389.228 |

These clocks remain separate; caches were not flushed. No hardware-cold or
summed startup clock is claimed. These are native-only comparisons against
c186, not fresh vLLM ratios. The full target/counts stay unchanged.

The first auditor passed all72 outputs, routes, clocks and original bounds,
then expected `<profile>.executed.py`. The actual frozen comparator uses
`with_suffix`, producing three shared per-model expanded script files across
six argv-distinct profiles. All three files are byte-identical to the same
regenerated generic comparator; all six recorded commands/cases remain distinct.
Repair only the pathname assertion to follow that exact original operation,
and record three actual files/six profile checks. No runtime, benchmark, input,
sampling or numeric bound changes, no initial GPU rerun. The partial auditor
JSON was rejected until the complete repaired process exited0.

Original controller67688 and dependent queues73358/83151 exit1 and are frozen
in45-file failure archive SHA256
`252aeda0ce11fbe3849f9127111504715f16cc8f4ceb17362bab6d3c28168c94`.
[Preserved failure and frozen scripts](auditor-path-failure/archive-verification.json).
The initial full-model archive retains the original failing auditor unchanged;
`verify_vector_vision_native_profiles_repaired.py` pins it by SHA256 and changes
only that final path check, preserving every original output/numerical gate.

After archive transfer terminal0 and both-host audits, new controller93884
reuses the complete admitted initial cohort and launches unchanged confirmation
PID93898 at `logs/vector-vision-native-confirm-20261016`. No favorable repeat
search: this is the one predeclared confirmation. Old failures are not rewritten.
Native inference sources/library/cases and sampling remain identical; added
audit/queue helpers are provenance tools, not measured inference changes.
No promotion or fresh vLLM target pass before repeated actual evidence.
