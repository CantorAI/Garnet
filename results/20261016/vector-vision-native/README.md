# Ordered float4 vision context: private native integration

Status: prospective native experiment; no full-model gain established.

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
