# Wide output block packed prefill rejected by timing

Actual Mac PID29834 exited0 in129.259064seconds. The fixed wide-output selection
passes all48 original numerical profiles but both M32/M64 tile policies are
slower than the original selected MPS/BF16 routes. Across4,800 raw timed samples,
GPU fractions are1.453310 to2.131512 and wall fractions1.228141 to1.754453.
All12 own-null profiles are stable within5%; no confirmation or native integration
is admitted. The full matched three-model target remains unmet and unchanged.

## Hypothesis and selection provenance

This follow-up was chosen after the [all-eligible block tile failure](../block-packed-prefill/README.md)
and before inspecting any timing for this new selection. It uses the same
N64/K32 contiguous BF16 layout and M32/M64 FP32 SIMD matrix kernels, but only
allocates/dispatches for original interleaved-eligible weights with output
columns strictly greater than1,024. All other projections keep their original
control path. The hypothesis is that wide gate/up, query and linear projections
provide more output tiles per row block and may benefit from contiguous loads.
It does not establish the cause of the preceding comparison failures.

The selected implementation is still the accepted d52 context; this test builds
a private context from frozen sources. FP32 activation/accumulation, original
checkpoint words, special key route, inputs, output caps, physical rows24/44,
26/52,80/160, original gamma and5e-6+5e-5*abs(control) comparison limits remain.
The same29 matrices and all six model/batch profiles are retained.12 matrices
use the new layout; the others execute the original path in the measured set.
No complete inference or vLLM measurement is included.

## Numerical and clock protocol

All three models complete numeric mode before any timing. Each of two policies,
two batches and four deterministic input patterns is compared against FP64 and
original control:48profiles and75,601,920 output values. All pass; worst control
and candidate FP64 ratio0.075892469, worst comparison ratio0. Numerical equality
here does not claim bitwise identity for signed zeros: no output memcmp was used.
Every packed/padded BF16 storage word is CPU-roundtripped before actual upload.

The runner metadata fixes the performance screen before execution: target text
B2, coderB2 and VL B1/B2 require5% improvement in GPU and wall clocks; all six
profiles must preserve both clocks within2%, and every own-null clock must be
within5%. A complete unchanged confirmation is required before any model
integration, with no favorable third repeat. The four targeted endpoints are
the outstanding TTFT workloads in the matched baseline; passing this component
would not finish the separate batch1 decode goals or any full-model requirement.

The runner verifies checkpoint slices before compilation; every execution rechecks
sources, and each timing process verifies layout and warms2seconds
per profile, and runs its own control/control null before control/candidate.
Each phase has10ABBA rounds,3warm calls/window and5samples/window. Fractions
below divide the arithmetic mean of100 raw candidate intervals by100 control
intervals per clock/phase. No filtering, null normalization or favorable subset.
Values above1 are slower. Policies1/2 mean M32/M64, both N64/K32 and256threads.

| Model | Batch | Policy | GPU fraction | Wall fraction | Null GPU fraction | Null wall fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 1 | 1.453310 | 1.228141 | 1.001472 | 1.005604 |
| Qwen3-0.6B | 2 | 1 | 1.702332 | 1.430575 | 1.006009 | 1.001777 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 1 | 1.726462 | 1.413292 | 1.024436 | 1.009980 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 1 | 1.707471 | 1.429962 | 0.997436 | 1.002789 |
| Qwen3.5-0.8B | 1 | 1 | 1.618370 | 1.512804 | 0.992636 | 0.996865 |
| Qwen3.5-0.8B | 2 | 1 | 1.660344 | 1.569948 | 0.998618 | 1.001558 |
| Qwen3-0.6B | 1 | 2 | 2.131512 | 1.687363 | 1.006152 | 0.998964 |
| Qwen3-0.6B | 2 | 2 | 2.014927 | 1.653236 | 1.003486 | 0.996083 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 2 | 2.108778 | 1.623424 | 1.006160 | 1.002583 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 2 | 1.911627 | 1.574619 | 1.005998 | 1.017109 |
| Qwen3.5-0.8B | 1 | 2 | 1.918649 | 1.754453 | 0.998217 | 1.001243 |
| Qwen3.5-0.8B | 2 | 2 | 1.786611 | 1.682008 | 1.003337 | 1.001661 |

Neither policy passes any profile's performance screen; both are rejected without
an unchanged confirmation or native model integration. The null results make
an unstable baseline an insufficient explanation for these large measured
regressions. Instruction scheduling, register use, occupancy and generated GPU
assembly were not measured, so the precise hardware cause remains unknown.
Contiguous BF16 loads and reduced shared scratch alone did not beat the original
selected MPS/BF16 projection sets. Do not rerun unchanged or claim a TTFT benefit.

## Costs and evidence limits

Additional BF16 buffers total88,735,744bytes (84.625MiB) across the three separate
numeric processes, reused for both batches. Upload/preparation totals427.354077ms,
including original preparation, new layout and bit verification. These are
component sums, not full-model storage, added setup latency or peak RSS. Both
measured policies/control carry the same allocations. Transient allocation,
whole-model initialization and first-request latency were not measured.

The original CPU layout proof and inherited irregular/padding coverage are in
the [prior preparation](../block-packed-prefill/preparation/preparation.json).
The wide source preflight produced hashes identical to Mac generation. A GPU
ownership/abort/tail suite and sequential trained model checks remain absent;
no integration was admitted. No vision-tower, output/prefix/image caching,
speculative decoding, fresh matched vLLM cohort or target-count change occurred.
No hardware cache flush or separate thermal observation was taken.

The runtime guard checks before/during each executed stage and rejects competing
inference/probes. Both controllers were terminal and archived before moving to
the next run; no consumed runner, generator or context was changed while live.
The numeric/timing runner was frozen before launch. The external verifier was
written separately; a row-identity assertion was added before inspecting clocks
and uploaded into a new separate audit-tool directory. Selection thresholds and
measured source were unchanged. This is not a formal publication preregistration.

## Reproduction

```text
python3 launch_wide_block_packed_prefill.py ROOT ROOT/logs/wide-block-packed-prefill-20261016
python3 archive_wide_block_packed_prefill.py ROOT ROOT/logs/wide-block-packed-prefill-20261016 ARCHIVE
python -X utf8 test2026/metal/verify_wide_block_packed_prefill.py RESULTS/initial VERIFICATION_JSON
```

[Mac audit](initial/mac-verification.json), [Windows audit](initial/verification.json),
[Manifest and raw source/clock/numeric/command records](initial/evidence/manifest.json),
[Archive byte audit](initial/archive-verification.json),
[Windows source preparation](preparation/preparation.json).

The63-file archive covers21 consumed source identities, four regenerated
sources, exact defines, all stage/guard/exit records, raw numerical and interval
records, configs and size/hash identities of omitted executable, raw slices and
checkpoints. Both hosts produce identical audits and rejection decisions.
Archive `c329a1c9493e45a182eee97be8f374e14c285b3ab927b5077c245a6c2f997012`.
Reference library `d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e`.
Environment inherits the verified M4/16GiB, macOS15.6.1, clang17, Python3.14.6 host;
separate pre/post thermal measurements are not present in this component archive.
