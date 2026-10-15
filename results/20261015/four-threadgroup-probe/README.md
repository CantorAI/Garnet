# Original four-output threadgroup tuning: audited; no setting selected

Hypothesis: the existing four-output compact kernel may prefer different
threadgroup sizes for individual projection shapes. Unlike the rejected
two-output candidate, this changes no shader, pipeline, payload, arithmetic,
lane reduction or weight representation. A private context hook changes only
threads/group and corresponding outputs/group. Control always uses64 threads;
candidates use64/128/256. Candidate64 is an explicit null GPU-work comparison.
CPU hook/counter overhead remains inside measured wall time.

The original20 selected matrices are measured individually at physical rows1/2,
with all29 learned assets still checked by the unchanged CPU packing admission
and runtime roundtrip proof. Four patterns, original exact output bits, original
FP64/native bounds and189 ownership/fallback/abort safety profiles are retained.
Expected counts are480 numerical records and7,680 timed submissions per cohort.
Failed numerical policies receive no timing windows. These are expected counts,
not an execution result. All weights remain resident; repeated single-projection
cache behavior differs from full inference.

Predeclared selection rule: require GPU time fraction<=0.95 and wall fraction<1
in BOTH initial and confirmation cohorts for every observed duplicate matrix
sharing a geometry. Its64-thread null GPU/wall fractions must stay within5%
of1 in both cohorts. Pick the qualifying thread count with minimum worst GPU
fraction; keep original64 elsewhere. This is a conservative selection heuristic,
not a confidence interval. Do not rerun rejected settings seeking a favorable
sample. Any selected table still needs fresh full-model validation/confirmation
and repeated matched vLLM measurements before a target claim.

Private route counters assert actual tuning selection, including the64-thread
null. There is no additional compiled Metal shader; a retained header copy is
archived solely to verify original source identity. Local context/header/harness
generation and expanded runner syntax checks pass. The separate frozen auditor
materializes every original packing/numerical/safety/raw-window assertion from
the prior verifier stack and regenerates the new private sources from snapshots.
Its SHA256 is
`f2007893419f58f0d0cff50afc19f6be89873747acc100dea5b830c2fd2478de`.
This was the launch-time prepared protocol. Actual terminal outcomes follow;
no model speed gain is established.

```text
python -X utf8 test2026/metal/freeze_four_threadgroup_verifier.py
python3 launch_four_threadgroup_probe.py ROOT ROOT/logs/four-threadgroup-probe-20261015 component
# Only after exact terminal capture:
python3 archive_four_threadgroup_probe.py ROOT ARCHIVE [TRIAL]
python -X utf8 test2026/metal/verify_four_threadgroup_probe_frozen.py AUDIT AUDIT_RESULT
```

ROOT is /Users/zsdx/garnet-metal-investigation. The launcher requires the last
BF16 component to be terminal, archived and audited, then its own quiet guard.
Sources remain frozen until terminal archival. Existing source/build defaults
and the protected GPT-OSS/Pi checkouts remain unchanged. Full goal remains unmet.

## Actual initial and confirmation cohorts — 2026-10-15

Initial PID74228 exited0 in31.738427s; confirmation PID74935 exited0 in
31.689417s. Both Mac and Windows frozen audits pass every original exact-bit,
FP64/native-envelope, packing and safety assertion. Each cohort covers20
selected matrices,480 numerical records,189 safety profiles and7,680 timed
executions. Sources, generated code, flags, original learned assets, versions,
hardware and binary match across cohorts. Original CPU packing proof is reused,
not independently repeated. The original shader and pipeline serve both arms;
no additional shader was compiled.

Initial59-file archive SHA256:
`2594e5f9bc1e9caace3176eb5bde5b296e9b9e34aba0ae5a653e96dc20102af5`.
Confirmation59-file archive SHA256:
`5a4a9a56e5bd455dff80908d32541c5d7928b38987216e5f101b87e87661cde7`.
[Initial audit](initial/verification.json),
[confirmation audit](confirmation/verification.json),
[joined observations and predeclared selection](confirmation-comparison.json).

Zero geometries qualify under the original selection rule. The only geometry
passing all initial criteria (physical rows2, output1024, inner1024 at128 threads)
does not pass both cohorts. Keep original64; no native integration or fresh
vLLM comparison is warranted for this selected set.

The64-versus64 null uses identical GPU work; CPU hook/counter overhead remains
in wall clocks. Null duration observations are:

| Cohort | GPU fraction min / max | GPU outside5% /40 | Wall fraction min / max | Wall outside5% /40 |
|---|---:|---:|---:|---:|
| initial | 0.862106 / 1.143511 | 18 | 0.878356 / 1.083377 | 12 |
| confirmation | 0.851710 / 1.347176 | 16 | 0.895276 / 1.202355 | 15 |

This is observed variability in this short repeatedly resident one-projection
protocol; it does not identify a causal clock/power/OS mechanism or establish
an uncertainty interval for other workloads. Small earlier single-projection
signals remain exploratory. They were already rejected by fresh model tests;
this calibration does not alter their raw data or the full-model/vLLM cohorts.

Next scheduling evidence should come from actual full-model work using the same
library/shader across thread settings, with actual route/thread counters and
all original correctness gates. This remains a plan, not an implementation or
execution result. Separately, installed SDK inspection located the public
MPSNDArrayMatrixMultiplication class, its4-D matrix discussion and double alpha/
beta properties. Mixed-storage support, original-value compatibility and speed
are unproven; no new prefill API benchmark has run. The retained matched chart
and full2x decode/half-TTFT goal remain unchanged and unmet.
