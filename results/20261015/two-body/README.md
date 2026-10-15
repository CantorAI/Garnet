# Two-output SIMD decode: repeated component signals, no model gain established

Hypothesis: reduce per-SIMD accumulator work while preserving the existing
compact four-output storage. Two adjacent SIMDs select the lower/upper half
of the same original quad. BF16 learned bits, 32-lane K partition, explicit
FP32 FMA and original fallback/prefill paths remain unchanged. Private copies
add actual route counters; there is no precision lowering or weight repacking.

Initial PID39399 exited0 in20.580033s; confirmation PID39933 exited0
in19.949173s. Each audited cohort passed72 mandatory exact-bit/FP64/native
envelope records and189 safety profiles before1,152 timed ABBA samples.
Coverage includes all29 original projection assets, batch1/2, four input
patterns and64/128/256-thread policies. Binary, runtime source, flags, assets
and versions match across cohorts; binary SHA256:
`2c091311f4a419f23cec945840535a5239be09e2030768c334d4997d0efeb541`.

Six of18 policies improved wall time in both runs; seven improved GPU time;
five improved both directions in both runs. Several gains are very small and
there is no formal confidence estimate. Text B1 mostly regressed. Text B2's
64-thread wall gain had an inconsistent GPU direction, so it is not a joint
gain. This motivates individual-shape selection instead of one global policy.

The exact initial archive SHA256 is
`b18dd030eccd5e0a6818611f7271daa4b95d050e488ee49facbe547bafe48004`;
confirmation SHA256 is
`495b6a231a98df78f09a3607e89f1ec29733d579a54e28471f0974c692bbe60c`.
Each contains54 files. [Initial numerical audit](initial/verification.json),
[confirmation numerical audit](confirmation/verification.json),
[joined observations](confirmation-comparison.json).

The initial verifier's descriptive scope text said 'eight-route' despite
asserting the two-route counters. The description was corrected and both
audits rerun; no actual numerical predicate changed. All jobs were terminal
and archived before their inputs could change. Original CPU packing proof is
reused, not independently repeated. These are repeated resident component
measurements, not full-model or vLLM performance results.

```text
python3 launch_two_body.py ROOT ROOT/logs/two-body-20261015 component
python3 archive_two_body.py ROOT ARCHIVE [TRIAL]
python -X utf8 test2026/metal/verify_two_body.py RESULTS RESULTS/verification.json
python -X utf8 test2026/metal/compare_body_probe_confirmation.py REPORT REPORT/confirmation-comparison.json
```

ROOT is /Users/zsdx/garnet-metal-investigation. Exact executed commands and
source snapshots are in the archives. [Per-matrix follow-up](../two-matrix-probe/README.md)
selects candidate shapes for fresh model validation. Full goal remains unmet.
