# Per-matrix two-output probe: three dispatch candidates selected

Both cohorts completed and passed independent numerical/packing/timing audits.
Initial PID40855 exited0 in31.793750s; confirmation PID41602 exited0 in31.822521s.
Each verifies20 originally selected matrices,480 mandatory exact-bit plus
unchanged FP64/native-envelope records,189 safety profiles and7,680 timed
executions. All29 original trained assets remain covered by the packing proof;
only the20 originally admitted matrices are timed individually. The original
CPU packing proof is reused, not independently replicated.

The runtime sources, generated context/header/harness, compiler flags, assets,
hardware/runtime versions and binary match across cohorts. Initial56-file
archive SHA256:
`022c70ae014a49126effbaf126ac512bfa576901ce48c390504c6ee44463a373`.
Confirmation56-file archive SHA256:
`d26e23980ee67d443453039c7a543e9fddbe77421707f3f9f9f8c6a274a61cc6`.
[Initial audit](initial/verification.json),
[confirmation audit](confirmation/verification.json),
[joined selection and all observations](confirmation-comparison.json).

Selection rule: require at least2% lower GPU time and lower wall time in
every observed matrix/cohort for a geometry. Across eligible thread policies,
choose the lowest worst GPU-time fraction. Retain the original four-output
kernel for all other geometries. This rule was applied after examining the
probe data: it is selection/training evidence and needs fresh confirmation.

| Physical rows | Output N | Inner K | Threads | Worst GPU-time fraction | Worst wall-time fraction |
|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 1024 | 64 | 0.977068 | 0.991750 |
| 2 | 896 | 4864 | 64 | 0.977937 | 0.987545 |
| 2 | 6144 | 1024 | 128 | 0.939315 | 0.984252 |

The probe keeps all original weights resident but repeatedly submits only one
matrix. Cache residency and per-submission costs differ from real decode.
Do not multiply these gains across layers, treat samples as independent
devices, or claim model speed/TTFT/vLLM improvements from this table.

```text
python3 launch_two_matrix_probe.py ROOT ROOT/logs/two-matrix-probe-20261015 component
python3 archive_two_matrix_probe.py ROOT ARCHIVE [TRIAL]
python -X utf8 test2026/metal/verify_two_matrix_probe.py RESULTS RESULTS/verification.json
python -X utf8 test2026/metal/compare_two_matrix_confirmation.py REPORT REPORT/confirmation-comparison.json
```

ROOT is /Users/zsdx/garnet-metal-investigation. Exact invocations/source are
archived. A separate shape-selected source copy now runs fresh28 original
validation stages and extends the native coalesced primitive from80 to104
cases, retaining the original cases. Owned validation PID42435 is running;
no full-model gain is established. The graph route remains disabled, original
four-output control remains available and the matched target remains unmet.
