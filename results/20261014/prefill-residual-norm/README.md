# Packed Add/Layout/RMS fusion: small native latency benefit, no promotion

The prior adjacent Add/Norm trial produced no useful native decode gain and
did not cover prefill's intervening reshape. This method combines prefill's
three original dispatches while preserving all three outputs: packed residual,
reshaped residual and normalized result. The same rounded FP32 addition,
128-thread RMS reduction, learned values, epsilon and weight offset are used.
Padded packed and normalized strides remain independent. No precision, batch,
model values, output caps, image bytes or cached results change.

Initial component PID 56946 exited 0 after 89.52 seconds. It covers six trained
layer-zero normalization weights across all three models, batch 1/2 and chunks
8/16/22/26/32/64/80. All **29,730** fresh-input executions retain all three
outputs and padding bit for bit, with four input variants and independent FP64
normalization checks. Each profile has two ABBA rounds, 192 timed executions
and at least 250 ms alternating common warm-up. Setup, input copies and checks
are outside the recorded clocks.

Confirmation 58492 was interrupted by an unrelated xlang3 runtime and is
discarded. Quiet-launch retry 59605 failed before inference because its guard
lacked the `output` variable; the corrected retry 59985 was also interrupted
by an unrelated runtime. No unknown actor was excluded or stopped. Their
observations are preserved, but none contribute to speed claims. The quiet
guard and 0.5-second monitoring remain mandatory; shorter external activity
cannot be categorically excluded.

Grouped retry PID **61209** exited 0 after 434.38 seconds. Every group waits for
five quiet seconds and runs one trained norm in a fresh process. Only an
interrupted group is discarded/retried. All 12 clean groups passed: six repeat
groups for the original 84 profiles, plus six groups covering chunks 2/4/24/48.
Chunk 24 is required by the frozen text-B1 workload. The grouped result covers
132 profiles / **53,902** bitwise executions / 1,056 timing windows. Combined
with the initial run, **83,632** executions passed. Worst FP64 normalized error
is 0.00377899613 under the unchanged prior envelope.

Mean per-profile candidate/control GPU fractions by model/batch are roughly
0.44–0.52 initially and 0.43–0.55 in the expanded grouped confirmation. Wall
fractions are approximately 0.90–0.96 initially and 0.83–0.94 in the grouped
run. Exact values and individual rounds are in `summary.json`,
`confirmation.json` and `confirmation/groups`. These are component durations,
not observed whole-model TTFT or TPS. OS caches and GPU frequency are
uncontrolled; no hardware-cold claim is made.

## Optional native experiment

`GARNET_METAL_PACKED_PREFILL_NORM` defaults OFF. The matcher requires adjacent
Add -> unmasked Layout -> Norm, compatible packed inputs/output, matching
normalized layouts, batch 1/2 and a bounded chunk larger than one. It writes
all three existing output allocations. Resident execution requires all three
operations; operation profiling retains the original dispatches. Native API
validation rejects aliases, foreign batches, uninitialized inputs, bad
layouts/chunks and invalid epsilon/offset. The former decode fusion stays OFF.

Context/session counters reset on Begin/Abort/clear/reset/run and publish only
after successful GPU submission. Both adapters aggregate actual prefill
dispatches into `prefill_profile` for the optional feature. Comparisons require
the actual total to equal layers times the observed chunk count, with zero in
the control. The backend marker is `packed_prefill_norm`, distinct from the
old decode feature marker.

Twelve candidate source identities matched their branch-base hashes before
installation; exact prior bytes are retained in
`ROOT/logs/packed-prefill-norm-original-source-20261014`. The isolated candidate
has built and passed native CTest. Its tests compare all three outputs bitwise,
verify downstream availability, cover odd widths/batches/chunks/input scales
and reject bad inputs/aliases/parameters while checking counter resets.
Candidate library SHA-256 is
`9286535724998269f7c871e853f24ca74ff11a526e6d34c83c17c697faab44a3`.

Initial validation PID **68972** exited 1 after 357.47 seconds when an unrelated
xlang3 runtime interrupted the first dense-tail stage. Its dependency PID
**70597** exited 1 without running benchmarks. This is guard interruption,
not a numerical failure. Both initial jobs are terminal and must not restart.

Resumed validation PID **75785**, directory
`ROOT/logs/packed-prefill-norm-resume-checks-20261014`, reuses only the pinned
candidate build and the three clean configure/build/CTest stages. All source
and library identities are rechecked first. It executes the original eight
dense-tail fixtures individually, preserving their random sequence by consuming
the skipped draws. `fixture-rng-verification.json` verifies all 48 input hashes
at both trained hidden widths. Shapes, three stateful iterations, paused rows,
slot reuse, invalid-input tests and numerical bounds remain unchanged.
Only interrupted complete cases are discarded/retried. All other validation
stages retain their original commands and coverage, with clean-interval guards.

Resumed validation 75785 exited 0 after **1166.26 seconds**, completing all 19
stages with the full original case coverage. Dependency 78978 at
`ROOT/logs/packed-prefill-norm-resume-native-20261014` verified the actual live
validation handle, then completed all six sequential ABBA profiles after exit
0. It exited 0 after **1134.42 seconds** including waiting. Both resumed jobs
are terminal and must not be restarted.

All **72** initial complete outputs match the frozen references, including
prompt IDs and finished flags. Actual aggregate prefill counts equal layers
times chunk count: 28 for text, 24 for coder/VL B1 and 72 for VL B2, versus zero
in the control. Every VL request freshly re-encodes the same image and matches
its patch hash. The audit checks every accepted validation case, source/library
identities, actual counters and independently recomputes request timing means.

| Initial native profile | Control -> candidate row TPS | Control -> candidate warm TTFT |
| --- | ---: | ---: |
| Text B1 | 58.9778 -> 59.6246 | 52.0077 -> 51.5846 ms |
| Text B2 | 57.6740 -> 57.6098 | 85.4079 -> 86.4755 ms |
| Coder B1 | 71.9390 -> 71.7760 | 35.8599 -> 34.9205 ms |
| Coder B2 | 70.9201 -> 70.2299 | 56.2549 -> 55.8590 ms |
| VL B1 | 47.3539 -> 47.8549 | 204.9279 -> 203.7787 ms |
| VL B2 | 44.4494 -> 44.4243 | 381.4346 -> 376.6184 ms |

Prefill GPU time falls in all initial profiles, but total TTFT changes are
small and mixed. Encoding changes are mixed. The phase data are in
`native/prefill-phases.json`; full precision means and separate initialization/
first-use observations are in `native/verification.json`.

Confirmation PID **99505**, directory
`ROOT/logs/packed-prefill-norm-confirm-native-20261014`, exited 0 after 74.23
seconds. It repeats the coder-B1 and VL-B2 latency signals and text-B2's lower
GPU time/higher total latency discrepancy. All **36** additional complete
outputs and actual prefill counts match under the same sequential guards.

| Repeat profile | Control -> candidate row TPS | Control -> candidate warm TTFT |
| --- | ---: | ---: |
| Text B2 | 57.8198 -> 57.4243 | 84.2267 -> 83.2828 ms |
| Coder B1 | 71.9435 -> 71.9288 | 35.9395 -> 35.7657 ms |
| VL B2 | 44.4745 -> 44.2428 | 380.4169 -> 376.4610 ms |

Coder-B1's initial 2.62% TTFT decrease shrinks to 0.48%; text-B2's direction
changes, with a 0.68% repeat decode decrease. VL-B2 repeats a small TTFT
decrease (1.26% / 1.04%) with a 0.52% repeat decode decrease. No meaningful
decode improvement is established. The native control is the identical
attention build, while VL B2's selected baseline remains the separate primary
library, so a deployment conclusion would require a new selected-baseline
comparison. The primitive remains optional and default OFF; no promotion or
fresh vLLM comparison is made. All owned jobs in this report are terminal.
The overall 2x decode / half-TTFT goal remains unmet. Protected GPT-OSS/Pi
checkouts and selected primary/attention libraries are untouched.

## Commands and evidence

```text
python3 launch_prefill_residual_norm_probe.py ROOT ROOT/logs/prefill-residual-norm-probe-20261014
python3 launch_prefill_residual_norm_probe.py ROOT ROOT/logs/prefill-residual-norm-confirm-20261014
python3 launch_quiet_prefill_residual_norm_probe.py ROOT OUTPUT regular
python3 launch_grouped_prefill_residual_norm_probe.py ROOT ROOT/logs/prefill-residual-norm-grouped-20261014
python3 archive_prefill_residual_norm_probe.py ROOT ARCHIVE
python verify_prefill_residual_norm_probe.py RESULTS REPORT JOB_NAME
python verify_grouped_prefill_residual_norm_probe.py RESULTS REPORT
python prepare_packed_prefill_norm_sources.py
python3 install_packed_prefill_norm_sources.py ROOT
python3 launch_quiet_packed_prefill_norm_checks.py ROOT ROOT/logs/packed-prefill-norm-checks-20261014 checks
python3 queue_packed_prefill_norm_profiles.py ROOT ROOT/logs/packed-prefill-norm-native-20261014
python3 launch_resume_packed_prefill_norm_checks.py ROOT ROOT/logs/packed-prefill-norm-resume-checks-20261014
python3 queue_resume_packed_prefill_norm_profiles.py ROOT ROOT/logs/packed-prefill-norm-resume-native-20261014
```

Initial archive SHA-256:
`1c05ee8075a5768ff03ef27da61bed10f468e9d731f5263e9edb332eb00ebc5b`
(35 files). Combined archive:
`d487d7af646f37e59e862c1dde5d81c892fb4e96ec3f58712b41e6eb00ef554f`
(470 files). Verification checks file/source/shader identities, exact trained
BF16-to-FP32 expansion, terminal stages, common warming, three-output/padding/
FP64 gates and independently recomputes all timing ratios. Interrupted attempts
are retained and excluded. Native snapshots are historical. Initial native
archive SHA-256:
`88582e7382eb500f7578d8d19b28303b368741bc2bd5e32da9fb908bfc221422`
(655 files). Combined native/confirmation archive:
`ecc86079e3e2680215e27e9f09c29b3e6868d3633f2e9ad2babe6f08dcb168b8`
(816 files), stored in `native/evidence`. Independent verification covers
all 108 complete requests and actual prefill counts. OS/driver caches were
not flushed; no summed or hardware-cold latency claim is made.
