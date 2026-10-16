# Prospective null-first timing design for vector vision

Status: two fresh corrected-runner null cohorts completed, were archived and
passed independent Mac/Windows audits and all prospective calibration gates.
The new full candidate cohort PID57747 completed and passed its screen;
unchanged full confirmation PID60693 is running. Original setup/parser
failure remains preserved and excluded. No new candidate or model gain has
been established for trained/full-model inference. Overall goal remains unmet.
The [original initial cohort](../../20261015/vector-vision-context/README.md)
remains rejected: identical-kernel B1 null drifts11.8%, beyond the declared5%.
No old samples are filtered, corrected by a null ratio, or reused as new gains.
No unchanged favorable repeat of the failed cohort is admitted.

Observed old null GPU window means fall from1.58 to0.79ms. Frequency/power
causality was not instrumented. Test a different timing design prospectively:
fixed2s common alternating conditioning per shape (old0.25s), ten ABBA rounds
with five samples per position (old two rounds/twenty). Retain every measured
sample and the original three warm calls per window. These changes reduce
window duration and extend fixed conditioning; they do not guarantee stability.
The scalar/vector shaders, inputs, shapes, four changed-input generations,
output/padding bit checks and original FP64 bounds stay identical.

Before candidate gains: run only variant0, the same scalar pipeline object
and identical geometry in both arms, through all24 original shapes. Require
both GPU and wall null fractions in[0.95,1.05] at BOTH serving batches
(216 keys,12 heads,64 dimensions) in two fresh unchanged null-only cohorts.
Only if the first passes is its one confirmation admitted. A failure stops
this design, with no repeated attempts to find favorable calibration.
No vector candidate executes during those cohorts, though its shader may be
compiled in the shared library. Timing sample count is4800 per null cohort.

Only after both null cohorts pass, admit a new complete three-policy candidate
cohort under this exact design, with all72 original numerical cases. Keep the
old gain rule: all24 cases per vector width pass; at BOTH serving batches,
GPU AND wall fractions<=0.95 with own null within5%. Initial failure stops;
initial qualification admits one unchanged fresh full confirmation. Both must
qualify. If both widths qualify, choose lower geometric-mean serving wall
fraction across both cohorts; exact tie chooses float2. No native/full-model
or vLLM claim follows from synthetic component qualification.

The old cohort is used only to motivate a measurement-design hypothesis,
not to select a vector width. This is a new prospective study, not a repair
of old ratios or a tolerance relaxation. Any native candidate still requires
trained vision, full original validation, complete native output/latency/memory
cohorts and fresh repeated matched vLLM tests before goal completion.

```text
python3 launch_calibrated_vector_vision.py ROOT OUTPUT calibration
python3 archive_calibrated_vector_vision.py ROOT ARCHIVE TRIAL
python -X utf8 test2026/metal/verify_calibrated_vector_vision.py EXTRACTED_RESULTS AUDIT_JSON
python -X utf8 test2026/metal/analyze_calibrated_vector_null.py RESULT_ROOT calibration-initial
# Candidate mode requires the separately audited two-cohort null decision.
python3 launch_calibrated_vector_vision.py ROOT OUTPUT candidate
```

## Preserved parser failure and fresh setup

PID54090 exits1 in61.967176416s after the compiler and variant0 GPU process
both exit0. The binary generated all24 original exact-output/FP64 checks and
4800 measured submissions. The Python summary parser still asserted the old
eight windows/twenty samples, while the predeclared new binary generated40
windows/five samples. It stops at that assertion and writes no summary.
This is a harness error, not a failed numerical check or an admitted null ratio.

Freeze32-file archive before editing any consumed source, SHA256
`7af5fad7078e3869d313bf12d2725b802d19969d01557feb6415a2f2219ac409`.
Mac/Windows [failure audits](parser-failure/failure-verification.json) regenerate
the exact shader/harness, verify every raw original numerical record, all
4800 positive finite samples and uninterrupted GPU-stage guards. They compute
no calibration ratios. [Manifest](parser-failure/evidence/manifest.json).

Repair only summary window/round/sample-count assertions (40/10/5). All GPU
shader and harness bytes remain identical, verified locally against the frozen
failure. Numerical bounds, conditioning, samples and prospective5% gates are
unchanged. Exclude all failed-setup measurements from selection; require two
fresh unchanged corrected-runner null cohorts. No favorable repeat is chosen
from those excluded timings. Fresh initial PID55356 uses
`logs/vector-vision-calibration-repaired-20261016`; its result follows.

## Actual independent null calibration

Initial PID55356 exits0 in61.974344042s; confirmation PID56568 exits0 in
62.002792667s. Each archive has33 files. Both-host audits verify all24 original
shape checks, no numerical failures and4800 timed submissions per cohort.
Only variant0 executes; all output/padding bits match. Complete captured sources,
generated shader/harness, variants, hardware, versions and timing design match
between cohorts. All samples remain; no old failed-setup measurements are reused.

| Cohort | Batch | Null GPU fraction | Null wall fraction |
| --- | ---: | ---: | ---: |
| calibration-initial | 1 | 0.978676 | 0.970952 |
| calibration-initial | 2 | 1.003490 | 1.002114 |
| calibration-repeat | 1 | 1.003274 | 1.004099 |
| calibration-repeat | 2 | 0.997444 | 1.000960 |

Both serving batches pass GPU ANDwall within5% in both fresh cohorts, admitting
new candidate collection. This is measurement calibration, not an algorithmic
gain or proof of hardware frequency stability.
[Initial audit](calibration-initial/verification.json),
[confirmation audit](calibration-repeat/verification.json),
[calibration decision](calibration-decision.json).

Initial archive SHA256:
`a0dc7f6577389435d1fd4a1a0ce25f748b4880e51bd3cb95132f78925f46f208`.
Confirmation archive SHA256:
`8ca90d1b52006c687973c59f767c06e009e82f74f4f6d22960f87e89d41539ae`.

After both terminal/archive/audits, dispatch full three-policy candidate cohort
PID57747 at `logs/vector-vision-calibrated-20261016`, with original numerical
gates and prospective timing/gain/null rules above. Its result follows; no
width is selected before an unchanged full confirmation.

## Initial candidate cohort passes; confirmation pending

PID57747 exits0 in169.318523958s. Freeze37-file archive, SHA256
`bcdc7fd237ff515d56604ef89200cecd5cffaff44358d60bf26d0e282f6e5fec`.
Mac/Windows regeneration/raw audits verify72 original numerical records, zero
failures, all output/padding bits and unchanged FP64 bounds, and14,400 timed
submissions. The prospective2s/10x5 design is recorded in metadata and audited.
All profiles and samples, including non-serving shapes, remain available.
[Initial audit](initial/verification.json), [decision](decision-initial.json).

| Width | Batch | GPU fraction | Wall fraction | Own-null GPU fraction | Own-null wall fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 0.632454 | 0.729740 | 1.010532 | 1.008367 |
| 2 | 2 | 0.645435 | 0.693187 | 1.001815 | 0.998157 |
| 4 | 1 | 0.615212 | 0.731072 | 1.010532 | 1.008367 |
| 4 | 2 | 0.602147 | 0.649052 | 1.001815 | 0.998157 |

Both widths pass the original gain/null rule in this new initial cohort:
both serving batches improve GPU ANDwall by>=5%, with own-null within5% and
every original numerical shape passing. Admit one unchanged complete
three-policy confirmation, PID60693 at
`logs/vector-vision-calibrated-confirm-20261016`. It is running and unproven.
No width chosen, native integration, complete-model TTFT or fresh vLLM gain
is established by this initial synthetic result.
