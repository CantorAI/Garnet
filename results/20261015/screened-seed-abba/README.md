# Direct exact-seed comparison and native candidate

Two actual same-process trials compare the original 8192-row screened-head
seed against a 1024-row seed. Both use original BF16/FP32 exact seed scores,
the same strict-math INT8-derived filtering index, unchanged error allowances
and original BF16/FP32 refinement. All captured predictions match. The smaller
seed reduces component GPU/wall time in all six ordinary model/batch groups
in both trials; the VL B1 improvement is small and variable. This is a private
greedy-head experiment, not a replacement quantized model.

The complete raw full-logit kernel runs before admission and matches captured
original GPU bits. Both seed policies then check every retained score and every
rejected score against the original full logits, and check greedy/error output.
The candidate additionally executes the existing zero/tiny/large/NaN/Inf/mixed
unsafe-input fallback checks. No numerical allowances were widened. These are
sampled learned-query checks, not certification for all future inputs.

Initial PID 97810 exited 0 in 38.1126 s; repeat PID 98647 exited 0 in 38.1690 s.
The independent audit checks 80 archived files, exact generated code and input
hashes, 102 captured-query pairs (156 row states per policy), 66 candidate
unsafe-input checks, 6,528 raw timed executions, guards and both ABBA rounds.
Each query uses common warming for at least 250 ms, then two ABBA rounds of
eight samples per window. Both timed policies include seed projection,
threshold/norm preparation, screening/refinement and the original argmax in
one command buffer. The timing control is the 8192-row screening policy.

The following means average each ordinary captured step's candidate/control
fraction, excluding the separate mixed-row fixture. Lower is better.

| Model / batch | GPU fraction initial / repeat | Wall fraction initial / repeat |
| --- | ---: | ---: |
| Qwen3-0.6B / 1 | 0.95967 / 0.96025 | 0.96338 / 0.96418 |
| Qwen3-0.6B / 2 | 0.96129 / 0.95863 | 0.96533 / 0.96340 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | 0.96693 / 0.96227 | 0.97613 / 0.97124 |
| Qwen2.5-Coder-0.5B-Instruct / 2 | 0.95636 / 0.95870 | 0.96680 / 0.97422 |
| Qwen3.5-0.8B / 1 | 0.97696 / 0.99254 | 0.97714 / 0.99265 |
| Qwen3.5-0.8B / 2 | 0.96117 / 0.96923 | 0.96489 / 0.96898 |

## Optional native integration under validation

`GARNET_METAL_SMALL_SCREEN_SEED` defaults OFF and requires SCREENED_HEAD.
It changes the exact seed allocation/dispatch to 1024 rows and supplies that
same constant to the strict Metal compiler. The resulting shader functions
match the audited 1024 component byte for byte after macro substitution.
Original head-shape/batch admission and fallback policies remain unchanged.
The native context records the actual seed dispatch width. Decoder sessions
publish it after successful submission, clear it on run/reset/close/abort,
and expose it for per-prediction benchmark checks. Legacy 8192 controls report
their frozen configured width; the new candidate reports the actual counter.

Synthetic and trained tests compare rejected scores against the smaller actual
seed threshold, rather than the old 8192-row threshold. They verify actual
seed width, retained bits, outside-seed lowest ties and recovery. Additional
trained session checks observe the new property through generic-logit and
float-head fallback, nonfinite poison, reset, private sessions, close and rebuild.

All previous native trials were terminal and archived before source changes.
Installation pins 18 files and preserves 13 verified existing originals from
base `09893ade4d80d39e602000b0e716ff878f82c44d`. An initial staging attempt
found an absent generator script and stopped before extraction. It was
reclassified as a new file; the existing partial backup was verified and
reused without deleting any files. All installed hashes then matched.

Validation PID 99758 subsequently passed all 27 stages. Native PID 294 and
fresh confirmation PID 26793 both completed all six profiles. Independent
audits verify 144 complete outputs and actual seed widths. The
[native measurements](../small-seed-native/README.md) show small, mixed gains:
text B2 decode improves 1.9–2.4%, while text B1 TTFT and VL B2 TTFT regress
twice. The optional policy remains OFF by default. These component ratios
are not whole-model gains; no new vLLM ratio is claimed and the full goal
remains unmet.

Initial archive SHA-256:
`1ecc41c4d43892e913c328bf6f17fa138b598ea763dd56457488d7af1f0a63eb`.
Combined archive SHA-256:
`57e4fcd7965a19d8c449d68b77fff08745202f5410f4bc3bb31052a95ffda0f9`.

```text
python3 launch_screened_seed_abba.py ROOT ROOT/logs/screened-seed-abba-20261015 checks
python3 launch_screened_seed_abba.py ROOT ROOT/logs/screened-seed-abba-confirm-20261015 checks
python3 archive_screened_seed_abba.py ROOT ARCHIVE screened-seed-abba-20261015 screened-seed-abba-confirm-20261015
python -X utf8 verify_screened_seed_abba.py RESULTS VERIFICATION_JSON
python3 launch_small_seed_checks.py ROOT ROOT/logs/small-seed-checks-20261015 checks
python3 queue_small_seed_profiles.py ROOT ROOT/logs/small-seed-native-20261015
```
