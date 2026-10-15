# Native guarded greedy-head screening

Status: native validation PID 91274 exited 0 in 1219.3464 s. The original
19 stages, screening CTest and added 51 trained-head queries passed. Queued
PID 3415 exited 0 in 760.4973 s including its dependency wait, six trained
B1/B2 session/recovery cases and all six full native inference profiles.
Independent audit covers 1396 files and 72 complete matching requests.
Confirmation PID 17037 exited 0 in 152.4481 s. Independent audits now cover
1696 files and 144 complete matching requests across both trials.
No fresh vLLM ratio is established yet. The full goal remains unmet.

`GARNET_METAL_SCREENED_HEAD` defaults OFF. An eight-bit derived row index
screens vocabulary candidates; retained scores use unchanged original BF16
weights and FP32 accumulation. Public complete logits retain the full head.
Unsafe query magnitudes/nonfinite values use full original refinement.
The component method, limits and repeated head-only observations are recorded
in [the component study](../screened-head/README.md).

Native validation additionally checks padded strides, lowest-index ties with
the winner outside the seed, mixed safe/unsafe rows, pending/aborted outputs,
foreign ownership and unsupported batch fallback. Actual session tests check
full-logit parity, float-head fallback, reset, private state, nonfinite poison
and recovery, close/rebuild and actual screening counts.

Full inference compares the selected global-alpha control with the optional
screened-head build in sequential ABBA workers. It requires frozen complete
token sequences, prompt IDs, stop flags and one actual screened dispatch after
prefill and each decode call. Both VL policies use the same combined threaded
decode and exact-coordinate strip resize, with fresh image encodings. There
is no answer/prefix/image reuse or speculation. Initializations and first
requests after initialization remain separate from warm means; OS and driver
caches are not flushed. Peak individual worker RSS is recorded separately.

## Initial full-model native ABBA

Four warm requests per policy/profile, excluding each worker's first request.
TPS is per batch row; batch-2 aggregate throughput is twice the listed TPS.
These are contemporary Garnet-versus-Garnet measurements. Do not combine
them with an older vLLM cohort to claim a fresh matched speed ratio.

| Model / batch | Control / screened TPS | Control / screened TTFT ms | Decode gain | TTFT change |
| --- | ---: | ---: | ---: | ---: |
| Text / 1 | 58.3748 / 64.4276 | 49.7274 / 45.5462 | +10.37% | -8.41% |
| Text / 2 | 57.8400 / 62.1763 | 80.6881 / 76.6020 | +7.50% | -5.06% |
| Coder / 1 | 72.3976 / 78.6663 | 32.6951 / 31.9066 | +8.66% | -2.41% |
| Coder / 2 | 70.5932 / 76.5865 | 51.4565 / 48.5997 | +8.49% | -5.55% |
| VL / 1 | 48.0703 / 52.2473 | 189.3121 / 181.7598 | +8.69% | -3.99% |
| VL / 2 | 44.2529 / 48.3005 | 347.4341 / 347.8004 | +9.15% | +0.11% |

Index construction adds about 0.42–0.77 s to initialization in this trial.
First-request TTFT after initialization is mixed and remains separate in
`verification.json`; the warm gains do not establish a cold-latency win.
Full source snapshots, accepted gate attempts, trained head/session checks,
commands, guards, complete outputs and immutable-asset hashes are preserved.
Initial archive SHA-256:
`d5782545e86492ac0ad7e3aa349561702b24928560d9790b75acadb2a611c7e4`.

## Fresh-worker confirmation

| Model / batch | Screened TPS | Screened TTFT ms | Decode gain | TTFT change |
| --- | ---: | ---: | ---: | ---: |
| Text / 1 | 64.7219 | 44.7997 | +10.14% | -8.82% |
| Text / 2 | 62.5544 | 77.0444 | +8.16% | -6.21% |
| Coder / 1 | 78.9069 | 31.4460 | +9.59% | -6.95% |
| Coder / 2 | 76.8147 | 48.4098 | +8.62% | -4.98% |
| VL / 1 | 52.5571 | 178.8665 | +9.99% | -2.42% |
| VL / 2 | 48.1895 | 347.8380 | +8.85% | -0.29% |

All decode gains repeat. Five warm TTFT improvements repeat; VL B2 remains
essentially flat. This still does not demonstrate the agreed vLLM targets.
Peak individual RSS and observed first-request/init values are independently
recomputed from the raw worker logs in the two verification JSON files.
Combined native archive SHA-256:
`d0e224b893277ee368eb8b3f1181879460f932fe7e176fdc363939f18055f5ca`.
Validated candidate dylib SHA-256:
`2b44d93ab1e1980845be89ea59cbaedbebcb66ad385ba24df3d52af2b6bacc23`.

Fresh matched comparator PID 20154 exited 0 after all six sequential
comparisons and three actual GPU FP32 dtype audits. Its separate pinned
selection and common combined CPU frontend apply equally to both runtimes.
An independent audit checks 507 files and 72 complete matching outputs.
[The fresh comparison](../screened-head-vllm/README.md) records four initial
decode-threshold passes and every remaining target. The option remains OFF;
no general promotion or full-goal completion is implied.

```text
python3 launch_screened_head_checks.py ROOT ROOT/logs/screened-head-checks-20261015 checks
python3 queue_screened_head_profiles.py ROOT ROOT/logs/screened-head-native-20261015
python3 archive_screened_head_inference.py ROOT ARCHIVE
python -X utf8 verify_screened_head_inference.py EVIDENCE VERIFICATION_JSON
python3 launch_confirm_screened_head_profiles.py ROOT ROOT/logs/screened-head-confirm-native-20261015
```
