# Residual addition plus RMS normalization: no useful native decode gain

This experiment combines two adjacent dispatches while preserving both the
rounded FP32 residual and normalized output. The original 128-thread RMS
reduction, learned weights, epsilon, weight offset and float operations remain
the same. A device/threadgroup barrier makes the rounded residual available to
normalization. There is no reduced precision, cached result or changed batch.

The first setup job (PID 33876) exited 1 before timing because the checkpoint
selector also matched Qwen3.5's MTP tensors. Restricting selection to `model.`
fixed it. The corrected job 33973 and fresh-process confirmation 34552 exited
0 after 32.35 / 32.39 seconds. Each covers six canonical trained layer-zero
normalization weights across three models, with rows 1, 2, 16, 32 and 64.

All **42,588** validated executions across both successful runs match both
original outputs and padding bit for bit, using four freshly uploaded input
variants. Independent FP64 normalization checks have worst normalized error
0.00351983185 under the unchanged 5e-6 + 5e-5*abs(reference) envelope. Each run
has 360 ABBA timing windows and 8,640 timed executions, with a shared alternating
warm loop of at least 250 ms per profile. Input copying and validation are
outside the recorded timing. Sources and raw observations are frozen locally.

GPU duration is lower in every pooled component profile in both runs; the
confirmation reductions range approximately 4–48%. Wall duration is mixed,
including regressions; isolated command submission overhead is substantial.
These are component measurements, not observed whole-model TPS or TTFT.
Full precision means and individual rounds are in `summary.json` and
`confirmation.json`. OS caches and GPU frequency are uncontrolled, and the
0.5-second process guard cannot exclude arbitrary shorter external activity.

## Optional native trial

`GARNET_METAL_RESIDUAL_NORM` defaults OFF. Graph encoding admits only adjacent
Add -> Norm operations with identical activation layouts, and writes both
existing output allocations. Resident execution requires both operations to
be selected. Profiling mode retains separate operations. Inputs must be ready
or produced in the current batch; output/input aliases, output/output aliases,
invalid epsilon, incompatible layouts and foreign batches are rejected.

The current matcher covers decode's adjacent Add -> Norm. Prefill adapters
reshape the packed residual through an explicit Layout operation, giving
Add -> Layout -> Norm; that sequence is not admitted by this candidate.
The component's prefill-sized speed reductions therefore cannot be attributed
to this native prefill path. Supporting and validating all three operations
together is a separate follow-up needed to pursue the remaining TTFT targets.

Actual dispatch counters reset on Begin/Abort and session clear/reset/run;
the session publishes them only after successful GPU submission. Worker logs
record the feature marker and last-session count after the timed request.
Native CTest covers bitwise outputs, downstream consumers, both offsets,
odd/tail widths, four input scales, batches/prefill rows, rejected aliases,
uninitialized inputs, invalid parameters/layouts and counter reset.

The isolated candidate has built and passed CTest. Owned numerical validation
uses library SHA-256
`409246a7ab8e069d048177eb490fdf34cd354c8f4e2cb4b718ceead32c49d2ce`.
Snapshots in `native/pending` are historical observations, not terminal proof.
Owned numerical validation PID **35350**, directory
`ROOT/logs/residual-norm-checks-20261014`, exited 0 after 1007.65 seconds with
all 19 model/vision/cache/prefill stages passed. Dependency PID **36987**,
directory `ROOT/logs/residual-norm-native-20261014`, verified that exact live
validation process, then completed all six sequential ABBA profiles after
validation succeeded. It exited 0 after 1073.79 seconds including waiting.
Both jobs are terminal and must not be restarted.

All **72** complete generated outputs match their frozen references, including
prompt IDs and finished flags. Actual last-session fusion counts are 28 for
text and 24 for coder/VL in both batches, versus zero in the control. Every
image request freshly re-encodes the same image and matches its patch hash.

| Native profile | Control -> candidate row TPS | Control -> candidate warm TTFT |
| --- | ---: | ---: |
| Text B1 | 58.8862 -> 58.5137 | 52.9490 -> 51.1108 ms |
| Text B2 | 57.4610 -> 57.3060 | 86.6573 -> 87.3352 ms |
| Coder B1 | 71.7284 -> 71.8605 | 35.5821 -> 36.2782 ms |
| Coder B2 | 70.5903 -> 69.7342 | 56.7633 -> 59.6107 ms |
| VL B1 | 47.3199 -> 46.7115 | 215.4341 -> 210.2973 ms |
| VL B2 | 43.8902 -> 43.5855 | 385.1809 -> 385.9222 ms |

Full precision values are authoritative in `native/verification.json`. Decode
is flat or worse across the profiles. The small coder-B1 increase does not
establish a useful gain; favorable individual TTFT changes are unconfirmed and
mixed, with no admitted prefill fusion. No unchanged-candidate repeat or runtime
promotion is justified. Dense decode GPU phase means are also flat/worse, while
encoding changes are mixed; `native/decode-phases.json` records these separately.
The optional primitive stays default OFF for follow-up investigations.

Candidate sources were installed only after all ten prior hashes matched,
with prior bytes preserved in `ROOT/logs/residual-norm-original-source-20261014`.
Protected GPT-OSS/Pi checkouts are clean and untouched. Selected primary and
attention libraries remain unchanged. No runtime promotion or fresh vLLM
comparison is claimed. The 2x decode / half first-token latency goal is unmet.

## Commands and evidence

```text
python3 launch_residual_norm_probe.py ROOT ROOT/logs/residual-norm-probe-20261014
python3 launch_residual_norm_probe.py ROOT ROOT/logs/residual-norm-retry-20261014
python3 launch_residual_norm_probe.py ROOT ROOT/logs/residual-norm-confirm-20261014
python3 archive_residual_norm_probe.py ROOT ARCHIVE
python verify_residual_norm_probe.py RESULTS_DIRECTORY REPORT_JSON SUCCESSFUL_JOB_NAME
python prepare_residual_norm_sources.py
python3 install_residual_norm_sources.py ROOT
python3 launch_residual_norm_checks.py ROOT ROOT/logs/residual-norm-checks-20261014
python3 queue_residual_norm_profiles.py ROOT ROOT/logs/residual-norm-native-20261014
```

Initial archive SHA-256:
`677f55fed27d5df5c2b846fd40d3add60da30e1f27fc88a978285567414acb85`
(41 files). Combined archive:
`63e40587f9666045cf5bc9f8eda39fe4ba2fc0257edea9e117cb20b12b5e81f9`
(64 files). Local verification checks all hashes, executed sources, exact
BF16-to-FP32 learned-value expansion, shader identity, terminal stages,
both-output/padding/FP64 gates and independently recomputes timing ratios.
Full native raw evidence is archived in `native/evidence`. Native archive SHA-256:
`c9e9d1f1e5b44335f06a41dbef93bd732f68f833e4c552cb2750f5ace1189dbb`
(450 files). Initialization and first request after initialization are separate
observations in the derived verifier report; OS/driver caches were not flushed.
`archive_residual_norm_native.py` refuses live or failed jobs, pins candidate
and control libraries and preserves executed source identities and prior bytes.
`verify_residual_norm_native.py` checks all six profiles, 72 full outputs,
actual last-session dispatch counts, fresh VL encodings and independently
recomputes warm means, aggregate throughput and first-use latency after
initialization. Its result explicitly retains the unmet overall goal.
