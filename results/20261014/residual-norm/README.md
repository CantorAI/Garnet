# Residual addition plus RMS normalization: native validation running

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
Owned numerical validation
PID **35350** is running at `ROOT/logs/residual-norm-checks-20261014`. The latest
snapshot records completed configure/build/CTest, both dense exact-tail suites,
all three private-KV suites, serving and mixed-image-pipeline checks. Qwen3.5
layer validation is running; all 19 model/vision/cache/prefill stages must pass.
Owned dependency PID **36987** at
`ROOT/logs/residual-norm-native-20261014` verifies that exact live validation
process before waiting, then starts six sequential native ABBA profiles only
after validation exits 0. It requires full generated-output matches and
positive actual candidate counters. These are running jobs, not completed
speed results; inspect their current authoritative status before continuing.

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
Full native raw evidence will be archived after the jobs become terminal.
`archive_residual_norm_native.py` refuses live or failed jobs, pins candidate
and control libraries and preserves executed source identities and prior bytes.
`verify_residual_norm_native.py` checks all six profiles, 72 full outputs,
actual last-session dispatch counts, fresh VL encodings and independently
recomputes warm means, aggregate throughput and first-use latency after
initialization. Its result explicitly retains the unmet overall goal.
