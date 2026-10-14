# Global lossless half scale through MPS alpha: trained components pass, native validation running

Native exact-half prefill currently chooses each output row's lossless scale,
runs mixed-storage MPS multiplication, then launches a separate rescale kernel.
This investigation asks whether one exact common matrix scale can place the
inverse power of two in MPS's alpha coefficient, avoiding the temporary and
rescale dispatch. Learned values and FP32 activations/output remain the same.
The optional native experiment described below now implements this route.

A CPU-only audit of 22 sampled layer-zero body matrices finds 20 with a common
exact scale: text 7/7, coder 7/7 and VL 6/8. Every accepted encoded value restores
the original BF16-expanded FP32 bits exactly. VL in_proj_qkv and in_proj_z have
unrepresentable values under the common scale and must retain fallback paths.
This is eligibility evidence, not GPU model compatibility or performance.

Owned toy gate PID 5180 exited 0 after 3.67 seconds on Apple M4. Real execution
uses FP32 inputs, exact scaled FP16 weights and FP32 output with alpha 2^-16,
without a separate rescale. Shapes (M/K/N) 3/7/5 and 16/1024/64 pass independent
FP64 checks under 5e-6 + 5e-5*abs(reference); maximum absolute errors are
4.38558e-9 and 2.45605e-7. The weights round-trip bit for bit. No trained GPU
projection, native full-model inference or speed claim is established yet.

## Trained projection evidence

Initial projection PID **6347** and fresh-process confirmation **9473** exited
0 after 152.88 / 148.48 seconds. Each covers 20 admitted trained body matrices
at 13 row counts: 8/16/22/24/26/32/44/48/52/64/80/128/160. Two VL matrices are
excluded for inexact common-scale storage. Every accepted weight restores the
original learned bits. All output elements and padding are checked with four
fresh power-of-two input variants, alternating warming of at least 250 ms and
two ABBA rounds per profile. Full independent FP64 projections verify both
routes under the unchanged 5e-6 + 5e-5*abs(reference) envelope.

All **212,350** validated executions across both successful runs have zero
numerical difference from the row-scaled control. This is numerical equality,
not a signed-zero bit identity claim. Each run contains 260 checked profiles /
2,080 timing windows. Worst normalized candidate FP64 error is 0.514653261,
below 1. Setup, uploads and validation are outside the recorded clocks.

| Model | Mean per-profile GPU fraction, initial / repeat | Wall fraction, initial / repeat |
| --- | ---: | ---: |
| Text | 0.8189 / 0.8142 | 0.9357 / 0.9298 |
| Coder | 0.8044 / 0.8177 | 0.9428 / 0.9544 |
| VL | 0.8380 / 0.8617 | 0.9406 / 0.9527 |

Encoding duration increases roughly 8%; GPU and wall durations decrease.
These are component clocks, not whole-model TPS or TTFT. Full precision data
and excluded cases are in `projections/summary.json` and
`projections/confirmation.json`. OS caches/GPU frequency are uncontrolled, and
0.5-second guarding cannot exclude arbitrary shorter external activity.

## Optional native experiment

`GARNET_METAL_GLOBAL_HALF_ALPHA` defaults OFF and requires exact-half prefill.
Only modest body matrices with entirely exact common-scale encoding use it.
The existing raw BF16 decode buffer stays intact. Inexact common-scale matrices
retain the prior per-row representation or existing FP32 fallback. The route
uses the existing half buffer and writes directly to the FP32 output, avoiding
the explicit rescale encoder and intermediate result matrix. Internal MPS
dispatch counts are not claimed. The MPS cache key includes the common scale
as well as shape, preventing reuse of a product with an incorrect alpha.

Counters reset on Begin/Abort/session reset/clear/run and publish after a
successful GPU submission. Adapters record aggregate actual prefill dispatches
only when the backend marker is present. Benchmark comparisons require positive
actual candidate counts and zero in the control. Other experimental norm/head/
joined/reset flags remain OFF in this candidate.

The twelve prior source hashes matched before installation; exact originals
are preserved at `ROOT/logs/global-half-alpha-original-source-20261014`.
The isolated candidate builds and passes native CTest, including differently
scaled same-shape cache entries, independent FP64 checks, counter resets and
common-scale rejection with lossless row-scale fallback. Library SHA-256 is
`887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f`.

Owned validation PID **12334** is live at
`ROOT/logs/global-half-alpha-checks-20261014`, executing all 19 original
model/cache/vision/prefill stages. Complete tail fixtures retain their original
random sequence and coverage and retry only on guarded contention. Dependency
PID **15843**, directory `ROOT/logs/global-half-alpha-native-20261014`, verifies
that exact live handle before waiting and starts all six sequential matched
native profiles only after validation exits 0. Inspect current authoritative
state before continuing. Historical snapshots are in `native/pending`.
No whole-model speed gain or runtime promotion is claimed. The unchanged
overall 2x decode / half-TTFT goal remains unmet; selected libraries and
protected checkouts remain untouched.

```text
python3 audit_global_half_scale.py ROOT REPORT_JSON
python3 launch_mps_global_alpha_probe.py ROOT ROOT/logs/mps-global-alpha-toy-20261014
python3 archive_mps_global_alpha_toy.py ROOT ARCHIVE
python verify_mps_global_alpha_toy.py RESULTS VERIFICATION_JSON
python3 launch_global_half_alpha_projection.py ROOT OUTPUT
python3 archive_global_half_alpha_projection.py ROOT ARCHIVE
python verify_global_half_alpha_projection.py RESULTS REPORT [JOB_NAME]
python prepare_global_half_alpha_sources.py
python3 install_global_half_alpha_sources.py ROOT
python3 launch_global_half_alpha_checks.py ROOT ROOT/logs/global-half-alpha-checks-20261014 checks
python3 queue_global_half_alpha_profiles.py ROOT ROOT/logs/global-half-alpha-native-20261014
```

Archive SHA-256:
`23c3283c17056fea76d4f1e21d2642751264900cf776261c9c62a037956021bb`
(24 files). The local verifier checks all hashes, executed sources, terminal
stages and the actual toy output gates. Feasibility/compatibility evidence
does not establish end-to-end latency or throughput.

Initial trained projection archive:
`55c744fa3ebb05ea8e605580efaa097e63a13c2bd3005cf534e2c3deb3c69aed`
(26 files). Combined trained archive:
`c0061d7d9591621169a738afb766b4be745da089950dcf7d23dc83216234217b`
(48 files plus 22 canonical matrix proofs). Local verification checks every
hash, executed sources, learned-matrix identities, complete output/padding/F64
gates, common warming, sample counts and independently recomputed timings.
