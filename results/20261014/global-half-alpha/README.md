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

Validation PID **12334** completed successfully in 1203.47 seconds. All 19
model/cache/vision/prefill stages passed, including all eight complete dense-tail
fixtures per dense model with their original random sequence and coverage.
Dependent benchmark PID **15843** completed all six sequential native profiles.
The local verifier checked 641 archived files, all 72 complete output requests,
actual positive global-alpha counters versus zero control counters, source and
library identities, contention monitors, fresh image preprocessing and independently
recomputed warm means. Historical pending snapshots remain in `native/pending`.

| Model / batch | Control -> candidate decode tokens/s per row | Control -> candidate warm TTFT ms | Warm TTFT reduction |
| --- | ---: | ---: | ---: |
| Text / 1 | 59.0555 -> 59.1669 | 52.4403 -> 49.5402 | 5.53% |
| Text / 2 | 57.9704 -> 57.0927 | 87.4583 -> 82.3701 | 5.82% |
| Coder / 1 | 71.6899 -> 72.1945 | 36.2565 -> 34.3173 | 5.35% |
| Coder / 2 | 70.8400 -> 70.9176 | 56.1233 -> 51.9464 | 7.44% |
| VL / 1 | 48.2369 -> 47.4741 | 208.5971 -> 195.9944 | 6.04% |
| VL / 2 | 44.3473 -> 44.6104 | 380.6332 -> 360.9455 | 5.17% |

Four warm requests per policy/profile, from two fresh sequential workers; each
worker's first request is excluded from the warm mean. B2 aggregate TPS is twice
the per-row rate. Decode changes range from -1.58% to +0.70% and are inconclusive.
Prefill GPU durations decrease across all profiles, while encoding duration
increases; exact phase observations are in `native/phases.json`.

First request AFTER initialization is recorded separately: control -> candidate
TTFT ms is text B1 150.15 -> 156.87, text B2 188.13 -> 184.08, coder B1
121.81 -> 165.96, coder B2 143.88 -> 185.38, VL B1 352.05 -> 374.76 and
VL B2 571.23 -> 685.93. Five profiles therefore regress on first use. These
are observed phases with OS caches unflushed, not hardware-cold end-to-end sums.

Initial archive SHA-256
`7748269dd13518fb7305ee27ab2d8f5030093d5cf41755cd4acf386d82763133`.
Fresh full-six confirmation PID **36981** completed successfully in 144.81 seconds.
Combined archive SHA-256
`d848315ee193958771d117a0aa347dee2f6383bc702cfaac3f7281dd1596d727`.
All 950 preserved files and both sets of 72 complete requests independently
verified. Full precision results are in `native/verification.json` and
`native/confirmation.json`; separate prefill phases are also recorded.

| Model / batch | Repeat control -> candidate tokens/s per row | Repeat control -> candidate warm TTFT ms | Repeat warm TTFT reduction |
| --- | ---: | ---: | ---: |
| Text / 1 | 59.1268 -> 58.7013 | 52.0985 -> 49.4033 | 5.17% |
| Text / 2 | 57.9222 -> 57.6509 | 86.2570 -> 80.5807 | 6.58% |
| Coder / 1 | 72.0331 -> 71.8944 | 36.1704 -> 33.1865 | 8.25% |
| Coder / 2 | 70.7938 -> 70.7996 | 56.2728 -> 49.7387 | 11.61% |
| VL / 1 | 47.7617 -> 47.5913 | 203.4895 -> 193.7058 | 4.81% |
| VL / 2 | 44.3583 -> 44.4210 | 380.6422 -> 362.6300 | 4.73% |

The warm latency direction repeats across all six profiles, though magnitude
varies. Decode changes in the repeat are -0.72% to +0.14%, with no useful speed
gain. Repeat first requests AFTER initialization are text B1 153.46 -> 153.91,
text B2 190.23 -> 188.14, coder B1 122.47 -> 121.90, coder B2 142.28 ->
142.64, VL B1 341.91 -> 331.90 and VL B2 574.80 -> 558.35 ms. Large first-use
penalties from the initial run do not repeat. OS/driver caches were not flushed;
the repeat must not erase those initial observations or establish cold improvement.

The candidate stays optional. The main trials above use attention control for
VL B2, while its prior selected library is the original resident-KV build. A
separate selected-control trial, PID **40062**, exited 0 in 36.01 seconds:
VL B2 warm TTFT 383.1646 -> 361.5074 ms (**5.65% lower**), per-row decode
44.4372 -> 44.4278 tokens/s (**-0.02%**). First request after initialization is
576.6335 -> 550.9718 ms. This ABBA uses two fresh workers per policy and four
warm requests per policy; all 12 complete outputs and fresh image identities
verify. The combined 1,043-file archive SHA-256 is
`1152d10b18bc72480dd00f2be9c776192cd52f4088685b8d107f794113a2d63c`.
Local verification is `native/selected-vl2.json`. The historical native records
must not be used as a new vLLM result.

A separate proposed full-six bundle is frozen in `vllm/selection.json`, using
global-alpha candidate SHA-256 above across all models and B1/B2. The original
selection file and original libraries remain intact. Fresh full-six sequential
comparison PID **40956** was launched at
`ROOT/logs/global-alpha-vllm-20261014`. It runs three real FP32 GPU dtype audits,
then native/vLLM ABBA, fresh complete images and unchanged caps/prompts with
prefix caching and speculation disabled. Inspect the actual current handle or
terminal record; a launch is not performance evidence. No fresh vLLM speedup
claim is yet established. The overall 2x decode / half-TTFT goal remains unmet;
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
python3 archive_global_half_alpha_native.py ROOT ARCHIVE
python verify_global_half_alpha_native.py RESULTS VERIFICATION_JSON
python3 launch_confirm_global_half_alpha_profiles.py ROOT ROOT/logs/global-half-alpha-confirm-native-20261014
python prepare_global_alpha_selected_vl2.py
python3 launch_global_alpha_selected_vl2.py ROOT ROOT/logs/global-alpha-selected-vl2-20261014
python verify_global_half_alpha_native.py RESULTS SELECTED_REPORT --selected-vl2
python prepare_global_alpha_vllm.py
python3 background_comparison.py launch ROOT/logs/global-alpha-vllm-20261014 python3 ROOT/source/test2026/metal/run_global_alpha_vllm.py ROOT ROOT/logs/global-alpha-vllm-20261014/jobs
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
