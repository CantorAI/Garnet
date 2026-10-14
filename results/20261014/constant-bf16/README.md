# Constant-address BF16 weights: no useful native decode gain

The component changes only the original shader's weight address-space
qualifier from device-readonly to constant, including its ushort4 casts.
Weights retain the same buffer, BF16 bits and original FP32 math. Both
native small-output and regular decode kernels are tested. The initial
attempt stopped at shader compilation because vector casts still used
device space; its timings are discarded. The corrected and fresh-process
confirmation jobs (PIDs 8012 / 8994) exited 0 after 35.64 / 35.59 seconds.

Both successful runs verify all three checkpoint identities and canonical
BF16 tensor slices, covering 34 fixtures / 68 batch profiles. All 121,652
validated warm/timed executions retain complete output/padding bits with
four freshly uploaded input variants. Each run also has 272 independent
FP64 reference executions, 6,528 timed executions, and a common alternating
250 ms warm stage per profile. Worst FP64 normalized error is
0.0100017036, below 1 under the unchanged prior envelope. Encoding, wall
and GPU clocks are separate; setup/checks/input copies are outside timing.

| Model/batch | Pooled constant/device wall fraction, initial / repeat |
| --- | ---: |
| Text B1 | 0.99247 / 0.98746 |
| Text B2 | 0.98009 / 0.99249 |
| Coder B1 | 0.97987 / 0.99532 |
| Coder B2 | 1.00235 / 1.00810 |
| VL B1 | 0.93113 / 0.98917 |
| VL B2 | 1.02018 / 1.01834 |

These are components, not observed full-model TPS. Large B1 head wall
fractions in four independent ABBA rounds (initial then fresh-process
repeat) are text 0.9960/0.9773/0.9752/0.9965, coder
0.9624/0.8931/0.9586/0.9324 and VL 0.7969/0.9618/0.9439/0.9922.
Magnitude varies substantially, but the B1 head direction repeats. Body
profiles are mostly flat/mixed, and B2 does not justify broad admission.
OS caches were not flushed, GPU frequency is not pinned and the 0.5-second
guard cannot rule out arbitrary shorter external activity.

## Native experiment, not a completed speed claim

`GARNET_METAL_CONSTANT_BF16_HEAD` defaults OFF. Its candidate route keeps
the original shader code and float operations, changing the two weight
pointer qualifiers only. Admission is exactly one row with K/N equal to
896/151936, 1024/151936 or 1024/248320, matching the measured vocabulary
heads. All other cases keep the existing path. The session exposes actual
dispatch counts, reset on clear/reset/begin/abort and published only after
successful submission. Worker logs capture the feature marker and actual
last-session dispatch count after the timed request has completed.

The isolated candidate builds and passes native CTest. All 19 trained-model,
cache, vision and extended-prefill validation stages subsequently passed,
with exit 0 after 1006.73 seconds.
Candidate library SHA-256 is
`df1c1912b2daf40c86925d323747d60db6a4950723ac04562f06854da1c7aaa3`.
Owned validation PID **10827** is terminal, directory
`ROOT/logs/constant-head-checks-20261014`. Historical build/CTest snapshots
are in `native/pending`; the full remote terminal logs remain authoritative.

Owned PID **16406** verified the predecessor's successful terminal state,
then completed three sequential B1 comparisons against retained attention.
All 36 complete outputs match, and candidate dispatch counters equal 1
versus control 0. Its directory is
`ROOT/logs/constant-head-native-20261014`; its exit code is 0. Both owned
jobs are terminal and must not be restarted. Full raw native evidence is
archived in `native/evidence`; `native/initial-summary.json` captures this
initial result and `native/verification.json` independently audits it.

| Initial native B1 result | Control → candidate TPS | Control → candidate TTFT |
| --- | ---: | ---: |
| Text | 59.2747 → 58.9827 | 51.9618 → 52.8059 ms |
| Coder | 72.0785 → 71.3736 | 35.5724 → 36.0902 ms |
| VL | 48.0947 → 48.1353 | 209.9577 → 206.6803 ms |

The component head signal does not establish a whole-model decode gain:
text/coder regress slightly and VL decode is essentially flat. Its single
small VL latency reduction is unconfirmed. No favorable-number repeat is
warranted without a new change or a specific unresolved concern. The route
remains optional and default OFF; no runtime promotion is made.
Selected primary/attention binaries remain unchanged (73c10e5d / f2c5c448).
These are initial native comparisons, with no fresh vLLM comparison for
this candidate. The full 2x decode / half-TTFT goal remains unmet.

## Commands and evidence

```text
python3 launch_constant_bf16_probe.py ROOT ROOT/logs/constant-bf16-probe-20261014
python3 launch_constant_bf16_probe.py ROOT ROOT/logs/constant-bf16-retry-20261014
python3 launch_constant_bf16_probe.py ROOT ROOT/logs/constant-bf16-confirm-20261014
python3 archive_constant_bf16.py ROOT ARCHIVE
python summarize_constant_bf16.py EVIDENCE SUMMARY [SUCCESSFUL_JOB_NAME]
python3 install_constant_head_sources.py ROOT
python3 launch_constant_head_checks.py ROOT ROOT/logs/constant-head-checks-20261014
python3 queue_constant_head_profiles.py ROOT ROOT/logs/constant-head-native-20261014
```

The installer verifies prior source hashes and preserves original bytes
before installing only owned investigation sources. Native validation uses
the retained attention configuration; direct-state-zero and joined-prefill
experiments are disabled. The conditional runtime changes do not promote
the candidate or change the protected checkouts.

Initial component archive SHA-256:
`1948fc753cea51110a3ccf8085d37eb8474eaaf4ea046c640ad4e3a9d68bfe79`
(42 files). Combined confirmation archive:
`5161df64338193a2741aaa0e60d6aded6c6f54396f80efcf8525eca1d9022551`
(65 files). Local verification checked all hashes, terminal stages,
source/shader/binary identities, sample counts, exact/FP64/padding gates,
common warm durations and recomputed means. Raw evidence retains its byte
identities. The two derived component reports are `summary.json` and
`confirmation/summary.json`.

Native archive SHA-256:
`d186619c05a5e22d6bf0bff0a71453b7e8d5065260fc0640510da0df9d745575`
(293 files). The local verifier checks all file hashes, 19 validation stages,
36 complete outputs, actual dispatch counts, input identities, fresh image
encoding, library identities and independently recomputed warm means.
