# Decode-only fused head: repeated full runs verified

The all-head variant repeated a 0.39–2.26% whole-model decode improvement,
but text B2 first-token latency regressed in both runs. This experiment keeps
the existing projection/argmax path when the session's physical chunk is
larger than one, and fuses only single-token runs. Agreed serving prefill
profiles use chunks of at least two; a direct single-token prefill API call
would use fusion as well. No precision, checkpoint, resolution, prompt, output
cap, request cache or prediction semantics are changed.

`GARNET_METAL_FUSED_HEAD_DECODE_ONLY` is OFF by default and requires the existing
optional fused-head flag. The previous candidate library and archived results
are preserved. Five route/measurement files are installed only after verifying
their exact original hashes. Raw fused shader arithmetic is unchanged.

Fresh build `build-decode-only-head` passed both actual CTests, including
the full-bit/FP64 fused-kernel, layout, ownership and recovery checks. All
original 19 model stages completed at PID 96768, exit 0 in 1201.70 seconds.
PID 98218 then completed six trained session/fallback/recovery cases and all
six matched native ABBA profiles, exit 0 in 1302.31 seconds including its
dependency wait. Independent verification covers 1,225 files and 72 complete
matching requests. The route counts are verified in every complete record.

The benchmark asserts zero fused dispatches for prefill's first prediction
and exactly one for each subsequent decode prediction. Every complete output,
prompt ID and finished flag must match; images must be freshly encoded. All
three models and both batches remain in scope. Initial whole-model measurements
are below; they are native control/candidate results, not new vLLM ratios.

| Model / batch | Control row tokens/s | Candidate row tokens/s | Decode ratio | TTFT fraction |
| --- | ---: | ---: | ---: | ---: |
| Text / 1 | 59.1251 | 59.8052 | 1.0115 | 0.9998 |
| Text / 2 | 57.8751 | 58.6436 | 1.0133 | 0.9745 |
| Coder / 1 | 71.9537 | 72.3788 | 1.0059 | 1.0287 |
| Coder / 2 | 71.0042 | 71.9356 | 1.0131 | 1.0042 |
| VL / 1 | 48.2817 | 48.8417 | 1.0116 | 0.9933 |
| VL / 2 | 44.3603 | 45.2198 | 1.0194 | 1.0093 |

Warm means use four requests/policy, first request excluded. Decode direction
is positive by 0.59–1.94%, but TTFT remains mixed.

Full six-profile confirmation PID 21906 completed with exit 0 in 144.98 seconds.
The combined archive has 1,523 independently verified files, all original 19
gates, six trained recovery cases and 72 matching requests/run (144 total).

| Model / batch | Repeat control row tokens/s | Candidate row tokens/s | Decode ratio | TTFT fraction |
| --- | ---: | ---: | ---: | ---: |
| Text / 1 | 59.1906 | 59.8678 | 1.0114 | 1.0014 |
| Text / 2 | 58.2480 | 58.7954 | 1.0094 | 0.9992 |
| Coder / 1 | 72.4016 | 73.1634 | 1.0105 | 0.9998 |
| Coder / 2 | 71.0377 | 71.9162 | 1.0124 | 1.0108 |
| VL / 1 | 48.1548 | 48.8477 | 1.0144 | 1.0065 |
| VL / 2 | 44.4670 | 45.2840 | 1.0184 | 0.9997 |

Decode improves in both runs for every profile, but the full-model gain remains
0.59–1.94%. Text B2 TTFT is 2.55% lower then essentially flat, so the earlier
all-head variant's repeated penalty does not reappear in these trials. This
is a within-run direction, not a direct statistical comparison between the
two candidate variants. Coder B2 TTFT is 0.42%/1.08% higher in both runs; other
small latency effects are mixed. Retain the policy as optional and OFF by
default. It is not a uniform two-metric selection or a new comparator result.

Candidate library SHA-256:
`5f40fdfb1921a0024bb1119fe6c2b2925498ef3e3256eab6133431c66f49795c`.
Initial archive SHA-256:
`a9c5ba32379735eb7a23c95fff54daaa51c4ad73c0882b6442dd71473166e744`.
Combined initial/confirmation archive SHA-256:
`a213c946edcc2c8f87c5700a458883ef7f5e14ac55beb81fa5ec8eb811392a3e`.
Initial verification preserves initialization and first request AFTER init
separately from warm means. Caches are unflushed; no sum is a measured hardware-
cold end-to-end clock. Compiler/Python/OS/hardware observations after the build
are preserved in `versions-post-build.json` within the archived validation logs.

```text
python3 prepare_decode_only_head_sources.py
python3 install_decode_only_head_sources.py ROOT
python3 launch_decode_only_head_checks.py ROOT ROOT/logs/decode-only-head-checks-20261015 checks
python3 queue_decode_only_head_profiles.py ROOT ROOT/logs/decode-only-head-native-20261015
python3 archive_decode_only_head_inference.py ROOT ARCHIVE
python -X utf8 verify_decode_only_head_inference.py EVIDENCE VERIFICATION_JSON
python3 launch_confirm_decode_only_head_profiles.py ROOT ROOT/logs/decode-only-head-confirm-native-20261015
python3 archive_decode_only_head_inference.py ROOT ARCHIVE decode-only-head-native-20261015 decode-only-head-confirm-native-20261015
python -X utf8 verify_decode_only_head_inference.py EVIDENCE CONFIRMATION_JSON decode-only-head-native-20261015 decode-only-head-confirm-native-20261015
```

The full 2x decode / half-TTFT goal remains active and unmet. The GPT-OSS and Pi
checkouts remain unchanged. The next decision requires actual complete-output
measurements and repeated warm/cold observations, not the earlier component
gain or a different comparator baseline.
