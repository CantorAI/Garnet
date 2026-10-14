# Deferred request-state zero: repeated full-model latency tradeoff

The optional `GARNET_METAL_DEFERRED_STATE_ZERO` reset queues each private state
matrix for row-preserving GPU blit fills at the start of the first inference
command buffer. No separate reset submission/wait is added. Owner snapshots
still return logical positive-zero values immediately; other contexts cannot
read or modify the stale physical allocation while its clear is queued.

Correctly sized host updates and direct zeros cancel a queued clear. Invalid
updates preserve it. Begin transfers ownership into the existing pending-output
set, so all GPU consumers are ordered after the fills and buffers survive until
completion. Abort/destruction invalidate pending/queued contents. Reset after
a poisoned session requeues every state. Logical row padding is preserved,
and actual materialized byte/state counts are recorded in the prefill profile.
No clears are expected on subsequent decode commands.

The feature is OFF by default and cannot be combined with the older direct-
zero or dirty-prefix reset policies. This experiment uses the selected global-
alpha compute options with fused-head variants disabled, isolating reset work.
Checkpoints, FP32 state/math, prompts, output caps, batches and images are frozen.

Fresh build `build-deferred-state-zero` passed both actual CTests in 2.43 seconds.
The new test verifies 45 nonzero/NaN/infinity transitions at B1/B2/padded B3,
positive-zero snapshots and GPU reads, ownership, duplicate resets, host-write
cancellation, invalid input handling, later-command clear elision, queued and
submitted abort recovery, partial submissions and owner destruction.

Owned validation PID 26187 passed all original 19 model gates, exit 0 in
1238.93 seconds. Two interrupted attempts were discarded when an unverified
runtime appeared; only complete clean cases are accepted. Guards are unchanged.
PID 31105 completed six trained session/recovery cases and 72 exact matching
requests in six native ABBA profiles, exit 0 in 1166.84 seconds including its
dependency wait. Full six-profile confirmation PID 52005 also exited 0, in
144.75 seconds. Both independent audits cover 1,551 archived files, all 19
original gates, six recovery cases, every clear count and all complete outputs
(144 matching requests across the two native runs).

| Model / batch | Initial TTFT control → candidate ms | Repeat TTFT control → candidate ms | Initial / repeat decode ratio |
| --- | ---: | ---: | ---: |
| Text / 1 | 51.015 → 44.149 | 49.493 → 44.719 | 1.0067 / 0.9947 |
| Text / 2 | 81.590 → 65.244 | 82.092 → 64.596 | 0.9774 / 0.9723 |
| Coder / 1 | 33.019 → 33.369 | 33.870 → 32.164 | 0.9988 / 0.9995 |
| Coder / 2 | 50.821 → 49.467 | 50.989 → 48.427 | 0.9963 / 0.9942 |
| VL / 1 | 202.299 → 192.016 | 196.259 → 192.852 | 0.9985 / 0.9995 |
| VL / 2 | 360.404 → 353.823 | 358.800 → 353.622 | 0.9964 / 0.9945 |

Warm means use four requests/policy/run, first request excluded. Text TTFT
falls 9.6–21.3% across the two runs, but text B2 decode loses 2.3–2.8%.
Most other decode rates are slightly lower as well. Keep this optional and OFF
by default; it is not a uniform throughput/latency selection. No new vLLM
comparison or 2x / half-latency result is claimed.

`phases.json` and `confirmation-phases.json` retain reset, prefill, initialization
and first request AFTER initialization separately. Candidate reset-call time is
logical marking; physical clearing remains in prefill. For example, initial
text B2 reset-call time falls 21.783→0.052 ms while prefill inference rises
58.272→63.731 ms. The whole TTFT gain is measured, not obtained by subtracting
component times. Encode/GPU/wait clocks overlap and must not be added. Caches
are not flushed; no sum is an observed hardware-cold end-to-end clock.

Ten changed/new files are hash-pinned and installed only after nine original
hashes match; exact originals are preserved in the investigation logs. Earlier
selected and experimental libraries are retained. GPT-OSS and Pi checkouts
remain unchanged. The full 2x decode / half-TTFT goal remains unmet.

Candidate library SHA-256:
`6cc869948b05a5d44708120fca3a7dfd13268cede7dc5233bd270011bcedcb31`.
Initial archive SHA-256:
`494ed0725ce7bf51e22797b87d0be4fff4006d965098107697f95109ba626e67`.
Combined initial/confirmation archive SHA-256:
`7e749db5696f5b21a25d04784c47fc9a80e9dcc1233cf7e6d574317f3192d750`.

```text
python3 prepare_deferred_state_zero_sources.py
python3 install_deferred_state_zero_sources.py ROOT
python3 launch_deferred_state_zero_checks.py ROOT ROOT/logs/deferred-state-zero-checks-20261015 checks
python3 queue_deferred_state_zero_profiles.py ROOT ROOT/logs/deferred-state-zero-native-20261015
python3 archive_deferred_state_zero_inference.py ROOT ARCHIVE
python -X utf8 verify_deferred_state_zero_inference.py EVIDENCE VERIFICATION_JSON
python3 launch_confirm_deferred_state_zero_profiles.py ROOT ROOT/logs/deferred-state-zero-confirm-native-20261015
python3 archive_deferred_state_zero_inference.py ROOT ARCHIVE deferred-state-zero-native-20261015 deferred-state-zero-confirm-native-20261015
python -X utf8 verify_deferred_state_zero_inference.py EVIDENCE CONFIRMATION_JSON deferred-state-zero-native-20261015 deferred-state-zero-confirm-native-20261015
```
