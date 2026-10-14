# Deferred request-state zero: full validation in progress

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

Owned validation PID 26187 is running all original 19 model gates at
`ROOT/logs/deferred-state-zero-checks-20261015`. Two attempts were discarded
when an unverified runtime appeared; only complete clean cases are accepted.
The guard remains unchanged. PID 31105 waits for successful terminal validation
before six trained session/recovery cases and six complete native ABBA profiles.
Recheck actual handles for live state; no model latency gain is yet established.

Ten changed/new files are hash-pinned and installed only after nine original
hashes match; exact originals are preserved in the investigation logs. Earlier
selected and experimental libraries are retained. GPT-OSS and Pi checkouts
remain unchanged. The full 2x decode / half-TTFT goal remains unmet.

```text
python3 prepare_deferred_state_zero_sources.py
python3 install_deferred_state_zero_sources.py ROOT
python3 launch_deferred_state_zero_checks.py ROOT ROOT/logs/deferred-state-zero-checks-20261015 checks
python3 queue_deferred_state_zero_profiles.py ROOT ROOT/logs/deferred-state-zero-native-20261015
```
