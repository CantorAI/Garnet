# Direct resident-state reset: validation in progress

This optional experiment removes temporary zero-filled CPU vectors from the
resident session reset. It clears logical float elements directly in the
existing shared Metal buffers, preserving row padding, positive-zero values,
allocation identity, readiness rules and the closed-command requirement.
The `GARNET_METAL_DIRECT_STATE_ZERO` option defaults OFF. Decode arithmetic,
checkpoint precision, graph scheduling and public reset behavior are unchanged.
The primary serving library and protected GPT-OSS/Pi checkouts are untouched.

Why investigate: the latest warm matched text B2 request takes 86.19 ms to
first prediction while its resident prefill session takes 62.62 ms. Reset,
embedding preparation and wrapper work occupy the difference. This difference
does not itself attribute the cost to reset. New request telemetry records
reset separately on both candidate and control, without removing it from TTFT.
The VL `vision_seconds` interval still includes reset, as in the prior runner;
the added reset field permits distinguishing it. GPU decode remains a separate
optimization requirement.

The candidate successfully configured and built on the actual M4 Mac mini.
CTest passed the direct-zero reference and existing MPS matrix reference.
The direct-zero test covers repeated nonzero-to-positive-zero transitions,
row padding shapes, real GPU consumption, empty matrices and attempts to
clear during an open command batch. Trained-layer, exact-tail, extended VL,
private-cache isolation/reset and complete-request checks are still pending.
No latency, memory or throughput improvement is claimed yet.

## Live work and exact commands

Owned root: `/Users/zsdx/garnet-metal-investigation`. Source installation
verified all ten old/new file identities and saved original files in
`logs/direct-state-zero-original-source-20261013`. The local
[source manifest](source-manifest.json) records those identities.
Source bundle SHA256:
`8941aa256fc62380accf5db9ad90ffa0da0bff206d742a055e657a9712e4a710`.

The durable checks wrapper is PID 6169, last verified live while running the
trained text exact-tail suite. The inference wrapper is PID 8442, queued
behind the checks' successful terminal exit. Inspect both PIDs and their
`exit.json` files before continuing; do not restart because an observation
times out.

```text
/opt/homebrew/bin/python3 source/test2026/metal/background_comparison.py launch logs/direct-state-zero-checks-20261013 /opt/homebrew/bin/python3 source/test2026/metal/run_direct_state_zero_checks.py ROOT logs/direct-state-zero-checks-20261013/jobs
/opt/homebrew/bin/python3 source/test2026/metal/background_comparison.py launch logs/direct-state-zero-inference-20261013 /opt/homebrew/bin/python3 source/test2026/metal/run_after_owned_job.py logs/direct-state-zero-checks-20261013 /opt/homebrew/bin/python3 source/test2026/metal/compare_direct_state_zero.py ROOT logs/direct-state-zero-inference-20261013/jobs
```

The archived launcher records the full absolute commands. `ROOT` above means
the owned root. The comparator uses native ABBA order for all three models,
batches 1/2, unchanged prompts/caps and fresh image encoding. Both builds
have the same attention-prefix policy so the change being tested is reset.
Control library SHA256:
`f2c5c448ec4f3c8431be37b48a2634d8fa4cc31de4aa163ec367a319c9263088`.
VL B2's use of this control is solely for isolating reset; it does not promote
the otherwise inconclusive attention-prefix VL B2 policy. Runtime/process
guards enforce exclusivity and a 2 GiB free-space floor. A repeat, complete
output checks and a terminal evidence archive are needed before drawing a
performance conclusion. This experiment is not a new vLLM comparison.

The full goal is still unmet. The latest
[actual vLLM comparison](../selected-attention-vllm/README.md) retains all
six measured profiles and the missing decode/first-token targets.
