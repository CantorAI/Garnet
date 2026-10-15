# MLP graph command-buffer encoding: 64-case private safety probe passed

Hypothesis: retain the complete FP32 gate/up–SiLU–down graph, but encode it
into the existing native MPSCommandBuffer rather than flushing and waiting
on the CPU after every MLP. The synchronous integration was
[rejected on full-model latency and memory](../mlp-prefill-native/README.md).
This probe changes scheduling only; it does not remove graph compilation or
expanded FP32 weight-storage costs, and it establishes no speed gain.

Actual PID5137 exited0 in10.621082s on the M4 16 GiB investigation after the
exclusive quiet guard. The private Objective-C++ context compiled; all64
original primitive cases passed with an added native output consumer. The
[independent audit](verification.json) regenerates the context, graph header
and test from the frozen sources, checks the unchanged numerical bound,
guarded compile/execution and actual assertion-bearing test output.

Source factory `make_mlp_command_buffer_context.py` leaves serving sources
unchanged. It replaces the synchronous graph call only in a private context
copy. The installed SDK headers were inspected and preserved with hashes:
MPSGraph accepts MPSCommandBuffer and may call `commitAndContinue` internally;
callers must obtain its current `rootCommandBuffer` before further encoding.
The prototype retains an internally committed original root, continues native
encoding on the current root, checks asynchronous graph errors/completion at
final submission, and drains committed work before abort invalidates outputs.
It does not assume the original underlying buffer remains uncommitted.

The original rows3/24/26/44/52/80/128/160, widths8/16, four input patterns,
pending native producers, ownership rejection, alias/uninitialized rejection,
abort/recovery and unsupported-row fallbacks remain. Each successful graph
is followed by native addition; both graph output and consumer output must
match their corresponding original native control within the same
`5e-6+5e-5*abs(native)` bound. Actual graph count is1 and explicit native
Flush count0 per admitted invocation. That counter does **not** count any
internal MPSGraph commits. Compiler emits existing fastMathEnabled deprecation
warnings; the compile and test exit successfully.

This is a small synthetic primitive test. The dependent chain extension below
is also synthetic; all trained checkpoints/heads, full vision grids, original
model outputs, first/init/warm latency and memory still need validation.
Native model comparison and fresh repeated matched vLLM cohorts remain
required. No option/default is promoted and the full goal remains unmet.

```text
python3 launch_mlp_command_buffer_primitive.py ROOT ROOT/logs/mlp-command-buffer-primitive-20261015 primitive
# After terminal status, before any consumed source changes:
python3 archive_padded_mps_body.py ROOT ARCHIVE mlp-command-buffer-primitive-20261015
python -X utf8 test2026/metal/verify_mlp_command_buffer_primitive.py RESULTS RESULTS/verification.json ARCHIVE
```

Frozen archive SHA256:
`1e87242bb5a9fc6f77ec9c6d2f73cd5dfb4236019f5ad2531557bf5e6f510a2f`.
38 files include original/generated sources, SDK headers, exact commands,
monitors and terminal records; compiled binary bytes are represented by hash
and size. This archive contains no timing benchmark or trained-model result.

## Dependent chains passed; full native validation running — 2026-10-15

Fresh PID5521 exited0 in11.264293s. The
[38-file chain audit](chained-verification.json) verifies the64 original cases
plus64 true dependent two-graph cases: native producer → first graph → native
addition → second graph → native consumer, followed by two-graph abort and
recovery. Both graph and consumer results use the unchanged native bound.
Actual graph count2, explicit native Flush count0. This extends ordering
coverage; it is not another replication of the original test or a speed test.
Chain archive SHA256:
`398edcdc8bc6f4b7af2adc53899d6448e93c98b250a33eb7afb02e7d5d19ff5b`.

`prepare_command_buffer_native_checks.py` created
`ROOT/mlp-command-buffer-investigation-20261015`, with1,240 copied source
files and exactly three transformed copies: native context, graph header and
primitive test. The original sources/builds stay unchanged. Existing model,
baseline build and historical-log paths are linked for workflow reads only;
the new candidate build and validation outputs are separate. Preparation
records every copied source hash and all three before/after identities.

Fresh PID5961 runs the complete original28-stage wrapper, including the
outer appended wide reference, rather than resuming a truncated generated
driver. Log folder is `ROOT/mlp-command-buffer-investigation-20261015/logs/mlp-command-buffer-native-checks-20261015`.
Build, original trained/recovery validation, terminal archival and independent
audit remain pending at this observation. The new tree's consumed sources
remain frozen through terminal capture and archival. Only after successful
audit may native output/timing comparison proceed.
