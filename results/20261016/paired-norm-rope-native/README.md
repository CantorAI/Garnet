# Private native paired normalization and RoPE investigation

Status: the private library builds and its linked vector/paired primitives,
CTest stage, full vision grid and trained real-head checks have exited 0.
The full 34-stage native correctness job is still running. No complete native
validation or model timing result is established; serving code is unchanged.

After two complete unchanged component cohorts, select only Qwen3-0.6B and
Qwen2.5-Coder-0.5B-Instruct matrix rows 1/2. VL and larger rows retain their
original operations. This physical-row rule also covers tiny 1–2-row prefill
chunks; the agreed serving benchmark's prefill shapes remain on the original
path. [Repeated component selection](../paired-norm-rope-chain/README.md).

Freeze a new source tree on accepted d52 vector vision: 1,411 files,
12 transformations and 5 added helpers, preserving every source/before/after
identity. Local regeneration matches the actual Mac preparation exactly.
[Frozen preparation](preparation.json).

The OFF-by-default GARNET_METAL_PAIRED_NORM_ROPE feature recognizes consecutive
Q/K NormRope operations only when they share positions/frequencies and original
normalization/rotary parameters, match a selected geometry, and both outputs
are live. Keep the original per-head arithmetic, independent output allocations,
projections, keys, fallbacks and public writes. Profiling retains isolated
operations. No packed weights, output views, speculative result or model cache.

The graph and resident session expose actual paired dispatch counters. Workers
record prefill and every decode-step count. Serving prefill and VL are expected
to use zero paired calls in the agreed profiles; text/coder decode must report
the actual full-attention layer count. No component ratio predicts model gain.

## Native correctness protocol

Run every original 30 stages afresh, linked paired primitive compile/execution,
and two additional trained text/coder graph/session reference suites, 34 total.
The added suites retain reproducible fixtures at chunks 1/2, both batches and
first/last layers: 16 profiles, 48 carried state steps, 36 selected-row and
12 fallback route checks. Preserve all original hidden/key/value error bounds,
invalid controls, pauses, resets, close behavior and explicit/resident equality.

The generated driver's intermediate counts are 25/31/34 before execution.
Local preflight regenerates the complete runner and independent auditor and
parses every source transformation. No stage-count repair, numerical tolerance
change or reused original validation stage is involved in this native run.

At the latest saved observation, check PID 54126 and finalizer 58273 are
verified live. Sixteen stages have exited 0, including dense tails, private KV,
serving and mixed-image tests. Remaining stages continue; no full34 gate is
published. [Process and stage observation](observation.json).
Current candidate library SHA256:
`0c0faf7191ee59b82b54dca630e21e98590d66a4c8032054e3f5bb04f9308d0d`.
Accepted d52 control remains:
`d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e`.

Finalizer 58273 watches that exact check PID. It archives terminal success or
failure, then independently regenerates/audits a successful full34 archive.
Only complete success publishes the correctness gate. It launches no model
benchmark. An observation deadline never stops or restarts the check job.
Consumed source, runtime and drivers remain frozen until terminal and archived.

Separate, unused comparison helpers are prepared and copied to the Mac.
The comparator and original independent 72-output auditor regenerate and parse
locally with actual prefill/every-decode route checks. The launch requires the
finalizer to be terminal, an exact archive/gate identity and a complete raw34
re-audit before exclusive inference. It compares against accepted d52, keeps
vector vision enabled in both policies, preserves original inputs/limits and
checks all six profiles. These are preparation checks, not native timings.

## Prospective complete model gate

After full correctness, compare all six profiles against accepted d52 in two
complete unchanged native cohorts. Require >=1% decode improvement at at least
one currently unmet batch-1 text/coder profile, with the same profile improving
in both cohorts. Require every profile's decode fraction >=0.98 and TTFT
fraction <=1.02, all original complete outputs and every route check. Report
initialization, first-after-init, warm and peak RSS separately; caches remain
unflushed. No native-only ratio updates the matched-vLLM target count.

Commands executed, ROOT=`/Users/zsdx/garnet-metal-investigation`,
CANDIDATE=`ROOT/paired-norm-rope-native-investigation-20261016`:

```text
python3 ROOT/source/test2026/metal/prepare_paired_norm_rope_native.py ROOT CANDIDATE
python3 ROOT/source/test2026/metal/launch_paired_norm_rope_native_checks.py CANDIDATE CANDIDATE/logs/paired-norm-rope-native-checks-20261016 checks
python3 ROOT/source/test2026/metal/finalize_paired_norm_rope_native_checks.py CANDIDATE ROOT/logs/paired-norm-rope-native-finalize-20261016 54126
```

The full 2x decode / half-TTFT goal for all matched models and batches remains
active. Fresh repeated vLLM-Metal measurements are required after any retained
native change.
