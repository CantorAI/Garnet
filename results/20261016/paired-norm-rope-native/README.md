# Private native paired normalization and RoPE investigation

Status: all 34 fresh native correctness stages pass, with identical independent
audits on Mac and Windows. The first complete six-profile native performance
cohort is running. No model speed improvement is established; serving code
is unchanged. Preserve the original failed archive audit described below.

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

The earlier [process and stage observation](observation.json) records 16 live
stages. Check PID 54126 subsequently exited 0 in 1,898.893526708 seconds,
with all 34 stages passing. The complete independent audits regenerate all
12 source changes and the actual executed driver; no validation stage or
compile stage is reused. [Windows full audit](checks/verification.json),
[identical Mac full audit](checks/mac-verification.json),
[published gate](checks/gate.json).
Current candidate library SHA256:
`0c0faf7191ee59b82b54dca630e21e98590d66a4c8032054e3f5bb04f9308d0d`.
Accepted d52 control remains:
`d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e`.

The linked primitive checks 60 cases and 3,710,720 FP32 output values bit for
bit. Added trained suites pass all 48 carried steps, 36 selected-row and
12 fallback routes; their worst original-bound ratio is 0.1624941610. The
wide VL reference's worst ratio is 0.5432836183. Ratios must remain <=1.
The legacy audit field `paired_rope_primitive_output_bits` counts FP32 values,
not individual bit positions.

Original finalizer 58273 exited 1 at the archive audit: the private source
tree predates `verify_vector_vision_native_checks.py`, which was not included
as an external dependency. Continuation 78736 preserved the terminal finalizer
and exited 1 before any model profile launched. Preserve both terminal records,
logs and consumed tools: [finalizer failure](finalizer-failure/archive-verification.json),
[continuation failure](continuation-failure/archive-verification.json).
The original incomplete checks archive remains unchanged on the Mac.

After both controllers are confirmed terminal and archived, repair packaging
only: include the vector/MLP auditors separately, check their identities against
the original controller snapshot, pin the vector auditor's original hash,
restore its required `re` import, and route gate/launch/archive paths to a
separately named corrected archive. Keep private source, dylib, executed driver,
all raw stage outputs, parameters and numerical assertions unchanged.
Finalizer 88835 exits 0 in 14.309746209 seconds, auditing 1,915 archived files.
Both hosts reproduce the same complete audit. No correctness rerun is claimed.

| Evidence | SHA256 |
| --- | --- |
| Original checks archive, omitted auditor | `a48c860b65fab26727327cc3fd0f4781dd0b38c59784bf914f0be2ee8f6dbf72` |
| Original terminal finalizer, 14 files | `9ca7c92a7824189dc37e44feda6fa7af6f6881f74bdc92f44cad16bac0a9e633` |
| Original terminal continuation, 1,245 files | `7c8c72d7a5c0ad835c25ceac20410eeb002c7953070c03aa0bcbd8d94b9d8c47` |
| Corrected full34 archive, 1,915 files | `d06d521f2a26d5f5e9c28f488e32393e4549ac2c477150833136c27966aa055a` |

Continuation 89012 snapshots all shared direct tools and launches first native
cohort 89030 only after terminal finalizer, archive/gate identity, complete raw34
re-audit and exclusive quiet interval. At 05:44:42 UTC both processes are live,
with the first text batch-1 profile started. It compares against accepted d52,
keeps vector vision in both policies, preserves original inputs/limits and checks
all six profiles. Its 72-output independent auditor also verifies actual
prefill/every-decode route counts. No complete timing result is available yet.
All consumed shared tools/private source/library remain frozen until terminal
and archived. Observation deadlines never stop or restart model jobs.

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
python3 ROOT/source/test2026/metal/finalize_paired_norm_rope_native_checks.py CANDIDATE ROOT/logs/paired-norm-rope-native-finalize-rearchive-20261016 54126 --rearchive
python3 ROOT/source/test2026/metal/continue_paired_norm_rope_native.py CANDIDATE ROOT/logs/paired-norm-rope-native-continuation-rearchive-20261016 88835 --finalizer-folder ROOT/logs/paired-norm-rope-native-finalize-rearchive-20261016
```

The full 2x decode / half-TTFT goal for all matched models and batches remains
active. Fresh repeated vLLM-Metal measurements are required after any retained
native change.
