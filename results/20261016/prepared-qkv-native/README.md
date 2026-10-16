# Private native text Q/K/V fusion after repeated component selection

Status: rejected after a trained key-cache numerical failure. The accepted
implementation passes the identical saved fixture. No candidate native timing
or promotion was admitted. The historical launch notes below are superseded
by this terminal result; the serving implementation remains unchanged.

## Final rejection

The original driver exited 1 after 25 passing stages because its intermediate
stage-count assertion still expected 23. Preserve that failed run and both
terminal controllers before resuming only the missing stages. The resumed
driver retains the original 25 records, source and library unchanged; six fresh
recovery stages pass. The new trained test then fails at layer 0, batch 2,
chunk 22, physical rows 44, first state step. The suite has 31 passing stages
and an incomplete failing trained stage, not a full 33-stage pass.

| Implementation on identical saved input | Worst original key error bound ratio | Result |
| --- | ---: | --- |
| Fused Q/K/V candidate | 1.516545714435 | Fail |
| Accepted d52 control, all three state steps | 0.520408486838 | Pass |

A ratio greater than 1 violates the unchanged bound. The diagnostic verifies
all 21 saved input and expected-buffer identities and uses the original
accepted driver. Both candidate session and graph fusion counters equal 1
on the failing case. Exact packed weight values and FP32 arithmetic did not
ensure acceptable downstream key-cache numerics. This observation does not
identify an internal MPS rounding algorithm.

The failed validation and its stopped controllers are frozen. The rejection
bundle contains 2,195 files, original and resumed failures, reference fixture,
control driver, and successful identical-input diagnostic. Both Mac and Windows
audits pass. Archive SHA256:
`bca55a5986356f6bf6f7c2320d59fb4d71f37f4a40f6b0cb4b027f1a6be6e556`.
[Rejection audit](rejection/verification.json),
[archive identity](rejection/archive-verification.json),
[failed controller identities](count-failure-controllers/archive-verification.json).

The full-QKV performance continuation stopped at the validation failure; no
native model benchmark ran. The next algorithm packs Q/V only and preserves
the original K GPU projection, with separate primitive, learned numerical,
timing and native gates. Full-QKV component measurements cannot establish a
Q/V or model gain.

Selection: two complete unchanged learned component cohorts select only
Qwen3-0.6B. Keep coder excluded despite a favorable repeat after initial failure;
VL is slower. [Complete component evidence](../prepared-union-qkv/README.md).

Copy the retained validated d52 vector-vision source into a fresh private root.
Apply exactly12 transformations and add5 source helpers; retain original copies
and every before/after/source hash. Build an OFF-by-default feature option
GARNET_METAL_PREPARED_QKV_TEXT. Original independent weights, output allocations,
all three projection writes, FP32 arithmetic and unsupported fallbacks remain.
No activation/input/prefix reuse or speculation. [Frozen preparation](preparation.json).

The graph recognizes only three consecutive projections of the same input,
with exact named q/k/v BF16 weights in the same model.layers attention prefix,
confirmed widths2048/1024/1024 and inner1024. Only physical rows24/44 encode the
prepared route; profiling and unsupported layouts/shapes retain original code.
The context rejects input/output or output/output alias, unready input and
foreign pending writers before encoding any candidate work. All three outputs
remain materialized, so later consumers/public results are preserved.

The resident session has its own GPU context. During session build, prepare
each recognized triple for its supported physical prefill row count on that
session context, using unchanged shared BF16 owners also used by future prefill
graphs. Do not pack lazily during the first encoded frame. Standalone graph.run
prepares on its own context before Begin and includes that cost in its first
execution. Report model setup, first-after-init and warm timings separately;
do not claim preparation is free or hardware caches were flushed. Extra union
payload/scratch are real costs, and preserved original FP32 key caches can have
different first-use allocations. Actual native RSS remains to be measured.

Publish actual graph/session/prefill counters and every decode-step counter.
Predeclared later native selection requires both text batches to reduce warm
TTFT by>=3% in two complete unchanged native cohorts, with no profile decode
reduction>2%, all six model/batch profiles and every original output/numerical
gate. Retained d52 vector control library SHA256:
`d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e`.
Do not compare against old scalar control to attribute vision gains to QKV.
No native win may be divided by an old vLLM time to claim a fresh target ratio.

## Fresh correctness protocol and actual running status

Require all original30 validation stages, linked actual-library QKV primitive
compile/execution, and one additional trained random-input text reference stage
(33 total). The new trained stage preserves original dense block bounds and
state checks while testing first/last learned layers, B1/B2 and chunks24/22:
four serving and four fallback profiles, three changed-input/state steps each.
Assert both explicit graph and resident-session actual counters. Preserve every
fixture/expected file by identity for reproduction. Retain invalid-control
rejection, private state, full-output equality, paused/tail, reset and closed
graph controls. No numerical bound is widened.

Actual launcher10766 is confirmed live; configure/build, vector primitive,
QKV compile/primitive, all7 CTests, full vision grids and trained real heads
have exited0. Original exact dense tail cases are executing. This is partial
validation, not a full33 pass. The main/retained libraries stay untouched.

Finalizer15326 watches only10766, verified live. It archives terminal success
or failure, independently regenerates and audits a successful full33 suite,
and publishes its gate file only on complete success. It does not launch any
performance comparison. Source/runtime/driver files remain frozen until the
job is terminal and archived; an observation timeout is not a restart signal.

Executed commands, ROOT=`/Users/zsdx/garnet-metal-investigation`,
CANDIDATE=`ROOT/prepared-qkv-native-investigation-20261016`:

```text
python3 ROOT/source/test2026/metal/prepare_prepared_qkv_native.py ROOT CANDIDATE
python3 ROOT/source/test2026/metal/launch_prepared_qkv_native_checks.py CANDIDATE CANDIDATE/logs/prepared-qkv-native-checks-20261016 checks
python3 ROOT/source/test2026/metal/finalize_prepared_qkv_native_checks.py CANDIDATE ROOT/logs/prepared-qkv-native-finalize-20261016 10766
```

Local preflight regenerates all12 source transformations and complete33 runner/
auditor expansions without invoking inference. An initial dry-generation call
needed its output parent directory created; after that setup correction,
generation succeeds. No GPU run, runtime or tolerance was repaired by that
local path fix. Final native result and cost tables remain pending.

## Sequential continuation armed after the exact validation finalizer

Actual continuation26534 watches only verified-live finalizer15326. Once that
process is terminal, it first archives the finalizer's commands/source/logs and
published gate, then proceeds only on complete success. Native launch re-audits
the frozen full33 archive with its actual archive identity before timing.

The continuation runs one complete six-profile native cohort, waits for its
exact launcher to become terminal, archives success or failure, independently
audits all72 complete outputs and actual per-request/prefill/decode routes,
and applies the declared3% both-text/2% all-profile-decode screen. Only a
passing complete initial screen admits one unchanged complete confirmation.
No runtime overlaps another, and no observation timeout triggers a restart.
Accepted vector vision is enabled in both policies; only candidate text can
report prepared QKV calls. Old scalar control is excluded from attribution.

Before deployment, expanded comparator and original72-output auditor syntax/
transformation preflights pass. The known historical expanded-filename issue
uses Path.with_suffix consistently here. The full source manifest includes
small binary fixtures and docs; native archives explicitly retain every source
and snapshot byte instead of silently dropping suffixes. All executed commands,
raw output records, source/binary/asset identities, clocks, first-use costs,
mean individual RSS, exclusions and decisions remain in the terminal archives.

The finalizer alone does not benchmark; this separate gated continuation does.
At the last authoritative observation, validation10766, finalizer15326 and
continuation26534 are live. Sixteen complete validation stages have exited0;
remaining trained checks are running. No native performance result is claimed.

```text
python3 ROOT/source/test2026/metal/continue_prepared_qkv_native.py CANDIDATE ROOT/logs/prepared-qkv-native-continuation-20261016 15326
```
