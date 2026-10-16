# Private native width64 recurrent prefill investigation

Status: all 32 fresh native stages and all 72 complete native outputs pass
identical independent Mac/Windows audits. The first native model cohort fails
the predeclared TTFT retention/preservation screen. Reject without confirmation
or promotion; accepted serving source remains d52. No new matched-vLLM target
count is established.

The [two complete component cohorts](../delta-value-tiles-gpu/README.md) select
only width-64 prefill. Copy accepted d52 into a new private tree with 1,410
source files, 12 exact transformations and 4 added helpers. Preserve all
before/after/source identities. The actual Mac preparation matches Windows
regeneration of every transformation and added helper.
[Frozen preparation and prospective gates](preparation.json).

## Native policy and implementation

GARNET_METAL_DELTA_VALUE_TILE64 defaults OFF. Append the confirmed kernel to
the original main shader only under that compile flag. Select only batch 1/2,
chunk 80, 16 heads, key/value dimension 128, aligned FP32 layouts with zero
buffer offsets, and supported 128-thread/32-lane pipelines. Divide the original
16-column group count by four when selecting width 64. Decode and other shapes
retain the original kernel and dispatch geometry.

Reuse the original state-kernel argument, initialized-producer, device,
input/output alias, pending-writer and secondary-output validation. Preserve
original independent output/state allocations and writes; no packed weights,
state-layout changes, numerical approximation or input/prefix/image reuse.
Keep all per-value arithmetic expressions, Q/K normalization and barriers.

Expose real context, graph and resident-session dispatch counters. Reset them
on Begin/Abort, graph/reset/close and session/reset/failure. Capture actual
prefill and every decode step in workers. Agreed text/coder and all decode
requests must report zero selected calls. For VL, count the 18 linear layers
for each actual chunk-80 prefill in the original plan. Original graph/session
trained references assert the selected/fallback counters on every carried step.

## Validation and current observation

Run all original 30 native stages afresh plus linked width64 primitive compile
and execution, 32 total. Before execution, regenerate/parse the full driver and
independent auditor, preserving checkpoints 25/31/32 and all original numerical
bounds. A local AST-only expansion initially used a dummy source root and could
not find the frozen template; correct only that local preflight path before
deployment. No runtime/driver or bound repair was made.
[Preflight event record](preflight-events.json).

The actual candidate library SHA256 is:
`03ac447d4db2c18e03f8090e79eefa1e0519358e09d9b89a66e1d37e28bf9ebb`.
Accepted d52 control remains:
`d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e`.

Check process 94442 builds the library and its actual public-API linked
primitive. Compare against the original width-16 Metal kernel, three carried
steps over batches 1/2/3, chunks 1/16/80, heads 2/16 and values 96/128/256.
All 162 checks pass: 6 selected and 156 original-fallback cases, 74,805,120
logical output/state FP32 words bit-identical. Changed prepared inputs, finite
results, pending unreadability, abort invalidation/recovery, state alias,
uninitialized producers and foreign writers are checked. CTest and the
existing vector primitive also pass. This linked primitive result does not
establish complete native correctness or model performance.

At 06:14:33 UTC, checks 94442, finalizer 98584 and continuation 1941 are verified
live. Nine whole stages pass, including full vision grid and real-head checks;
the suite is within text exact-tail cases. No full32 gate or model performance
job exists. [Exact process/stage/primitive observation](latest-observation.json),
[earlier observation](observation.json).

SSH later times out over both saved VPN routes while the local adapters remain
up; the Mac's VPN peer briefly reports offline. Do not infer process termination
or restart any job. At 06:26:11 UTC, the connection recovers and all three exact
processes are live. The check and finalizer birth times match their original
watcher metadata. Twenty-four whole stages pass, including both text/coder
tail suites, private KV, serving, mixed-image pipeline, layers, heads and
trained vision checks. No full32 gate or native model timing exists at that
observation. [Recovery and exact stage record](network-recovery-observation.json).

The finalizer watches that exact check process, archives terminal success or
failure, then independently regenerates/audits all 32 successful stages. The
archive explicitly includes the original vector/MLP auditor dependencies and
all declared source bytes. Require the linked primitive coverage and all 48
original extended/wide trained route steps: 12 selected, 36 fallback. Preserve
the original error bounds, hidden/conv/state/KV comparisons, invalid controls,
resident/explicit equality, resets and close behavior.

Only a successful complete audit publishes a gate. No source/runtime/driver
consumed by a live job changes before terminal archive. The continuation also
snapshots and pins all direct shared Mac tools; freeze that directory until
the continuation is terminal and archived. An observation deadline stops no
job and never authorizes a restart.

## Prospective model performance screen

Behind the terminal archived/audited full32 gate and a fresh raw re-audit,
run all six matched native profiles sequentially against accepted d52. Keep
original models, learned values/FP32 arithmetic, images/preprocessing, prompts,
limits, complete output checks and both VL B2 chunk plans [80]. Keep vector
vision enabled in both policies and verify actual width64 routes on all 72
complete outputs, including prefill and every decode step. Initialization,
first-after-initialization, warm results and individual-worker RSS remain
separate; hardware caches remain unflushed.

The gate fixed in preparation requires >=1% warm TTFT reduction at each VL
batch in two complete unchanged cohorts, every profile's decode fraction
>=.98 and TTFT fraction <=1.02, with all original outputs/routes passing.
Only a complete passing initial cohort admits one unchanged full confirmation;
no favorable third repeat. Report all failures and setup/memory costs. This
retention screen is separate from the unchanged full 2x decode/half-TTFT goal.
Fresh repeated matched-vLLM measurements remain mandatory after any retained
implementation. Component kernel ratios never update target counts.

## Executed commands

ROOT=`/Users/zsdx/garnet-metal-investigation`,
CANDIDATE=`ROOT/delta-value-tile64-native-investigation-20261016`:

```text
python3 ROOT/source/test2026/metal/prepare_delta_value_tiles_native.py ROOT CANDIDATE
python3 ROOT/source/test2026/metal/launch_delta_value_tiles_native_checks.py CANDIDATE CANDIDATE/logs/delta-value-tile64-native-checks-20261016 checks
python3 ROOT/source/test2026/metal/finalize_delta_value_tiles_native_checks.py CANDIDATE ROOT/logs/delta-value-tile64-native-finalize-20261016 94442
python3 ROOT/source/test2026/metal/continue_delta_value_tiles_native.py CANDIDATE ROOT/logs/delta-value-tile64-native-continuation-20261016 98584
```

The model comparison helpers' comparator, outer and inner independent 72-output
auditors regenerated and parsed locally without executing inference. That
preflight established preparation only; terminal measurements are recorded
below. Full matched-vLLM goal remains active.

## Terminal full32 audit and native model rejection

Check 94442 exits 0 in 1,452.645279584 seconds with all 32 stages fresh and no
reused compile/validation stages. Finalizer 98584 exits 0 after the complete
1,880-file archive audit. The full audit regenerates all 12 changes and the
executed driver, retains all original checks and passes every 48 trained-route
step: 12 selected and 36 fallback. Worst original trained-bound ratio is
0.5999873838; worst wide-reference ratio is 0.5432836183, each <=1.
[Windows full32 audit](checks/verification.json),
[identical Mac full32 audit](checks/mac-verification.json).

The first six-profile native cohort 22203 exits 0 in 192.920540834 seconds.
Continuation 1941 archives it, audits all 72 complete outputs and actual routes,
then rejects the fixed screen and exits 0. Its 1,312.108444209-second duration
includes waiting for correctness; it is not inference latency. The 6,092-file
archive preserves 1,410 source files and 1,241 external tools. Complete raw
checks and the same native decision reproduce independently on both hosts.
[Complete native model audit](initial/verification.json),
[identical Mac model audit](initial/mac-verification.json),
[fixed initial decision](decision-initial.json),
[terminal controller](continuation/archive-verification.json).

VL B1 warm TTFT fraction 0.998322 misses the required <=.99; VL B2 fraction
0.988069 passes. Coder B2 fraction 1.037333 exceeds the 1.02 preservation limit.
All decode fractions stay >=.98. Reject without confirmation, promotion or a
favorable retry. The changed kernel is not selected for coder, so its observed
latency difference does not establish a causal width64 regression. Do not
replace complete-model measurements with the prior 9–13% component kernel gain.

Warm decode TPS is per row; aggregate throughput is TPS times batch. Each
policy has four warm complete requests, with initialization and first-after-init
separate. These are one complete native cohort, not repeated matched-vLLM ratios.

| Native model | Batch | Decode TPS control / candidate | TTFT ms control / candidate | Decode fraction | TTFT fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 73.9653 / 74.8021 | 36.0923 / 36.5307 | 1.011313 | 1.012148 |
| Qwen3-0.6B | 2 | 71.4519 / 71.0217 | 56.7331 / 56.3283 | 0.993979 | 0.992865 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 87.7866 / 88.6825 | 27.7902 / 27.3399 | 1.010205 | 0.983798 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 83.6361 / 84.1895 | 43.0413 / 44.6482 | 1.006617 | 1.037333 |
| Qwen3.5-0.8B | 1 | 60.3426 / 61.5050 | 174.8774 / 174.5839 | 1.019265 | 0.998322 |
| Qwen3.5-0.8B | 2 | 55.3409 / 54.8818 | 291.6713 / 288.1915 | 0.991704 | 0.988069 |

| Setup model | Batch | Init seconds control / candidate | First TTFT ms control / candidate | Mean individual-worker peak RSS delta MiB |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 3.547632 / 3.504776 | 130.1541 / 129.2086 | -7.1562 |
| Qwen3-0.6B | 2 | 3.530979 / 3.531000 | 142.7121 / 143.2173 | -29.0703 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 2.835301 / 2.826777 | 107.0367 / 105.2469 | 2.3125 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 2.836665 / 2.836302 | 118.8019 / 118.7585 | 6.6172 |
| Qwen3.5-0.8B | 1 | 5.490732 / 5.492733 | 266.0298 / 266.1791 | 6.3828 |
| Qwen3.5-0.8B | 2 | 5.474836 / 5.481856 | 378.8402 / 385.9722 | -6.6172 |

Peak RSS spans initialization and all worker activity; the difference does not
isolate this kernel's allocations. Hardware caches remain unflushed, and
first-after-init is not a hardware-cold observation. All consumed processes are
confirmed terminal and archived before starting the next isolated experiment.

| Archive | SHA256 |
| --- | --- |
| Fresh full32 checks, 1,880 files | `a847754655e4bb477102db80678b72d687cf6c739cff9fcf6f74d715eef7ecda` |
| Complete native cohort, 6,092 files | `6e2a1cf78b3ebda2ee3cff23eb9fb2d5ad4aa44deb7568150a490289097473a4` |
| Terminal continuation, 1,270 files | `b22c3531007f4d234e276820c22418bce688558993937006fc8de1e2af8a2054` |
| Terminal finalizer, 17 files | `a674015d4245b03847be10d056770a28f9819f4e9ab504abf027a90527aa0fa6` |

