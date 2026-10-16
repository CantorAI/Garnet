# Private native width64 recurrent prefill investigation

Status: the private library builds, the actual linked primitive and CTest pass,
and nine complete validation stages have exited 0 at the saved observation.
The full 32-stage correctness suite is still running. No complete native model
timing or new matched-vLLM result is established. Accepted serving source stays
on d52.

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
auditors regenerate and parse locally without executing inference. They are
prepared/armed, not a performance result. Full matched-vLLM goal remains active.
