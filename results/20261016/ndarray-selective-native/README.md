# Selective no-exact-half NDArray: isolated native validation

Status: initial PID22114 failed one original CTest after successful compile
and selective cache primitive. Failure archived/audited; packed-layout guard
repaired in a new isolated source/build. Fresh full30 PID23582 is running.
Live observation: repaired compile, selective primitive and all seven CTests
have passed; subsequent trained gates are running. This is partial validation.
Complete validation and performance remain unproven until terminal evidence
is archived and audited. Retained serving library
`c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1`
and defaults remain selected. Overall three-model matched target is unmet.

Admission: both [trained component cohorts](../ndarray-selective-prefill/README.md)
pass the predeclared >=5% GPU and wall improvement screen for both VL batches
with corresponding nulls within5%. All selective numerical checks pass under
the original bounds. This justifies a native test; it does not predict a
complete-model speedup.

Private Mac directory:
`/Users/zsdx/garnet-metal-investigation/ndarray-selective-native-packed-investigation-20261016`.
Copy1371 source files and apply eight confined transforms, recording the full
before/after closure in `preparation.json`. Baseline parent/library, original
Mac checkout, GPT-OSS and Pi checkout stay unchanged. Add counters to native
session, prefill profile and workers, with backend marker
`ndarray_prefill_selective_80_160`.

The repaired route requires packed FP32 physical rows80/160, non-vision, non-special key projection,
and absence of BOTH original exact-half matrix and scales. Preserve all other
native routes. The full checkpoint audit predicts only VL first-layer QKV
will qualify: **one NDArray prefill dispatch per serving image, zero per decode**,
with the existing182 common-half calls retained. Text/coder expect zero NDArray.
Actual counters and complete chunk plans must prove those claims during native
inference; profile labels alone are insufficient. Extra logical F32 payload
24MiB is an estimate until observed; peak RSS is measured separately.

Use two-source FP32 unit-alpha products, exact BF16 expansion, virtual weight
transpose, hoisted source pointer, strong owners and256-product cache with
original native fallback after the cap or on older OS. A dedicated actual-library
primitive covers unsupported tiny-normal values, both serving row shapes,
unchanged admitted-half routing, fresh input updates, cache256/fallback14,
cached-after-cap reuse and abort/recovery. It supplements all original28 stages.
Existing exact-owned-primitive process-group guard repair is reused; competing
inference remains forbidden.

Before native timing: require fresh validation, all original numerical and full
output predicates, original FP64 bounds, source/library/asset identity, and
independent frozen archive audit. Then run all six complete native ABBA profiles,
keeping initialization, first-after-init, warm TTFT, decode and peak RSS separate.
For the native screen, require at least3% warm TTFT reduction in both active VL
batch profiles in an initial complete cohort before seeking an unchanged fresh
confirmation; require that gain in both cohorts and no repeated >=2% decode
regression in any profile before promotion. Reject failed screens without
searching favorable repeats. No new vLLM ratios from native-only timings.
The complete goal still requires fresh repeated matched vLLM results across
all six profiles; this gate does not redefine that goal.

```text
python3 prepare_ndarray_selective_native.py PARENT CANDIDATE
python3 launch_ndarray_selective_native_checks.py CANDIDATE OUTPUT checks
```

Only AST/source-generation preflights preceded the repaired launch. They are
not evidence of a successful repaired native compile or inference. Freeze all serving source and
external drivers until the specific validation process is terminal and archived.

## Initial validation failure and packed-layout repair

Initial PID22114 exits1 in53.492590250s. The new library compiles; its actual
selective primitive passes exact/update/cache256/fallback14/cached-after-cap/
abort-recovery/both serving shapes/native-half tests. All six other CTests
pass, but `metal_wide_batch_prefill_reference` fails with
`NDArray prototype requires packed FP32 matrices`; CTest exits8 without a
guard interruption. Later trained/model stages never execute. This is a real
compatibility defect, not a contention event or a stale benchmark assertion.

The row160 extension intercepted an unsupported padded FP32 layout, then
threw at the prototype's packed-layout assertion. Add packed FP32 type/stride
checks to the outer admission predicate, leaving those layouts on the original
native fallback. Do not relax the existing test or numerical bounds. Packed
trained-component shapes still take the same tested NDArray operation. Old
candidate/source/build and failure remain preserved; no patch to that snapshot.

Archive1366 files, SHA256
`a04efb9061130bb856a399168e23967f5ecedd6a9254632507a20508b15dbf68`.
Mac and Windows failure auditors verify all members, regenerate eight source
transforms, four initial accepted stages, six passing/one failed original
CTest and uninterrupted process guards. [Failure audit](initial-validation/failure-verification.json)
and [raw manifest](initial-validation/evidence/manifest.json).

Only after terminal/archive/audits, update factory/driver and prepare a new
directory/build `build-ndarray-selective-prefill-packedguard`, with no validation
reuse. Original retained context SHA
`813c5679777de90d487ae5e5a481b0b076e09e4325add22b42ea3bd6f6818a59`;
initial transformed SHA
`488005e0f17894e06aa294b3d160869bc65f0a04467c1449066588613b565c60`;
repaired context SHA
`7410a44d4028d47188baa97e9ab030780c8438ddcc65de913e86577c142be3ae`.
All are source hashes, not compiled library hashes. All original full30 stages
must pass freshly before any native timing admission.

Controller PID25901 waits the exact validation PID23582, then archives its
terminal success/failure and independently audits successful full30 evidence.
It performs no inference and cannot promote the candidate or mark the overall
goal complete. Controller/source identities, observations and results live at
`logs/ndarray-selective-native-finalize-20261016` on the parent Mac investigation.

A local extraction attempted before SCP13062 completed received Windows file-lock
`PermissionError`; it created no evidence folder and admitted no result.
After that exact transfer reached terminal exit0, extraction and both-host
failure audits succeeded. This local workflow mistake does not change the
captured CTest failure or require rerunning the failed GPU dataset.

## Full-model comparison prepared and gated

New continuation PID33009 waits exact controller25901. Only terminal full30,
frozen archive and successful independent audit permit dispatch of the original
six native ABBA profiles, sequentially. It then freezes/audits72 complete outputs
and applies the existing both-VL >=3% TTFT screen. A fresh unchanged six-profile
confirmation is dispatched only if that initial screen passes; failed screens
stop without searching favorable repeats. Source/drivers stay frozen through
terminal/archive. The controller cannot promote or mark the full goal complete.

Comparator preserves original matched prompts, outputs, preprocessing, chunk
plans,32-bit arithmetic, original bounds and all clock/reset/ownership checks.
It copies the exact captured full checkpoint admission report into every profile,
checks its SHA and learned tensor shapes/checkpoint identity, and derives both
the selective calls and retained common-half calls from that report. Every
candidate VL image must execute1 NDArray prefill/zero decode plus182 common-half
calls; text/coder must keep0 NDArray and their original168/143 common-half calls
per eligible chunk. Actual values must still prove these predictions.

A local [guard preflight](profile-guard-preflight.json) evaluates the exact AST
guard expressions against36 frozen actual retained-control requests from the
previous complete cohort. All observed common-half counts agree; new candidate
counts are explicitly unexecuted expectations in that preflight. This avoids
the old broad-NDArray assumption that any new route eliminates common half.
The old data is used only to check guards, never reused for new timing ratios.

The controller's first full30 audit is preserved. Before profiling, repeat the
same CPU-only evidence audit with the explicit archive argument to include its
SHA in a separate file; compare all original audit fields unchanged. No GPU
stage is repeated. Generated comparison scripts get distinct filenames for
each profile. The frozen standalone native auditor also regenerates them,
checks every original output/clock predicate, all72 retained common-half
counters, source closures, initialization/first/warm TTFT and worker peak RSS.

```text
python3 continue_ndarray_selective_native.py CANDIDATE OUTPUT 25901
python3 launch_ndarray_selective_native_profiles.py CANDIDATE OUTPUT profiles
python3 archive_ndarray_selective_native_profiles.py CANDIDATE ARCHIVE TRIAL
python -X utf8 test2026/metal/verify_ndarray_selective_native_profiles.py EXTRACTED_RESULTS AUDIT_JSON TRIAL
python -X utf8 test2026/metal/analyze_ndarray_selective_native_results.py RESULT_ROOT initial
```

Prepared/dispatched watcher is not evidence of full30 or performance success.
No fresh vLLM comparison or goal-target update has occurred.
