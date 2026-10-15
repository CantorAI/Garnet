# Native row80 NDArray candidate: validation in progress

Status: initial build/self-guard attempt terminal and archived; corrected
actual PID84121 running fresh validation. Its build, NDArray primitive, seven
CTests, full vision grid and trained head stages have completed. The entire
30-stage suite and its terminal evidence audit remain incomplete.
No full-model timing or promotion. Retained serving libraryc186 remains selected.

Two [trained component cohorts](../ndarray-prefill/README.md) select only
Qwen3.5 B1 F32 under the predeclared >=5% GPU/wall and null-calibration rule.
Integrate this path in a separate Mac directory:
`/Users/zsdx/garnet-metal-investigation/ndarray-native-investigation-20261015`.
The parent investigation, original Mac checkout, GPT-OSS and Pi sources are
untouched. [Preparation identities](preparation.json) list1,333 copied files and
eight transformed existing files. This manifest describes prepared sources;
it does not prove a successful build or inference.

Exact change: private native context uses the original learned BF16 bits
expanded exactly into retained FP32 NDArray weight storage; activations/results
are FP32, RHS uses deferred transpose, sourceCount2/unit alpha. Only non-vision,
non-float-key projections with physical80 input rows qualify. Decode rows1/2
remain native. The prototype's global policy variable is removed. No model-type
API is implied by a shape gate: validation must establish its supported scope.

At most256 view/product entries retain input/output/weight owners. New entries
after the bound use the original native path; existing entries remain usable.
Older OS versions retain native fallback. The FP32 weight cache adds memory;
actual memory, internal MPS packing, first-use time and numerical behavior at
other layers are unmeasured. Allocation/type errors still fail and are not
silently reclassified as success. New session/backend/worker counters record
actual prefill calls and verify decode exclusion in a later model comparison.

Run the original28 fresh stages, without test reuse, plus a new compile/run
primitive covering exact integer products, input update,256 admitted views,
14 bounded-cache fallbacks, reuse after the cap, aborted-output rejection,
recovery and row79 fallback. Primitive source is
[ndarray_prefill_cache_gpu.mm](../../../test2026/metal/ndarray_prefill_cache_gpu.mm).
The30-stage runner retains original trained references, complete-grid vision,
head queries, numerical bounds, prefix/screened recovery and carried-state tests.
A passing primitive alone does not prove full validation or model performance.

## Archived harness failure and scoped repair

Initial PID81353 compiled the native library and cache primitive, then its
guard mistakenly treated that exact owned executable as a foreign GPU probe.
Thirteen primitive attempts were interrupted; no primitive result was accepted.
Send SIGINT only to verified Python driver group81353 (members81353/81446)
between primitive attempts. It exits1 in120.330340375s. Foreign jobs untouched.
Preserve1,374-file archive
`2f827d302fa75759c9a85c0cafdf5ff1c65a21235b913be5016039113bb8ce1e`
before any repair. [Mac/Windows failure audit](self-guard-failure/verification.json)
regenerates all eight changes and verifies1,251 archived text source identities,
three accepted setup stages and all interrupted attempts. Built library
`8e4e9b4ea9da3dec731ed4f9452bf4d585de099b8d6d1d88702b3d57b5b42c15`
is not promoted or claimed fully validated.

The repaired external driver excludes only the exact expected primitive path
when its process-group ID equals the current owned stage's group. All unknown
or foreign GPU/inference processes remain subject to the original guards.
Private serving source and original driver bytes remain unchanged. Use a
fresh `build-ndarray-prefill-shape80-guardrepair` and rerun all30 stages without
reusing the previous build/tests. External driver snapshots and expanded source
are captured in the new log; runner checks their identities at completion.
Only descriptor semantics and the guard are repaired; no numerical limit changes.

```text
python3 source/test2026/metal/prepare_ndarray_native.py ROOT ROOT/ndarray-native-investigation-20261015
python3 source/test2026/metal/launch_ndarray_native_checks.py CANDIDATE CANDIDATE/logs/ndarray-native-checks-20261015 checks
python3 source/test2026/metal/archive_ndarray_native_checks.py CANDIDATE ARCHIVE
python3 source/test2026/metal/launch_ndarray_native_checks_repaired.py CANDIDATE CANDIDATE/logs/ndarray-native-checks-repaired-20261015 checks
python3 source/test2026/metal/archive_ndarray_native_repaired_checks.py CANDIDATE REPAIRED_ARCHIVE
python3 source/test2026/metal/verify_ndarray_native_checks.py REPAIRED_EXTRACTED VERIFICATION.json
```

Placeholders refer to the verified Mac investigation and confined candidate.
Exact commands and stdout/stderr live under that candidate's logs. No consumed
runtime/source/driver is edited until the job is terminal and archived. Future
work must archive/audit failures before repair, verify all30 stages before
promotion, then execute complete-model ABBA with matched full outputs, separate
init/first/warm measurements and RSS. Only a repeated gain warrants fresh
matched vLLM cohorts. No comparator or other inference runs concurrently.

## Complete-model comparison prepared and conditionally queued

Controller90454 is waiting on exact validation PID84121. It will preserve the
terminal validation archive, extract and independently audit all30 stages,
and launch native ABBA only after success. Failure is archived and prevents
benchmark admission. No restart follows from an observation timeout.
The expanded comparison and all original-gate verifier transformations pass
CPU-only syntax/needle checks locally; expanded comparison also passes on Mac.
These checks do not establish actual trained inference or performance.

Measure all six original model/batch profiles, even though only VL B1 should
execute row80 NDArray. Run control/candidate/candidate/control as four separate
sequential workers per profile, with three fresh requests per worker:72 complete
outputs in the full cohort. Require exact complete token IDs, prompt IDs,
finished flags, image/preprocessing identities, actual original latency bundle
and unchanged limits/chunk plans. Prefix/speculative/image/input reuse stays off.
Independent checkpoint-header BF16 layer shapes and physical chunk plans derive
the expected NDArray count; measured counters must match, including zero decode
dispatches and zero calls in unaffected workloads. Do not infer routing from
the backend label alone.

Control is retainedc186; candidate is the freshly validated guardrepair build.
Init, first-after-init, warm and peak process RSS remain distinct. OS/driver
caches are unflushed; no hardware-cold claim. Added exact-F32 weight storage,
lazy allocation and any first-use regression must be reported. Full source and
external-driver closures are snapshotted and rehashed at completion; preserve
every consumed runtime/driver until terminal archival.

Before native timing, declare an advancement screen: VL B1 warm TTFT must be
at least3% lower in both native cohorts, with no model/batch showing a repeated
decode regression of2% or more. These pragmatic thresholds are not confidence
intervals. Report all unaffected-profile variation and every accuracy/output
failure. Only an admitted first cohort warrants an unchanged confirmation;
do not search repeats for a favorable mean. Promotion is distinct from the
full goal: >=2x decode AND <=0.5x TTFT still need two fresh matched vLLM cohorts
for all six workloads. No new target pass follows from this queue.

```text
python3 source/test2026/metal/finalize_ndarray_native_checks.py CANDIDATE CANDIDATE/logs/ndarray-native-finalizer-20261015
python3 source/test2026/metal/archive_ndarray_native_profiles.py CANDIDATE NATIVE_ARCHIVE
python3 source/test2026/metal/verify_ndarray_native_profiles.py NATIVE_EXTRACTED NATIVE_VERIFICATION.json
```
