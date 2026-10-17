# Garnet Metal optimization research log

Purpose: maintain an evidence-linked laboratory record for a future thesis or
paper. This is a research record, not a publication draft or a claim that the
performance goal has been achieved. Started 2026-10-15; historical experiments
are indexed from their original reports rather than reconstructed from memory.

## Evidence and coverage

The [experiment index](metal-method-evidence/INDEX.md) covers all locally present
experiment README reports under `results/` and `test2026/metal/results/`, including
rejected variants and failed attempts documented within those reports.
[JSON](metal-method-evidence/index.json) preserves complete report text, report
SHA-256, documented commands/archive hashes, and adjacent verification paths.
[CSV](metal-method-evidence/index.csv) provides one record per report.
These are report records, not a count of independent experiments or replicates.

Historical implementation context also remains in
[the Metal README](../../test2026/metal/README.md) and
[bring-up history](../../test2026/metal/BRINGUP_HISTORY.md).
Original result reports and raw artifacts remain the authority. A historical
report saying "running" is not a current job-status assertion; dated journal
entries below record later observations without rewriting frozen evidence.

## Research question and fixed comparison

Can native Garnet Metal/MPS achieve at least 2x vLLM-Metal per-row decode tokens/s
and at most half its TTFT on the same Apple M4 Mac mini, across the agreed three
models and batch sizes 1/2? Both targets must repeat for every agreed profile.

- Hardware: Mac16,10; M4, 10 CPU cores (4 performance/6 efficiency), 10 GPU cores,
  16 GiB unified memory; macOS 15.6.1. Other Apple chips are unmeasured.
- Models: Qwen3-0.6B; Qwen2.5-Coder-0.5B-Instruct; Qwen3.5-0.8B with image input.
- Preserve checkpoint identities, learned weight values, prompts/token IDs,
  image preprocessing profile, output caps, stop rules and numerical bounds.
- The matched comparator cohort uses FP32 arithmetic and lossless FP32-expanded
  comparator weights. Garnet can retain BF16 storage or an exact representation
  while using FP32 arithmetic. This is a precision-matched comparison with
  different storage implementations, not a benchmark of stock BF16 vLLM.
- Run runtimes sequentially, with resource guards. No competing inference,
  prefix reuse, speculative decoding or reuse of a prior image encoding.
- Separate initialization, first request after initialization and warm requests.
  OS/driver caches are not flushed; do not call these hardware-cold measurements.

Comparator versions in the latest archived cohort: vLLM 0.30, vLLM-Metal 0.30,
MLX 0.32.1, mlx-lm 0.32, mlx-vlm 0.6.17, Torch 2.13 and Transformers 5.19.
Use archived environment records for exact package/build identities.

## Experimental method and reporting rules

1. Profile the current implementation, including instrumentation overhead.
2. State a mechanism and isolate one candidate against a contemporaneous control.
3. Test original checkpoint values against original-bit comparisons and/or the
   independent FP64/reference envelope appropriate to that operation. Preserve
   the original tolerances; test tails, fallback, ownership, abort and recovery.
4. Time an admitted component using sequential ABBA windows. Component timings
   exclude stated setup costs and do not establish full-model gains.
5. Integrate promising candidates behind explicit options. Run trained-layer,
   vision-grid, head, state/recovery and complete-output gates before serving tests.
6. Run full-model fresh-worker ABBA and confirmation. For native comparisons,
   use control/candidate/candidate/control; for the matched comparator use
   Garnet/vLLM/vLLM/Garnet. Each profile has two fresh workers per policy and four
   warm requests per policy; each worker's first request is reported separately.
7. Audit original commands, terminal status, source/library/asset hashes,
   complete token sequences and measured counters. Preserve failed archives
   before making repairs. Freeze running sources until terminal and archived.
8. Run a fresh matched vLLM cohort after selecting a new native implementation.
   Never divide a newer native-only result by an older comparator measurement.

Definitions: TTFT is request-to-first-output latency with the scope defined by
the executed runner. Decode tokens/s is per row, excluding first-token work as
defined by that runner. Aggregate batch throughput can be `batch * row TPS`
only where equal row lengths/timing have been verified. Command GPU spans,
host encoding, waiting and wall latency can overlap; do not add them blindly.
Peak child RSS is process memory, not unique Metal allocation size.

Current repeats provide descriptive measurements. Two cohorts and repeated
warm requests do not establish a population confidence interval. Report raw
windows, sample units and all exclusions; do not treat correlated requests as
independent hardware replications or claim statistical significance without
an appropriate analysis.

## Methods tried and decisions

Each evidence link includes implementation details, executed commands and raw
record references. "Retained" describes a measured optional route or historical
bundle; it does not mean a universal default or a production deployment.

| Method / hypothesis | Observed result and decision | Evidence |
| --- | --- | --- |
| GPU-resident decoder state, shared weight graphs, BF16 weight storage with FP32 accumulation, parallel prompt/vision operations: reduce host transfers and repeated work | Implemented and validated during early bring-up. Earlier Garnet-versus-Garnet targets are separate from the later vLLM goal; historical timing exclusions remain documented. | [Implementation and original comparisons](../../test2026/metal/README.md) |
| Sparse GPU timestamp sampling: identify expensive encoder shapes with lower profiling overhead | Full sampling slowed observed decode 25–45%; sparse sampling reduced observed overhead to about 1.6–2.4%. Diagnostic only. Heads are oversampled; raw ranks are not unbiased time shares. | [Full counters](../../results/20261015/counter-trace/README.md), [sparse counters](../../results/20261015/sparse-counter-trace/README.md) |
| Guarded greedy vocabulary-head screening: refine candidate rows with original weights instead of scoring every row equally | Repeated full-model decode gains around 7.5–10.4% in the initial cohort, with confirmation recorded. Complete logits preserve the full head; unsafe queries fall back. Index construction adds initialization cost. | [Native screening](../../results/20261015/screened-head-inference/README.md), [component bounds](../../results/20261015/screened-head/README.md) |
| Ordered prefill streaming and partial state reset: reduce submissions and unnecessary zeroing | Evaluated in combined native bundles; attribute effects only to the corresponding isolated/control comparison. Recovery, reset and actual route counts are included. | [Combined prefill/reset](../../results/20261015/combined-prefill-reset/README.md), [streaming](../../results/20261015/prefill-stream/README.md), [reset](../../results/20261015/screened-prefill-reset/README.md) |
| Original-order vision prefetch: reduce vision-stage waiting while retaining original numerical order | Two native trials reduced VL TTFT about 2.9–3.8%; text B1 TTFT regressed. Remains optional. Full vision embedding values match bit-for-bit. | [Native vision study](../../results/20261015/serial-vision-native/README.md) |
| Wide VL batch-2 prefill: admit 160 physical rows so both 80-slot prompts use one pass | VL B2 TTFT fell 8.25% initially and 7.14% in confirmation. Other profiles mixed; option OFF by default. Four new trained references and 12 carried steps passed. | [Wide prefill](../../results/20261015/wide-batch-prefill/README.md) |
| Four-output interleaved BF16 body: improve decode weight loading while keeping FP32 order | Two full native trials: text decode gains about 4.7–6.1%; coder B1 regresses; VL decode gains about 1.5–3.9% with roughly 1 GB extra RSS. Optional text candidate; no broad promotion. | [Native body integration](../../results/20261015/interleaved-body-native/README.md) |
| Eight-output BF16 body: increase work shared within a SIMD group | Numerical/safety checks passed after an archived compile failure. Mixed timings, mostly regressions; no broad promotion or unchanged repeat. | [Eight-output study](../../results/20261015/wide-body-decode/README.md) |
| Lossless four-output BF16 block storage with coalesced vector loads: reduce weight traffic while reconstructing original bits before FP32 math | Two component trials passed 144 numerical checks and 378 safety profiles. Fixed 64-thread policy improved all six component wall groups, roughly 1.8–14.3%. Full-model benefit was not established by these component timings. | [Repeated component study](../../results/20261015/coalesced-body-block/README.md) |
| Native integration of the lossless block route: test whether component gains survive complete execution | Seven CTests, including the new 80-case suite, passed; full 28-stage validation and initial six-profile benchmark are terminal exit 0 as observed in the dated entry below. Full-output/performance audit and confirmation remain pending. | [Native integration record](../../results/20261015/coalesced-body-native/README.md) |
| Full-checkpoint CPU coverage audit: check compact representation beyond the first layer | 378 admitted matrices (168 text/72 coder/138 VL) reconstruct exactly; compact block fractions about 96%. Added one-copy buffer estimates are 591.06/451.21/705.66 MiB. These are not GPU usage/RSS or speed measurements. | [Coverage source/report](../../results/20261015/coalesced-body-native/checkpoint-coverage) |
| FP32 prefill tiles reading packed four-output weights: reuse decode-friendly storage for prefill | Nine numerical comparisons failed the original envelope; admitted timed policies were slower, about 1.73–3.88x component wall time. Rejected. Do not time or promote failed policies as valid candidates. | [Packed prefill](../../results/20261015/packed-body-prefill/README.md) |
| Pad physical prefill rows to 32/64/128 while preserving logical rows: improve MPS shape efficiency | Nine original-bound failures; mixed/no-op results and substantial regressions in admitted profiles. No broad promotion. One isolated VL signal is not a whole-model TTFT result. | [Row alignment](../../results/20261015/aligned-row-prefill/README.md) |
| Per-projection MPSGraph: test FP32 weight placeholders or exact-half weights cast to FP32 | 400 numeric checks passed for admitted routes after an archived API compile failure. FP32-storage graph results mixed; cast-half graph slower in every group. Four special routes excluded explicitly. No broad prefill replacement. | [MPSGraph study](../../results/20261015/mpsgraph-prefill/README.md) |
| Combined multi-result MPSGraph: amortize submissions; compare FP32 placeholders and static FP32 weights | Includes all 29 first-layer matrices and four input patterns. Private component job queued/running; no audited numerical or performance conclusion at this snapshot. Compile/setup and extra constant storage must be reported separately. | [Combined graph study](../../results/20261015/combined-graph-prefill/README.md) |

Other explored variants include projection grouping, split-K, padded MPS,
activation coalescing, mixed accumulators, alternative head indices/seeds,
state-zero strategies, image preprocessing and encoder/submission changes.
Their individual reports, failures, gates and decisions are retained in the
complete index; the table above is a selected methodological narrative, not
an exhaustive count or an assertion that those variants succeeded.

## Earlier matched comparator cohort: wide-prefill control

Source: [two-cohort budget summary](../../results/20261015/wide-batch-vllm/remaining-budgets.json)
and [comparison report](../../results/20261015/wide-batch-vllm/README.md).
144 complete matching outputs and six actual GPU dtype audits are recorded.
Confirmation warm means are shown below; do not relabel these as the newest
lossless-block candidate's results.

| Model / batch | Garnet row tokens/s | vLLM-Metal row tokens/s | Garnet TTFT ms | vLLM-Metal TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| Text / 1 | 62.6849 | 36.9923 | 38.1346 | 76.0602 |
| Text / 2 | 59.9916 | 27.3401 | 59.5096 | 89.0885 |
| Coder / 1 | 79.7018 | 47.1194 | 26.2606 | 57.6205 |
| Coder / 2 | 76.6158 | 28.4510 | 41.9173 | 83.2458 |
| VL / 1 | 54.0630 | 26.4811 | 178.8031 | 260.1136 |
| VL / 2 | 50.5653 | 17.2623 | 302.3464 | 427.4645 |

Across both cohorts, 4/6 profiles meet the decode target, 1/6 meets the TTFT
target, and **0/6 meet both targets in both cohorts**. The overall goal is unmet.
No quantization, long-context, continuous batching, camera-rate, larger-model,
other-device or general production superiority claim follows from this cohort.

## Dated laboratory entries

### 2026-10-15 — paper log established; native block experiment advances

The evidence index is generated from existing local reports with their original
wording and SHA-256. Historical failed/rejected trials remain included. This
curation introduces no new performance measurements and does not rerun benchmarks.

A key-only read of the exact owned remote job records observed:

| Remote log folder | PID | Observed terminal record |
| --- | ---: | --- |
| `coalesced-body-native-checks-20261015` | 88930 | exit 0; wall 1419.8315604170784 s |
| `coalesced-body-native-20261015` | 88983 | exit 0; wall 1591.253100541886 s |
| `combined-graph-prefill-20261015` | 6026 | PID record present; no terminal record observed |

Folders are under `/Users/zsdx/garnet-metal-investigation/logs/`. Benchmark
wall time includes the dependency wait and is not request latency. The native
benchmark's successful process exit does not substitute for the pending archive,
independent complete-output audit, repeat or fresh vLLM comparison. No new native
TPS/TTFT value is asserted here. Do not alter pinned source while the graph job
is live or before its dependencies are archived.

Control library SHA-256:
`9ee337d9db7d14bcff29a90970c4b5ac9021aee9f89a38a7f6f364051927d77b`.
Native block candidate SHA-256:
`c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1`.

Next: terminal graph capture; archive and independently audit both experiments;
repeat the native block cohort without source changes; promote only if repeated
whole-model benefit and memory/init costs justify it; then rerun matched vLLM.

### 2026-10-15 — initial native block and combined-graph audits

The native initial archive passes independent verification of 2,845 files and
72 complete matching outputs. Actual encoded counters confirm the candidate
route. Decode improves 9.1–16.1% across six profiles against the retained
wide-prefill control, with mixed TTFT. Extra peak individual RSS is about
1.04–1.65 GiB, and initialization adds about 0.84–1.36 s. This comparison
includes both interleaved and compact-block changes, so it does not isolate
the incremental block benefit. Full figures and separate first/init clocks:
[native report](../../results/20261015/coalesced-body-native/README.md).

The combined-graph job exited 0 in 71.6171 s. Its independent audit checks
29 files, 48 numerical records, 12 first-graph records and 512 timed submissions.
Eight admitted profiles reduce component wall latency about 5.5–22.8%; four
text/coder batch-2 policy profiles fail the original native-comparison bound
and have no timed windows. They pass the independent FP64 envelope, which
does not waive the failed comparison gate. First graph calls span
7.22–109.81 ms. Peak RSS and isolated compile time are unmeasured here.
[Graph report](../../results/20261015/combined-graph-prefill/README.md).
No full-model graph TTFT improvement is established.

Native confirmation PID19668 subsequently exited 0 in 178.8188 s. Its combined
archive is being retrieved for independent audit. Successful exit is not yet
an audited repeat result. A new comparator driver is prepared to require two
72-output native audits and a 144-output identity join before freezing the
candidate and running fresh GPU dtype checks and sequential matched profiles.
The overall goal remains unmet.

### 2026-10-15 — native repeat audited; fresh comparator launched

Both native audits and the identity join pass: 3,159 combined archived files,
144 matching complete output requests, identical assets/libraries/sources/cases,
and repeated positive decode gains across all six profiles. Text improves
about 14.8–18.0%, coder 9.1–11.9%, and VL 9.1–14.3% across both trials.
Coder and VL B1 TTFT regress in both trials; memory/init overhead remains.
See the [repeated native report](../../results/20261015/coalesced-body-native/README.md)
and its `verification.json`, `confirmation-verification.json`, and
`combined-verification.json`. The joined request count does not double-count
primitive/reference validations as new independent tests.

The fresh precision-matched comparator launched as PID23200. Its launcher
requires both native audits and their identity join; it freezes the candidate
library and source identities. Three GPU dtype audits precede six sequential
matched profiles, and actual encoded block counters are checked. Results are
pending in [the comparator record](../../results/20261015/coalesced-body-vllm/README.md).
No ratio is inferred from older comparator timings. The full goal remains unmet.

### 2026-10-15 — exact graph error diagnosis queued; current phase clocks

Per-projection numerical diagnosis is queued as PID26997 behind the exact
matched-comparator PID23200. It preserves combined-graph mathematics and all
original patterns/bounds, removes timed windows, and reports each projection's
error location. Expected coverage is 464 per-projection and 48 aggregate checks;
execution and diagnosis are pending. See [the diagnostic record](../../results/20261015/combined-graph-error-diagnostic/README.md).
No failed profile has been excused or silently removed.

The [phase summary](../../results/20261015/coalesced-body-native/phase-means.json)
recomputes 24 warm means from the two audited native trials, preserving source
file hashes. At repeat VL B1/B2, candidate prefill GPU spans average
85.44/147.05 ms, vision wall clocks 60.27/109.62 ms, and common preprocessing
27.56/29.04 ms. These are overlapping scoped clocks, not additive stages or
new comparator measurements. Large GPU prefill spans justify pursuing graph
execution after resolving its numerical failures. They do not predict a gain.

### 2026-10-15 — fresh matched cohort audited; graph error isolated

The fresh lossless-body cohort exited0 in558.0408s. Independent audit verifies
1,457 files, three real GPU dtype audits and72 complete matching outputs.
**5/6 profiles meet2x decode;2/6 meet half-TTFT;1/6 meets both**, text B1.
These initial results do not establish repeated completion. Exact means and
remaining gaps are in [the fresh comparator report](../../results/20261015/coalesced-body-vllm/README.md)
and [initial budgets](../../results/20261015/coalesced-body-vllm/initial-budgets.json).
The earlier comparator table remains a historical control cohort.
Confirmation PID34285 is running with the same frozen policy. No current
target is inferred from older vLLM measurements.

Diagnostic audit:34 files,48 aggregate and464 per-projection checks,12 first
calls, zero timed windows. Four failed checks are exclusively text/coder B2
special key projections under pattern1 in both policies. All other tested
records pass. Worst comparison ratios1.1462492/1.0439333 are not excused.
[Diagnostic evidence](../../results/20261015/combined-graph-error-diagnostic/README.md).
This is fixture-level route diagnosis, not a general library-defect claim.

Hybrid study PID35785 waits behind exact confirmation PID34285. It retains
every special key on the original native route and graphs all other matrices,
including both workloads inside candidate wall time. All inputs/bounds remain;
fallback outputs must match original bits. Execution is pending.
[Protocol](../../results/20261015/hybrid-combined-graph-prefill/README.md).
The full goal remains active and unmet.

### 2026-10-15 — comparison exclusion and next decode hypothesis

The confirmation guard rejected text B1 attempt0 on detecting inference-worker
PID36306; attempt1 completed cleanly. The trace identifies a detected worker,
not its provenance. Do not assume external contention or a guard ownership
race without more evidence. Preserve the rejected attempt and exclude its
timings from the accepted cohort. The live confirmation remains pending audit.

Coder B1 is the only initial fresh profile below2x decode. The existing body
route requires at least1,048,576 weight values. An unexecuted hypothesis is to
test a524,288-value cutoff in a private context, preserving every other shape,
precision and fallback rule. Checkpoint-verified metadata identifies two added
coder896x896 projections and two sampled VL512x1024 projections; text adds none.
[Shape-only hypothesis](../../results/20261015/coalesced-body-native/small-body-admission-hypothesis.json).
These counts do not establish compact coverage, graph usage, allocation,
correctness or speed. Production code is unchanged. Test original bits/FP64,
all workloads, memory/init costs and repeated timing before integration.

Terminology for the paper: historical component reports use "first-layer"
as shorthand for early projection fixtures. The VL fixture includes first
occurrences of both linear-attention and full-attention types (layers0 and3).
Always identify the exact checkpoint tensor list; these component sets do not
represent every layer or a complete dependency graph. Full-checkpoint CPU
coverage and full trained-model validation are separate evidence scopes.

### 2026-10-15 — matched confirmation audited; hybrid and smaller-body evidence

The matched identity join passes across144 complete outputs and six actual GPU
dtype audits. Selection, versions, sources, assets and cases match exactly.
Across both cohorts,4/6 profiles meet2x decode,2/6 meet half-TTFT, and0/6 meet
both. Text B1 repeat decode is1.984810x, not a pass after rounding. Both-cohort
means and remaining gaps: [matched report](../../results/20261015/coalesced-body-vllm/README.md),
[budgets](../../results/20261015/coalesced-body-vllm/remaining-budgets.json).
The initial marginal text result is superseded by the repeat requirement.

Hybrid native-key graphs pass all48 original numerical checks and key bit
comparisons. Text/coder component wall time regresses10.6–38.8%; VL improves
11.1–28.5%. No broad integration or whole-model claim follows. Asynchronous
key study PID47481 tests removal of the intermediate CPU wait, keeping all
work inside the timing and original math/bounds. Its result is pending.
[Hybrid evidence](../../results/20261015/hybrid-combined-graph-prefill/README.md),
[asynchronous-key protocol](../../results/20261015/async-key-graph-prefill/README.md).

Smaller-body study PID46404 passes24 exact-bit/FP64 and90 safety profiles,
with384 timed submissions. Initial coder gains2.0–2.7% are smaller than the
3.6–6.6% variation of mathematically unchanged text controls. VL is mixed;
no gain is established yet. Confirmation PID47024 exited0; audit is pending.
[Smaller-body evidence](../../results/20261015/small-body-threshold/README.md).
The full goal remains active and unmet.

### 2026-10-15 — repeated components: reject smaller layout, retain graph candidate

Smaller-body confirmation reverses the initial coder wall signal (1.3–2.2%
slower). Combined audit:97 files,48 original-bit/FP64 checks,180 safety
profiles,768 timed submissions. The source/config/asset identity join passes.
Reject native admission expansion; unchanged text no-op timing variation
reinforces caution about the small initial signal.
[Decision and both trials](../../results/20261015/small-body-threshold/README.md).

Asynchronous-key graphs pass96 numerical checks across both trials,24 first
calls and1,536 timed submissions;67 combined files and identical sources,
versions, assets and configuration are audited. Placeholder policy1 is faster
in all six groups both times, but text B1 falls to1.25% gain on repeat.
Constant policy2 coder B1 reverses. Retain a fixed placeholder candidate for
further dependent-subgraph work, not a serving promotion or a significance
claim. [Repeated graph evidence](../../results/20261015/async-key-graph-prefill/README.md).

Explicit core-only graph compilation does not repair the four text/coder B2
key failures. Audit:34 files,48 numeric records,12 first calls and512 admitted
timed submissions. No bound relaxation or broad integration.
[Core compiler study](../../results/20261015/core-graph-prefill/README.md).

Source/API inspection identifies the next integration requirement: the current
Metal scheduler has real gate/up/SiLU/down and attention dependencies, while
the component fixture is independent. Test a complete dependent MLP subgraph
before changing the scheduler. A native graph path must preserve pending-output
ownership, abort/recovery, command ordering and complete GPU-span accounting;
Apple's MPSCommandBuffer may commit-and-continue and replace its root buffer.
The macOS SDK API was inspected, but no native graph integration or dependent
MLP performance is claimed here. The overall2x/half-TTFT goal remains unmet.

### 2026-10-15 — complete dependent MLP experiment launched

Owned PID52062 tests complete gate/up-SiLU-down MLP chains using four trained
blocks (text/coder layer0, VL layers0/3). VL blocks have separate graphs and
sequential calls; no artificial fusion across layers. Native and graph each
use one execution boundary per MLP, and all work is included in set wall time.
Original FP32 values and all four input patterns remain. CPU FP64 evaluates
the complete nonlinear chain; the existing5e-6+5e-5 reference/comparison
bounds gate every final value. Failed native or graph profiles are not timed.
The compile and text/coder stages have completed; VL and audit are pending.
[Protocol and exact scope](../../results/20261015/dependent-mlp-graph/README.md).
No dependent-MLP speed result or native integration is claimed yet. The full
2x decode/half-TTFT goal remains active and unmet.

### 2026-10-15 — dependent MLP results, reference correction and native prototype

The initial dependent-MLP experiment is terminal and audited:30 files,
48 numerical records,12 first calls,256 timed submissions. All48 graph/native
comparisons pass, but8 text/coder profiles fail the whole-chain FP64 reference
in both implementations and are excluded from timing. The initial description
overstated equivalence to the existing trained reference: that reference rounds
each projection/activation to FP32. Preserve the first study as a stress test.
[Original study and correction](../../results/20261015/dependent-mlp-graph/README.md).

The separate stage-rounded correction changes only CPU reference arithmetic,
with no tolerance/input/native/graph changes. Audit:36 files,48 checks,12 first
calls,384 timed submissions. Six text/coder profiles still fail the independent
reference in BOTH native and graph; none is timed. All48 graph/native comparisons
pass. Only coder B1 and VL profiles pass every component gate; their graph wall
means are2.4–16.6% lower. These are not full-model or replicated gains.
[Corrected study, exact means and remaining failures](../../results/20261015/stage-rounded-mlp-graph/README.md).

Native prototype PID54555 is in validation with a frozen17-file source manifest.
The build and CTest phases completed according to the observed driver; full
trained-model/recovery validation and independent audit remain pending. An
OFF-default complete-MLP MPSGraph uses the same queue after native Flush and
waits for completion. Synchronous per-MLP waits, lazy compilation and extraFP32
storage may outweigh gains. No full inference/performance result is claimed.
[Prototype, counters, ownership design and pending gates](../../results/20261015/mlp-prefill-native/README.md).

Latest fully audited matched evidence remains the c186 native library's two
fresh vLLM cohorts:144 complete matching outputs, six GPU dtype audits,
4/6 repeated decode targets,2/6 repeated TTFT targets,0/6 repeated joint targets.
The graph prototype does not inherit those numbers. The overall goal is unmet.

### 2026-10-15 — guard diagnosis archived; retry output isolation repaired

Original native graph checks PID54555 were deliberately interrupted while
waiting on an obsolete WorldSense supervisor fingerprint. Exit1,969.807181s;
archive audit verifies1,291 files and nine completed stages, including8 CTests.
Diagnostics pin the changed supervisor source, show that its sole Garnet
literal forwards the budget setting, and find no loaded inference/GPU libraries
in the supervisor/child. Only that exact fingerprint was added to the existing
guard; unknown workers and inference/GPU library loads still reject. Foreign
processes were untouched. Full-suite validation did not complete in this trial.

Fresh guard-fixed PID71863 built a separate candidate and passed8 CTests,
then failed after82.209367s. Vision attempt0 was rejected on unknown PID73159;
its provenance remains unproven. Attempt1 hit FileExistsError on the partial
vision output directory. The second archived failure verifies1,143 files.
This is a harness retry failure, not an admitted numerical or timing result.
[Both failed archives, hashes, commands and diagnostics](../../results/20261015/mlp-prefill-native/README.md).

Resumed PID73896 reuses only the second trial's exact configure/build/CTest
evidence and library23bff0f7..., whose source identities are unchanged. Every
remaining original check reruns. The repaired retry preserves rejected outputs
inside the rejected attempt and records a relocation map before retrying the
unchanged command. No tolerance or guard is relaxed. Vision full-grid and
trained heads have passed; exact-tail cases are progressing at this observation.
Full-suite audit, complete outputs, memory/init/first/warm measurements and
matched confirmation remain pending. Native numerical sources are unchanged
by these three helper repairs; the current manifest contains18 files.

Dependent controller PID82980 waits on exact PID73896, then archives and runs
the separate frozen-evidence audit. Only a successful audit with unchanged
source/library identities admits dispatch of the six-profile native ABBA.
Any failure stops the sequence. A controller exit0 records dispatch only,
never a benchmark pass or performance gain. The controller and downstream
commands/logs are preserved in `ROOT/logs/mlp-prefill-native-finalize-20261015`.

A separate compact eight-output decode experiment is prepared locally. It
shares each activation vector across two original compact four-output blocks,
retaining the current weight payload and reduction order. Control remains
compact four-output64-thread; candidate policies64/128/256 threads. This is
distinct from the earlier rejected raw eight-output layout. Only generation/
Python syntax checks have run; no Mac compile or GPU execution is claimed.
[Prepared protocol and required exact-bit/FP64 gates](../../results/20261015/coalesced-eight-body/README.md).
It must run sequentially after owned-job terminal capture/archival. No component
or full-model gain is established for this kernel. The full goal remains unmet.

### 2026-10-15 — comparison audit bookkeeping; decode probe queued in sequence

Resumed validation has passed all trained-layer/head and vision checks at this
observation; extended prefill/recovery remain. No full-suite or model speed
result is established yet. Controller82980 still gates the native benchmark
on terminal success, archival and the separate frozen-evidence audit.

Static review found an audit bookkeeping mismatch: the original validation
archive has fewer files than the combined validation-plus-native-comparison
archive. The frozen auditor incorrectly required equal counts. A new auditor,
`verify_mlp_prefill_native_comparison.py`, checks every combined file, records
the original count separately and preserves every numerical/coverage/source/
library equality. This was discovered before executing that audit; it is not
an observed numerical failure or a relaxed accuracy gate.

Controller88447 waits on exact controller82980 and its recorded native worker.
It archives/audits the complete native comparison before dispatching the
compact-eight decode probe. Any failed dependency/audit stops dispatch.
The probe remains unexecuted, with only generation/Python syntax checked.
[Native sequence and audit scope](../../results/20261015/mlp-prefill-native/README.md),
[queued decode protocol](../../results/20261015/coalesced-eight-body/README.md).
The full2x decode/half-TTFT goal remains active and unmet.

### 2026-10-15 — completed MLP validation, audit repairs and rejected integration

PID73896 exited0 after1292.240134s, but its generated driver contained27 of
the original28 required stages. The strict auditor correctly stopped the
82980/88447 controller sequence. The missing wide-batch stage had been
appended by the original outer wrapper and was absent from the generated
resume driver. Preserve both the1,465-file incomplete archive and15-file
controller failure archive; neither is a complete validation pass.

Completion PID97189 exited0 after101.555720s, copying27 frozen stages and
running the missing wide reference freshly. The independent audit accounts
for28 stages,8 CTests,64 primitive cases,51 trained-head queries/78 rows,
4 vision-grid profiles,6 screened recovery and6 prefix recovery cases,
4 wide profiles/12 carried steps. Worst original wide-bound ratio0.543284.
The27 reused stages include configure/build/CTest previously reused from
PID71863: this assembled coverage is not28 fresh stages or a replication.

Controller99015 admitted full native ABBA PID99045. It exited0 in211.133414s:
all72 complete outputs match, all6 profiles exercised expected graph routes,
and all1,131 source-closure identities are frozen. Synchronous per-MLP graphs
regress warm TTFT by6.2–60.4%, with0.85–1.11 GiB extra peak process RSS.
First-after-init TTFT rises from104–390ms to1917–2193ms. Initialization
remains separately recorded; graph compilation is lazy. Decode ratios versus
the same compact native control are0.957–0.991 despite an unchanged decode
implementation; this cohort does not isolate the cause. Reject promotion,
retain OFF default and c186 control. No fresh vLLM ratio or goal pass.

Controller99421 stopped during the independent archive audit:13 `.metal`
files were omitted by the archive suffix whitelist, in both source and
snapshot, although their26 hashes were recorded. A corrected archive adds
only the exact frozen bytes and terminal controller diagnostics; original
records and the pinned original manifest are independently preserved. A
second audit reached the final asset assertion but incorrectly looked for
tokenizer JSON in `omitted` rather than verifying its archived bytes. The
corrected auditor checks actual tokenizer hashes, checkpoint hashes and every
unchanged numerical/output assertion. Local and Mac audits then passed.
All failures and archive hashes are recorded in the
[native integration report](../../results/20261015/mlp-prefill-native/README.md).

### 2026-10-15 — compact-eight decode rejected; command-buffer scheduling probe

PID3818 compiled and executed the initial compact-eight shader, exiting0 in
19.352752s. Independent audit:49 files,72 numerical records,189 safety
profiles. All72 pass the original FP64/native numerical envelopes; only18
are bitwise equal. Every model/batch/policy has unequal required patterns,
so all18 profiles are excluded and **zero timing samples** are admitted.

A lexical accumulator follow-up first failed from a generator identifier
collision (PID4314, exit1 after11.658886s), preserved before correction.
After whole-token substitution and generated-name checks, PID4586 exited0
in19.247992s. Independent audit:50 files,72 numerical records,189 safety
profiles, again54 unequal records and zero admitted timing samples. Replacing
indexed accumulator arrays did not restore exact bits; no compiler cause is
established. Reject both variants without changing the exact-bit gate.
[Sources, bounds, failures and frozen audits](../../results/20261015/coalesced-eight-body/README.md).

Next hypothesis targets scheduling: encode the unchanged FP32 MLP graph into
the native MPSCommandBuffer, refresh its underlying root after any internal
commitAndContinue, and defer graph error/completion checks to final native
submission. Abort drains submitted work before invalidating pending owners.
This is a private generated context; native serving sources are unchanged.
The installed SDK headers were inspected and preserved as API evidence.

PID5137 exited0 in10.621082s. Its38-file independent audit regenerates source
and confirms all64 original synthetic graph safety cases, with an additional
native consumer and the unchanged native numerical bound. Explicit native
Flush count is0 per graph; internal MPS commits are not counted by that field.
No trained-model correctness or speed result follows. Fresh PID5521 exited0
in11.264293s; the38-file independent audit verifies64 original plus64 true
dependent two-graph cases, including native consumers and abort/recovery.
The identical original numerical bound remains mandatory. Both primitive
archives are preserved before any source transformation.

A separate source/build tree at ROOT/mlp-command-buffer-investigation-20261015
copies1,240 source files, transforming only the context, graph header and
primitive test. All copied source hashes and original/transformed hashes are
recorded; existing sources/models/builds remain untouched. Fresh PID5961 runs
the complete original28-stage wrapper with its appended wide reference.
It is live at this snapshot; validation, archive audit, complete generation
and model timing remain pending. Consumed sources remain frozen until terminal
and archival. No performance or goal-completion claim follows from dispatch.
[Private command-buffer protocol and evidence](../../results/20261015/mlp-command-buffer-primitive/README.md).

The latest completed matched vLLM cohorts still meet decode in4/6 profiles,
TTFT in2/6 and both targets consistently in0/6. None of these component or
rejected native trials changes that result. Full goal active and unmet.

### 2026-10-15 — fresh command-buffer candidate and full native audit protocol

Fresh build612af105... is running the original28-stage suite in its confined
source tree. At this observation configure/build/8 CTests, vision full-grid
and trained-head checks have passed; text exact tails are progressing.
No complete-suite or timing claim is made. Controller13152 waits on exact
PID5961, then archives terminal evidence and independently audits every
original numerical assertion plus generated-source provenance. Only successful
audit admits the72-complete-output native ABBA. Audit replacements are limited
to the old build/resume provenance, new root path, explicit native Flush count
and graph-test marker; two-graph64-case coverage is additionally required.
All actual sources and external drivers are snapshotted before benchmarking.
[Source/library identities, gate scope and exact command templates](../../results/20261015/mlp-command-buffer-native/README.md).

Derived phase analysis of the already audited rejected synchronous cohort
shows control text/coder prepare+encode around0.62–0.69ms per decode step,
with GPU spans10.51–12.67ms and full wall windows11.71–13.97ms. GPU span
overlaps wait; these clocks cannot be summed into disjoint contributions.
This supports GPU kernel work for the remaining decode gap without proving
a particular bottleneck or achievable speedup. VL control B1/B2 stage means
are27.17/28.38ms common preprocessing,58.27/110.07ms vision and85.16/154.12ms
language prefill. These describe the same warm requests, not a new cohort.
[Reproducible phase summary and interpretation](../../results/20261015/mlp-prefill-native/README.md).

The matched vLLM target remains unchanged:4/6 decode,2/6 TTFT,0/6 both across
the two completed cohorts. The new candidate has no admitted timing result.

### 2026-10-15 — explicit FP32 FMA decode candidate prepared, not executed

The two archived eight-output variants pass their original numerical envelopes
but fail exact bits in54/72 records. An explicit-FMA candidate changes only
four named accumulator expressions in the lexical shader. This tests a
contraction hypothesis without treating it as an established cause. Original
control, learned values, packing, input patterns and precision stay fixed;
all original exact-bit, FP64/native-bound and safety gates remain mandatory.
The earlier rejected records remain rejected, with zero admitted timings.

Only local source generation and Python syntax checks have run. The launcher
requires terminal/archived/audited completion of the active command-buffer
native cohort before any GPU execution, then a separate quiet guard. It has
not been dispatched. No source consumed by the current native job is changed.
[Prepared source transformations, admission protocol and limitations](../../results/20261015/eight-fma-body/README.md).
The full target remains unmet; this candidate has no compatibility or speed
result, and the current command-buffer validation remains the active run.

### 2026-10-15 — command-buffer full validation passed; frontend-path failure repaired

Validation PID5961 exited0 in1432.475943s. Independent Mac audit verified
1,614 files and all28 fresh original stages:8 CTests,64 original plus64
dependent-graph primitive cases,51/78 trained-head queries/rows,4 vision grid
profiles, both6-case recovery suites and4 wide profiles/12 carried steps.
No compile or stage reuse. Worst original wide-bound ratio0.543284. Archive
1835c361... is frozen; local re-verification waits for transfer after active
GPU benchmarking, to avoid unnecessary competing work.

Controller13152 exited0 after dispatching native PID31600. That worker exited1
in125.648896s after four complete text/coder profiles: the first VL control
command could not find ALT/comparators/vllm/venv/bin/python and exited127
before VL inference. The1,240-file source copy had omitted the comparators
link. Preserve the4,842-file failed cohort archivecc2e35df... before repair;
partial timings do not establish a complete native cohort or any target pass.

The repair checks frozen failure, library and copied-source identities, then
links the existing isolated frontend environment. CPU-only imports confirm
Python3.12.15/Pillow12.3.0/NumPy2.3.5; no packages or inference source change.
Fresh PID34111 reruns all six profiles with the original output checks and
timing design. No old profile is reused. Its sources remain frozen through
terminal capture and archival. Explicit-FMA remains locally prepared only.
[Full identities, commands, scope and failure record](../../results/20261015/mlp-command-buffer-native/README.md).
The full matched target remains active and unmet.

### 2026-10-15 — command-buffer retry rejected after full independent audits

Fresh native PID34111 passed all72 complete outputs after a six-profile ABBA
retry. Both Mac and Windows audits verified the5,004-file b750c7b4... archive,
all source closures and unchanged28-stage validation. No old partial profile
was reused. Zero of six profiles improved both decode and warm TTFT; five
TTFT profiles regressed. Additional RSS ~0.84–1.10GiB and ~1.9–2.1s lazy
first-request graph cost reject promotion. Retain c186. Phase timestamps
overlap; similar encode times and longer GPU spans do not isolate a cause.
[Full times, source proofs and failure repair](../../results/20261015/mlp-command-buffer-native/README.md).

### 2026-10-15 — explicit-FMA correct but repeated eight-output timing rejected

PID38096 and38768 passed72 exact/numerical plus189 safety records each,
2,304 timed samples total, identical code/flags/assets/versions/binary. Zero
of18 policies improved both GPU/wall direction in both cohorts; the apparent
initial coder gain reversed. Restored bits do not establish the cause of old
compiler differences. Auditor version skew was repaired by using the stronger
regeneration assertion after jobs were terminal/archived, without relaxing a
gate. [Full method and both audits](../../results/20261015/eight-fma-body/README.md).

### 2026-10-15 — two-output component signals, per-shape selection and native trial

PID39399 and39933 passed the same exact/numerical/safety gates. Five of18
policies improved both wall and GPU directions across both cohorts, but text
B1 mostly regressed. A stale descriptive 'eight-route' label was corrected;
the underlying assertion already checked Two counters. No model claim.
[Aggregate method and repeated observations](../../results/20261015/two-body/README.md).

Per-matrix PID40855 and41602 each passed480 exact/numerical checks plus189
safety profiles before7,680 timings. Both archives and all source identities
were independently verified. Across observed duplicate shapes, require GPU
fraction<=0.98 and wall fraction<1 in every cohort; choose minimum worst GPU
fraction, retaining four-output elsewhere. Three shapes qualify. Selection
used both datasets and therefore requires fresh model confirmation. Repeated
single-matrix cache effects do not establish end-to-end gains.
[Selection policy, all observations and archive identities](../../results/20261015/two-matrix-probe/README.md).

The isolated shape-selected source tree changes decode dispatch only for
those three geometries; it adds actual session counters and extends all80
coalesced primitive cases to104. Graph prefill stays disabled. Fresh28-stage
validation PID42435 is running. Existing serving, GPT-OSS, Pi and original
Mac source trees remain untouched. Retained matched results are still
4/6 decode,2/6 TTFT and0/6 both consistently across repeated cohorts. No new
component observation is counted as a full-model target improvement.

### 2026-10-15 — isolated shape dispatch built; fresh model gates running

Private native tree preparation freezes1,275 copied files and nine changed/added
paths. Fresh build and seven CTests passed, including all80 original coalesced
cases plus24 added selected-route cases. Exact validation PID42435 continues
through the original28 trained/native stages; no full pass is claimed yet.
Controller47970 observes that exact handle, archives any terminal result and
requires the frozen original28 numerical audit before native ABBA admission.
The prepared six-profile comparison requires72 complete outputs and derives
selected-route counts independently from original checkpoint tensor shapes.
[Implementation identities, commands, gate design and current scope](../../results/20261015/shape-selected-two-native/README.md).

The new table changes decode only. It cannot by itself resolve the remaining
TTFT gaps; full-model selection/rejection and further prefill/vision work are
still required for the original goal. No component improvement is promoted
into the retained matched chart. Runtime sources remain frozen until terminal
archival, and competing inference runtimes remain sequential.

### 2026-10-15 — BF16-storage graph probe prepared; runtime result pending

Hypothesis: remove the explicit retained FP32 MLP weight expansion while
feeding original BF16 values and casting exactly to FP32 inside the graph.
All arithmetic/activation/product/result tensor dtype assertions stay FP32;
native ordering, owners, completion checks and abort barriers are unchanged.
Installed MPSCoreTypes.h exposes BFloat16 from macOS14, but this is header
availability evidence only. Local transformation and expanded-script syntax
checks pass. No Mac compile, execution, memory saving or performance claim yet.

Controller53701 observes exact controller47970, then the actual native handle,
archives and audits72 complete native outputs, and only then admits the new
64+64-case primitive behind a quiet guard. Success or failure is archived before
interpretation. This retains the original rejection of the FP32-cache graph
and the prior slower cast-half component. A new primitive pass would still
need trained numerical and performance evidence before any integration.
[Exact method, dependencies and limitations](../../results/20261015/mlp-bf16-feed-primitive/README.md).

### 2026-10-15 — trained BF16-feed MLP timing protocol prepared

The follow-up preserves complete gate/up–SiLU–down dependencies, the existing
FP32 stage-rounded CPU reference and unchanged tolerance coefficients. Before
timing it requires exact GPU BF16-to-FP32 roundtrips of every learned value in
12 tensors/four fixture MLP blocks. Four patterns and all failed native or
candidate profiles remain; failed profiles have no timing windows. Local
generation and expanded-driver/auditor syntax checks pass. No Mac compile,
execution, memory saving or speed is claimed yet. Queue60730 waits on exact
controller53701's audited primitive completion, then preserves/audits its own
actual component handle. This is sequential and does not change live inputs.
[Protocol, scope and reproduction templates](../../results/20261015/mlp-bf16-feed-primitive/README.md#trained-component-follow-up-prepared).

### 2026-10-15 — selected-shape validation passed; actual native cohort started

Exact PID42435 exited0 in1423.387570s. All28 original stages were freshly
executed, including seven CTests and104 coalesced cases. Mac frozen-evidence
audit passed1,615 files,51/78 trained-head queries/rows, four full/small vision
profiles, both six-case recovery suites and four wide profiles/12 carried
steps. Worst unchanged wide bound ratio0.543284; no compile/stage reuse.
Archive5b9ee0dc... and candidate libraryc22fb8f2... are frozen. Local transfer
and re-audit wait until the current performance work completes.

Controller47970 exited0 with native_benchmark_dispatched after successful
audit. Actual native PID68805 is running; text B1 completed its original
complete-output and actual-route predicates. The whole six-profile cohort
is incomplete and no new speed/TTFT/vLLM ratio is claimed. Controllers53701
and60730 continue to wait in sequence; BF16 GPU work has not started.
[Identities, protocol and scope](../../results/20261015/shape-selected-two-native/README.md).

### 2026-10-15 — selected-shape full-model cohort passed; promotion rejected

Native PID68805 exited0 in192.245289s. Mac and Windows audits pass the5,094-file
f18be819... archive,72 complete outputs,1,199 serving and980 external-driver
identities, fresh full28 validation and actual routes. None of the three
active selected-kernel workloads improves both decode and warm TTFT. Coder B2
decode regresses~1.7%, VL B2~1.5%; VL B1 gains~0.8% decode with~4.5% worse
TTFT. Text/coder B1 execute zero selected kernels; their positive observations
cannot establish benefit from executing this kernel. Reject promotion, retain
c186, and leave the repeated matched chart unchanged. No fresh vLLM ratio.
[All profiles, raw evidence and route-aware decision](../../results/20261015/shape-selected-two-native/README.md).

### 2026-10-15 — BF16-feed primitive passed; missing-auditor setup failure repaired

Primitive PID72603 exited0 in11.376850s;42-file archive914fce25... passes
all64 original plus64 dependent-chain cases on Mac and Windows. Controller53701
failed before audit because a base verifier had not been copied; queue60730
stopped without trained execution. Preserve the17-file setup failure2ee0de40...
before interpretation. Add the missing helper after terminal runtime archival;
three other dependencies match their prior frozen hashes exactly. Re-audit
the same primitive without rerunning it or changing a numerical predicate.
CPU-only expanded-runner preflight passes before direct trained launch73088.
[Failure, repair identities and primitive proof](../../results/20261015/mlp-bf16-feed-primitive/README.md).

### 2026-10-15 — trained BF16-feed graph slower; rejected

Direct PID73088 exited0 in39.209634s;38-file archivebec11fda... is independently
audited on Mac and Windows. All12 complete learned tensor casts reproduce
44,531,712 FP32 values bit exactly. There are48 numerical records and12 first
graph calls. Six profiles fail the original stage-rounded reference in BOTH
native and candidate and remain excluded; every candidate/native comparison
passes. The six admitted profiles execute384 timed samples and ALL regress
wall time by~23.5–65.4%. Reject integration; no complete-model or memory saving
claim, no new vLLM ratio, no favorable-repeat search. The full goal remains
active and unmet, and existing native prefill/defaults remain selected.
[Exact timing table, rejected records and cast coverage](../../results/20261015/mlp-bf16-feed-primitive/README.md#actual-trained-component-reject-promotion).

### 2026-10-15 — same-pipeline four-output geometry probe prepared

After the selected two-output model rejection and BF16 graph rejection,
test scheduling in the actual existing four-output pipeline. Control64 versus
candidate64/128/256 preserves shader/pipeline/weight bytes/FP32 arithmetic;
candidate64 is a null GPU-work comparison. Keep original20 selected matrices,
all29 packing assets, four patterns, exact bits, original bounds and safety.
Local generation and expanded-runner syntax pass; no Mac execution yet.

Before timing, declare selection requiring>=5% GPU gain and wall gain in two
cohorts for all duplicate geometry observations; null GPU/wall fractions must
stay within5%. No statistical-confidence or model gain follows from this rule.
Fresh full-model and matched confirmation remain required. The frozen auditor
retains all original assertions while eliminating live wrapper dependencies.
[Protocol, source identity and limitations](../../results/20261015/four-threadgroup-probe/README.md).

### 2026-10-15 — same-pipeline scheduling calibrated; no geometry selected

Actual PID74228 and74935 exited0 in31.738427s and31.689417s. Both Mac and
Windows audits pass59 files per cohort, all480 exact/numerical records plus189
safety profiles before7,680 timings each. Runtime sources/flags/assets/versions/
binary match; control and candidate use the same original shader/pipeline.
The predeclared>=5% GPU/wall/null rule selects zero geometries across cohorts.

Null GPU fractions span0.862107–1.143511 initially and0.851710–1.347176 in
confirmation, with18/40 and16/40 outside5%. CPU hook overhead remains in wall
time. These are observed short-burst variations, not a causal attribution or
confidence bound for other workloads. Earlier small single-projection signals
remain exploratory; full-model rejection and matched benchmark data stand.
[Complete calibration, all observations and decision](../../results/20261015/four-threadgroup-probe/README.md).

Keep original64. The next scheduling study should use the same library across
settings and actual full-model timing/counters rather than select from these
short cache-resident windows. Prefill remains a separate priority: local SDK
inspection found MPSNDArrayMatrixMultiplication and its alpha/beta API, but no
mixed-storage execution, precision compatibility or latency claim exists.
No new serving defaults or matched target passes are claimed. Full goal active.

### 2026-10-15 — NDArray coefficient/API gates, native failure and descriptor repair

Inspection found an already tested physical-half-transpose method in the
20261013 report; do not rerun it unchanged. Test the distinct installed-SDK
MPSNDArray product API. Actual PID76186 compiles but exits1: two-source alpha2^-8
fails both storage modes on the first numerical record. Native controls pass.
Preserve24-file archive1274759a... before diagnosis; inverse scaling is only
diagnostic and never admits the failed output. No universal API-bug inference.

PID76652 exits0 in13.493525500s: mixed/F32 unit-alpha and explicit-third-zero
alpha paths pass64 original numerical/guard/immutability checks, independently
reconstructed in FP64 on Mac and Windows. Preserve34-file archive4bc6ecef...
before using the coefficient path. This is synthetic API compatibility only.

First actual trained-context PID77664 compiles, then exits1 in12.306445375s
on `NDArray transposed view failed`, before any emitted numerical result.
Archive52 files9d037622... before source repair. Separate PID77981 exits0 in
13.520210875s:32 alias-view observations establish physical K,N array lengths
versus logical N,K descriptor shape;64 original numerical gates pass. Preserve
35-file archive528ed48d..., independently audit Mac/Windows, then change only
the native logical-shape assertion to query the descriptor. Original arithmetic
and bounds remain intact. Local archive reads during active transfers failed
file locking; wait for terminal transfer and re-audit the same bytes. No GPU
rerun follows from observation/transfer errors.

Repaired trained PID78431 is running with actual native MetalContext, all29
original first-layer assets, four patterns, original envelopes, two ABBA rounds,
mixed three-source alpha and exact-F32-cache policies plus unchanged native null.
Declare before repaired timing: repeat only a passing candidate with >=5% GPU
AND wall gain plus model/batch null within5%; confirmation/full validation/full
model/fresh vLLM required for promotion. No synthetic ownership suite, memory
saving or model-level speed claim. Retainc186 serving library and matched chart.
[API failures and compatibility proof](../../results/20261015/ndarray-matrix-api/README.md),
[native protocol and archived failure](../../results/20261015/ndarray-prefill/README.md).

### 2026-10-15 — NDArray trained confirmation selects a confined VL candidate

PID78431 exits0 in79.905263500s, archive2675b4e3...; PID79830 exits0 in
77.154079084s, archived658c582.... Each59-file cohort is audited on Mac and
Windows:72 numerical records,768 actual timed executions. Six candidate groups
fail original native-comparison bounds on pattern1 (text B1/coder B1/B2, both
policies), retain failures and have no timing. All FP64 envelopes pass, which
does not waive comparison rejection. Source/flags/SDK/assets/versions match.

Only VL B1 exact-F32 NDArray qualifies in both cohorts: GPU fractions
0.878001/0.880955 and wall0.823963/0.833840; own native-null remains within5%.
Other signals fail numerical, gain or calibration criteria. No fresh matched
vLLM ratio or complete-model TTFT improvement follows from these components.
[Full table and selection](../../results/20261015/ndarray-prefill/README.md#repaired-trained-execution-and-confirmation).

Prepare1,333-file separate native source copy, eight changed files; retainc186
parent source/library. Candidate removes global policy state, restricts F32
NDArray to physicalrow80 nonvision/nonfloat-key, bounds strong-owner view cache
at256 with original native fallback and adds actual session/worker counters.
CPU-only transform/expanded30-stage syntax passes. Actual PID81353 starts
fresh original28 plus new cache/update/abort/shape-fallback primitive. Build,
all stages, memory and model timing are unverified at this entry. Preserve
live consumed sources until terminal/archive. Full goal active and unmet.
[Candidate scope and live validation](../../results/20261015/ndarray-native/README.md).

### 2026-10-15 — native NDArray self-guard failure preserved; fresh repair running

PID81353 builds library8e4e9b4e... and primitive but guard mistakes its own
exact `_gpu` executable for a foreign probe. Thirteen interrupted primitive
attempts, zero accepted numerical tests. Interrupt only verified Python group
81353/81446 between attempts; terminal1 in120.330340375s. Preserve1,374-file
archive2f827d30... before repair. Mac/Windows audits regenerate eight native
changes, verify1,251 text source identities and three accepted setup stages.
Do not call this an accuracy/performance failure or a validated native build.

Leave private source/old driver frozen. New external driver excludes only the
expected primitive path AND verified current owned process group. Original
unknown/foreign rejection and every numerical predicate stay intact. Fresh
guardrepair build and all30 stages, no build/stage reuse. Local/Mac expanded
runner syntax passes, actual PID84121 starts. It has completed build, cache
primitive, seven CTests, full vision grid and trained heads; whole suite is
still live, terminal audit pending. No complete-model speed or memory claim.
No fresh vLLM inference and no target pass is added. Goal active/unmet.
[Failure provenance, precise repair and pending validation](../../results/20261015/ndarray-native/README.md#archived-harness-failure-and-scoped-repair).

### 2026-10-15 — native NDArray full-model comparison queued behind exact validation

Validation PID84121 remains live and advances through original trained text
tail cases. Controller90454 observes that exact live PID, then will archive,
extract and independently audit all30 stages before dispatching any native
timing. Terminal failure is preserved and stops benchmark admission. No
runtime/consumed driver edits, no competing inference and no timeout restart.

Prepare all six native ABBA profiles and original72 complete-output predicates.
Candidate adds independently derived NDArray prefill counts from checkpoint
headers/physical chunk plans, zero decode counts, distinct first/init/warm/RSS,
both source closures and unchanged input/stop/image policies. Expanded comparison
passes local/Mac syntax; auditor transformations preserve original bounds and
pass local expansion/AST. These are preparation checks, not full-model results.

Before timing, require VL B1 warm TTFT>=3% improvement in both native cohorts,
with no repeated>=2% decode regression in any workload, to advance this candidate.
No statistical-confidence claim; report unaffected-route variation, first-use
costs and extra FP32 storage. Only an admitted initial cohort warrants a fixed
confirmation. Fresh repeated vLLM comparison/all six joint target passes remain
required for full goal completion. Methods/results already retained are not
redefined as the goal. No new accepted speed/TTFT gain or target pass yet.
[Protocol, exact queue and admission rule](../../results/20261015/ndarray-native/README.md#complete-model-comparison-prepared-and-conditionally-queued).

### 2026-10-15 — vector-dimension vision component prepared locally; no execution

During the verified wait on PID84121/controller90454, inspect prior methods:
parallel contraction failed trained accuracy; scalar eight-key prefetch is
retained; old fused softmax/context is not a broad repeated gain. Prepare a
distinct local-only float2/float4 adjacent-dimension contraction. Share probability
loads while preserving all per-output key FMAs and the tail loop. Current
selected scalar prefetch8 is the exact source control, both128-thread, with a
same-pipeline null. No weight/input/resolution/softmax/precision change.

Retain24 shapes, four generations, output/padding bit identity, original FP64
bound and ABBA raw samples. Predeclare>=5% GPU/wall gain at BOTH serving batches
with null within5%, all numerical gates and unchanged full-cohort confirmation;
tie selection by repeated serving geometric wall fraction then float2. No
statistical-confidence/model claim. Local source generation and Python AST
pass only. Metal/C++ uncompiled; nothing copied to Mac or launched. Keep the
current validation/native chain exclusive and frozen. Goal active/unmet.
[Prepared scope, source differences and future admission rule](../../results/20261015/vector-vision-context/README.md).

### 2026-10-15 — native NDArray header route/payload budget calculated

During the verified live validation wait, compute route/payload arithmetic from
byte-verified frozen checkpoint headers; require matching full-checkpoint SHA
identities against actual trained component metadata. Planned physical rows
24/44,26/52,80/160 predict186 NDArray prefill calls for VL B1, zero for all
other profiles and decode. That path adds1,990,459,392 logical FP32 weight
bytes (~1.854GiB), excluding MPS internals and metadata/owner overhead. This
is not measured RSS or proof that subsequent requests retain stable keys.
Record the cost before timing, rather than claiming a memory saving from
deferred aliases. Actual counters, outputs and RSS remain benchmark gates.
No Mac source/driver mutation or competing inference. Qwen3.5 trained layer
checks have completed and text layer validation continues. Goal active/unmet.
[Header provenance, complete tensor list and limitations](../../results/20261015/ndarray-native/header-coverage.json).

### 2026-10-16 UTC — NDArray fresh30 passed; audit setup repaired; native ABBA live

PID84121 exits0 in1440.044627875s; archivef60abf2c... contains1,693 files.
Every30 stage fresh. Controller90454 archives/extracts but its auditor rejects
the new silent successful compiler log. Preserve14-file ce435dc7... failure
and faulty auditor e1fe8dba... before repair; zero native inference dispatched.
Do not relabel executed runtime validation as failed accuracy/performance.

Correct only the new `ndarray-cache-compile` nonempty-log assumption, retain
the original28 log predicates and every numerical/guard bound, add matching
compiler/executed artifact identity and actual primitive PASS. Auditor9ec11306...
passes the same frozen archive on Mac:30 fresh stages, seven CTests,80 body
cases, cache256/fallback14,51/78 trained heads, four vision profiles,6+6 recovery,
four wide profiles/12 carried steps, worst original bound0.5432836183. No GPU
rerun or serving-source change. Windows transfer/audit waits until timing ends.

Actual native PID10943 dispatched after successful direct re-audit, all six
profiles and original72 complete outputs/counts pending. Keep runtime and
consumed external-driver closure frozen until terminal/archive. No additional
inference, GPU probe or large transfer during timing. Init/first/warm/RSS stay
separate. Retainc186, no fresh vLLM ratio or target pass yet. Goal active/unmet.
[Exact stage/repair identities and current scope](../../results/20261015/ndarray-native/README.md#actual-fresh30-completion-and-silent-compiler-audit-repair).

### 2026-10-16 UTC — first native cohort guard failure audited; fresh full cohort running

Native PID10943 exits1 in137.186741167s on an inherited `legacy_alpha>0`
benchmark assertion. Archive5,194 files8288bb6c... before repair. Mac/Windows
independent audits confirm54 complete matching outputs across18 successful
workers: four full text/coder profiles plus three control/three candidate VL B1
requests. Each candidate image executes186 NDArray prefill calls, zero decode
and zero legacy calls; control has182 legacy calls. This is actual routing,
not inference from a label. The stale predicate expected a replaced route to
execute. No numerical/runtime failure or admitted full-cohort timing result.

After all original jobs/archives are terminal, transfer/re-audit fresh30 F60
on Windows successfully. Reproduce14-file silent-compiler audit failure locally
using its frozen old auditor, expanded line17. Preserve every old error; no
model test rerun or numerical predicate changes. Fix only the new comparison's
route-aware legacy expectation, strengthen frozen audit equivalently, retain
the same validated8e4 source/library. Expand/AST preflights pass locally/Mac.

Fresh whole six-profile native PID13840 starts after successful failure audit
and transfer completion, all72 outputs/timings/RSS required, no reused partial
timings. Original repeated>=3% VL B1 warm-TTFT/no repeated>=2% decode-regression
rule unchanged. No favorable-repeat search from incomplete data. Source and
external drivers frozen until terminal/archive; no other inference/large copy.
[All failed output/counter evidence and repair scope](../../results/20261015/ndarray-native/README.md#initial-native-cohort-stale-legacy-counter-assertion-preserved).

### 2026-10-16 UTC — full NDArray native cohort correct but not useful; reject promotion

Fresh native PID13840 exits0 in199.862149500s. Freeze5,320-file archive7ed9d48e...
before interpretation. Mac and Windows audits verify72 complete outputs,
1,251 serving/1,028 external-tool identities, fresh30 validation and actual
186-call prefill/zero-decode routes for VL B1, zero NDArray in all other profiles.
The legacy counter repair is exact: zero on full NDArray, positive on control/
unaffected paths. Old failed54-output dataset remains separate, not reused.

VL B1 warm TTFT181.973854 ->181.494229ms, only0.264% lower; fails declared3%
screen. Decode61.370864 ->58.948862steps/s/row, ~3.947% worse. First-after-init
TTFT268.619520 ->3354.833000ms, peak individual RSS +1.716GiB. Init5.519764 /
5.536633s distinct. Logical added weight storage1.854GiB is not observed RSS.
Other profiles execute no NDArray; mixed timing changes cannot establish a
benefit from executing the new operation. Preserve every profile/cost.

Reject8e4 broad-F32 promotion, retainc186, no favorable confirmation search,
no fresh vLLM ratio and no new matched target passes. Component improvements
did not predict full inference. Goal stays active/unmet. Next inspect full
weight admission (common versus row-wise exact half) before selectively using
F32. Control182 legacy versus186 NDArray calls suggests a small exception set,
but not its identities or original fallback type. A per-value `MTLBuffer.contents`
read in expansion is a possible first-use bottleneck; hoisting remains an
untested hypothesis. No claims or source edits to archived8e4 from that inference.
[All native timings, initial costs, RSS and decision](../../results/20261015/ndarray-native/README.md#actual-complete-native-cohort-reject-broad-f32-ndarray-promotion).

### 2026-10-16 UTC — full-checkpoint admission narrows the next candidate

CPU Python scan exits0 in63.683980791s; independent original-C++ codec audit
exits0 in1.497496542s. Freeze24-file archive66d403a0... before interpretation;
Mac and Windows provenance/counter audits pass. Cover498 BF16 body matrices:
text168/168 common/exact, coder143/144, VL182/185 out of186. Reconcile18
actual retained-control requests with full chunk plans. NumPy bitwise weight
roundtrips and5 exported non-common matrices through original compiled codec,
19,548,160 admitted-value bitwise roundtrips. Omitted raw assets by identity;
host evidence audit is not an arithmetic rerun of missing weights.

Only first-layer VL linear-attention QKV lacks exact half. Fourteen tiny
normal BF16 rows need scale138, beyond+/-64; no values dropped and no bounds
relaxed. Coder gate and three other VL non-common matrices already have exact
row-wise paths. Selective candidate must use !ExactHalfReady(), not absence
of common scale. Logical added F32 payload24MiB versus broad1.854GiB; not RSS.
Review existing20261013 expanded-scale report: correct bit rescale but no
useful component gain, so do not repeat unchanged. Retainedc186 unchanged.
[Full identities, coverage, caveats and CPU records](../../results/20261016/ndarray-weight-admission/README.md).

### 2026-10-16 UTC — selective F32 trained-component initial screen passes

Predeclare before execution: original29 fixtures/four patterns/all6 physical
row shapes, same FP64 and original-control bounds; fullF32 reference, only
no-exact-half F32, native null. All selective numerical groups must pass.
Require>=5% GPU AND wall gain in both VL batch shapes and nullwithin5% in
two fresh cohorts before isolated native integration. Exact zero-operation
policies require bit identity; no inference/vLLM claim from component gains.

Initial PID19306 exits0 in77.300869250s; archive62 filesf70a9eb2...;
Mac/Windows independent regeneration/raw audits pass72 numerical records,
960timed executions. Selective executes exactly1 projection per VL set,
zero for text/coder. Original comparisons reject fullF32 textB1/coderB1/B2;
no timings admitted for failed groups. All selective groups pass unchanged
bounds. VL1 selective GPU/wall fractions.899819557/.923042519, null
1.032385329/1.029009783; VL2 .828327395/.849457303, null
1.015988762/1.022997518. Initial screen passes, no native promotion.

Fresh complete confirmation PID20686 launched after initial terminal/archive
and audited initial screen. Source stays frozen until terminal/archive.
New path hoists buffer-contents read out of per-value loop for both reference
and selective products; no isolated first-use causal claim. Strongowners,
cachecap256/nativefallback preserved. Goal active/unmet.
[Predeclared protocol and evidence](../../results/20261016/ndarray-selective-prefill/README.md).

### 2026-10-16 UTC — selective confirmation passes; native compatibility failure preserved/repaired

Confirmation PID20686 exits0 in77.263889209s;62-file archivea4de900f...,
Mac/Windows audits pass. Same sources/assets/generatedcontext/header/harness/
macros as initial. Both cohorts:24/24 selective checks,24/24 null checks;
fullF32 reference21/24, original pattern1 failures remain excluded. Each has
960timed executions. Repeat selective VL1 GPU/wall .815967281/.852873274,
null1.024950009/1.015859208; VL2 .888714569/.907303111,
null1.036873670/1.037127831. Both predeclared screens pass; admit isolated
native investigation only, no promotion, vLLM ratios or new goal passes.

Native initialPID22114 exits1 in53.492590250s. Build and actual selective
cache/route/abort/native-half primitive pass; six original CTests pass, but
wide-batch prefill fails because row160 intercepts padded FP32 then throws
the packed assertion. CTestexit8, no guard interruption. No later trained
stages or model timing. Freeze1366-file archivea04efb90... before any edits;
both-host failure audits regenerate8transforms, verify4accepted stages,
6pass/1fail CTests and guards. This is a compatibility bug, not contention.

After terminal/archive/audits, add packed FP32 dtype/stride admission to
the outer selective native predicate; preserve original padded fallback and
all original tests/bounds. New isolated1371-file source and fresh packedguard
build; full30 PID23582 launched, no partial validation reuse. Candidate
context7410a44d... vs initial488005e0..., original813c5679.... All serving
and external drivers frozen until terminal/archive. Retainedc186 unchanged.
Local early extraction of still-copying SCP13062 received file-lock denial;
wait its terminal0, then fresh extraction/audit succeeds, no result admitted
from the incomplete copy. Record workflow failure separately from GPU result.
[Full native failure, hashes and repair scope](../../results/20261016/ndarray-selective-native/README.md).

### 2026-10-16 UTC — repaired partial validation passes original CTests

LivePID23582: new packedguard build, selective cache primitive and all7CTests
pass, including original padded wide-prefill failure. Vision fullgrid and
trained head stages also pass so far; model/tail/recovery stages still run.
No full validation claim from partial status. New unused controllerPID25901
waits exact23582, archives terminal success/failure, then audits a successful
full30 without rerunning GPU stages. Controller identities frozen separately;
no concurrent inference, driver/source mutations, promotion or goal completion.

### 2026-10-16 UTC — guarded full native pipeline queued after exact validation handle

Prior turn made progress: full checkpoint evidence, repeated selective component
screen and actual native failure/repair; no new matched goal pass. Revalidate
exact23582/25901 live before new work. Prepared full6 native profiles retain
all original72 output/clock predicates. Derive new selective and retained common
half counts from frozenf9505c75... admission report; exact168/143/182 counts
per eligible text/coder/VL chunk, newVL1 prefill/0decode, others0. Offline exact
AST formula preflight matches36 real old retained-control requests; no old
timings reused, candidate counts remain predictions pending execution.

New continuationPID33009 waits exact25901, then only terminal/archive/audited
full30 can launch guarded original6 native ABBA. Archive/audit before applying
both-VL>=3% warm-TTFT initial screen; only passing initial permits one unchanged
fresh full6 confirmation. No promotion or goal completion from this pipeline.
First full30 audit preserved; repeat only CPU evidence audit with explicit
archiveSHA before native launch, original fields must match. Correct per-batch
physical-row metadata80/160; distinct expanded-script filename per profile;
independent profile auditor regenerates comparators and checks source/counters.
All live serving/external drivers frozen. Retainedc186, goal active/unmet.
[Prepared pipeline and limits](../../results/20261016/ndarray-selective-native/README.md#full-model-comparison-prepared-and-gated).

### 2026-10-16 UTC — CPU exponent-escape study: modest modeled ranked packet saving

Actual CPU-only first-layer analysis exits0 in4.830290125s. Freeze11-file
archive94fba9ef...; Mac/Windows provenance/histogram/budget audits pass. Full
checkpoint and29fixture slice hashes verified. Retained large-body geometry
admits20 matrices (6text/3coder/11VL),129984 blocks,125525 compact4459raw.
Seven-exponent window plus original exponent escapes reconstructs64268800
compact entries exactly as a CPU expression; no actual packet/GPU kernel.

Correct descriptive budget adds identical4-byte raw-fallback metadata omitted
from initial fraction denominators; retain original report. Ranked/aligned
packet/control fractions .985794610/.983626959/.985190216; ideal free-rank
.938683336/.936669964/.938101777. Original master/fallback buffers remain;
variable addressing/lookup unimplemented, no GPU traffic/RSS/model speed claim.
Prioritize no GPU candidate from1.4--1.6% ranked payload estimates alone; ideal
~6% excludes lookup work. Decision is after descriptive CPU results, not a
predeclared speed gate or universal rejection. CPU analysis overlaps only native
numerical validation, no timed inference comparison; preserve commands/scope.
[Histograms, corrected table, source identities and decision](../../results/20261016/compact11-escape-cpu/README.md).

### 2026-10-16 UTC — exact validation still live; vector next-step preparation only

Revalidate exact23582/25901/33009 via process handles, not lock-file presence.
Prior goal turn made progress through CPU evidence and guarded native pipeline.
Fresh30 now passes dense tails/layers, all privateKV, trained vision, extended
Qwen3.5 prefill and all prefix/screened recovery suites; final trained wide
batch gate still executes. Partial successes are not full30 or model speed proof.

While that exclusive job runs, prepare only unused vector-vision launcher,
synthetic-only terminal archiver and fixed predeclared analyzer. Copy them and
the earlier vector generators/auditor to Mac; local/Mac generation/AST passes,
no Metal compile/GPU probe dispatched. Launcher requires current native chain
terminal success, archived/independently audited72-output cohorts, retainedc186
hash and quiet guard. Analyzer requires identical complete sources/variants/
environment across confirmation, all24 numerical cases per width, >=5% GPU
ANDwall at both216/12/64 serving batches and matchingnullwithin5%. This changes
no predeclared numerical/gain bound and makes no vector speed claim.
[Prepared vector protocol](../../results/20261015/vector-vision-context/README.md).

Review existing joined-prefill/compact-joined and single-dispatch projection
reports before proposing more projection fusion: joined warm effects were small
and memory/first-use costs material; compact allocation did not repeat its
initial latency gain. Raw grouped QKV was mixed, including coderB1 regression.
Do not repeat those unchanged methods or call unimplemented Q/V view/packing
ideas tested optimizations. Current immediate decision still requires the
selective native cohort and original broader matched target.

### 2026-10-16 UTC — selective full30 and full native cohort audited; reject promotion

Fresh30 PID23582 exits0 in1426.546372500s; freeze1732-file142964df... archive.
Mac/Windows audits pass all original stages,7CTests/80coalesced cases, cache256/
fallback14/abort/nativehalf,51headqueries78rows,4vision,6+6recoveries and4wide
profiles12carriedsteps, worst original bound.5432836183473805. No reused stages.
Candidatef6e855e0..., retainedc186 unchanged. Exact new full30 driver/source
regeneration passes on both hosts. Controllers25901 then33009 execute in order.

Native PID49121 exits0 in192.727560875s. Freeze5484-file1861996b... before
interpretation. Both-host audits verify72 complete outputs,1287 serving-source
and1068 external-tool identities, all original gates, exact72 common-half
counts and regenerated comparators. ActualVL1NDpre/0decode+182common retained;
other profiles0ND. All original image/patch/prompt/cap/stops preserved.

VL B1 warm178.288667 ->179.360740ms (fraction1.006013); B2 296.004115 ->
296.515594ms (1.001728). Both fail declared>=3% TTFT screen; no favorable
confirmation or promotion. Decode ratios1.000584/.998940, near control in one
cohort; do not claim repeated equivalence/regression. Preserve all6 timing,
init/first and mean worker peak RSS rows. Lower observedVL peak RSS does not
prove saving: logical F32 allocation24MiB remains, time-l peak spans init and
inference. Cold hardware not asserted, component first-layer gain not extrapolated.
No freshvLLM ratio, matched target counts unchanged, goal active/unmet.
[All complete native data and decision](../../results/20261016/ndarray-selective-native/README.md).

### 2026-10-16 UTC — vector vision initial numerically exact, null calibration rejects selection

Only after native terminal/archive/audits and SCP20205 terminal0/local full
audits, dispatch vectorPID53012. It exits0 in38.676133083s,33-filee2ec3e35...
archive. Both-host audits:72 original shape/variant records,0numeric failures,
11520timed submissions. Full output/padding bit identity and oldFP64 bound pass.
Apparent vector2 GPU/wall fractions B1 .678234/.764214, B2 .719771/.757169;
vector4 B1 .669169/.780027, B2 .660777/.700194. But B1 same-pipeline null
1.118033/1.117906 exceeds original5% calibration. Neither width qualifies;
no unchanged favorable repeat, no trained/native gain claim.

All null windows retained: B1 round0 GPUmeans1.578144/1.599190/1.416550/
.954771ms; round1 .798069/.786873/.800398/.786079. Both arms use the same
pipeline object/geometry. Nonstationary timing observed; no instrumented
frequency/thermal cause. Never filter round0 or normalize candidate by null.
[Original failed screen, raw windows and bounds](../../results/20261015/vector-vision-context/README.md).

### 2026-10-16 UTC — new prospective calibration-only design, before candidate recollection

Predeclare separate fixed2s conditioning (old.25s),10ABBA rounds x5 samples
(old2x20), all measured samples retained. Scalar/vector shaders byte-identical,
all original24 shapes/four changed inputs/bit+FP64 gates unchanged. First two
fresh cohorts run variant0 only,4800timings each; both serving batches must
haveGPU ANDwall nullwithin5% in both cohorts before any new candidate gains.
Initial failure stops this design, not repeated until favorable. New candidate
cohorts, only if admitted, retain all72 original records and old5%gain+null5%
both-batch repeated screen. Olde2ec failed cohort remains failed, never reused.

Local/Mac generation preflights confirm identical shaders, only timing-harness
changes. Launch initial null-onlyPID54090; no new vector candidate executes.
No tolerance relaxation, source mutation of live drivers, model/VLLM gain or
goal completion implied by new sampling design.
[Prospective design and limits](../../results/20261016/vector-vision-calibrated/README.md).

### 2026-10-16 UTC — null-only GPU dataset complete; parser failure preserved and repaired

Null setupPID54090 exits1 in61.967176416s. Compiler and actualvariant0 GPU
exit0, all24 original bit/FP64 checks and4800samples exist, but Python summary
parser retained old8windows/2rounds/20samples rather than new40/10/5. No summary
or null-gate result admitted. Freeze32-file7af5fad7... before editing source.
Mac/Windows failure audits regenerate shader/harness and verify every raw
numerical/window/sample and guard; do not calculate/select from failed ratios.

After terminal/archive/audits, fix only three summary-count assertions. Generated
shader and C++ harness match frozen failure bit-for-bit; no GPU algorithm,
conditioning, sample or numerical/5% gate changes. Require two fresh corrected
null cohorts, exclude all failed-setup timing data. Initial repairedPID55356
launched; sources frozen until terminal/archive. Current native/vision methods
remain unpromoted, retainedc186, full goal still active/unmet.

Documentation checker variable-shadow defect in newly added table code was
caught from reports_verified=2 despite163-report inventory; rename local table
list to retain the inventory set. Recheck reports_verified=163 and all table/
link/source hashes. This changes no benchmark evidence or numerical predicate.
[Failure provenance and repair limits](../../results/20261016/vector-vision-calibrated/README.md#preserved-parser-failure-and-fresh-setup).

### 2026-10-16 UTC — two fresh null calibrations pass; new full candidate collection starts

Corrected initialPID55356 exits0 in61.974344042s;33-filea0dc7f65... archive.
Confirmation56568 exits0 in62.002792667s;33-file8ca90d1b... archive. Both-host
audits pass24 original shape/numerical records and4800timings each, onlyvariant0.
All captured sources/shader/harness/variants/environment/timing design match.
Initial B1 GPU/wall .978676390/.970951999; B2 1.003489961/1.002114045.
Repeat B1 1.003273801/1.004098940; B2 .997444067/1.000960291. Both meet5%
in both cohorts, no sample removal, old failed setup excluded entirely.

Only after terminal/archive/audits, launch fresh full3-policy candidatePID57747
atvector-vision-calibrated-20261016. Candidate gains are still unproven; all72
original numerical checks and new complete timing records required. Same
>=5%GPU ANDwall both-serving-batch + ownnullwithin5% screen, unchanged fresh
confirmation before any width choice; olde2ec cohort never reused. No trained
vision/native/VLLM gain or goal completion from a null gate.
[Calibrated prospective study and all source/artifact identities](../../results/20261016/vector-vision-calibrated/README.md#actual-independent-null-calibration).

### 2026-10-16 UTC — prospective full vector initial passes; unchanged confirmation running

Full candidatePID57747 exits0 in169.318523958s; freeze37-filebcdc7fd2... archive.
Both-host raw/regeneration audits:72 original shape/variant records, no numerical
failures, full bit/padding identity and FP64 bound unchanged,14400timings.
Source/shape/conditioning/10x5 sampling preserved, no filtered warm windows.

Width2 initialB1 GPU/wall .632454/.729740, B2 .645435/.693187;
width4 B1 .615212/.731072, B2 .602147/.649052. Own null B1
1.010532/1.008367, B2 1.001815/.998157. Both widths qualify in initial
new cohort, but no repeated width selection or model benefit yet. Admit exactly
one unchanged full3-policy confirmationPID60693 after terminal/archive/audits;
it is running. Retainedc186 unchanged; no freshvLLM ratio or new target pass.
[All raw prospective data, decisions and limits](../../results/20261016/vector-vision-calibrated/README.md#initial-candidate-cohort-passes-confirmation-pending).

### 2026-10-16 UTC — calibrated confirmation selects width4; native correctness running

Full confirmation60693 exits0 in169.352535625s; both-host37-file archive/raw
audits pass72 original numerical records and14,400 submissions, no dropped
windows or relaxed bounds. Width4 repeatB1 GPU/wall .623559/.726689, B2
.604956/.658583; nullB1 .992744/.972613, B2 1.000306/1.011905.
Both widths qualify in both complete fresh cohorts. Predeclared geometric-mean
wall choice selects width4 .690318881 versus width2 .712403945. Selection
admits private native integration; no full-model or freshvLLM gain yet.
[Confirmation raw evidence and two-cohort decision](../../results/20261016/vector-vision-calibrated/README.md#confirmation-passes-width4-selected-for-native-investigation).

Create private `vector-vision-native-investigation-20261016` source closure:
1406 files, eight transformations. Exact selected shader appended to strict
prefetch library; adjacent outputs vectorized without changing FMA order.
Guard shape/FP32/alignment/thread limits, preserve scalar fallback, add actual
graph/frame dispatch counters. No new model weights or input reuse; baseline
c186 remains retained. Before native timings predeclare both VL batches >=2%
warmTTFT reduction in two complete cohorts, decode regression <=2%, original
correctness unchanged; this gate does not replace the full2x/halfTTFT goal.

Launch fresh30 original numerical and actual native primitive stages,
PID64007. Configure/build and primitive compilation/execution are terminal0;
new native primitive verifies96 original-FMA bit cases:8 serving/88 fallback,
four changed-input patterns, pending-output rejection and abort/reset/recovery.
Other trained numerical stages remain running, so no full correctness or
performance claim. No live-job consumed source/driver modified.
Two ad hoc read-only process-guard probes initially omitted imports and failed
with NameError before guard completion; full original import closure then
passes. A guessed standalone guard filename was absent. These inspection
failures launched no inference and supply no performance evidence.
[Native design, tests and prospective rule](../../results/20261016/vector-vision-native/README.md).

### 2026-10-16 UTC — compact-aware grouped decode prepared and queued

While exact vector validation64007 remains live, implement a separate decode
component in new unused helpers. This addresses residual textB1/coderB1 gaps
without altering the consumed validation source/driver. Different from earlier
raw128 mixed fusion: compact64/current small64 control, one fused64 grid,
unchanged independent output buffers, original compact/raw/small per-output
accumulation/reduction; raw large group stride8 preserves two-SIMD coverage.
Dense special FP32-key raw route retained. Reject unsupported interleaved-only
control rather than substituting another kernel. CPU fallback/compact bit
roundtrip gates, exact four-input-scale output/padding parity and original FP64
envelope remain mandatory. Weight setup duration/bytes separate from timings.

Prospective2s alternating conditioning,10 complete ABBA rounds x5 samples,
three window warm calls, all records retained; identical-pipeline null arm.
Learned group must improve GPU ANDwall >=3% BOTH B1/B2, own-null GPU ANDwall
within5%, all16 numerical group/batch cases, one unchanged complete confirmation.
All groups including irregular/negative outcomes retained; no blanket native
policy or matched-vLLM gain from components. Generated source/runner syntax
checks pass locally; no actual packing/compile/GPU numerical/timing claim.

Queue73358 watches exact native continuation67688, verified live, waiting for
terminal/archive/audits before any compilation/GPU use. It then archives/audits
each cohort and applies the predeclared repeat gate. New consumed helpers frozen
while queued; existing live numerical/full-model chain remains untouched.
[Complete design, control, exclusions and command](../../results/20261016/compact-fused-projection-groups/README.md).

### 2026-10-16 UTC — retained control phase analysis identifies prefill work

Derive all six c186 warm control clock descriptions from frozen1861996b...,
checking18 raw file hashes and four warm rows/profile against the existing
72-output audit. VL B1 TTFT178.289ms, vision58.630ms, prefill wall89.603ms,
prefill GPU83.781ms; B2 TTFT296.004ms, vision113.156ms, prefill wall151.072ms,
GPU144.813ms. Text/coder prefill GPU spans also remain material. GPU spans
overlap encode/wait, never add them; worker components exclude frontend/IPC,
whose residual is not an isolated causal measurement. This motivates prefill
GPU work alongside vision/decode, without predicting new speed or updating
matched ratios. No fresh GPU execution in this analysis.
Initial local helper matched a generated `.executed.py` as a profile directory
and failed before output; add a directory filter and rerun successfully. No
raw benchmark edits or failed-analysis timing admitted.
[Source identities, complete table and limits](../../results/20261016/vector-vision-native/README.md#retained-control-clock-breakdown-for-next-optimization-priorities).

### 2026-10-16 UTC — exact three-half representation CPU evidence; MPS cost probe queued

Respond to measured prefill GPU cost with a distinct precision-preserving
representation hypothesis: bounded sourceFP32 equals three half-storage pieces
at fixed scales0/11/22, rather than the earlier lossy BF16-feed activation cast.
Actual Windows CPU study exits0:2,826,240 bounded synthetic source values
reconstruct bitwise over six model/row shapes x4 scales. Finite exponent/mantissa
edge sample4590 admits769, requires originalFP32 fallback3821. No universal
coverage, actual model activation range/frequency, runtime fallback or speed
claim. Final CPU evidence pins prospective probe source847f1dc1...; repeat
after unqueued diagnostic reporting additions leaves all admission counts and
numeric data unchanged.

Prepare synthetic MPS product cost floor on model gate shapes: oneF32xF16
versus threeF16xF16 products intoF32, scales/alpha1/2^-11/2^-22, beta0/1/1.
Precomputed source planes and synthetic structured weights deliberately exclude
GPU decomposition and learned/native admission. Original FP64/comparison bounds,
source bits and all output/padding checks precede2s/10x5ABBA timings and own-null.
Four finite diagnostics test product and scale/beta behavior, without assuming
outputF32 proves every internal accumulator instruction. Apple API/spec fetch
did not establish a precision guarantee; no third-party claim is used.

Queue83151 watches exact compact controller73358, verified live; terminal,
archived/audited dependencies and quiet guard required before compile/GPU.
One probe only, preserve negative/exception outcomes; no native selection or
favorable-repeat search. Freeze six consumed helpers. Current vector validation
64007 remains live with16 completed stages, including all privateKV cases,
serving, mixedimage and Qwen3.5/text layers. No full validation/TTFT gain yet.
[Complete equations, CPU evidence, limits and command](../../results/20261016/half-triplet-lower-bound/README.md).

### 2026-10-16 UTC — vector native all30 pass; first model cohort running

Exact validation64007 exits0 in1437.991319083s. Finalizer66876 exits0 after
terminal archival and independent Mac raw/regeneration audit:1764-file
239dbf64... archive. All30 fresh stages,7CTests,96 vector primitive cases,
80 retained compact primitive cases,51trainedhead queries/78rows,4vision grids,
6+6recoveries and4wide profiles/12steps pass. Worst original wide bound fraction
.5432836183473805; all eight changes and owned driver regenerate. Candidate
libraryd52c37e0..., retainedc186 unchanged. Windows archive audit deferred until
measured work is terminal to avoid heavy copy during performance.

Continuation67688 admits initial fullsix-profile cohort89884 only after audit,
verified live. No native timing gain yet; current full-model outputs/counters,
TTFT/decode/init/first/warm/RSS will be independently audited before selection.
Compact and synthetic triplet queues remain sequential behind that chain.
[Archive/library identity and test scope](../../results/20261016/vector-vision-native/README.md#full-native-validation-passes-model-timing-running).

### 2026-10-16 UTC — native initial passes after preserved auditor filename repair

Initial native89884 exits0 in192.701290041s: allsix profiles and72 complete
outputs. Freeze5611-file4391422a... full archive. Original auditor passes all
raw outputs, routes and clock/bound checks, then fails expecting per-profile
`name+'.executed.py'`; frozen comparator actually uses `with_suffix` and writes
three per-model generic expanded files shared by six argv-distinct profiles.
Generated source is identical for both batches. Controller67688 and waiting
queues73358/83151 stop1; no component/capability GPU job launched. Freeze all
three terminal controllers/sources in45-file252aeda0... failure archive before
repair. No runtime/benchmark/numerical gate or initial data changed/rerun.

New audit shim pins frozen original e53bbdd3..., repairs only pathname lookup
to exact comparator behavior and records3 distinct files/6 full profile checks.
Mac and Windows full archive/audits now pass; Windows also independently audits
1764-file full30 and45-file failed-controller archive. The partial audit JSON
remains rejected until repaired process0; no provisional metrics admitted.

Initial VL B1 control/candidate warmTTFT177.9961/172.5206ms, B2
294.5845/287.8972ms:3.076%/2.270% reductions. Decode61.1809/61.2709 and
55.1446/55.2624 TPS:flat+0.147%/+0.214%. Every profile decode fraction>=.98.
Actual candidate VL counts12vector/image; text/coder/control0. Other model
variation is not attributed to vision. Init/first/warm/RSS all retained; small
peak process RSS changes do not prove logical/GPU memory savings.

After completed archive transfer and both-host audit, new controller93884
reuses initial complete cohort and launches one unchanged confirmation93898.
No favorable-repeat search. New dependency-only wrappers queue compact94685
after93884, synthetic triplet94813 after94685; numerical source/gates and
sampling for both remain unchanged. Do not mutate consumed live source. No
promotion, fresh vLLM ratio or goal target-count update.
[Complete initial table, cost clocks and failure evidence](../../results/20261016/vector-vision-native/README.md#initial-model-cohort-passes-after-audit-path-repair).

### 2026-10-16 UTC — actual vector native confirmation passes; compact component running

Confirmation93898 exits0 in192.768228250s; repaired controller93884 exits0.
Freeze5615-filee7536a9b... archive and Mac independent72-output/route/raw audit;
all original numerical bounds and three-script/six-profile regeneration pass.
Serving source/library/cases/limits unchanged. RepeatVL B1 TTFT179.4877->
174.9992ms (-2.501%), B2 297.7506->287.7258ms (-3.367%). Decodefractions
.996983/1.001774; all six repeat decode profiles>=.98. Both complete native
cohorts pass the prospective2% both-VL screen. TextB1 repeatTTFT35.8427->
36.7988ms cost remains recorded; do not attribute/exclude it. No freshvLLM ratio
or fullgoal/targetcount update, retainedc186 unchanged.

Compact component97616 actually starts after native terminal/archive/audit;
MPS synthetic probe waits behind its exact controller. Defer heavy Windows
repeat archive transfer until measurements are terminal, then independently
audit before installing any selected source change. Current source/driver
remain frozen while live.
[Repeat scope, costs and archive identity](../../results/20261016/vector-vision-native/README.md#confirmation-passes-native-screen-on-mac-windows-archive-audit-pending).

### 2026-10-16 UTC — two actual follow-up experiments complete; both stop before integration

Compact component97616 exits0 in51.396588375s, archived33-file28e0d5b2...,
queue94685 exits0 after independent Mac raw audit. All16 group/batch original
bit/padding/FP64 checks pass,6400 samples retained. No learned group passes
both-batch>=3% GPU ANDwall plusownnull5%. Some coder/text B2 groups improve,
but B1 gains/nulls fail; VL misses required gain. No filtered/null-normalized
windows, hardware-cause claim, favorable repeat or native integration.
[Complete observed limitations and archived identity](../../results/20261016/compact-fused-projection-groups/README.md#actual-initial-result-numerical-pass-no-selection).

Synthetic MPS probe98524 exits0 in28.241084708s,20-fileee9d6968... archive,
queue94813 exits0 after independent Mac raw audit. All32 finite precision/scale
diagnostics and six synthetic shapes/original bounds/padding pass;2,826,240
host source-bit reconstructions and2400 timings. Three products cost2.806–
2.897xGPU and1.972–2.238xwall before GPU splitting; null within5%. Stop this
separate-product MPS strategy, no learned/native selection or favorable-repeat
search. Finite numeric feasibility does not prove general internal accumulator
dtype or model admission; no fused-Metal cost claim. Windows transfer/raw audits
pending with all measured jobs terminal; do not treat pending copied files as
complete until SCP session reports0.
[Actual capability/cost scope](../../results/20261016/half-triplet-lower-bound/README.md#actual-mps-result-numeric-feasibility-unfavorable-cost).

### 2026-10-16 UTC — both-host repeat audit complete; validated vision source retained in branch

SCP67223 completes0 before extraction. Windows verifies e7536a9b...5615 files
and all72 repeat outputs, original gates/routes/clocks/source regeneration.
Both-host initial/repeat decisions agree: native confirmation passes. Initial
and repeat total144 complete outputs. Integrate exactly eight validated source
files into feature/qwen35-metal only after asserting current before hashes and
all tested after hashes. Runtime is guarded strict float4 vision context with
original scalar fallback and published actual dispatch counts, under existing
serial-vision option. No weight copies, input caching or arithmetic change.
Remote defaultc186 and frozen d52 comparison builds stay untouched; source
identities recorded in branch-source-integration.json. Full2x/halfTTFT goal
remains active and requires fresh repeated matched vLLM evidence.

Windows also independently audits compact33-file28e0d5b2... (16 checks,
104093 validated executions,6400 timings, original worst FP64 .0090262) and
MPS20-fileee9d6968... (32 finite diagnostics,6synthetic shapes,2400 timings).
No compact group selected; no learned/native integration for separate MPS
triplets. Keep all negative raw data/source/decisions. All measured jobs and
controllers are terminal and archived; no GPU work overlaps evidence transfer.
Next optimization work must address remaining decode and prefill GPU gaps;
neither component failure nor this modest retained VL improvement completes
or narrows the original goal.
[Full repeat/cost tables and exact integrated source hashes](../../results/20261016/vector-vision-native/README.md#both-host-confirmation-audit-and-branch-integration).

### 2026-10-16 UTC — final controller provenance preserved; this chain quiescent

Archive all four successful current/repaired controllers, snapshots, commands,
resume records and wrapper dependencies after verifying their exact PIDs are
gone and exit0.96-file63348bf4... bundle also transferred terminal0 and audited
on Windows. Old failed controllers remain separately in252aeda0..., without
rewritten statuses. This chain has no live GPU/measurement/controller jobs.
The retained native vector source is installed in this investigation branch;
remaining full goal work is active, not blocked or complete. Do not poll stale
PID files as evidence of a live wait on the next continuation.
[Final controller artifact](../../results/20261016/vector-vision-native/final-controllers/archive-verification.json).

### 2026-10-16 UTC — all-layer union eligibility and prepared-context primitive

The preceding status-only turn changed no evidence. Revalidate the authoritative
archives before continuing: both hosts independently pass the 38-file prepared
context primitive audit, and the full learned CPU admission audit passes.
Neither changes the latest matched-vLLM target counts.

Hypothesis: prepare a common-scale gate/up union before inference, broadening
coverage beyond pairs with identical independent scales, then encode one MPS
product and one activation-only kernel. Preserve the original independent
weights and required fallbacks. Compared with the previous lazy joined method,
this explicitly separates CPU packing/setup from encoding and avoids separate
gate/up output writes. It adds a union weight buffer and scratch; lazy first
GPU compilation remains possible. No full-model gain is inferred.

All 76 learned pairs / 152 matrices receive independent NumPy and original C++
codec checks. Of 561,512,448 values, 561,512,447 restore to the original bits;
one coder layer8 value rejects its entire pair. Union admission is text28/28,
coder23/24 and VL24/24, versus prior equal-scale coverage17/28,13/24 and0/24.
Host execution times are28.6250 seconds for NumPy and3.7440 seconds for original
C++ including compilation. Logical duplicate payload estimates if all admitted
pairs are prepared:336/382.375/336 MiB; not measured RSS or memory savings.
[CPU provenance, raw offsets and both-host audit](../../results/20261016/joined-union-half-admission/README.md).

Private prepared-context primitive PID914 exits0 in11.2132 seconds including
quiet interval and compilation. Twenty-four changed-input exact dyadic cases
cover all six serving physical row counts, intermediate products, activation
bits, pending/foreign ownership, alias, abort/recovery, unsupported shapes,
nonexact weights and bounded128-pair/2-reject ownership controls. Both-host
audit regenerates the exact generated source from frozen d52 context/factory.
No learned GPU or inference performance admission from this synthetic test.
[Primitive evidence, costs and prospective learned design](../../results/20261016/prepared-union-mlp/README.md).

Before learned GPU measurements, declare all76 pairs/both batches/four changed
structured-input patterns under original numerical bounds, then only first
and last layers of each model for timing (12 profiles). Common2-second warmup,
10ABBA rounds,3 warm calls and5 retained samples/window. Each own-null precedes
the comparison; means use every sample, never null normalization. A model needs
>=5% GPU AND wall reduction at both endpoints/both batches, own-null within5%,
all numerical/fallback checks, then one unchanged confirmation. Full graph
liveness/correctness and native/matched comparator repeats are separate gates.
Learned PID1987 launched after archived prerequisites; compile exits0, numeric
screen is actually executing. Sources remain frozen until terminal archive.

### 2026-10-16 UTC — prepared union learned component completes; confirmation rejects integration

Initial1987 exits0 in120.124820292s and is frozen43-filea3e4df6e... before
analysis. Mac audit verifies all152 numerical profiles,608 changed-input cases,
849,807,360 checks; original bounds and required coder layer8 fallback pass.
Twelve predetermined first/last-layer profiles retain4800 timing samples,
including own-null controls. Only coder meets all four required5% GPU/wall
gains and own-null gates. This admits one unchanged complete confirmation,
not a full-model improvement. Sources and design are checked unchanged before
launching4097; initial archive/audit/decision hashes are captured at launch.

Confirmation4097 exits0 in119.978341708s; freeze44-file954c6ff4... before audit.
All original numerical/fallback gates pass unchanged, and all repeat nulls
pass. Coder last-layer B2 candidate/control GPU .976858 and wall .969217 miss
the .95 requirement, despite passing other endpoints. Text B1 and VL first-layer
B2 also miss. Initial VL first-layer B1 null GPU1.063222 was outside5%; it
remains an exclusion, not a correction or inferred hardware cause. The combined
initial/repeat rule selects no model. Do not choose favorable layers, lower
the threshold or rerun unchanged seeking a third favorable cohort.

Both-host audits pass304 all-layer profiles and9600 retained timing samples,
regenerating private source/header, compile flags, asset identities and guards.
Full tables, costs, warnings and failures are retained. First-shape preparation
costs~343..486ms/pair initially, plus original uploads; hypothetical retained
duplicate payload336/382.375/336MiB. No full-model startup sum or RSS inference.
This exact prepared activation-only version stops before graph integration;
accepted vector serving sources and matched target counts remain unchanged.
All jobs are terminal/archived; no live measurement is inferred from PID files.
[Complete two-cohort tables, controls, source identities and decision](../../results/20261016/prepared-union-mlp/README.md#two-terminal-learned-cohorts-numerically-valid-no-repeated-selection).

### 2026-10-16 UTC — distinct heterogeneous Q/K/V union CPU prerequisite passes

Inspect current lowering: Q, K and V projections are consecutive from the same
input, before bias/normalization/rope consumers. New hypothesis joins three
unequal-width outputs, then scatters to the existing independent outputs. It
does not repeat the rejected activation-only MLP method or its component data.
Include current FP32-stored text/coder key matrices in learned-value checks;
exact packing does not itself admit a changed MPS arithmetic route.

Actual CPU6423 exits0 in12.951632625s including NumPy, compilation and original
C++ codec. All58 separate triples/174 matrices checked: text28/28 and coder24/24
admitted; VL4/6. Preserve original fallback for VL dense layers7 and23, each
with one rejected common-scale value. Existing18 VL combined linear-attention
layers remain outside this separate-projection hypothesis.173,670,398 values
roundtrip to original FP32 bits;2reject.52 original FP32 key matrices included.
Logical extra union payload estimates224/47.25/40MiB; no GPU allocation or
full-model RSS claim. No source integration or performance result follows.

Freeze20-file31424748... archive after terminal; both Mac/Windows independent
provenance audits pass with regenerated exact original-C++ arity/count adaptation.
Next safe work is new private primitive and all-learned GPU numerical screening,
including original FP32-key controls and both mandatory fallbacks, before any
predeclared timing experiment. No active measured jobs remain in this chain.
[Full eligibility, source identities and remaining GPU requirements](../../results/20261016/qkv-union-half-admission/README.md).

### 2026-10-16 UTC — heterogeneous Q/K/V primitive and complete learned GPU screening pass

New private generated context retains all independent weights and three output
allocations. Prepare exact common union/MPS product/scratch outside Begin;
encode one joined product plus scatter, with bounded64-group strong ownership,
original readiness/foreign-writer/alias/fallback guards and actual counters.
FP32 key originals remain the control; no original weight cache is mutated.
This prototype adds union storage and scatter traffic, with no model gain yet.

Actual primitive6834 exits0 in11.194792416s including quiet interval/compile.
All24 unequal-width dyadic cases, original FP32-key controls, joined/output bits,
pending/foreign ownership, alias, abort/recovery, unsupported/nonexact fallback
and64-group/2-reject cache controls pass. Freeze39-file11a891b1... before any
further stage. Both Mac/Windows independent exact source-regeneration/audits
pass. Logical outputs are covered, not arbitrary physical padding or captured
model activations. The initial union fixture uses336 bytes across six shapes.

Before learned GPU execution, declare all58 groups/both batches/four structured
patterns under original gamma/sumabs and5e-6+5e-5 bounds. Timing endpoints are
fixed by CPU eligibility alone: text0/27,coder0/23,VL3/19 at both batches; all
interior and unsupported groups remain in numeric screening. Null then candidate
comparison each uses2s alternating warmup,10ABBA rounds,3 warm/5 retained samples.
Both endpoints/batches need>=5% GPU AND wall means, each own-null within5%, all
numeric/fallback checks, then one unchanged full confirmation before graph work.

Actual learned7300 compiles and completes116 numeric profiles/464 cases,
188,276,736 reference values and59,482,112 exact scatter bits.52 original FP32
key matrices and16 fallback cases pass unchanged. Worst gamma .0213529,
projection-comparison .0678973,control/candidate .0507740; peak component RSS
323,338,240bytes. The predetermined endpoint timing stage is actually running;
sources remain frozen until terminal archive. No native/full-model/fresh-vLLM
gain is inferred from these component correctness results.
[Protocol, scope and primitive evidence](../../results/20261016/prepared-union-qkv/README.md).

### 2026-10-16 UTC — Q/K/V repeated component selects text only; both-host audits complete

Initial7300 exits0 in85.327812625s; freeze45-file890cb0e8... before audit. Every
original numerical/FP32-key/scatter/fallback gate passes and all4800 timing
samples are retained. Text meets>=5% GPU AND wall at all four predetermined
endpoint/batch profiles with own-nulls within5%. Coder misses first-layer B2
wall gain and last-layer B1 own-null; VL is slower despite lower CPU encode
time. Admit exactly one unchanged full confirmation, not a model performance
claim. Launch8868 only after comparing every measured source hash and record
the exact initial archive/audit/decision identities at launch.

Confirmation8868 exits0 in85.406805458s; freeze46-fileed28aa1c... before audit.
Both-host exact source/header regeneration, all232 numerical profiles,52 original
FP32-key matrices/cohort,32 fallback cases across cohorts and9600 timing samples
pass. Text repeat GPU fractions .719165/.650452/.698773/.655579; wall fractions
.847805/.795903/.828562/.809635, with own-nulls passing. Only text is selected.
Coder's favorable repeat cannot rescue its failed initial gates; VL remains
slower in both. No threshold change, filtered windows, null normalization or
third repeat. First-shape text preparation costs238.485..246.950ms/triple
initially; duplicate logical text payload224MiB. Process peaks and full cost
ranges remain component-only, not measured model startup/RSS.
[Complete tables, exclusions, clocks and frozen artifacts](../../results/20261016/prepared-union-qkv/README.md#complete-repeated-component-result).

### 2026-10-16 UTC — private text graph/session integration builds; full33 validation running

Inspect actual resident architecture before integration: per-graph and resident
session GPU contexts differ. Preparing only the graph context would miss the
resident cache and could put packing into first inference. Recognize only named
consecutive Q/K/V with one input and confirmed2048/1024/1024-by1024 text geometry.
Prepare supported24/44-row unions on the resident session context during build,
using unchanged shared BF16 owners also used by future prefill graphs. Standalone
graph.run prepares on its own context before Begin and includes that first-use
cost. Keep all three original output writes, aliases/readiness/foreign ownership
guards, original fallbacks and actual graph/session/prefill/decode counters.
No input cache, speculation or arithmetic change is introduced.

Freeze a private source copy on top of accepted d52 vector vision,1411 files,
12 exact transformations/5 added helpers, with all original/before/after hashes.
OFF-by-default feature targets only confirmed text shapes; retained/serving,
coder/VL and protected GPT-OSS/Pi sources remain unchanged. Declare later native
rule before any performance: both text batches>=3% warm TTFT reduction in two
complete unchanged cohorts, no>2% decode regression at any of six profiles,
complete outputs/all original gates; init/first/warm/RSS separately. Control
must be accepted d52, not obsolete scalar c186. Fresh vLLM matching still needed.

Actual native10766 compiles, links and passes vector/QKV primitives, all7 CTests,
full vision grids and trained real heads. Original exact tail cases are running;
no full33 or model gain is claimed. Require original30 plus QKV primitive
compile/run and a trained random-input Qwen3 text graph/session stage at chunks
24/22, first/last layers and both batches (8 profiles/24 changed state steps,
12 serving/12 fallback route checks), preserving original full hidden/key/value
bounds, invalid controls, paused inputs, reset/close and explicit/resident equality.
Keep reproducible fixtures/expected buffers by identity. Local source/runner/
auditor expansion preflights pass after creating the dry-run output parent;
that local setup issue did not change a measured runtime or tolerance.

Finalizer15326 watches only verified-live10766, archives terminal failure or
success, audits successful complete33 evidence, and publishes the gate only on
success. It does not start inference/performance comparisons. Freeze all source,
runtime and drivers until terminal archive; no restart from observation timeout.
[Private preparation, methods, costs and live-stage scope](../../results/20261016/prepared-qkv-native/README.md).

### 2026-10-16 UTC — exact validation continuation armed; native performance not yet claimed

Full33 validation10766 is still verified live and16 complete stages have exited0,
including all7 CTests, linked vector/QKV primitives, full vision grids, trained
heads, both original dense tail suites, private KV, serving and mixed-image
checks. Remaining trained layers/recovery/wide/new24-row checks are running.
No full33 or native performance pass is inferred from this partial result.

New continuation26534 watches verified-live finalizer15326 only. It archives
that terminal controller/gate before the native launch re-audit adds the checks
archive identity. It then admits one full six-profile native ABBA, waits the
exact process, archives terminal success or failure, independently rechecks all
72 complete outputs, current d52 control, candidate QKV counters and all decode
steps, and applies the previously declared3% both-text/2% all-profile-decode rule.
Only a complete passing initial screen permits one unchanged confirmation.
No third favorable-repeat search, overlapping runtime or restart from an
observation timeout. Every source/driver/runtime remains frozen until its own
terminal archive. Adding the new unused helpers does not alter the live check
driver's five-file snapshot or private1411-file source closure.

Expanded comparator and original72-output auditor transformation/syntax
preflights pass before deployment. Preserve the known Path.with_suffix naming
semantics for six distinct argv profiles/three expanded script files. Because
the new full source gate includes docs and small binary fixtures, the native
archive explicitly carries all original source/snapshot bytes; no suffix-based
omission may invalidate later independent checks. Fresh full-model timing and
Windows raw audits remain pending; the full matched-vLLM goal stays active.
[Exact continuation and current live-stage record](../../results/20261016/prepared-qkv-native/README.md#sequential-continuation-armed-after-the-exact-validation-finalizer).

### 2026-10-16 UTC — full QKV native integration rejected on original key bound

The original validation driver exits 1 after 25 passing stages because its
intermediate assertion still expects 23; freeze that run and both stopped
controllers before repair. A separate resume driver copies the original 25
records unchanged and executes only missing suites against the same e34
candidate library. Six recovery stages pass, then the added trained random
test fails at layer 0, batch 2, chunk 22, physical rows 44, first state step.
There are 31 passing stages, not a full 33-stage success. No performance
continuation or candidate model benchmark is admitted.

The candidate key-cache worst original-bound ratio is 1.5165457144345686.
The accepted d52 implementation passes all three state steps on the exact
saved fixture, worst ratio 0.5204084868377323. All 21 saved input/expected
buffer identities and the original accepted driver are verified; candidate
graph and session counters both equal 1. Preserve the original bounds. Reject
full-QKV integration despite its repeated text component timing. Packed
weight equality and FP32 arithmetic are insufficient to infer downstream
numerical equivalence; the observation does not identify an internal MPS
algorithm. Both-host rejection audit verifies 2,195 frozen files. The next
variant must retain the original K GPU projection and receive new evidence.
[Terminal failures, same-fixture control and archive hashes](../../results/20261016/prepared-qkv-native/README.md#final-rejection).

### 2026-10-16 UTC — integer half inverse passes exhaustive Windows and Mac CPU tests

Hypothesis: replace the expensive half-to-float plus ldexp verification in a
future setup loop with an exact integer bit inverse. This is separate from
the Q/V GPU experiment and is not installed in any runtime. The same header
passes Windows MSVC x64 and actual Apple clang ARM64 CLZ-branch exhaustive
tests: 8,454,144 original-codec cases, 8,454,144 half-word cases and 393,216
invalid-scale cases per host. All 1,122,816 admitted codec results and
8,189,952 finite half reconstructions match original FP32 bits/independent
floating reference; rejected outputs remain unchanged.

Preserve the first Mac compile failure, cstdint not found from a direct
resolved compiler invocation, before repairing only the runner to invoke
xcrun clang++. No mathematical header or acceptance condition changes.
Windows archive and failed/successful Mac archives are verified locally;
successful Mac evidence is also audited there. Setup performance, learned
packing and runtime effects remain unmeasured.
[Counts, compiler commands and frozen proofs](../../results/20261016/prepared-half-inverse/README.md).

### 2026-10-16 UTC — QV scatter variant preserves key bits but fails complete timing screen

Change the rejected full-QKV algorithm: pack only Q/V, retain conservative
previous triple admission/common scale, and invoke original EncodeBF16MatVec
for K. Inspection confirms text/coder floatPrefill uses the split BF16 key
projection with FP32 accumulation, not an FP32-weight MPS call. The archived
factory comment about an original K MPS call is imprecise; the actual function
call and bit-identical K outputs are checked. Keep all original independent
outputs, alias/readiness/foreign ownership checks and fallbacks.

Primitive 42044 exits 0 in 11.290640791 s. Learned 42323 exits 0 in
83.629794833 s, with all 116 numerical profiles, 464 changed structured-input
cases, 52 original FP32-accumulation key matrices, 16 fallback cases,
177,553,408 reference-value checks, 48,758,784 scatter-bit checks and
10,723,328 original-K bit checks. Both-host source/context regeneration and
complete data audits pass. The previous eligibility excludes two VL groups;
this is not complete independent pair admission.

Keep first/last endpoints and both batches, 2 s warmup, own-null and 10 ABBA
rounds, all 4,800 samples. No model meets >=5% GPU AND wall gain at all four
profiles with own-null within 5%. Text GPU fractions 1.063689, 1.096286,
1.071997, 1.075359 are worse despite approximately 42% less CPU encode time.
Coder is mixed; VL is slower. Reject without confirmation or native build.
Extra duplicate QV payload estimates are text 168 MiB, coder 42 MiB and admitted
VL 36 MiB. Text preparation is 195.198459–200.536792 ms per first shape/group;
these component costs are not measured model setup. The integer inverse is
not used. Freeze all source/logs before evaluating the next algorithm.
[Full endpoint table, costs and frozen artifacts](../../results/20261016/prepared-union-qv/README.md).

### 2026-10-16 UTC — canonical QV output views remove scatter but fail full model endpoint rule

After the scatter variant is terminal and archived, prepare a new private
context component. Keep original K, previous triple eligibility/common scale
and exact Q/V weights; create canonical disjoint column views of one joined
output, with one extra guard row. Adapt only Download to honor offsets. The
candidate directly writes those Q/V views through the joined product and
encodes no scatter. No downstream GPU/host-write or native graph/session
support is claimed. An unused compiled scatter pipeline remains outside the
candidate encode path. Serving source/library are unchanged.

Primitive 44008 exits 0 in 11.285164042 s; all 24 changed-input offset/guard,
original-key, ownership, pending, abort/recovery, fallback and bounded cache
cases pass. Learned 44331 exits 0 in 83.593469917 s; all 116 profiles, 464
changed structured-input cases, 177,553,408 reference checks, 48,758,784 exact
view-bit checks and 10,723,328 original-K bit checks pass. Source/context
regeneration, all 4,800 timing samples and original numerical gates are audited
on both hosts. The guard row is checked at every candidate validation.

Retain the unchanged >=5% GPU AND wall gain at all four endpoints/model and
own-null within 5%. Only five individual profiles pass. Text last-layer
profiles miss the gain rule; last-layer batch-2 null wall fraction .948814
also fails. Coder first-layer batch-2 GPU fraction .969518 misses; VL misses
GPU gains. No model qualifies for confirmation, so reject without repeating
or integrating the unchanged method. Better individual component times do
not establish a new native or vLLM ratio. Additional guard storage and
unimplemented consumer support are explicit limitations.
[Complete output-view evidence and timing exclusions](../../results/20261016/prepared-union-qv-views/README.md).

Next investigation: inspect whether Q and K normalization/RoPE can share one
dispatch while preserving the existing per-head SIMD reduction, precise
trigonometry, output writes and original FP32 behavior. This is a proposed
method, not an executed optimization. It could affect both decode and
prefill, unlike the prefill-only Q/V union experiments.

### 2026-10-16 UTC — paired norm RoPE passes original bits; first timing cohort needs calibration

Generate a new private context from accepted d52. Pair Q/K normalization and
RoPE in one dispatch while retaining the original per-head arithmetic body
verbatim, 32-lane SIMD reduction, precise trigonometry and independent output
writes. No packed weights/output views, input cache or speculation. Require
alias/readiness/foreign writer guards and original fallback behavior.

Primitive 46016 exits 0 in 10.590449334 s: all 60 changed synthetic cases and
3,710,720 original FP32 values match bit for bit, including partial rotary, unusual
head sizes, normalized/unnormalized modes and offset variants. Learned 46457
exits 0 in 119.748470250 s: actual native Q/K projection fixtures and checkpoint
norm weights cover every separate full-attention layer, 232 decode/prefill
profiles, 928 changed-input cases and 47,321,088 bit-identical output values.
This includes 68 learned norm matrices, coder's 48 all-ones placeholders and
the two VL layers excluded by prior union packing. The timed endpoints use
the true first/last separate layers, including VL 3/23, fixed before execution.

All 9,600 samples/1,920 windows, 2 s warmups, ABBA order, four changed host
fixtures, original output-bit checks and source/context identities are audited
on both hosts. Numeric/timing process peaks 253,640,704/132,186,112 bytes are
component-only. Single-pair GPU intervals are short; only 8/24 own-nulls stay
within 5% for both GPU and wall time. Preserve all exclusions and do not infer
a stable model speedup, normalize by nulls or start a favorable repeat.

Next measurement is a separately identified fixed 32-stage chain per command
batch, with carried outputs, independent allocations, original bit checks,
all endpoints/samples and unchanged own-null exclusion. Longer work changes
timing granularity and workload; it is not confirmation of the first cohort.
No new calibration, native integration or fresh vLLM benchmark has run yet.
[Complete method, raw evidence, all endpoint clocks and limitations](../../results/20261016/paired-norm-rope/README.md).

### 2026-10-16 UTC — fixed carried-chain calibration confirms text and coder component decode

Preserve the unstable one-pair cohort and change the measurement workload:
32 dependent pairs per command batch, each consuming the previous output,
with independent allocations and all stage outputs retained/bit-checked.
Weights/positions remain those of one learned layer across the chain; this
is not a model stack. The normalization arithmetic/context stays unchanged.
Regenerate the distinct C++ and driver from frozen sources and record hashes.

Initial 48751 exits 0 in 149.624152750 s. All 232 numerical profiles/928 changed
cases and 1,514,274,816 original FP32 values match bit for bit; all 24 endpoint nulls
are within 5%. Keep all 9,600 timing samples, 1,920 ABBA windows, 2 s warmups
and original first/last layers/all four row profiles. Declare >=5% GPU AND wall
gain at all four endpoints per model/workload, with every own-null within 5%.
All three decode workloads pass initially; every prefill workload fails.

Complete unchanged confirmation 51322 exits 0 in 148.896526375 s. All source,
context, generated harness, flags, assets/config and design identities match;
all numerical coverage repeats. Only text/coder decode passes both cohorts.
VL decode fails repeat layer 3/rows 2 own-null wall fraction 1.060406, despite
favorable comparison; exclude it without a third repeat. Coder prefill and VL
prefill also have repeat exclusions, and initial failures remain failures.
Across both complete cohorts, 3,028,549,632 numerical-stage FP32 values match bit for bit and
19,200 timing samples are audited on Mac and Windows. Preserve the local
premature-extraction error: wait for the original SCP session to exit 0,
then extract/audit the same bytes; no runtime, sample or source was repaired.
[Full table, fixed selection rule, exclusions and archive hashes](../../results/20261016/paired-norm-rope-chain/README.md).

### 2026-10-16 UTC — private row1/2 norm RoPE native build starts full34 validation

Copy accepted d52 source to a fresh private root, 1,411 files, with 12 exact
transformations/5 added helpers. Select only confirmed text/coder rows 1/2;
this matrix rule also includes tiny prefill chunks, while serving benchmark
prefill and all VL shapes retain original operations. Recognize only adjacent
Q/K norm/RoPE with shared controls and matching original parameters, retain
both original output writes and projection paths, profiling/fallback behavior,
and publish actual graph/session/prefill/every-decode counters. No model cache,
speculation, packed weights or output views. Feature defaults OFF.

The actual private library builds: 0c0faf71...308d0d. Linked vector and paired
primitives, CTest, full vision grid and trained real-head stages pass; actual
54126 is still live in original dense tails, 9 completed stages, not a full34
success. Require all original30 afresh plus linked paired primitive compile/
execution and text/coder trained rows/fallbacks: 16 added profiles/48 carried
steps, original hidden/key/value bounds, private states, invalid/pause/reset/
close controls, explicit/resident equality and 36 serving/12 fallback routes.
Preflight all stage checkpoints25/31/34 and complete independent auditor before
launch. All 12 local before/after and 5 added-source identities match Mac prep.

Finalizer 58273 watches only the verified check PID, archives terminal failure
or success, then audits successful full34 data and publishes a gate only after
complete correctness. No model benchmark has run. Freeze consumed source,
library and drivers until terminal archive. Prospective native retention
requires two complete all-six-profile cohorts, >=1% decode gain at the same
currently unmet B1 text/coder profile in both, every decode fraction >=.98,
every TTFT fraction <=1.02 and all original outputs/routes. Init/first/warm/RSS
stay separate; fresh matched vLLM measurements remain mandatory.
[Frozen native preparation, commands, limits and current job](../../results/20261016/paired-norm-rope-native/README.md).

Prepare unused native comparison/launch/archive/audit/analysis helpers while
the exact correctness job runs; do not modify its consumed files. Local
comparator and original complete72-output auditor expansion/AST preflights
pass with actual prefill/every-decode pair counts. Copy only the six unused
helpers to Mac. The model launch requires a terminal successful finalizer,
exact archive/gate hashes and a complete raw34 re-audit, then the exclusive
quiet guard. No model performance job is started by this preparation.

At 05:19:54 UTC, fresh process-handle observation confirms 54126 and 58273
live, 16 completed stages all exit 0, including dense tails, private KV,
serving and mixed-image tests. No terminal record or published full34 gate is
present. Save the exact observation; an incomplete suite remains incomplete.

### 2026-10-16 UTC — full34 passes and archive-only recovery admits native comparison

Fresh check 54126 exits 0 after 1,898.893526708 seconds with all 34 stages.
Original finalizer 58273 archives it but exits 1 because the private source
predates the vector auditor and the archive omitted that external dependency.
Continuation 78736 archives the terminal finalizer and exits 1; no profile
launch occurs. Confirm all three exact PIDs terminal and preserve both failed
controllers, their consumed tools and the original checks archive identities
before any repair. No runtime/test/tolerance change is needed or made.

Package both original vector/MLP auditors as external validation dependencies,
matching their previously frozen controller source hashes. Pin the original
vector auditor identity, restore the needed re import, and publish separately
named corrected archive/audit paths through the gate and launcher. Never
overwrite the original failure archive. Repaired finalizer 88835 exits 0 in
14.309746209 seconds. All 1,915 archive files and the complete numerical audit
pass independently on Mac and Windows with identical records: 12 regenerated
private changes, 34 fresh stages, no reused stages, 60 linked primitive cases,
3,710,720 bit-identical FP32 values, 48 added trained carried steps and
36 selected/12 fallback routes. Original bounds remain fixed; worst added
trained ratio 0.1624941610 and worst wide VL ratio 0.5432836183.

After completed transfers, continuation 89012 and first native cohort 89030
are verified live at 05:44:42 UTC, first text B1 profile started behind raw34
re-audit and quiet interval. Freeze all consumed shared tools/private source.
Only the fixed complete all-six-profile screen can admit one unchanged
confirmation; no timing gain or matched-vLLM target update is established.
[Commands, archive hashes, failures and independent full34 records](../../results/20261016/paired-norm-rope-native/README.md).

### 2026-10-16 UTC — wider delta value tile CPU indexing prerequisite

Hypothesis for VL first-token work: widen the existing 16-value recurrence tile
to 32/64 values per 128-thread group, sharing the identical Q/K norm prefix
across extra float4 bands. Retain literal per-value arithmetic and original
barriers. More registers may outweigh reduced duplicated work. This differs
from the previously rejected extra-dispatch norm cache. No Mac tools change.

Preserve an initial CPU audit authoring failure before enumeration: the control
keeps original variable names while the assertion expected renamed candidate
names. Fix only control-block membership, with identical generated shader.
Corrected enumeration and independent frozen-source reproduction pass all
12 width/batch/padding profiles, 4,718,592 state words and 3,575,808 output words
at chunks 1/16/80. Original recurrence matches accepted d52 and private paired
source. The 2.047-second CPU duration is not GPU/model latency; compilation,
numeric/state equality, learned references and timings remain unmeasured.
[Method, preserved failure, sources and reproduced CPU proof](../../results/20261016/delta-value-tiles-addresses/README.md).

### 2026-10-16 UTC — complete native paired norm RoPE screen rejected

All six native profiles finish without interruption in 192.471940959 seconds.
Continuation 89012 exits 0 in 212.306754458 seconds after archiving, independent
72-output audit and fixed analysis. Reproduce the 6,096-file archive/full audit
and identical selection on Windows; all original output, clock/protocol and
actual paired-route assertions pass. Preserve both terminal controllers before
changing any consumed tool. No model source is promoted.

Text B1 decode fraction 1.012209 and coder B1 fraction 1.018985 exceed the fixed
1% improvement threshold. Every decode fraction remains >=.98. However text
B2 TTFT fraction 1.041447 and VL B1 fraction 1.039354 exceed the predeclared
1.02 preservation limit, so reject without an unchanged confirmation. VL
does not select paired execution; its observed timing difference cannot be
attributed causally to the changed primitive. Keep all requests and no favorable
retry. Init/first-after-init, warm and worker RSS remain separately recorded.
These native-only figures do not update matched-vLLM results or target counts.
[All six timings, setup/RSS costs, raw evidence, hashes and fixed decision](../../results/20261016/paired-norm-rope-native/README.md#complete-native-screen-rejects-integration).

### 2026-10-16 UTC — GPU wider value tiles confirm prefill component gain only

After the complete native paired-norm rejection and terminal controller archive,
prepare a separate private GPU probe of original width 16 and wider 32/64 value
tiles. Preserve first setup process 92974 exit 1 before compile/GPU: the reused
guard needs the output alias. Freeze 19 source/log/metadata files, then define
the alias. Before fixture execution, correct the chunk 1 extension so a learned
single-token fixture is active rather than inheriting the inactive final token.
Assert actual active asset values. The shader/math/tolerance is unchanged.

Corrected process 93093 exits 0 in 90.803874875 s, compile/prepare/numeric/timing.
All 208 profiles and 1,872 output/state checks match original bits over three
carried steps with changed queries, both batches, widths 16/32/64, padding 0/4,
synthetic inactive/correlated/zero-decay modes and learned first/last linear
layers 0/22 at chunks 1/80. The 553,986,648 compared FP32 words include guard words
and control self-checks. The NumPy fixture maker uses real learned projections
with synthetic hidden/prior state; this is not Garnet full-model inference.

All 8 learned timing endpoints and both widths retain 6,400 samples, 16 primitive
dispatches/command, 2 s common warmup/width, own-null before comparison, 10 ABBA
rounds, 3 warm calls and 5 raw samples/window. Fixed statistic is ratio of raw
arithmetic means, no filtering or null normalization. Require own-null GPU/wall
within 5% and >=5% gain for both clocks at all four endpoints/workload.
All 16 initial nulls are stable. Only width 64 prefill passes all four endpoints.

One unchanged complete confirmation 93509 exits 0 in 90.376146083 s. All 13 source
identities, generated shader/preparation, flags, model/config, assets and design
match. Repeat every numeric/timing profile and both widths, retaining initial
failures. All 16 repeat nulls are stable; only width 64 prefill confirms. Initial
GPU fractions .869049–.915350, repeat .873431–.906314; wall fractions initially
.876546–.912910, repeat .880689–.903724. Both hosts reproduce both complete
audits and identical two-cohort decision: 416 numeric profiles, 3,744 checks,
1,107,973,296 word comparisons and 12,800 raw timing samples. No third repeat.

Keep rejected widths/decode endpoints and setup failure. Isolated control
prefill is roughly .326–.331 ms B1/.626–.629 ms B2 per dispatch; measured differences
are only .040–.043 ms B1/.053–.056 ms B2. Do not equate kernel ratios with TTFT or
extrapolate a whole-model gain. Native policy/ownership/graph numerical checks,
complete native model cohorts and fresh repeated vLLM comparisons remain.
No serving source or target count changes.
[All endpoint tables, protocols, source/hash records, setup failure and both-host audits](../../results/20261016/delta-value-tiles-gpu/README.md).

### 2026-10-16 UTC — guarded native width64 prefill implementation enters full32 checks

After both complete unchanged component cohorts select only width64 prefill,
copy accepted d52 to a fresh private root: 1,410 source files, 12 exact changes
and 4 added helpers. Actual Mac preparation matches every local regenerated
before/after and added-source identity. Feature defaults OFF and selects only
B1/B2, chunk80, heads16, key/value128, aligned FP32 zero-offset layouts and a
supported 128-thread pipeline. Keep original argument/ownership/alias checks,
independent output/state writes, normalization and per-value arithmetic.
Other geometries and decode retain the original kernel. Expose/reset actual
context/graph/resident-session counters; workers retain prefill/every-decode
counts and original trained references assert each selected/fallback route.

Check 94442 builds candidate 03ac447d...9ebb and passes the actual linked native
primitive: 162 cases, 6 selected/156 fallback and 74,805,120 output/state FP32
words bit-identical to original width16 Metal across three carried steps.
Check pending outputs, abort invalidation/recovery, state alias, uninitialized
producer and foreign writer. CTest and original vector primitive also pass.
At 06:14:33 UTC, nine whole stages pass, including full vision grid/real heads;
the suite is within text tail cases. This is not complete native validation.

Run all original30 afresh plus new primitive compile/execution, 32 total.
Before execution, parse/regenerate the complete driver, checkpoints25/31/32,
and original independent auditor. Preserve original bounds. A local dummy-root
AST preflight path first misses the template; correct only that local harness
path before deployment, without changing native tests/runtime. Full32 requires
all original extended/wide trained state checks and 48 actual counter steps,
12 selected/36 fallback. Archive includes every declared source byte and
explicit vector/MLP auditor dependencies from fixed original hashes.

Finalizer 98584 watches exact native check 94442 and publishes a gate only after
terminal archive and complete independent audit. Continuation 1941 snapshots
all direct shared Mac tools, then can launch sequential six-profile native
ABBA only after that gate, raw32 re-audit and a quiet interval. Both watcher
handles are verified live. Freeze all consumed source/runtime/drivers until
terminal archive, including the complete shared-tools directory while the
continuation runs. Observation limits stop/restart no job.

Fix the native screen before model timing: >=1% warm TTFT reduction at both VL
batches in two complete unchanged cohorts; every decode fraction >=.98 and
every TTFT fraction <=1.02; all six complete original outputs/routes required.
Only complete initial success admits one unchanged confirmation. Init, first,
warm and worker RSS remain separate. Comparator and outer/inner original
72-output auditor expansion/AST preflights pass, without model execution.
No native model timing exists yet. No serving source or matched-vLLM target
count changes; the full 2x decode/half-TTFT goal remains active.
[Private policy, exact preparation, observed processes/results and commands](../../results/20261016/delta-value-tiles-native/README.md).

### 2026-10-16 UTC — VPN recovery preserves exact native jobs and reaches 24 stages

Read-only SSH observations time out over the current route and the previous
verified address. Local VPN adapters remain up; the target peer briefly reports
offline. Neither a transport timeout nor peer state proves any job terminal.
Do not restart or mutate the consumed native source/tools. After the route
recovers, all three exact process handles remain live; check 94442 and finalizer
98584 birth times match their original watcher metadata. At 06:26:11 UTC,
24 whole stages pass, including text/coder tails, private KV, serving, mixed
images, layers, heads and trained vision. Full32 and native timings remain
unproven; counters/performance gates are unchanged.
[Recovery observation and stage list](../../results/20261016/delta-value-tiles-native/network-recovery-observation.json).

### 2026-10-16 UTC — streamed decoded-weight lifetime source preparation only

While the Mac native jobs/tools stay frozen, prepare a distinct Windows-only
decode hypothesis. Keep accepted four-output/64-thread geometry, original
payload/tag/raw loads and reconstruction, K/j and SIMD reduction order. Consume
each reconstructed float4 immediately in its compact/raw branch rather than
retaining four vectors until a later loop. The compiler may already perform
this lifetime shortening; no register/occupancy or speed claim follows from
the source change. It is not the prior two/eight-output or explicit-FMA method.

Generate isolated source/header/counter and draft safety/numeric harness, and
reverse the shader moves to the original text. Preserve a local generator
assertion failure: two policy loop strings plus a separately named warm loop
were mistaken for three identical strings. Fix only those occurrence/policy
bounds before any execution. Add a separate prepared numeric/timing probe with
original exact-bit/FP64/finite/comparison gates, all-model numeric gating before
timing, fixed64 threads, 2 s common warmup, own-null before comparison, 10 ABBA
rounds and all 2,400 planned samples retained. No sample has run; the exclusive
runner, compiler/GPU validation and admission screen remain future work.
Source preparation and its byte record are not numerical/performance evidence.
No Mac deployment, serving modification or target-count update occurs.
[Prepared sources, provenance, authoring failure and limitations](../../results/20261016/streamed-body-weights-preparation/README.md).

### 2026-10-16 UTC — complete native width64 prefill screen rejected after full32 success

Check 94442 exits 0 in 1,452.645279584 s with all 32 stages fresh. Finalizer 98584
archives/audits all 1,880 files and publishes the full32 gate. Preserve original
bounds, 162 linked original-bit cases, 74,805,120 output/state words and all 48
trained graph/session route steps: 12 selected/36 fallback. Worst trained-bound
ratio 0.5999873838 and wide ratio 0.5432836183 stay <=1. No reused stages.

Behind terminal archive/raw re-audit/quiet guards, complete native model cohort
22203 exits 0 in 192.920540834 s. All 72 complete outputs and actual routes pass;
archive 6,092 files with 1,410 source and 1,241 external-tool records. Continuation
1941 rejects the fixed model screen and exits 0, then is itself archived before
any next deployment. Mac/Windows full32, full72 and decisions are identical.

VL B1 TTFT fraction 0.998322 misses the .99 limit; B2 fraction 0.988069 passes.
Coder B2 TTFT fraction 1.037333 exceeds the all-profile 1.02 preservation limit.
All decode fractions stay >=.98. Reject without confirmation/promotion/favorable
retry. The changed kernel is unselected for coder, so the observed regression
does not establish causation. Prior component gains are not TTFT gains. Keep
init/first-after-init/warm and worker RSS separately, with caches unflushed.
Source, full native timings and both terminal controllers are frozen on both
hosts; accepted d52 and matched-vLLM target counts remain unchanged.
[Complete tables, hashes, independent audits and fixed rejection](../../results/20261016/delta-value-tiles-native/README.md#terminal-full32-audit-and-native-model-rejection).

### 2026-10-16 UTC — streamed weight lifetime GPU component correct but unselected

After the native controller is terminal/archived and all transfers finish,
deploy the prepared variant into a separate private tools directory. Require
the exact prior-controller archive and current d52 identity. Preserve original
coalesced control, same 64 threads, unchanged data/FP32 math and all unsupported
paths. Counters assert the new kernel only in candidate calls. Finish all-model
numeric mode before any timing; no whole-model inference occurs in this probe.

Controller 26445 exits 0 in 35.186062542 s including compile/setup/numeric/timing.
All 29 matrix fixtures (text/coder layer0, VL layers0/3), 20 selected matrices,
24 original-bit/FP64 numeric profiles, 680,832 original FP32 output values and
63 ownership/tail/fallback/prefill/abort/recovery safety profiles pass. Keep all
2,400 raw clock samples, 2 s warmup/profile, own-null before comparison, 10 ABBA
rounds, 3 warm calls and 5 samples/window. Original error bounds are unchanged.

All six own-nulls pass the fixed 5% rule. Text B1 GPU/wall fractions are
1.007683/1.000537, coder B1 1.005412/1.021977; neither meets the required >=5%
gain on both clocks. Coder B2 wall fraction 1.028127 also fails 2% preservation.
Reject without confirmation or native integration; do not select favorable B2
or VL endpoints after the fact. Actual register allocation/assembly identity is
unmeasured. The pre-execution thresholds and post-execution/pre-inspection raw
mean aggregation are identified separately; no formal preregistration claimed.

The 58-file terminal archive and all generated/native/template sources, defines,
raw slice/checkpoint identities, actual dispatch, safety, monitor and clock
records audit identically on Mac/Windows, with the same rejection. Post-run
environment confirms M4/16 GiB; compiler/Python/macOS observations are scoped
after execution. Setup/packing are outside kernel intervals; init/RSS and actual
model decode/TTFT gains remain unmeasured for this variant. No serving or target
count update. Preserve the earlier Windows-only preparation separately.
[All endpoint clocks, costs, source records and original gates](../../results/20261016/streamed-body-weights/README.md).

## 2026 10 16 Contiguous BF16 prefill blocks rejected

Hypothesis: a K-major N64/K32 BF16 copy gives coalesced tile loads, while M32/M64
FP32 SIMD matrix tiles reuse input/result shared storage and skip inactive
M64 accumulator bands. This is different from the earlier four-output packed
prefill layout. Additional packing is lossless; no checkpoint, arithmetic
precision, prompt, output cap, batch or error threshold changes.

Actual Mac PID28401 completed in72.469517seconds, exit0, using the accepted d52
context and exact defines in a private standalone build.25 Windows CPU layout
cases cover204,800 padded arbitrary BF16 words, and actual Mac weight upload
verifies every packed/padded word against its original. The generation hashes
match Windows. All29 checkpoint matrices,20selected, run B1/B2 at the original
physical prefill rows and four patterns:48profiles and75,601,920 FP32 outputs.
Six comparisons fail the original5e-6+5e-5*abs(control) limit; worst normalized
ratio1.905077547. The unchanged FP64 envelope passes, worst0.075892469, which
does not override the separate control-comparison failures.

Both complete policies were excluded before any timing, with no favorable
per-profile subset, repeat or integration. The planned all-six5% GPU/wall screen
and own-null protocol therefore supply no measured speed result. Additional
BF16 storage totals126.9375MiB across separately executed model projection sets;
upload/preparation totals487.0775ms, with no full-model or peak-RSS extrapolation.
The50-file frozen archive and all source/numeric/route/guard records reproduce
identically on Mac/Windows. No GPU ownership/abort/tail suite or vision-tower,
sequential-network or matched vLLM comparison was attempted. The goal counts
remain4/6decode,2/6TTFT,0/6both consistently.
[Methods, original failures, setup cost, commands and audits](../../results/20261016/block-packed-prefill/README.md).

## 2026 10 16 Wide output block tiles correct but slower

Follow-up to the rejected all-eligible tiles: fixed output columns>1024, keep all
other original projection routes. The hypothesis targets wide gate/up, query and
linear matrices for more contiguous output-tile work; it does not establish why
the first design failed comparison. This is an evidence-informed selection,
chosen before any new timing, with the same six workload profiles and bounds.

Actual Mac PID29834 exited0 in129.259064seconds after complete three-model numeric
mode and separate own-null/comparison timing.29matrices,12selected,48numerical
profiles,75,601,920 outputs all pass, comparison ratio0.0, original FP64 bound
unchanged. The original BF16 word/padding roundtrip is also checked at upload.
M32/M64 use the same N64/K32 packed layout, FP32 accumulation and256threads.

All4,800 timing samples retained:12 complete model/batch/policy profiles, all own
nulls within5%, GPU fractions1.453310–2.131512, wall1.228141–1.754453. Every profile
fails the preservation/gain screen, so both policies are rejected with no repeat
or integration. The pre-execution screen targets the four outstanding prefill
TTFT workloads for5% GPU/wall improvement and requires every profile within2%,
plus an unchanged full confirmation; it does not shrink the full matched goal.

Additional buffers total84.625MiB and upload/preparation427.354077ms across
separate numeric processes; no full-model or peak-RSS estimate. The63-file archive
and both independently run audits agree on all sources, original checks, clocks
and rejection. External audit code was separate from immutable consumed runtime
sources. No fresh matched goal ratio or target count is claimed. Contiguous
packing/shared scratch reuse is insufficient in this implementation; preserve
the selected MPS/BF16 route. Register scheduling and exact GPU causes unmeasured.
[All12 GPU/wall/null rows, protocol, selection provenance, costs and commands](../../results/20261016/wide-block-packed-prefill/README.md).

## 2026 10 16 Smaller rejection indexes preserve answers but fail speed gates

Mechanism: reduce the private greedy-head approximate rejection index from eight
bits to four/six while retaining original BF16/FP32 exact seed/refinement,
upward residual/original/reconstructed norms, strict bounds and unsafe-query
fallback. This is not a reduced-precision final-logit model. Fixed1,024-row seed
and original inputs/queries stay unchanged; all retained and rejected original
scores are checked before timing.

Four-bit PID32841 exits0 in144.568587seconds.51original full-query bit checks,
102policy checks,66unsafe checks and102ownership cases pass, comparing47,488,512
scores.19,200timed samples, six stable grouped own-nulls; GPU fractions1.393651–
1.600489 and wall1.340444–1.563867 all regress. Mean kept rows per request row
rise from roughly1,026–1,036 to62,644–126,723. The survivor inflation is consistent with these regressions; timings do not
isolate its contribution from unpacking and scheduling costs. Reject confirmation/integration.
[Four-bit protocol, raw score/clock arrays, costs and both-host audits](../../results/20261016/nibble-screen-index/README.md).

Six bits pack four output tags into24bits and load three32-bit words per four K
terms. It is chosen from four-bit survivor evidence before six-bit timing.
Initial PID35707 exits0, but the agent-started prior-archive SCP overlaps its
execution. Before reading speed ratios, record unconditional exclusion of this
entire timing cohort. Preserve all records; no original failed result is hidden
or reclassified from observed speed. Only after actual transfer terminal0 and
initial terminal/archive/audit, run one fresh clean initial with identical source,
math, cases, sampling and thresholds. No transfer overlaps the clean run.

Clean PID38347 exits0 in130.993883seconds. Same complete original/safety counts
and19,200samples pass data audits. Every grouped null is within5%. Coder B1 GPU
improves3.304% but wall rises0.396%; text B1 GPU improves1.184% but wall rises2.084%.
B2 text/coder and both VL profiles regress. Both required5% GPU ANDwall gains
fail; no clean confirmation or integration admitted. Six-bit payload saves25%,
but added unpacking/survivor/encoding costs are not causally separated. A later
Windows-only exhaustive24bit-word proof checks67,108,864words at four positions;
it changes no measured source or result and is not a GPU tail/model proof.

All three complete cohorts and source generators are frozen on Mac. Windows
independently reconstructs all manifest files, borrowing only byte-identical
captured arrays from the fully verified four-bit archive to avoid duplicate
large transfers. Distinguish verified transport/file bytes from full original
six-bit gzip hashes observed on Mac. An external verifier-only Windows path
normalization repair changes no GPU predicate or threshold. All original
selected libraries and matched target counts remain unchanged; no fresh vLLM
ratio, TTFT or whole-model decode gain is claimed.
[Six-bit clean decision, excluded-overlap chronology, complete clocks, storage/preparation/RSS and archive proofs](../../results/20261016/sixbit-screen-index/README.md).

## Continuing this record

Append a dated entry for each new experiment or repair. Give it an identifier,
hypothesis, exact change and control, source/library/asset identity, command and
environment record, correctness gate, sample design, numeric results with units,
setup/memory costs, failures/exclusions, decision and remaining uncertainty.
Link raw data and the verification report. Mark planned, running, terminal,
audited, repeated and rejected states explicitly; preserve old entries.

Regenerate the report index with:

```text
python -X utf8 docs/research/build_metal_method_index.py
```

Future paper claims should cite a specific frozen cohort and code revision.
Component speedups cannot be multiplied together to predict end-to-end gains.
Candidate selection across many trials creates selection bias: use fresh
confirmation and a fresh comparator cohort after choosing the implementation.
External algorithm provenance, related work, formal statistical analysis and
publication figures still need their own review before submission.

Documentation checks can be repeated without launching inference:

```text
python -X utf8 docs/research/build_metal_method_index.py
python -X utf8 docs/research/verify_metal_research_log.py
```

The second command checks report hashes, curated local citations, verifier-file
identities and the displayed matched/component table values. It does not
replace the separate numerical or benchmark audits. The index captures the
base commit plus exact report bytes; uncommitted reports must not be represented
as if they were already present in that base commit.

## 2026 10 16 M5 4B low-copy chunk and cache profile

A separate MacBook M5/24 GiB serving smoke experiment applies six existing
chunk-recurrence/cache flags to Qwen3.5-4B. The current WorldSense build had all
optional kernel flags off. Preserve precision, pixels, checkpoint and prompts;
freeze consumed source and run control/candidate/candidate/control sequentially
with WorldSense paused and restored. Two cohorts retain eight complete matching
4B token outputs per runtime; a separate three-prompt 1.7B guard also matches.

Warm median total11.210→8.659s and11.587→8.664s; TTFT4.816→2.470s
and4.649→2.410s. First requests/init and process memory are separate. Matmul
reference passes; this does not substitute for original trained numerical gates.
Opt-in profile only; no default/CUDA change, quantization, general speed claim
or fresh M4/vLLM ratio. The handover's full six-profile goal remains unmet.
[Selection, full token outputs, all clocks, memory, source/library identities and limitations](../../results/20261016/4b-m5-chunk/README.md).

## 2026 10 16 Further M5 4B vision sharing and body specialization

First execute fixed-grid immutable-weight sharing against the already optimized
six-flag4B control. All eight outputs/policy match and total/TTFT improve, but
repeat decode11.5320→11.2561tok/s fails the predeclared2% limit. Reject without
an unchanged favorable retry. Preserve raw runs and selection.

A new candidate adds existing bounded body-only fixed-shape kernels to address
that preservation failure. Two fresh ABBA cohorts retain all complete tokens,
actual248 specialized last-decode projections and zero misses. Warm total
11.340→9.087s and10.781→8.915s; TTFT4.279→2.455s and3.605→2.233s;
decode11.389→12.154 and11.287→12.124tok/s. Initialization/first and memory
remain separate; no vLLM ratio or full numerical equivalence is claimed.

642,560 actual vision values match bits across three grids and return; source
close, failed replacement recovery and changed-input checks pass. After one
separate16-token car request at512capacity, all32 decoder-state layers match
bits; three1.7B text outputs match. Native matmul and model-contract tests pass.
Retain opt-in BODY_SHAPE plus4B Metal sharing. CUDA/default flags unchanged.
[Both candidates, original gates, full outputs, source/library hashes, state/vision checks, memory and deployment](../../results/20261016/4b-m5-vision-shape/README.md).

## 2026 10 16 Native image input through XLang3

Replace the Mac bring-up image subprocess with in-process C++ JPEG/PNG decode,
EXIF, fixed-point bicubic resize, FP32 normalization and patch packing. XLang3
executes all tests; Pillow/NumPy are only the offline/reference control. Codec
versions, geometry, inputs and reference hashes are retained. Three real frames
have1,173,504 exact patch values;21 edge fixtures include trailing PNG EXIF,
CMYK, alpha and16-bit gray. Invalid/truncated input and zero std are rejected.

Five alternating component pairs per image measure old helper275–285ms versus
native2–5ms. Do not apply54–130x component fractions to full VLM calls.
Two fresh ABBA cohorts retain eight complete outputs per policy. Warm TTFT
2.369→2.108s and2.307→2.093s; total9.450→9.457s and9.358→9.201s.
Decode11.550→10.989 and11.564→11.357tok/s: not a decoder speed gain.
Predeclared port gate is exact inputs/outputs and total/decode preservation
within5%; it is distinct from earlier pure-kernel improvement screens.

All32-layer state bits on a separate request and three1.7B text outputs match.
Final main-source native tests pass21 fixtures and serving path/bytes checks
with helper env absent. WorldSense stores a native-cpp4B observation after
deployment, with four files online. Initial formats are JPEG/PNG; no fallback
Python process. CUDA codec dependencies stay excluded via an unsupported stub;
no Windows build is claimed. Original flags/chunk16/precision/portal preserved.
The earlier rejected chunk32 attempt is retained in this evidence; no partial
validation is promoted. [Raw calls, component clocks, source/binary identities,
format/state checks, rejected attempt and deployment limitations](../../results/20261016/native-image-input/README.md).

## 2026 10 16 WorldSense-driven M5 4B optimization (running)

User requires future optimization performance decisions to use WorldSense's
Galaxy pipeline, with XLang3 orchestration and video-file inputs. Research
records are intended for paper preparation. Preserve unsuccessful trials;
do not reuse historical M4/smaller-model gains as M5/4B serving evidence.

Experiment WS-M5-01 is planned/running, not a selected optimization. Initial
600-second live-video capture uses WorldSense main 0ba2a57, Garnet
feature/qwen35-metal 6c4a42393, existing four mounted video sources and their
unchanged assignments. This observational cohort measures normal system
behavior, not controlled replay: loop phase, output length and sibling Skills
vary. Successful persisted calls exclude retries, failures, queues and loading.

Before candidate timing, add a fixed recorded-frame benchmark exposed through
WorldSense and executed by the same Galaxy Garnet-VLM filter. A current-frame
Skill evaluation must not commit WorldState or emit production events. Keep
ordinary live evidence as a separate integration workload. Fix source image
bytes, prompts, model weights, pixels, output limits and scheduling. Preserve
full outputs, stage timings, first/cold calls and setup separately. No competing
GPU benchmark may run during a timed window. Record source/library hashes and
machine/memory state. Future confirmatory runs are fresh, after selection.

First candidate hypothesis: existing opt-in fused greedy vocabulary projection
and argmax can reduce decode bandwidth/submission costs on 4B. Keep original
BF16 values and FP32 arithmetic; test original output/state/error behavior.
Initial screen requires all recorded-frame outputs/decisions to match, at
least 5% median decode improvement, and no median TTFT or total regression
over 2%. A screen pass is not publication confirmation. Candidate is not
deployed or measured yet at protocol declaration. A failed screen remains
archived without an unchanged favorable rerun. Instrumentation is identical
in baseline/candidate.

WS-M5-01 is now terminal/rejected. Four fresh WorldSense HTTP suites retain
72 complete matching model-call outputs. Actual fused-head counters change
only in the candidate. The first 4B comparison improves decode 15.15% and total
12.97%, but the second has 0.62% lower decode, 3.05% higher total and 2.42%
higher TTFT. The second text route also exceeds the total/TTFT preservation
limit. Do not discard control drift or promote the first favorable pair.
Original library/profile restored; no fresh favorable retry admitted.
Instrumentation smoke failures and repairs remain in the archive.
[Protocol, complete calls, stage metrics, memory observations and failed gate](../../results/20261016/worldsense-m5-fused-head/README.md).

WS-M5-02 planned before new candidate timing: exact-half/common-alpha prefill
on top of the original deployed seven-flag control, with fused-head OFF.
Existing admission preserves original BF16 decode buffers and only encodes
eligible matrices ≤16 MiB; every value must be exactly representable under
the admitted power-of-two scale. This trades additional retained half buffers
and setup work for faster mixed FP32/FP16 MPS prefill. It does not quantize
learned values or promise unchanged MPS accumulation results. Require full
output parity, ≥10% median 4B TTFT improvement and ≥5% median 4B total
improvement in both fresh WorldSense comparisons; preserve 4B/text decode and
text TTFT/total within 2%. Retain raw memory/swap and cold/warm records. Reject
on gate failure without an unchanged favorable retry. No result existed at
protocol declaration.

WS-M5-02 is terminal/rejected. Four fresh WorldSense suites retain all72
matching complete model-call outputs and actual nonzero common-alpha counters.
The first4B pair has14.12% higher total,26.29% higher TTFT and7.68% lower decode;
the second has1.05% higher total, only0.66% lower TTFT and0.29% lower decode.
Text preservation also fails in the first pair. Original library restored;
do not discard the control drift. Header-size audit finds only1.285% of4B
decoder layer-matrix bytes can pass the existing16MiB admission bound before
exactness checks. This motivates a distinct storage design, not a reinterpretation
of this failed screen. [All calls, size audit, actual counters, gates and tradeoffs](../../results/20261016/worldsense-m5-half-prefill/README.md).

WS-M5-03 planned/prototype, not measured: share one immutable exact scaled-half
weight allocation between MPS prefill and strict FP32 decode. Raise admission
to64MiB without retaining a second full BF16 body allocation; retain original
fallback when exact representation is unavailable. Integer reconstruction
avoids hardware half-subnormal flushing; dot/reduction order and bounded body
shape specialization are retained. Costs are integer decode instructions,
startup packing and added implementation complexity. No quantization, CUDA or
default-policy change is proposed. Source is isolated under the local work
tree. Require exhaustive GPU finite-half inverse bits, native decode bits/FP64
envelopes, one-buffer accounting, context isolation and abort recovery before
WorldSense timing. Screen gates declared before compilation: exact full outputs,
≥20%4B TTFT/≥10%total gain, ≤5%decode cost on either model and ≤5%text total/TTFT
regression, then fresh confirmation. This explicitly accepts a small decoder
tradeoff for large prefill gains; it does not relax the two rejected screens.
The actual M5 MPSMatrix BF16 capability probe succeeds for FP32 control and
aborts all three BF16 combinations, matching the earlier API limitation;
header enum availability alone is not runtime support. This is capability
evidence, never an inference-speed benchmark. The first isolated compile missed
the repository nlohmann header; its failed log is preserved before copying
the unchanged dependency and rebuilding. Native validation was pending at
that declaration.

WS-M5-03 is terminal/rejected. Exhaustive GPU inverse8,189,952cases and24
native decode/FP64/ownership/recovery cases pass. Actual checkpoint audit
admits247/248matrices,99.706%of body-matrix bytes, with4.309MiB logical retained
overhead rather than a second full weight copy. One32-layer/16-token request
has equal output and finite states but nonidentical state bits (max abs0.0001295;
pointwise diagnostic is not an original deep numerical-suite pass).
The first Documents-root service attempt blocks on an import filesystem open,
with no candidate model samples; stack, attempted control and logs are retained.
After identical-source packaging under CantorAI and fresh matched builds,
all72WorldSense model-call outputs match. TTFT improves11.09%/33.13%, but4B
decode falls27.33%/20.82% and total rises22.87%/6.81%. Reject and restore the
original profile. [Exact storage, source/root repair, raw calls, numerical caveats and failed gates](../../results/20261016/worldsense-m5-shared-half/README.md).

WS-M5-04 prepared/native-validated, timing running: coalesce original BF16
global reads along K and transpose into65-column padded threadgroup storage
for the existing16×64×64/32×64×32FP32 MMA kernels. Arithmetic/reduction order,
original weights, chunk16 and decode remain unchanged. Extra shared bytes are
256/128 per wide tile. Small-row/output/device fallback remains original.
Initial source extraction duplicated an adjacent Argmax declaration; preserve
the shader-assembly failure before correcting the extraction, with no numeric
or performance samples yet. Native48-case full FP32-bit/FP64/tail/route/recovery
tests and a full4B32-layer/16-token state-bit comparison pass.

Before any WS-M5-04 timing, replace coarse process-level attribution with a
same-binary dispatch ablation inside WorldSense. Prior controls drifted heavily.
Warm both policies; worker1 uses ABBA windows and a fresh prospective worker2
uses mirrored BAAB windows, each with the same three cases/query per window
and two measured repeats per policy. Actual counters require0 control and
positive candidate prefill use, and zero decode use. Native RAII restores the
live policy even after inference exceptions; XLang3 schedule/failure tests pass.
No direct-model timing replaces WorldSense evidence. Both fresh workers must
meet declared exact-output/state/route gates, ≥10%4B TTFT/≥5%total gain, ≤2%
decode regression on both models and ≤2%text total/TTFT regression. No result
or promotion is claimed yet; all roots and warmups are retained.


WS-M5-04 terminal/rejected: all72 outputs and actual routes match; all32-layer state bits pass. Fresh ABBA/BAAB workers increase4B TTFT16.36%/13.70% and total5.64%/4.63%, failing prospective gates. GPU prefill dominates the observed regression, while CPU encoding is approximately10.5ms. No promotion; original library restored. [Raw calls, exact analysis, failures, source and limitations](../../results/20261016/worldsense-m5-coalesced-prefill/README.md).

WS-M5-05 prospective distinct screen: remove65-column shared padding while retaining coalesced reads and original64-column alignment. Native reference tests and32-layer state-bit equivalence pass. Fixed car frame/current-event Skill only, both modes warmed, four measured samples per mode in ABBA blocks. Gates declared before timing:10%4B TTFT/3%total improvement, decode and text preservation within2%, exact outputs and actual route counters. This smaller component screen cannot promote without a fresh full three-frame/search confirmation. No result claimed at declaration.


## Paper evidence policy (2026-10-16)

The user intends these records for paper writing. Freeze each completed experiment with its prospective hypothesis, inclusion/exclusion rules, exact code/library/model/input hashes, machine/toolchain identity, execution order and warmup policy. Preserve every raw sample, failed attempt, correctness check and keep/reject decision. Report the sample unit (model call, frame/Skill invocation or fresh process); repeated calls and copied camera files are not independent devices or independent scenes.

Keep model total, TTFT, generated-token count, decode rate and native Skill wall separate. Record units and rate formulas. Archived-frame Skill timing excludes video decoding, scheduling, state/event persistence and retrieval/ranking, so it cannot support an end-to-end video-event latency claim. Live observational loops are not synchronized replay controls. Cold startup, warm measurements and compilation must remain separately labeled.

Record memory metric provenance; process RSS, system swap and logical weight bytes cannot establish peak Metal/GPU memory. Include power/thermal observations where available, without inventing missing telemetry or causal hardware explanations. Current small paired screens are descriptive engineering evidence: medians and percentage changes are not statistical significance, confidence intervals or broad hardware generalization. A paper-level inference needs an explicitly designed replication set; do not treat earlier exploratory screens as preregistered confirmatory trials.

Rejected candidates stay in the evidence index. Any later change to thresholds, workload or implementation becomes a separately labeled experiment before timing. Promote only after the stated correctness and fresh WorldSense confirmation gates pass. Freeze archive SHA-256 manifests after final README edits, preserve immutable raw data, and cite the containing Git commit plus archive identity. Secrets and authentication tokens must never enter publication artifacts.


WS-M5-05 terminal/rejected:20 model outputs/input hashes match, native routes validate and32-layer state bits pass. Four measured samples per policy/model:4B TTFT+35.47%,total+16.55%,decode−2.95%; text TTFT+12.14%,total+6.60%. Failed prospective screen, so no full confirmation or promotion. Original library/live assignments restored. [Raw calls, source, tests and limitations](../../results/20261016/worldsense-m5-coalesced-nopad/README.md).


WS-M5-06 terminal/rejected: existing smaller16×32×32BF16 tile for physical16-row prefill; reduced shared storage but more output tiles/input rereads. Prospective single-car paired WorldSense screen, both modes warm/twoABBA blocks/four measured calls per policy/model. Native reference and32-layer state bits pass, all20tokens/text/inputs match and routes validate.4B TTFT−1.42%,total−0.10%,decode−0.76%; text TTFT−4.12%,total−2.44%. Fails declared10%TTFT/3%total screen. No full confirmation or promotion; restore original library/live assignments. [Protocol, failures, raw samples, exact analysis and source](../../results/20261016/worldsense-m5-small-prefill/README.md).

Next bottleneck hypothesis, not yet implemented or measured: larger64-token prefill could reduce repeated weight traffic. Current4B packed-width validation admits only16 tokens; raising it requires native shape/memory guards and complete state/output checks before WorldSense measurement. Earlier32-token trial was rejected and must not be reinterpreted as success.


WorldSense per-call CSV exports retain every eligible root-cohort model record, including warmups, with raw artifact SHA-256 and phase/policy/repetition. Nested failed attempts are explicitly outside this table and remain in their archived directories. Decode rate is `(generated_tokens−1)/(total_seconds−ttft_seconds)`; generated token IDs exclude stop tokens, while elapsed decode may include the final stop prediction and host loop overhead. It is a serving-phase rate, not kernel-only throughput. Never pool rows as independent scenes or hide these metric boundaries.

WS-M5-07 prospective/prepared:64-token4B prefill with opt-in packed-width bound64×2560, unchanged chunk limit64 and original BF16/FP32 kernels. Native reference and one full32-layer/token state comparison pass; text reasoner stays16. Both chunk-keyed graph sets remain cached with shared weights for paired warming. Larger activation/cache memory and padding are costs. Before timing, declare same single-car WorldSense ABBA screen/four measured calls per policy/model, exact tokens/states and actual chunk-plan checks,10%4B TTFT/3%total gains, decode/text preservation2%. Fresh full three-frame/search mirrored confirmation required before promotion. No timing result existed at declaration.


WS-M5-07 terminal/rejected: all20 complete outputs and inputs match and actual chunk plans validate, with unchanged decode source.4B TTFT−34.34%,total−12.59%,decode−0.83%; reasoner TTFT+18.81%,total+9.96%,decode−1.10%. Native Skill wall13.7442→12.4779s. Reject because text preservation fails despite useful VLM component gain; no unchanged favorable rerun or causal claim. Original profile restored. [Raw data and scope](../../results/20261016/worldsense-m5-chunk64/README.md).

WS-M5-08 prospective distinct method: apply64-token prefill to BOTH4B VLM and1.7B reasoner against16/16 control, otherwise same BF16/FP32 kernels and budgets. Both chunk-keyed graph sets stay warm/shared-weight; larger activation/cache footprint is a cost. Native reference, all32VLM and28reasoner state/token bit checks pass before timing. Both model checkpoints/source/build/environment fingerprinted. Fixed-car WorldSense ABBA screen/four measured calls per policy/model, exact golden outputs and actual chunk plans required;≥10%4B TTFT/≥3%total gains,≤2%decode regression and≤2%text TTFT/total regression. Fresh full three-frame/search mirrored workers required before promotion. No result exists at declaration.

Evidence packaging correction: WS-M5-04/05 originally omitted native backend/CMake/test snapshots due to an incorrect archive path. Added retained isolated sources; raw results unchanged. Compatible reconstruction-base commits and source overlay patches are now explicit, distinguished from original execution HEAD. Patch applicability checks pass; they are reconstruction checks, not fresh benchmark or numerical results.


WS-M5-08 terminal/rejected AFTER screen pass. Full fresh ABBA/BAAB workers retain72matching outputs/inputs/actual plans, plus20screen calls.4B TTFT−26.17%/−23.42%,total−7.94%/−12.20%; reasoner TTFT+10.33%/+39.28%,total+15.77%/+23.00%. Native Skill14.9255→14.2447s and16.2493→16.2858s. Both text gates fail; second Skill wall fails improvement. No promotion despite favorable car screen. All raw rows/warmups and post-hoc workload diagnosis preserved; original profile restored. [Full terminal evidence](../../results/20261016/worldsense-m5-dual-chunk64/README.md).

WS-M5-09 prepared only, not built/validated/timed:4B prefill64 with dense1.7B reasoner32 versus16/16, same native kernels/budgets and preserved source. Native shape/state correctness and unchanged WorldSense10%TTFT/3%total/decode-text2%gates remain required before promotion, with fresh full mirrored confirmation. Record direct prompt SHA-256 and effective non-chunk budgets identically in both policies as additional invariants. This explores model-specific batch tradeoffs after WS08 failed broad workloads; no causal explanation is asserted and no GPU work overlapped its live confirmation.


WS-M5-09 terminal/rejected. Native matrix reference plus all32VLM/28reasoner state/token bits, finite-state assertions and capacity rejection/recovery pass before timing. All20WorldSense records match golden outputs/image-Skill hashes/token counts; new caller-prompt SHA and effective non-chunk-budget checks also pass.4B TTFT−39.47%,total−15.54%,decode−0.16%; reasoner TTFT−3.93%,total+0.50%,decode−3.42%. Native Skill13.8908→12.3632s. Reject because declared2%decode preservation fails. No full confirmation or unchanged favorable retry. This small screen does not establish a causal decode-kernel regression; the prospective engineering decision still stands. Original library/live assignments restored. [Raw calls, executed finite/recovery checks, source overlay and decision](../../results/20261016/worldsense-m5-mixed-chunk/README.md).

### Next mechanism hypothesis: native dense text-prefix state reuse (not implemented)

Repeated current-Skill verification and search planning re-prefill fixed instruction text on every call. Explore caching only the exact transformer KV state for an identical initial64-token prefix of the1.7B text model, with16-token prefill so the saved boundary preserves the existing arithmetic grouping. Keep at most two entries for event/search instruction families. Every image perception and answer remains newly computed; no observation, event, WorldState or answer cache is proposed. VLM64 remains a separate batching choice, so combined-policy gains must be labeled as such rather than attributed to one component without ablation.

Implement data movement in native C++: bounded compact per-head prefix arrays copied from completed shared Metal KV buffers, restored after an explicit fresh reset with the remaining cache slots zero. Unified-memory native memcpy is a hypothesis to measure, not a claimed zero-copy/GPU-resident cache. XLang3 chooses a key from exact token IDs/positions within an immutable model instance and updates logical slots after native restore. No CPython process or Python image helper enters serving. Capture/copy/reset overhead and bounded retained bytes must be reported, including cold misses and warm hits.

Require native shape/layout admission for dense attention only, context/session ownership, fresh-reset validation, graph/checkpoint-generation invalidation, bounded entry/byte limits, eviction and unknown/stale-handle rejection before mutation. Close/rebuild clears entries. Avoid deferred-zero/dirty-reset combinations until separately proved. Cache metadata never accepts arbitrary cross-model state. Restore exact prefix bytes, then compute the suffix using original positions/masks; an identical-token prefix must not be reused across images or recurrent VLM layers.

Before any performance claim, prove complete states/tokens against uncached inference across changing suffixes, alternating event/search prefixes, cold misses/hits, eviction, reset, overflow/failure recovery, stale graphs and cross-session misuse. Then preregister WorldSense fixed-frame/query gates and distinguish warm-cache gains from cold/startup costs. This is a mechanism design only: no source change, build, execution, benchmark result or promotion exists yet.


### WS-M5-10 — bounded native text prefix snapshots (provisional screen)

Combined candidate: Qwen3.5-4B prefill64 and Qwen3-1.7B prefill16 with an exact64-token native KV prefix; control16/16 without prefix reuse. Two session-private compact CPU snapshot entries,64MiB logical float-payload cap, bounded metadata; fresh full reset before restore. Images, suffixes and answers are freshly inferred. This intentionally differs from the older M4 comparator cache-disabled protocol and is a separate WorldSense experiment, not a vLLM ratio. Cold capture/finite scanning, reset/restore copies, retained memory and key overhead are explicit costs.

Native row/group copy tests,32 text-session safety checks, changing-suffix all28-layer state/token checks and 4B all32-layer state/token checks passed. Initial harness failures (different short-prefix IDs and unsupported reversed dict_values) are retained, not counted as model failures. Model asset bytes/hashes revalidated. No external CPython image processing or CUDA source changes.

Fixed-car paired ABBA WorldSense screen returned20 model calls including warmups; exact golden outputs/input identities, actual prompt hashes/non-policy budgets and native prefix counters passed. Median changes:4B TTFT−36.77%,total−13.38%,decode−0.50%;1.7B TTFT−19.44%,total−10.69%,decode+0.13%. Skill native wall14.1701→12.3493s. Screen passes the prospective gates; two fresh broader workers remain required. No promotion. Raw source, protocol, correctness results and measurements currently in local work/worldsense-prefix-cache; confirmation runs tracked separately.

WS-M5-10 confirmation complete: both fresh full workers separately pass all declared gates. Worker1/2 4B TTFT−39.97/−35.00%,total−10.84/−12.56%,decode+2.96/+1.10%;reasoner TTFT−23.37/−5.38%,total−18.56/+0.39%,decode+7.83/−0.38%. Skill wall improves in both.92 returned model calls across screen and confirmation, all warmups preserved. Archive results/20261016/worldsense-m5-prefix-cache includes72 confirmation calls, independent CSV, native checks and reconstructable source overlay against bfb570205; git apply --check passes. Initial packaging encountered derived binary __pycache__ files; excluded from text overlay, source hash manifest remains. Candidate passes experiment gates, but no promotion before default-OFF compatibility and deployment checks. Original library restored; four file sources online, camera unmounted, valid benchmark request404 after normal restart.

WS-M5-10 integrated and deployed: opt-in MetalWorldSense4BPrefix.cmake, default-OFF APIs/profile rejection and integrated native build/reference checks pass. All eight integrated overlay files equal confirmed frozen source. CUDA sources/default backend selection untouched; CUDA runtime tests unavailable here. WorldSense macOS profiles enable VLM64/text16+cache. Exact tested binary deployed; non-paired normal-profile smoke18 model calls preserves all Skill/search tokens/text, actual cache hits and routes. Live service four videos online, camera unmounted, benchmark disabled404. Slow433.698s smoke retained with11.57–65.58s VLM totals and memory observations; not substituted for matched speed evidence. Initial OFF build command lacked XMODEL option, retained/corrected. Detailed integration artifacts under results/20261016/worldsense-m5-prefix-cache/integration. Next bottleneck is native GPU decode (example9.136s GPU/9.227s decode), not host image preparation.


### WS-M5-11 — shape-restricted eight-SIMD-group body decode (prospective)

Hypothesis:128→256threads/16→32outputs per group for row1 body projections with inner2560/9216, output>256<100000, retaining original BF16 weights/FP32 per-lane accumulation and SIMD reductions. VLM64 and reasoner16+prefix remain enabled in BOTH policies. Same dynamic group-width shader/binary between policies; native route counters required. Tail-checked outputs or columns divisible32 only; columns congruent16 mod32 retain128 fallback because existing function specialization removes tail guards for16-divisible outputs. Added272/784 guards before numerical execution. Distinct M5 full WorldSense experiment, informed by prior M4 short smaller-model threadgroup rejection; no reuse of its speed claims.

Initial native execution failed Metal declaration validation (uint3 thread-count alongside scalar grid argument); no model/timing ran. Scalar argument correction rebuilt. Initial service cleanup bootstrap raced asynchronous unload, then validated service restored; controller now waits for unloaded state. Logs retained. Repaired native tests pass72 exact-bit/FP64/tail/fallback policy cases, prefix copy and matmul references. Full32-layer/token parity versus previous confirmed source and between modes, plus reasoner prefix checks, running before any speed sample. Screen gate declared before timing:≥5%4B decode and≥3%total gain, TTFT≤2%regression, text decode/TTFT/total preservation≤2%, exact full outputs/prompt/input/config identities and actual native routes. Fixed-car twoABBA blocks; only on pass require fresh full three-frame/query ABBA and BAAB workers each passing, no pooled rescue. No result or promotion at declaration.

WS-M5-11 terminal/rejected:20 complete WorldSense model calls, all golden outputs/input/Skill/prompt/config identities and actual decode/prefix counters pass. Frozen source/library and exactABBA schedule validated.4B decode−3.61%,TTFT+29.06%,total+22.56%;reasoner decode+4.22%,TTFT−24.15%,total−23.25%. Native Skill wall57.1627→64.5267s. Fails original screen; no full confirmation/promotion/unchanged retry. Swap4680.44→10573.31MiB and shifts in unchanged prefill/vision/text make this unsuitable as a pure kernel causal ratio; failure retained, no retroactive exclusion. Archive results/20261016/worldsense-m5-decode256 includes failures, native72policy tests, all32VLM/all28reasoner state proofs,20-call CSV and reconstructable source overlay. Earlier copied guard stdout says unchanged decode path, but grouping changes: only state/token equality is claimed. Original validatedWS10 library restored, four file sources online, camera unmounted, valid benchmark request404. Next: actual native Metal allocation/working-set diagnostics through WorldSense; resource pressure is a hypothesis requiring measurement.


### WS-M5-12 — completed native WorldSense memory diagnostic

UnchangedWS10 policy/math with passive Metal/task snapshots through realWorldSense:12 full calls across3 fixed frames/Skills+query,2windows,48request-phase snapshots plus model-init samples. All tokens/text/prompt/config/input counts/image-Skill hashes match prior confirmation. Sampled Metal maximum21914648576bytes(20.409GiB), device recommended19069665280bytes(17.760GiB); task footprint max18104565000bytes, compressed max14742028288bytes. Not peaks, metrics overlap, not freeRAM/per-model attribution. Window-end Metal21907488768→21914517504bytes plateau: no demonstrated unbounded request leak. Decoder-ready13.518GiB; vision15.432; VLMprefill17.028;reasoner prefill19.100;car tail16graph20.403GiB. Fixed inputs369/368tokens use64-only chunks;car393uses64+16. Resource pressure is measured; cause of every latency change is not established.

Native lifecycle test:4MiB allocation survives C++ lexical-reference scope, returns exactly to737280-byte baseline on explicit ObjC pool drain. Initial immediate-release assumption failed; retained test/log, revised test distinguishes lexical/pool lifetimes and preservesalias/release assertions.3native reference tests andXLang3 memoryJSON contract pass. Controlled autorelease retention does not by itself prove aWorldSense leak. Idle/loaded vmmap read-onlydiagnostics retained; no speed comparison/instrumentation promotion. Source/binary/model/SDK identities,12callCSV,48snapshotCSV,failures and sourceoverlay archived in results/20261016/worldsense-m5-memory. ValidatedWS10 restored;4files online,cameraunmounted,validbenchmark404. Next test scoped native request autorelease lifetime with fresh separateworkers, because mixed pooled/unpooled calls retain control resources acrosspolicy switches.


### WS-M5-13 — native serving-boundary autorelease scopes (prospective)

Hypothesis fromWS12: scoped native ServeModel/InferJson autoreleases may release retired resources, while strongly held graph/session/command owners survive. No modelmath/precision/preprocessing/prompt/profile orCUDA change. Candidate default-OFF CMake flag; control/candidate share instrumented source and differ only in scope flag. C++ exceptions captured inside lexical pool, normal pool drain, then original exception rethrown. Scope entry/drain cost remains; cleanup cannot release strongly held cached graph resources. No assumed memory gain.

Native ON/OFF tests pass normal values, livealiases, exceptioncleanup, nestedscopes and GPUoutput/pending-abort recovery plus memory/prefix/matmul references. Initial serving guard used16text output tokens against8-token reference; first8 matched, fixture corrected and raw attempt retained. Both native serving API guards then pass full32VLM/28reasoner state/token bits, cold/warm prefix and capacity recovery. Beforetiming a dedicated overlap test demonstrates global invocation counters conflate threads; fixed to thread_local caller counts and final native/model guards rerunning. No speed samples at declaration.

Fresh worker screen:control then candidate, onecar frame/currentSkill,1warm+4measured requests perworker; nevermix pooled/unpooled calls in aprocess. Preservation:bothmodels TTFT/total/decode≤2%regression;4B total≥3%gain;nativeSkillwall≥5%gain;sampled maximum post-request Metal allocation≥10%reduction; exact outputs/states/prompts/config/inputhashes/nativecounter evidence. Modeltotal excludesouter cleanup, nativeSkill/APIwall includesit (APIwall beforefinalJSONdump). Firstcalls/setupseparate, sampled memory notpeaks. Onlyifscreenpasses require two fresh full three-frame/query control/candidate and mirrored candidate/control pairs independentlypassing; no pooledrescue/unchangedfavorableretry. Fullprotocol in local work/worldsense-request-pool/protocol.json.

WS-M5-13 terminal/rejected: final ON/OFF native8test runs and realAPI full32VLM/28text state/token/cold-warm prefix/capacity-recovery guards pass; actual0/1per-thread scope counters validate.20complete model calls in fresh control/candidate car screen, all golden outputs/prompt/config/input/Skill/image identities equal.4B total−15.15%,TTFT−18.32%,decode−1.88%;reasoner total−23.17%,TTFT−24.60%,decode+12.52%;nativeSkill wall65.4823→51.9269s(−20.70%). Both sampled post-requestMetal maxima exactly21903376384bytes:0%reduction fails original≥10%memory gate. No fullconfirmation/promotion/unchangedfavorableretry; single favorable latency comparison not a confirmed causal gain. Scope unit retention mechanism is not the observed model allocation bottleneck. Archive results/20261016/worldsense-m5-request-pool includes sourceoverlay,final and preliminary native/state tests,fixture/counter failures,20-call resourceCSV,allfirst/warm calls and exactidentities. Old failed concurrency executable hash unavailable, reconstructablesource preserved. ValidatedWS10 library/normalservice restored,4filesonline,cameraunmounted,benchmark404. Next inventory unique graph/session buffers, including decoder/prefill/head/vision ownership and inactive packed-output storage, before memory optimization; preserve fulladmission.


### WS-M5-14 — completed native graph/session ownership inventory

Read-only native MTLBuffer identity dedup permodel, allocatedSize/length and graphslot/held-weight/session references, shape/input/output/weight/legacy/resident-mask/writer roles. No addresses/tensorvalues emitted.4native tests andXLang3 schema/invalid/closed/recovery checks pass. Initial inline graph exceeds practical IPC serialization budget and returnsstream-writefailure afternativeinference; source/attempt/error retained. FullJSON sidecars+compactSHA/path/size refs fix presentation without transport/maths change. CorrectedWorldSense12calls alltokens/text/prompt/config/input/image/Skillhashes matchWS10;12captureSHAs/lengths/schema/dedup/accounting verify. Diagnostic notperformancecomparison.

Largest4B unique14345887664bytes(13.361GiB),4543buffers/7186references; weights9.078GiB,graphinputs0.551,outputparts0.549,othergraph2.303,nonresidentwrittentemps0.705,session0.175. Text unique7556751792bytes(7.038GiB),2163buffers/3201references;weights3.205,inputs0.879,outputparts0.879,other0.537,nonresidentwrittentemps1.101,session0.438. Categories priority/overlap conventions; refs notARCcounts, collectionIDsnotcrossmodelunion. Scratch/driver maybemissing; noexactglobalcoverage/peak/safe-freeclaim.

Specific defaultstateinputslot1/2+inactive terminallegacyroot targets3036086272bytes(~2.828GiB) acrosslargestmodelcollections. NativeBuildInternal allocatessymbolsininputorder, verifyingindexmapping. Ordinarygraph.Run requiresdefaultstorage, so correctnextstepis lazyallocation/re-materialization, notblinddeletion. Needmixedgeneric/resident calls, two sessions, empty-placeholder layout/alias checks, closure/rebuild, failure recovery, fullstate/token parity, fulladmission andactualresource/WorldSensecontrols beforepromotion. Boundedbuffer-propertyautoreleasesmaybenecessarywhenslotsareactuallyretired; request-pool-onlymethod remainsrejected.

Archive results/20261016/worldsense-m5-inventory contains12nativecaptures,buffer/ref/captureCSVs,fullparity/snapshot audits,initialfailureandruntime,SDK/build/model/sourceidentities,sourceoverlay. Archive-relativecaptureaudits rerun successfully. ValidatedWS10 normalservice restored,4filesonline,cameraunmounted,validbenchmark404.


### WS-M5-15 — lazy resident default storage (terminal, rejected)

Default-OFF Metal prototype retires only singleton standard-layout default layer inputs1/2 and inactive terminal packed concat roots; excludes weights/aliases/live output parts, preserves metadata/identical-slot alias proofs and full2048 admission. Ordinary graph.Run re-materializes before GPU encoding. Native getter pools bound retired resource references. CUDA sources byte-identical toWS14; native C++ image preparation preserved.

OFF/ON four native references and real32VLM/28text state/token/prefix/capacity-recovery guards pass. Batch1/capacity8 and batch2/capacity64 trained-layer alternation/late-second-session/invalid/closed-rebuilt checks pass. Separate fault-only source/build forces second allocation failure after first succeeds, preserves host inputs/private sessions and recovers exact output on next call. Timed binaries lack fault hooks. Initial missing backend/XModel flags and cached compiler-detection/header-order configuration failures retained; clean explicit configuration succeeds. Verification keyword failure retained/corrected without rerunning timing.

Fresh separate WorldSense control/candidate car Skill screen,1warm+4measured each,20model calls: complete tokens/text/prompt/config/input/image/Skill hashes, interpreted Skill output and native retire/regrow counters pass. Model assets rehashed; common frozen source differs only lazy flag. Sampled post-decodeMetal21903376384→18867290112bytes, −3036086272bytes(2.828GiB,−13.8613%), exactly prior inventory target. Warm Skillp50 44.3147574→32.9034618s(−25.7506%).4B TTFT−32.9215%,total−22.6460%,decode+5.0262%;textTTFT−36.7439%,total−31.7204%,decode−3.0927%. Original≤2%decode-regression gate fails: REJECT; no fullconfirmation, promotion or unchanged favorable retry. Prepared full-case/audit scripts are explicitly unexecuted. No causal attribution of text regression; allocation snapshots not peaks/free RAM and timing not a confirmed general gain.

Archive results/20261016/worldsense-m5-lazy-defaults includes20-callCSV, every first/warm/setup/failure, exact identities, full-source closure plus compatible-base overlay, separate fault delta and numeric fixtures. WS10 validated binary restored; normal4file service online, camera unmounted, valid benchmark404. Next distinct design: retain actual retirement but use graph construction/ordinary-run/resident-run lifetime pools instead of frequent getter pools; test exceptions/owners and exact state before fresh WorldSense controls. Not implemented or causally established byWS15.


### WS-M5-16 — graph/session storage lifetime pools (terminal, rejected)

Retains WS15 lazy retirement but moves getter pools to native graph construction, ordinary graph.Run and resident session.Run boundaries. Native owners survive completed GPU work; C++ exceptions caught inside lexical pool then rethrown after drain. No model/context/precision/runtime/preprocessing/CUDA change. OFF/ON5native references each pass normal/nested/alias/exception/GPU output/abort checks; both32VLM/28text state/token/prefix/capacity-recovery guards pass. Mixed ordinary/resident/late second session/invalid/closed-rebuilt and separate partial allocation fault/recovery pass. Fault hooks absent from timed libraries.

Fresh WorldSense control then candidate car Skill workers,1warm+4measured each,20model calls. All full outputs/prompts/config/input/image/Skill identities, interpreted Skill semantics and native retirement/zero-regrowth checks pass. Frozen source/build flags/checkpoint identities; only lazy flag differs. Small future-cohort baseline-library copy overlapped control setup and is disclosed in cohort-host-activity.json; no build/archive export during timing, no measurements selectively removed.

Sampled post-decodeMetal21903376384→18867290112B(−3036086272B/2.828GiB/13.8613%). NativeSkillp50 38.1596916→38.7060935s(+1.4319%).4B TTFT+4.9454%,total+3.0431%,decode−2.1206%;textTTFT−13.8751%,total−11.1632%,decode−0.6087%. Original4B2%preservation gates fail: REJECT. No fullconfirmation/promotion/unchangedretry; prepared full cases/audits explicitly unexecuted.

Phase audit:4B warm decode CPUprepare10.083→9.589ms,encode77.477→66.842ms,GPU10612.159→10891.688ms,wait10802.470→11035.377ms. CPU prepare/encode do not support a simple increased-encoding-cost explanation. Vision8.040→8.859s,prefill10.619→10.931s. Sampled task physical footprint medians stay about18.05GB(decimal) despite fewer reported Metal resources; no demonstrated proportional RAM/headroom gain, no allocator/driver/VM causal attribution. Fixed-frame results exclude live decode/persistence/retrieval. Next distinct design: avoid initial allocation of admitted resident defaultstateinputs/terminalroots, preserve ordinary fallback and full context; not implemented here.

Archive results/20261016/worldsense-m5-storage-scopes includes20-callCSV, fullsourceclosure/compatiblebaseoverlay, native/fault sources/tests/fixtures, phase audit and raw setup/failures. Initial reused archive-finalizer path accidentally targeted local WS15 evidence; restored exact committed868cfe628 version, verified original SHA manifest, removed single extra test, corrected path and reran. No raw benchmark data altered. WS16 SHA audit passes. WS10 binary/normalservice restored,4filesonline,cameraunmounted,validbenchmark404.


### WS-M5-17 — explicit resident graph construction (terminal, rejected physical feasibility)

Default-OFF native build_bf16_resident/build_shared_resident omit first default state1/2 and final inactive packed-root allocations, preserve padded geometry/device limits and logical alias/writer/root proofs; normal APIs/generic materialization remain. Support4-input linear decode and6-input attention/prefill; one-input heads ordinary. First caller incorrectly routes head, then six-only caller leaves24linear graphs retiring defaults; both failures retained/fixed without weakening zero-retirement gate. Failed initial binarySHA notcaptured/earlybuildlogs overwritten; prepare.py+WS16seed+fixrecords reconstruct source, final measured source/build/model identities frozen.

OFF/ON6native references plus full32VLM/28text finite states/tokens/prefix/capacity checks pass. Native inventory confirms absent initial state/root buffers. Trained4/6input generic/resident/late session/inactive/invalid/shared-source-close/closed-rebuilt tests pass; separate4/6 second-allocation failure preserves private/caller state and recovers. No external CPython image helper/CUDA/modelmath/context change.

Prospective WorldSense physical-resource feasibility: fresh control/candidate carSkill1warm+1followup each,8actual modelcalls. Allgolden tokens/text/prompts/config/input/image/Skill hashes/semantic outputs/native avoidance with zero retirement/regrowth pass. Maximum post-decodeMetal21903376384→18867290112B(−3036086272B/2.828GiB/13.8613%);task footprint18058983800→18053707192B(−5276608B/~5.03MiB/0.02922%). Original>=10%physicalfootprint gate fails. REJECT; no speed screen/fullconfirmation/promotion/unchangedretry. Raw timings retained butnot speed evidence. Snapshot/notpeak/freeRAM and task/Metal overlap limits remain. Sparse/unwritten backing/allocator behavior are unproven hypotheses.

Archive results/20261016/worldsense-m5-resident-admission contains8callCSV, sourceclosure/compatiblebaseoverlay, realstate/native/fault tests+fixtures, all initial interface/coverage failures and provenance limits, final identity/gate audits. Post-cohort SDK/runtime identities explicitly labelledpost, notinvented pre-run capture. SHA verified. WS10 normalservice restored,4file sources online,cameraunmounted,validbenchmark404. Next design work/worldsense-resident-workspace/design.json targets actually written scratch via conservative exact-layout liveness/session-private reuse; notimplemented/notasafe-free claim.


### WS-M5-18 — active resident scratch workspace (terminal, rejected physical gate)

Session-private exact-layout interval colors share eligible temporary buffers across sequential layers. Protect weights/held records/constants/inputs/outputs/state/aliases/root; singleton standard tracked matrices only, one full writer before reads, strict lastRead<nextFirstWrite. Full-copy KVChunk admitted from scalar/vector/inactive source proof; in-place KVUpdate/uncertain partial Conv/Delta excluded. Pool ownership through GPU completion/abort, graph lock/binding restoration, generic separate materialization and graph lifetime invalidation audited. Multi-op fusion configs rejected. No MSL arithmetic/precision/context/CUDA/image changes.

OFF/ON7native refs pass (strict intervals, pendingreuse, NaNbeta0, KVscalar/vector inactive/tails, hazardproperty), full32VLM/28text finite states/tokens/prefix/capacity parity pass; actual pooled route/buffercount<bindings, protected slot/interval audit and4/6adversarialprivate/generic/sharedowner/invalid/rebuilt checks pass. Separate4/6two-layer fault tests partialgrow after nonzero state, generic2ndscratchfail and layer1 encoderabort after priorstateexchanges preserve appropriate state/poison/reset semantics; inventory confirms graph unbinding and exactrecovery. Fault hooks not in measured libs.

Prospective WorldSense resourcefeasibility: fresh control/candidate carSkill1warm+1follow each,8model calls allgolden output/prompt/config/input/image/Skill semantics and nativecounter checks pass. Frozen source/model/build+pre-cohortSDK/runtime/toolchain identities. MaximumMetal21903376384→18084315136B(−3819061248B/17.43595%);taskfootprint18057443728→17285478632B(−771965096B/~736.20MiB/4.27505%). Declared5%physical target fails: REJECT. No speed/fullconfirm/promotion/unchangedretry; raw timings retained butnot speedevidence. Snapshots notpeaks/freeRAM or kernel/DRAMcausality.

Target audit ofpriorWS14 otherGraphstorage identifieslarge VISION shapes[308,4096],[308,1024],[308,3072],[308,4928]; categorywasnot solelydecoder scratch. WS18 only3-output resident graphs, ordinary vision untouched. Priorinventory is prioritization, notWS18residual/safe-freeproof. Next design work/worldsense-graph-workspace/design.json adds conservative all-op ordinarygraph scratch pooling forone-outputvision, preserving3-outputfallback. Notimplementedhere.

Archive results/20261016/worldsense-m5-resident-workspace:8callCSV,sourceclosure/compatiblebaseoverlay,plan/hazard/ownership proofs,native/fault sources/tests,rawcases/model/build/dependency/gate identities. WS10normalservice restored,4filesonline,cameraunmounted,validbenchmark404.


### WS-M5-19 — ordinary vision graph workspace (terminal, rejected decode gate)

WS-M5-19 terminal: rejected companion decode preservation. No promotion, full confirmation or unchanged retry.

Default-OFF graph-owned exact-layout scratch pool for ordinary zero-outputParts graphs, added to WS18 resident pools and WS17 explicit resident construction. Plan covers all operations for ordinary graphs; protects inputs, root, outputs, weights, aliases, uncertain writers. Strict non-overlap, tracked resources, allocation before GPU begin, owners through wait/abort, NIL unbinding. Three-output generic fallback remains separate. NativeSession materializes formerly pooled head scratch before direct encoding. Custom Vision producers excluded; their operands extend liveness. Pool retains highwater until graph close/rebuild. Kernels, CUDA, precision/context/model assets and native image preparation unchanged.

OFF/ON seven native references; full32VLM/28text finite state/token/prefix/capacity parity; ordinary shape/tail/returned-output/input/shared-close/rebuild, trained4/6resident/private/generic and direct-head transition pass. Car/traffic/bunny/car multi-grid full vision embeddings bitwise equal across OFF/ON including shared replacement. Separate ordinary partial-growth/encoder-abort/unbinding/root-preservation/recovery passes. Fault delta three files CMake/Graph header/source, absent measured binaries; WS18 session-fault hooks not inherited in WS19 fault build.

Prospective resource stage:8actual WorldSense/Galaxy calls, fresh control/candidate one warm+one followup. All exact/native. Metal21903376384→17761353728B(−18.91043%), sampled task footprint18056165704→16965384280B(−1090781424B/6.04105%). Pass original10%Metal/5%physical gates. Snapshots not peaks or free RAM; task and Metal overlap.

New fresh car speed workers one warm+four measured each,20calls. All exact golden tokens/text/prompts/config/input/image/Skill identities and interpreted semantics/native routes pass. 4B TTFT−72.8832%,total−53.5895%,decode+9.0245%;text TTFT−76.1306%,total−61.5661%,decode−19.0002%. Skillp50 47.2395453→21.1143410s(−55.3037%). Text decode violates original2%preservation: REJECT. Large favorable single-screen timing is not a confirmed general gain. Full three-frame/query mirrored confirmation scripts prepared but explicitly UNEXECUTED. All cold/first/warm records retained, no samples removed.

Phase audit:4B vision8.981→1.456s,prefill16.844→5.505s,decode11.359→10.418s; text prefill7.976→1.639s,decode1.436→1.765s. Text decode GPU1358.736→1682.480ms,encode12.913→18.383ms,prepare3.541→5.110ms. Descriptive only; no CPU-only or allocator/VM causal attribution. Initial post-cohort phase audit assumed text vision_seconds and raised KeyError; corrected absent phase to null without any benchmark rerun/data changes. Tiny script/audit preparation during timing, no build or bulk source/archive copy; unrelated host activity not fully controlled.

Reconstruct source with recorded compatible committed base plus overlay, full closure hashes; base selected after experiment is not a claimed build checkout. Main resource records root; speed nested with its own CSV. Prepared confirmation separate and marked unexecuted. Fixed frames exclude video decode/live scheduling/persistence/retrieval/ranking. Four file sources online after validated WS10 restore, camera unmounted, benchmark valid request404. See restoration-verification.json. Research remains active.

Archive results/20261016/worldsense-m5-graph-workspace includes8resource+20speed call records, source closure/overlay, native/fault evidence and unexecuted confirmation plan.


### WS-M5-20 — private dense construction with VLM reuse (terminal, rejected Metal gate)

WS-M5-20 terminal: rejected resource feasibility; no speed screen, full confirmation, promotion or unchanged retry.

Distinct policy from WS19: default-OFF dense-private capability routes DenseMetalDecoder to atomic native build_bf16_private. Graph marker set under build lock after success; Clear/rebuild resets. Shared-weight construction inherits private source policy under both locks and suppresses resident default deferral. Private graphs skip retirement and workspace planning; VLM retains WS19 resident/ordinary-vision reuse. Native counters and inventory prove28dense decode plus shared-prefill graphs private, no dense workspace/avoided descriptors. Original kernels, CUDA, BF16/FP32 math, context, images and native C++ preparation unchanged. Dense retains more allocations; memory benefit can be insufficient. No causal claim that WS19 regression was due to pooling.

OFF/ON7native references each and private-policy3tail shapes pass shared inheritance/source close/returned-output/input/invalid construction/rebuild reset. Original trained4/6 generic/resident/private sessions, late second session, inactive/invalid/rebuild and pooled-head direct transition pass. Dedicated private4/6 OFF/ON batch1cap8/batch2cap64 and trained linear tests pass private ownership/shared-resident inheritance/source close/session isolation/invalid/rebuild. Full32VLM/28text finite state/token bits vsWS10, prefix reuse and capacity recovery pass; explicit inventory verifies default inputs1/2 and packed root retained for dense. Multi-grid car/traffic/bunny/car full native vision embeddings bitwise OFF/ON. Separate fault build7refs and ordinary partial pool growth/encoder abort/root/input retention/unbinding/recovery pass. Fault delta CMake/Graph header/source only; no hooks in measured builds.

Frozen common source and rehashed checkpoint assets, four OFF/ON policy flags, native binaries and exact request. Fresh real WorldSense/Galaxy carSkill workers control then candidate, one warmup+one followup each,8calls. All complete golden tokens/text/prompt/config/input/image/Skill hashes/semantic outputs and native VLM/dense-policy checks pass. Maximum postdecodeMetal21903376384→19725959168B(−2177417216B/9.9410117%), task footprint18054527352→17049466968B(−1005060384B/5.5668053%). Physical>=5% passes; Metal>=10% fails. REJECT without rounding or changing gates. Raw resource timings retained but not speed evidence. Prepared20call speed and mirrored full3frame/query suites explicitly unexecuted. Snapshot counts not peaks/free RAM; Metal/task overlap; fixed frames exclude decode/scheduling/persistence/retrieval/ranking.

Preserved preparation failures: WS19 fault patch apply-check conflicted; applied nothing, merged C++/header from exactWS19 main/fault bases; CMake append-only options explicitly verified/appended, separate fault tests pass. First request preparation assumed repeats4 but reused file already1, corrected prospective speed copy without measurement. Wrong assumed sdk/include root produced empty headers and assertion; shell lacked fail-fast and still dispatched controller. Correct SDK root hashes captured DURING first control startup, explicitly not pre-dispatch, equalWS19 recorded headers; runtime/Xcode/status capture same time. No invented pre-capture or cohort restart. Tiny script preparation during cohort; no source build/bulk archive during timing. External host activity not fully controlled.

Reconstruction compatible base chosen after experiment, not asserted as build checkout. Archive full source closure/overlay, separate fault delta, numeric fixtures, native/trained/private/model/vision/fault records,8callCSV, raw setup/failures, prepared-only nested suites and SHA manifest. ValidatedWS10 binary and normal service restored, four files online, camera unmounted, benchmark valid request404. Goal remains active.

Archive: results/20261016/worldsense-m5-vlm-workspace.


### WS-M5-21 — scalar Vision full-writer workspace (terminal, rejected decode preservation)

WS-M5-21 terminal: resource pass, speed preservation rejected. No promotion, full confirmation or unchanged retry.

Extends WS20 VLM reuse/dense-private policy with default-OFF GARNET_METAL_VISION_WORKSPACE. Existing exact-layout strict-liveness plan admits scalar custom Vision fullwriters0,2,3,4,5,6,9 only: bias, GELU, merge, positional addition, RoPE/copy V, flatten. Every logical output assigned; specialized norm/scores/context1/7/8/10 remain excluded. All operands conservative reads, full-one-writer/standard singleton tracked storage, protected inputs/root/weights/outputs/aliases and strict last<first unchanged. Native telemetry counts actual eligible vision bindings. Kernels/MSL, CUDA, BF16/FP32 math/context/image/native C++ preparation byte-identical to WS20. Pools highwater until graph close; memory/layout/performance tradeoffs remain.

Both8native references pass including56vision cases each: NaNoutput vs fresh bit equality, CPU reference tolerance3e-6, batch1/2,tokens4/8,columns7/12,merge4,queued reuse with intervening consumers and unsupported-mode abort. CPU oracle initially read irrelevant reshape mode9 index out of bounds; fixed on review BEFORE first test execution and final-source rebuild forced; no failed native sample hidden. Both full32VLM/28text finite state/token/prefix/capacity parity pass, original/private4/6trained graph/session/inheritance/source-close/invalid/rebuild pass, multi-grid car/traffic/bunny/car full vision embeddings bitwise OFF/ON pass. Correctness route exercises151eligible vision slots,12pooled buffers/298bindings; not performance evidence. Separate fault build8refs and ordinary partialgrow/encoderabort/root/input/unbind/recovery pass. Fault-source merge initially conflicts at independent EOF getter/test-hook additions; explicit combination and conflict text retained; first helper prints full conflict and missing builder fails, retry build log overwrites missing-builder log (failure recorded in fault-preparation.json). No fault hooks in measured binaries.

Fail-fast prep-only controller verifies prerequisites then freezes source, fiveflag OFF/ON differences, rehashed original model assets, library and SDK/runtime/toolchain/request identities BEFORE separate benchmark dispatch. WS10 backup SHA verified. No source/build/bulkarchive during timed cohorts; small script/audit preparation during runs disclosed; unrelated host activity not fully controlled.

Resource stage fresh WorldSense/Galaxy control then candidate, one warm+one followup,8calls exact full golden outputs/prompts/config/input/image/Skill semantics and native VLM/dense-private/newfullwriter checks. Sampled maxima Metal21903376384→19246563328B(−2656813056B/12.1296964%),task18057558416→16571513760B(−1486044656B/~1.384GiB/8.2294883%). Original10%Metal/5%physical gates PASS. Snapshots not peaks/free system RAM; task/Metal overlap.

NEW fresh speed workers one warm+four measured each,20calls all exact/native. 4B TTFT−76.03197%,total−55.89561%,decode+8.85274%;text TTFT−76.99056%,total−62.67886%,decode−26.04104%. Skillp50 49.8185326→21.3438966s(−57.15671%). Original2%companion decode preservation fails: REJECT. Single-screen favorable total timings are not a confirmed general speed gain. Full3frame/query mirrored suites prepared but UNEXECUTED; no gate relaxation, favorable retry or promotion.

Phase audit:4B vision9.60699→0.695707s,prefill17.03999→5.79824s,decode11.66435→10.71509s. Text prefill8.12750→1.77003s,decode1.28229→1.73704s; decodeGPU1234.648→1696.633ms,encode11.570→18.264ms,prepare3.266→4.633ms. Dense private buffers did not prevent observed regression; no pooling/clock/thermal/VM/allocator causal claim. Further diagnosis must isolate actual WorldSense workload/order before another mechanism, not rescreen same candidate.

Archive includes8resource+20speed call records/CSVs, allfirst/warm/setup, sourceclosure/compatiblebaseoverlay, separatefault delta3files, native/private/trained/model/vision/fault proof and failures, fixedfixtures, conditional unexecuted confirmation/integration plans and SHAmanifest. Compatible reconstruction base selected after experiment, not asserted as build checkout. Fixedframes exclude video decoding/live scheduling/persistence/retrieval/ranking. ValidatedWS10 normalservice restored,4filesonline,cameraunmounted,benchmark validrequest404. Goal remains active.

Archive: results/20261016/worldsense-m5-vision-workspace.


### WS-M5-22 — repeated search sequence diagnostic (complete, no promotion)

WS-M5-22 completed sequence diagnostic, not a speed screen or promotion. WS21 rejection unchanged.

Question: does companion decode behavior persist across repeated identical WorldSense search-planning calls and worker order? Exact frozen WS21 OFF/ON native source/libraries reused, no model/math/context/CUDA/image/serving-code changes. Existing authenticated featuregated benchmark runs fixedcar currentSkill then TWO identical 'a passenger car' PlanSearch calls perwindow,1warm+1follow. Fourfresh workers control1,candidate1,candidate2,control2,32calls. Galaxy PlanSearch source audited: each request calls native infer_json, benchmark returns raw response; no answer cache. All warmups/first/setup retained.

Before dispatch verified full frozen source/library hashes, current SDK/runtime/Galaxy/WorldSense backend identities and request, WS10 backup. Model asset hashes are exact WS21 reference copied, not a new rehash claim. Current environment captured separately from referenced original build environment. Small analysis/source-review scripts during cohort; no source/build/bulkarchive, unrelated host load not controlled.

All32calls valid/assignments restored, fullgolden tokens/text/prompt/config/input/image/Skill semantics andqueryplans match WS10; allactualVLMworkspace/fullwriter anddense28private/no-workspace routechecks pass. Samequeryinput411tokens/generated56 in allquerypositions; Skillreasoner input200/generated35, VLM393/94. Tasks differ, so only identicalqueryposition comparisons isolate request content. Prefixhits annotated; firstcoldquery miss and secondhit, allfollowup hits. No independent-sample/statistical or causalclaim.

Followup servingdecodeTPS: control1 VLM7.655,reasoner24.691,queries16.043/16.557;candidate1 VLM8.468,reasoner14.004,queries19.515/18.443;candidate2 VLM8.835,reasoner25.933,queries16.559/17.381;control2 VLM7.656,reasoner26.463,queries17.630/17.691. Candidate shortreasoner slowdown not consistent across two workers; querytimings differ too. These are diagnostic observations, not new optimization confirmation or a replacement WS21 pass.

Timing-boundary audit preserves original servingTPS=(generated−1)/(total−TTFT). total includes generation plus after-loop memory/ownership telemetry/tokenizerresponse work. Also report loopTPS, postloop=servingtail−loop, and loop−nativewait. Native GPU andwait overlap; neveradd orlabelresidual pureCPU. Candidate1 followupreasoner GPU1.3228s vscontrol1 1.3078, nativewait1.5917vs1.3419,loop2.2821vs1.3720,postloop.1458vs.0051. Candidate2 reasonerGPU1.2645/outsidewait.024, while itsfirstquery hasoutsidewait.572. Unattributed gaps move between requests, not uniformGPUkernelregression. Clock/thermal/VM/allocator/pool cause not established.

Static lifecycle audit: private marker suppresses workspace/retirement but globalLAZY_RESIDENT_DEFAULTS stillwraps graph/session Run in native StorageScope. Privatebuffer isolation is not lifecycleisolation. Session.Run may accept different compatible prefillgraphs eachcall; any future policy must inspect actualgraphs under sortedlocks, preservehead/graph lifetimes, Abort/unbinding and Build/Close synchronization. A Build-cached private flag alone unsafe. No codechange made from hypothesis. Next separate instrumentation should measure native call/body/drain, embedding/host work and processfaults before selecting mechanism.

Archive32rawcalls/nativeCSV+derivedphaseCSV, exactrequest/environment/source/library/dependencyIDs, partialaudits, allfirst/warm andtiming/lifecycle code evidence. Reconstruction references immutable committed WS21 archive868c8babb (fullclosure/compatiblebaseoverlay/faultsources/buildflags); diagnostic has no new nativecode. Fixedframes exclude livevideo decoding/scheduling/eventpersistence/retrieval/ranking. ValidatedWS10normalservice restored,4filesonline,cameraunmounted,benchmark validrequest404 (verificationrecord). Goalactive; no optimizationpromoted.

Archive: results/20261016/worldsense-m5-decode-sequence.


### WS-M5-23 — native and host latency tracing diagnostic (complete, no promotion)

WS-M5-23 completed instrumented diagnostic, no optimization/promotion. WS21 original failed speed gate unchanged.

Question: locate unaccounted WorldSense decode wall time after WS22 sequence variability. Isolated WS21-derived native source adds defaultOFF GARNET_METAL_LATENCY_TRACE, ON identically in both diagnostic variants; five workspace/private flags differ. Existing MSL/CUDA/genericCPU/image/modelmath/context unchanged. Native Session.Run uses same scoped/unscoped path and locks/exception/GPUAbort behavior. Caller-thread-local trace sequences measure native entry/body/prebody/postbody wall and getrusage processCPU/minor/majorfault deltas, sampling statuses/cost. Postbody includes scope-return/closure cleanup and scheduling, not pure ARC drain. Process counters include other threads, not perthread attribution or full Metal/VM residency.

Metal-only serving loop measures native embedding, step_hidden host/FFI, traceJSONread, nativebody/tail and sidecarexport. Generic CPU/CUDA decoder untouched. Perstep JSON/allocation/syscall/filework can perturb timings; data never compares as optimization against untraced WS22. Generation loop uses same native embedding/step_hidden/controlvalues and token/stop semantics. NativeGPU/wait clocks only aggregated percall, overlap and must not be added. Reference original servingTPS=(generated−1)/(total−TTFT), plusloopTPS/postloop views remain; no priorgate redefinition.

Both9native suites pass, including56Visionpoison/queuedreuse cases and new scope/no-scope callbacks, preserved exceptions, nested/caller completion sequences, two-thread isolation and nonnegative timing/processsamples. Real SDK success/error traceguard confirms native exceptions propagate withfailedtrace and recover. Full32VLM/28text finite state/token/prefix/capacity parity vsWS10, persteptracefileSHA/seq/scope/FFI-boundary checks, original/private4/6trained graph/session/inheritance/source-close/invalid/rebuild and visionmultigrid exactembeddings pass. Separate faultbuild9refs and ordinary partialgrow/encoderabort/root/input/unbinding/recovery pass; testonly Graph/CMake3filedelta absent measured binaries. No build/testfailure hidden; Galaxy/WorldSense backend dependency metadata was appended to the frozen environment before dispatch, no timedretry.

Failfast prep freezes fullsource, actualOFF/ONlibraries/commontracing and5policyflags, rehashes originalmodelassets, SDK/runtime/toolchain/request andWS10backup BEFORE dispatch. New workload is identical WS22 diagnostic: onefixedcar currentSkill then TWO identical 'a passenger car' actualGalaxy PlanSearch calls perwindow,1warm+1follow, fourfreshworkers control1,candidate1,candidate2,control2. All32calls exact fullgolden tokens/text/prompts/config/input/image/Skill semantics/queryplans, actualVLMworkspace/fullwriter+dense28private/no workspace andtrace sidecarSHA/recordcount/threadsequence/timing partitionchecks PASS. Allfirst/warm/setup retained; no timing gate, no selection of favorablecalls. Small scripts/source-review during runs, no build/bulkarchive; unrelatedhostactivity not controlled.

Warm maximum postbody sum is0.000409788s/response, embedding sum0.02894525s/response. These do not support removing autoreleasecleanup or embedding work as the main explanation of prior largegaps. Larger intervals inside nativebody/wait: candidate2reasoner body1.6884s vsGPU1.4533/wait1.6675; firstnative step body.294372s/processsystemCPU.189899/minorfault145/postbody3microseconds, latersteps~.04-.054s. Candidate1reasoner body1.4598/GPU1.3998; controls1.4402/GPU1.3658 and1.4386/GPU1.3881. Control1query1 body3.4299/GPU2.8433/wait3.1239, trace-read.0896. Variation also exists without candidate scopes. ProcessCPU/fault observations do not establish driver/queue/VM/clock/thermalcausality; zero majorfaults is not proof of no memory pressure. PerstepGPU/wait unavailable, so individual submission delays cannot be separated further from these records.

All32rawcalls/nativeCSV, perstep sidecars and component/ranked-span views retained. Copy every referenced benchmark AND correctness capture withSHA and portable path mapping; unreferencedstartup/live captures are not selected asbenchmarkcalls. Fullsourceclosure/compatiblebaseoverlay, separatefaultdelta, SDK/build/model identities, native/trained/private/model/vision/error proofs and limits included. Compatible reconstruction base chosen afterexperiment, not asserted asbuildcheckout. Fixedframes exclude livevideo decoding/scheduling/eventpersistence/retrieval/ranking. ValidatedWS10normalservice restored,4filesonline,cameraunmounted,validbenchmarkrequest404. Goalactive; nooptimizationpromoted.

Prior-method review: streamed-body-weights was a rejected decoded-vector register-lifetime microkernel screen on smallmodels, not GPU materialization of resident FP32 vision weights. Existing4B vision-sharing+BODY_SHAPE retained immutable cross-grid sharing and dtype/math; it did not stream perprojection weights. Any next mechanism must retain these outcomes, audit actual weight owners/layouts and declare new untracedWorldSense gates before measurement. No new mechanism implemented in WS23.

Archive: results/20261016/worldsense-m5-latency-trace.


### WS-M5-24 — exact vision-weight materialization (rejected)

WS-M5-24: Exact BF16 vision masters with reusable FP32 materialization

Status: REJECTED at independent fresh speed screen. Correctness and resource feasibility passed; companion decode throughput failed the unchanged2% preservation gate. No confirmation, promotion or unchanged retry.

Method: seed the untraced WS21 source, retaining its previously rejected measurements. Replace persistent FP32 copies of 96 BF16 block-projection weights with immutable, raw-only BF16 masters. Before each matrix use, a GPU integer kernel writes uint(ushort)<<16 into a transient FP32 buffer, zeroing row padding. The original MPS FP32 matrix multiplication remains unchanged. Distinct raw/master keys retain checkpoint metadata validation and shared owners; expanded buffers are excluded from immutable shared records. Conservative exact-layout, full-write, single-owner, hazard-tracked interval coloring permits reuse after the last consumer, protecting inputs, roots, aliases and masters. Original precision, model assets, context, pixel budget, CUDA and native image preparation are retained.

Tradeoffs: every frame adds 96 encoded expansions, reads the BF16 masters and writes FP32 workspace. Workspace pools retain their high-water allocations until graph close. The implementation initially allocates F32 shadow descriptors before pool binding, so there is no setup peak-saving claim. Header-only net payload estimate528MiB is distinct from measured whole-process results; this candidate also includes WS21 workspace/private policies.

Correctness: both nine-test native suites and the separate nine-test fault build pass. Primitive proof covers all65,280 finite BF16 bit patterns including signedzero/subnormals/extremes;24 varied-rounding MPS tail/batch/queued-reuse controls compare bitwise. Two official vision projections prove one reusable weight-buffer color, protected BF16 masters, changed input/output lifetime, shared source close and rebuild, and F32 checkpoint fallback. Full32-layer VLM and28-layer companion state/token/prefix/capacity checks pass. Four images with multiple grids and shared replacements produce identical embedding hashes. Original/private4/6-input trained graph/session guards pass. Separate test-only partial pool growth and four expansion/MPS abort boundaries preserve inputs/returned outputs/master identities, unbind slots, and recover exact results. Tests use native XLang3/Garnet; CPython coordinates and audits only.

Failures retained: first full vision attempt asserted pooled count before lazy plan construction; explicit plan then proved all96 and complete embedding equality without native changes. Initial fault-source preparation merged top-level CMakeLists rather than cmake/GarnetMetal.cmake, leaving test options uninitialized and hooks unavailable; corrected only the separate test source. Independent EOF merge conflicts and clean-merge assertion failure are documented. Resource workers both completed successfully; post-benchmark package startup raced launchd's Cantor, causing controller exit1. The wrapper restored normal WS10 successfully; original captures were analyzed separately, not rerun. Later controllers wait for native ProbeCantor PID to match launchctl before package start; setup is excluded from returned inference timings.

Resource feasibility:8 actual authenticated WorldSense/Galaxy model calls, two fresh workers, one warmup and one followup each. Exact tokens/text/prompts/configuration/input hashes/Skill semantics and native counters all pass. Maximum returned post-decode Metal allocations21903376384→18692915200B, reduction14.65738034%; sampled process footprint18056526176→16014785200B, reduction11.30749601%. Gates declared before implementation were10%Metal and5%physical. These are sampled resource snapshots, not peak usage, free system memory, or speed evidence.

Speed protocol: a NEW fresh control/candidate pair with one warmup and four measured car Skill repetitions each,20 model calls. Both4B VLM and1.7B reasoner TTFT/total/decode and complete native Skill wall must preserve the existing2% regression limits. If successful, two new full three-frame/query pairs in mirrored worker order must each pass independently. Preserve all first/warm calls and failures; no unchanged favorable retry or gate relaxation. No production replacement before the complete gates and integration checks.

Scope: recorded fixed frames traverse the real WorldSense Skill/Galaxy/Garnet and query-plan paths, excluding live decode/scheduling, event persistence and retrieval/ranking. The four mounted video files are separate sources; the copied car second view is correlated and is not independent camera-fusion evidence. No camera is used. Normal restoration verifies validated WS10, four online files and valid benchmark POST returning404.

Provenance: frozen source/build flags/model hashes/SDK/runtime/toolchain plus raw captures, request IDs and phase data are retained. The source overlay is relative to a compatible committed base selected for reconstruction, not a claimed source checkout identity. Primitive-proof identities precede graph integration and are explicitly post-test, not benchmark freeze identities. Existing three MSL blocks and21 CUDA files are byte-identical to the seed; one new integer kernel is added. CUDA is not executed on this Mac. GPU and wait timings overlap; do not add them. Serving decode rate includes host/post-loop work and is not a GPU-only rate.

Terminal speed screen:20 actual WorldSense calls all exact/native. 4B TTFT−75.28828%,total−54.95339%,decode+14.70518%;1.7B TTFT−80.17954%,total−66.57080%,decode−26.58625%. Complete Skill wall p50 53.968477708→22.812258563s(−57.73040%). These favorable VLM/Skill results do not override the companion decode gate. Both original control/candidate raw outputs and all warmups retained. Prepared full confirmations were not executed. Normal WS10 fourfiles/noCam/benchmark404 verified after screen; speed-stage controller setup ownership wait avoided the earlier restore race. No causal attribution of companion regression.

Archive: results/20261016/worldsense-m5-vision-weight-stream.


### WS-M5-25 — actual private execution and output head (rejected)

WS-M5-25: Actual private execution and dense private output head

Status: REJECTED at fresh speed screen. Correctness/resource pass; companion decode failed the original2%preservation gate. No confirmation, promotion or unchanged retry.

Prior evidence: WS24 exact BF16 vision-master materialization passed correctness and reduced sampled Metal/process allocations14.66%/11.31%, but companion1.7B decode fell26.59%, so WS24 remains rejected. WS23 found warm postbody≤0.000410s and embedding≤0.028945s; it did not support cleanup or embedding as the main cause. WS25 tests execution-policy isolation rather than claiming scope cleanup caused the regression.

Mechanism: defaultOFF Metal PRIVATE_EXECUTION requires DENSE_PRIVATE_STORAGE. Dense decoder layers/prefill graphs already had private storage, but its vocabulary head used ordinary build. New generic build_private preserves ordinary bf16Projections=false and the explicit head bf16_weight operation, then sets the immutable private marker only after successful construction. Dense head uses this method only when the new capability is enabled. Shared construction inherits the marker; failed build, Clear and rebuild reset it.

Native Session.Run accepts alternate graph lists, so policy cannot be cached at Session.Build. The call takes its existing session mutex and sorted distinct actual graph/head locks. It then selects the wholly private path only when every actual participant is ready/private. That path bypasses the reuse StorageScope and empty workspace/retirement bookkeeping. Mixed lists and ordinary heads retain scoped execution. Ordinary Graph.Run selects under its own graph mutex. Helper lambdas retain the same input preparation, GPU commands, exception propagation, Abort, poison/reset, state/output/input behavior and locks. The defaultOFF wrapper retains the old route. Native private/scoped counters increment only after successful calls, reset on Clear/Build, and are cumulative per object; Reset retains them. Serving timings also report private head state. Counters do not mean GPU-only timing or independent samples. Their instrumentation is active only with PRIVATE_EXECUTION; zero in disabled builds is not universal proof of absent lifecycle scopes. The measured control also has LAZY_RESIDENT_DEFAULTS OFF by frozen flags.

Correctness before measurement: OFF/ON native9 suites; generic FP32 private vs ordinary output parity/shared-source-close/invalid/rebuild/failure-reset; actual private/mixed layer lists and private/ordinary heads over both fixture sizes, exact states/predictions/input and head generation rejection; full32VLM/28text finite states/generated tokens/prefix/capacity recovery; four-image multigrid vision embedding equality; original/private trained4/6graph-session interoperability. Actual serving counters select private LLM/head and scoped VLM. No arithmetic or image/context/precision changes; all Metal kernels and21 CUDA files byte-identical to the WS24 seed. CUDA execution is not tested on this Mac.

Separate fault source: original ordinary pool growth/encoder abort and vision-weight partial growth/four expansion-MPS abort boundaries pass. Existing test-only graph encoder hook additionally accepts private graph encoding from Session.Run, enabling private Graph Abort and actual private/mixed-head session Abort, poisoned-run rejection, Reset and exact state/prediction recovery. Failed calls do not increment completed-route counters. Fault definitions/hooks are absent from measured source. The3file fault delta remains CMake/compiled_graph.h/cpp. CMake merge's shared-terminal-endif conflict was resolved by verifying/appending the exact old fault-only tail to the complete new main file, not blindly stripping conflict markers.

Retained setup failure: the first inherited weight fault script lacked its local graph-off.json reference. Native9fault, ordinaryfault and trained private checks had completed; those logs were retained. Fresh OFF/ON real two-projection guards generated the missing reference and proved exact output/reuse/F32 fallback; only remaining fault stages resumed. No native source change, performance retry or gate relaxation. Initial logs and explicit failure record remain.

Prospective protocol: NEW fresh actual WorldSense/Galaxy car Skill control/candidate,1warm+1followup each,8model calls. Require exact tokens/text/prompts/config/input/image/Skill identities and actual native route plus≥10%reportedMetal and≥5%sampled task-footprint reduction. If pass, another NEW fresh pair with1warm+4measured each,20calls; both models TTFT/total/decode and complete Skill wall must preserve2%regression limits. If pass, two new full3frame/query mirrored pairs independently pass before integration/defaultOFF checks/promotion. Preserve all first/warm/failed records; no unchanged favorable retry. Launchd Cantor LRPC ownership is verified before package startup; setup is outside returned inference timings. Final wrapper restores normal WS10 and verifies4onlinefiles/noCam/validbenchmarkPOST404.

Limits and paper provenance: resource readings are maximum returned post-decode snapshots, not peaks/free system RAM. Serving decode=(generated−1)/(total−TTFT), including host/postloop work. GPU and wait intervals overlap; do not add them. Fixed-frame real Skill and PlanSearch paths exclude live decode/scheduling, event persistence and retrieval/ranking. The copied car second view is correlated, not independent camera evidence. CPython only orchestrates/audits; native XLang3/Garnet handles image preparation/inference. Source/build7flags/model/SDK/runtime/toolchain/request identities freeze before cohort dispatch. Archive compatible source overlay identifies a reconstruction base selected after experiment, not a claimed checkout identity at build. Small source/record preparation occurs during the resource run; no build or GPU test is concurrent with performance. No causal explanation of prior companion variation is established by this experiment.

Resource result:8actualWorldSensecalls all exact/native. Sampled maximum Metal21903376384→18692915200B(−14.65738034%);taskfootprint18059032904→16014390952B(−11.32199029%). Bothworkers0, controller/wrapper0; normalWS10fourfiles/noCam/benchmark404 verified. These data pass the prospective10%/5% gates, not speed gates. No cross-experiment causal comparison with WS24. New20call fresh speed comparison dispatched after normal restore.

Terminal speed result:20actualWorldSensecalls all exact/native. 4B TTFT−76.68919926%,total−58.00783170%,decode+15.12665222%;companion1.7B TTFT−75.75986026%,total−61.75903458%,decode−23.91451812%fails2%limit. Complete Skill wall p50 51.433624625→21.183272479s(−58.81435027%). Favorable total latency does not override throughput gate. No fullconfirmation or unchanged retry; prepared scripts remain unexecuted. WS10normalfourfiles/noCam/benchmark404verified. Allroute counters, including fully private companion/head with no new scoped executions, pass. This does not establish lifecycle bookkeeping as the cause or cure of prior throughput gaps.

Archive: results/20261016/worldsense-m5-private-execution.


### WS-M5-26 — per-token GPU/native/wait/host diagnostic (complete, no promotion)

WS-M5-26: Existing native GPU/wait counters per decode token

Status: COMPLETE diagnostic;32actualWorldSensecalls and1928decode steps pass exact/native/profile integrity. No optimization/promotion or speed gain claim; WS25 rejection retained.

Question: locate the companion serving-decode loss seen in WS24/25. WS25 proved fully private layers/head and no new scoped executions on that route, yet companion decode fell23.91%, failing the unchanged2%gate. WS23 showed small warm postbody/embedding totals and native-body variability but lacked perstep GPU/wait spans. No cleanup, embedding, VM, driver or clock cause is assumed.

Implementation: defaultOFF DECODE_STEP_PROFILE capability ON identically in both diagnostic variants. Seven existing memory/private flags differ. Seed is untraced WS25, not WS23. Only CMake, native entry capability metadata and Metal serving code change; native decoder/kernel/model/image/CUDA sources remain byte-identical. AST audit verifies the original decoder.step call and existing profile-getter loop are unchanged. The runtime already reads prepare_ms,encode_ms,gpu_ms,wait_ms once per generated-step to form aggregate decode_profile; samples reuse those readings and cached decoder.last_session_ms, with no extra native getter calls. Two monotonic calls around the original step record host wall, plus index/input token/position/output prediction. Original loop/active/position/EOS/output-limit behavior and aggregates remain. Inline successful-call timing samples are serialized after generation; no perstep file/sidecar writes or process-counter syscall collection or Python image/inference adapter is added.

Clock semantics: host_step includes native embedding/controls/step and FFI, but excludes sample dictionary creation and later profile getter/record bookkeeping. Original decode_seconds includes loop/bookkeeping overhead. Native inference timer has existing native boundaries. Prepare+encode+wait are native sequential host intervals; GPU and wait overlap and must not be added. host_minus_native, native_unpartitioned and wait_minus_gpu are labeled arithmetic residuals, not causal attribution or pure CPU work. Tiny signed residuals are retained, not clipped. Integrity clock tolerance0.1ms and aggregate-sum tolerance1e-6ms were set before model tests.

Correctness before dispatch: both native9 diagnostic suites; full32VLM/28text finite state/token/prefix/capacity parity; multigrid vision embeddings; original/private trained4/6, actual alternate private/mixed layer and private/ordinary head policy, shared-source close/rebuild; generic FP32/private/output/input lifetime and real projections/F32 fallback. Trace indices/token transitions/positions/finite nonnegative clocks/native-host bounds and sums exactly match old aggregates. Outputlimit1 has zero decode steps, limit2 exercises a step, a temporary synthetic stop condition exercises zero-output EOS, canonical stop IDs are restored and capacity-error recovery matches reference. Separate defaultOFF build/native9/actual native inference proves absent tracing and exact original tokens/error recovery. Separate fault build/native9/ordinary/weight/private graph and session Abort/poison/reset, plus real infer_json injected encoder error, prove preserved errors, no stale successful trace and exact recovery. Fault hooks absent from measured source. No timing evidence is taken from correctness tests; temporary synthetic stops are never benchmarked.

Protocol: identical to WS22/23 diagnostic sequence: one fixed car Skill and two identical actual Galaxy PlanSearch requests per window,1warm+1follow, four NEW fresh workers control1,candidate1,candidate2,control2;32 model calls. Exact golden tokens/text/prompts/config/input/image/Skill semantics and query plans, actual native routes, common tracing, source/library/dependency identities and every step's integrity are required. No timing pass/fail selection, favorable retry, optimization confirmation or promotion. All first/warm calls retained. Step CSV rows link raw artifact SHA, worker, phase, repetition, query position, model/prompt and step. Canonical stop IDs for final EOS validation derive from frozen model/tokenizer assets. Current serving-source path must equal the frozen runtime path.

Provenance and scope: source/build/common trace plus7policy flags, original model assets, SDK/runtime/toolchain, WorldSense backend Python hashes, Galaxy binary and requests are frozen before dispatch. Compatible reconstruction base is selected after experiment, not asserted as build checkout identity. Fixed frames traverse real WorldSense Skill/Galaxy/Garnet and query planning, excluding live decode/scheduling, event persistence and retrieval/ranking. CPython coordinates/audits only; XLang3 and native Garnet perform inference/images. Four video sources remain mounted, camera unused; copied car view is correlated. Normal wrapper restores best validated WS10 and verifies four online files/no camera/validbenchmarkPOST404.

Observer limits: dictionaries, clock reads and larger JSON responses perturb execution, cadence and serialization. Native model timing and outer Skill wall have different serialization boundaries. Tracing ON both variants does not eliminate observer effects. Do not compare these durations to untraced WS25 as optimization gains or pool perstep observations as independent experiments. Host activity, queueing, VM and thermal state are not isolated by these clocks; retain all raw data and avoid unsupported causal claims. GPU/wait counters are existing synchronous native profile values, not new hardware performance-counter samples.

Final observations, all warm companion Skill requests retained: controlsGPU totals1272.5178/1215.6283ms; candidates1645.5163/1429.2997ms. First native steps controls114.666/66.441ms vs candidates42.207/39.988ms, so the companion loss is not consistently an initial-token penalty. Candidate1 GPU spans rise fromabout39ms toward53–55ms within its35steps; candidate2 spans rise fromabout36ms toward50ms. Controls subsequent spans remainmostly33–38ms. No GPU clock/bandwidth/thermal/VM/resource-placement cause is established.

Candidate2 warm companion step29 host273.641416ms vs native52.466ms, GPU50.972667ms and wait51.62125ms:221.175416ms host-minus-native residual. Its decode-loop time outside the measured step spans is81.798665ms; other warm companion workers0.739332/2.528166/1.539497ms. Postloop residuals0.850667/3.145083/2.748542/1.605625ms remain small. These clocks locate both GPU-duration differences and a host-side outlier, not pure CPU time or a root cause. Median-only summaries would hide that outlier; all1928rows and raw JSON remain.

Warm4B first steps also vary: controls native719.912/727.931ms withGPU481.685/535.940ms; candidates119.013/101.829ms withGPU114.855/97.455ms. Later GPU medians controls101.316/98.197ms vs candidates111.270/108.631ms. Do not interpret the first-step improvement as a universal nativekernel speedup or compare this traced workload as a gain over untraced screens. Repeated query and mirrored worker observations retained without selection. NormalWS10fourfiles/noCam/benchmark404 verified after the completed diagnostic. No builds/GPU tests during measurement; small record/analyzer preparation and one provisional first-worker integrity analysis occurred and are retained.

Archive: results/20261016/worldsense-m5-decode-step-profile.


### WS-M5-27 — dense BF16 decode grouping256 (rejected)

WS-M5-27: Dense inner2048/6144 BF16 decode grouping

Status: REJECTED at fresh speed screen. Correctness/resource passed, but companion decode failed the unchanged2%preservation limit. No confirmation/promotion or unchanged retry.

Prior evidence: WS26 retained32exact/native calls and1928perstep rows. Companion loss extended into later GPU spans and included a221ms host-minus-native outlier; it was not consistently an initial-token penalty. No clock/thermal/VM/driver cause was established. WS11was specifically4Binner2560/9216 256-thread grouping, not dense2048/6144; its rejection remains. Earlier M4/smallmodel outcomes remain separate. This is an exploratory launch-geometry test, not a claimed fix of a diagnosed hardware cause.

Scope: original checkpoint headers identify1961.7B BF16 body projections with inner2048/6144, output257..99999, and zero eligible4B matrices. Vocabulary head excluded. Runtime native counters must also prove companion positive/4Bzero; header classification alone is not execution proof. Seed is untraced frozen WS25; WS26tracing is not used for performance.

Implementation: defaultOFF DENSE_DECODE256 requires BODY_SHAPE. Append a specialized shader copied from existing body shader with only function name and output-group stride16→32 changed. Original four MSL literal blocks and original body_shape_shader.h remain byte-identical. Eight SIMD32groups/256threads write32outputs instead of four/128/16, retaining per-output BF16 bit expansion, FP32products/accumulation and SIMD32reduction order. Row1 and target inner/output dimensions only; batch2, small outputs, other inners and vocabulary route unchanged. Weight/scratch/model/context/image/CUDA sources and representations unchanged.

Body-shape cache entries now retain pipeline plus immutable wide flag under the existing shared mutex. Wide constants use no-tail iff outputs divisible32, protecting e.g.272columns. Failed/unavailable wide specialization retries the original baseline with original constants and a nonwide marker; encoding preserves legacy parameters unless wide. Device/SIMD/maxthread validation remains. Older hardware fallback is source-checked, not executed on this M5; do not claim older-device performance. Actual encoded-wide counter increments after encoder dispatch, cumulative per native context/session and may include later-aborted batches; not completedGPUoperations or perrequest counts. One postrequest native getter adds telemetry, no extra pertoken getter.

Correctness: both native10 suites pass. New77shapes×2varied mantissa/exponent/sign input patterns produce154complete logical FP32 output records bitwise identical OFF/ON. Retain binary word files and SHA/schema. CPUFP64 reference uses bound1e-4+2e-6×sum(abs(products)); it is a correctness bound, not a precision change. Tests cover target/tail272, unsupportedinners, batch2, small/vocab fallback, NaN-poisoned output, input preservation, invalid dimensions/no dispatch, queued reuse/source release, Abort read rejection and exact retry. Test source strengthened before first execution; no native failure or timing evidence from unit walltimes.

Both full32VLM/28text finite state/token/prefix/capacity checks and multigrid vision embeddings match existing reference bits. Actual serving encoded-wide counter positive only companion and zero4B. Original/private trained4/6, generic/projection/F32 fallback, actual private/mixed layers/head/generation/shared-owner lifetimes, separate nativefault10 and ordinary/vision-weight/privateGraph/session partial-grow/Abort/poison/reset recovery pass. Separate test-only3file CMake/Graph fault delta remains absent from measured source.

Protocol: NEW fresh realWorldSense/Galaxy carSkill control/candidate,1warm+1follow each,8modelcalls. Require all exact golden outputs/tokens/prompts/config/input/image/Skill hashes and native routes, actual serving-source path/library IDs,≥10%reportedMetal and≥5%sampledtaskfootprint reduction. If pass, NEW fresh20call screen1warm+4measured perworker; bothmodelsTTFT/total/decode andnativeSkillwall preserve original2%regression limits. If pass, two NEW full3frame/query mirroredpairs eachindependentlypass before integration/defaultOFF/publicprofile/normalservice validation. No unchanged favorable retry or gate relaxation. Preserve all first/warm/failed records. Prospective design and checkpoint-shape audit precede implementation.

Provenance/limits: source/build8policyflags/originalmodel/SDK/runtime/toolchain/request identities frozen before dispatch. Optional read-only pmset battery/AC state before/after each worker; no power-policy changes, no GPUclock/temperature attribution from that metadata. Resource values are postdecode sampled maxima, not peaks/freeRAM. Serving rate=(generated−1)/(total−TTFT), including host/postloop costs. GPU/wait overlap; do not add them. Fixedframes traverse realSkill/Galaxy/Garnet and PlanSearch, excluding live decode/scheduling/eventpersistence/retrieval/ranking. Fourfile sources/no camera; second car copy is correlated, not independent fusion evidence. CPython only orchestrates/audits; nativeXLang3/Garnet handles images/inference. Source reconstruction base chosen afterexperiment for compatible replay, not asserted as sourcecheckout atbuild. Normal wrapper restores validatedWS10 and verifies4onlinefiles/noCam/validbenchmarkPOST404.

Tradeoffs: scheduling/occupancy/input rereads, extra pipeline metadata/compilation and encoded-counter overhead. No extra weights/scratch intended; complete process resource readings determine feasibility. More SIMD groups may regress despite exact math. Combined candidate includes previously rejected memory/private policies, so any observed difference is a whole-profile retention result, not isolated causal attribution to group size. Prior results remain unchanged.

Resource result:8actualWorldSensecalls exact/native/source-identity checks pass. Metal21903376384→18692915200B(−14.65738034%);sampled task18070387016→16013621816B(−11.38196541%). Bothworkers/controller/wrapper0; normalWS10fourfiles/noCam/benchmark404 verified. These are resource gates, not speed proof. A NEWfresh20call speed pair starts after normalrestore. Optional battery/AC metadata retained without inferring clocks or thermal state.

Terminal screen:20actualWorldSensecalls all exact/native/source-identity checks.4B TTFT−73.34918292%,total−51.95416047%,decode+3.56358615%;companion1.7B TTFT−76.86967360%,total−65.48306959%,decode−11.55992607%fails2%limit. Native Skillwall p50 43.102269479→19.455598312s(−54.86177747%). All first/warm records retained; favorable overall latency does not override throughput gate. No full confirmation, promotion or unchanged retry. No cross-cohort purekernel gain claim vsWS25. Bothworkers/controller/wrapper0; normalWS10fourfiles/noCam/benchmark404 verified.

Archive: results/20261016/worldsense-m5-dense-decode256.
