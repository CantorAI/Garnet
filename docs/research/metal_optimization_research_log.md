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
