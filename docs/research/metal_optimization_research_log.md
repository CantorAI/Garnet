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
