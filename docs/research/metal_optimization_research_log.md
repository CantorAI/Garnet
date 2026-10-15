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

## Latest audited matched comparator result

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
