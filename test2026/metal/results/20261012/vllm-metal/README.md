# vLLM-Metal comparison evidence

Target: Garnet >=2x vLLM-Metal decode throughput and <=0.5x its first-token
latency on the same M4 16 GiB Mac and identical BF16 checkpoints. The first
clean matched baseline executed 48 requests. Garnet is slower in all four
profiles; neither target is met.

## Clean baseline

`baseline/vllm-matched-clean-baseline-20261012/jobs/summary.json` is the
authoritative repeated baseline for native build `d2a7146`. Each row below
averages four warm requests per runtime, following first-use requests in two
fresh processes. Full first-use results, generated token IDs, commands,
individual engine steps, process snapshots and model/library hashes are
retained in that folder. All 48 outputs and prompt token IDs match exactly;
all processes shut down cleanly.

| Model | Batch | Garnet tokens/s per row | vLLM tokens/s per row | Garnet TTFT, ms | vLLM TTFT, ms |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 37.62 | 73.83 | 150.22 | 52.07 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 62.11 | 92.24 | 96.79 | 34.77 |
| Qwen3-0.6B | 2 | 27.20 | 75.34 | 280.62 | 71.12 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 55.83 | 92.36 | 157.85 | 38.03 |

Hardware: Apple M4 Mac mini, 16 GiB, macOS 15.6.1. Comparator: isolated
CPython 3.12.15, vLLM 0.30.0+cpu, vLLM-Metal 0.30.0, MLX 0.32.1,
mlx-lm at `9e6acca691e64d6d8bb808c328fcdea459099cca`. Despite the vLLM wheel's
`+cpu` label, the Metal plugin and MLX GPU execution are active and the device
probe reports Apple M4. Both runtimes use the same BF16 checkpoint files.
Garnet uses FP32 activations/KV/accumulation; vLLM uses BF16 model/KV execution.
This compares checkpoint/weight precision, not identical internal arithmetic.

The frozen workload has capacity 256, greedy generation, output cap 64,
no prefix reuse or speculation, identical native prompt IDs, and batch 1/2.
Decode speed uses the complete first-to-final prediction wall interval.
Per-request reset/cache preparation is included in TTFT. Filesystem caches
are not flushed. RSS is individual-process high water, not aggregate worker
or unique GPU memory. Two existing WorldSense service processes were verified
to have no inference/GPU libraries and left running; their CPU/RSS snapshots
are saved. Competing inference runtimes never overlap.

## Visible-cache optimization

Attention now scores and accumulates only the causal visible prefix. Cache
storage, slot bounds, state reset, weight precision and context capacity stay
the same. The inactive-row controls retain their previous fallback behavior.
All 11 numerical/serving/vision validation jobs passed, followed by another
48-request matched suite with identical outputs and clean shutdown. Evidence:
`visible-cache/visible-cache-checks-20261012` and
`visible-cache/vllm-matched-visible-cache-20261012/jobs/summary.json`.

| Model | Batch | Garnet tokens/s per row | vLLM tokens/s per row | Garnet TTFT, ms | vLLM TTFT, ms |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 40.18 | 74.40 | 118.77 | 51.51 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 67.73 | 92.33 | 75.40 | 34.72 |
| Qwen3-0.6B | 2 | 28.49 | 75.53 | 214.42 | 70.55 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 61.01 | 92.00 | 113.86 | 38.30 |

Against the native baseline, decode improves 4.8–9.0% and TTFT falls
20.9–27.9%. Neither vLLM-relative target is met. GPU time remains the principal
decode cost; these gains are not evidence of a 2x comparator advantage.

## Rejected prefill experiment

`transposed-gemm/` retains a column-major shared-weight tile experiment that
uses a transposed SIMD-group load. All 11 validation jobs and the 48 matched
requests passed, including exact outputs. However, warm TTFT worsened by
6.4–22.8% against the visible-cache build. The small decode differences are
not persuasive because this experiment changes only GEMM prefill. It was
reverted. `rejected_kernel.diff` records the attempted kernel; the source
branch retains the original GEMM layout.

| Model | Batch | Experimental Garnet TTFT, ms | Visible-cache TTFT, ms |
| --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 136.14 | 118.77 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 92.58 | 75.40 |
| Qwen3-0.6B | 2 | 228.05 | 214.42 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 127.74 | 113.86 |

## Rejected decode-address experiment

`u32-decode/` retains a guarded 32-bit index variant of the unchanged BF16
GEMV arithmetic, with the original 64-bit fallback. The matrix reference
test (aligned/odd widths, padded rows and batches) and all three pretrained
vocabulary-head checks passed. All 48 matched requests had exact outputs and
clean shutdown. Decode rates were 40.35/68.22 tokens/s for text/coder batch 1
and 28.96/61.59 for batch 2, only 0.4–1.6% above the visible-cache run.
That difference does not establish a useful gain. The extra kernel/dispatch
path was reverted; `rejected_kernel.diff` preserves its implementation.

## Qwen3.5 image compatibility

Actual Qwen3.5-0.8B image inference succeeded on vLLM-Metal, using the same
existing checkpoint and RealWorld-04 image. The first probe reused its image
identity; those warm timings are not a fresh-frame baseline. The corrected
probe assigns a unique multimodal UUID per request, which identifies all
vLLM image/embedding caches, while disabling processor cache, prefix reuse
and speculation. It executed three requests with 79 prompt tokens and 54
visual tokens each, matching the native 288x192 profile's token geometry.
All produced the same 28-token completed caption and shut down cleanly.

The fresh-process first-use TTFT was 264.29 ms. The two loaded fresh-frame
TTFTs were 195.81 and 195.82 ms, with total response times 751.88 and
748.76 ms. Initialization and full per-step outputs are saved separately in
`u32-decode/vllm-qwen35-fresh-image-probe-20261012`. The earlier probe's
4.42-second first use includes a different compilation/cache state; these
are fresh-process measurements with filesystem/compiler caches unflushed.

These are compatibility measurements, not a matched Garnet speed claim.
The comparator opens/preprocesses the source image inside its request timer;
the previous native image harness consumes prepared FP32 patches. A shared
preprocessing/timing boundary and repeated native/comparator VL comparison
are still required. Internal vision arithmetic also differs (native FP32
versus comparator BF16) despite identical BF16 checkpoint files.

## Optional MPS prefill and operation profiling

`mps-prefill/vllm-matched-mps-prefill-20261012` executed another 48 matched
requests with exact token outputs and clean shutdown. The optional `mps`
prefill policy expands original BF16 weights exactly into FP32 MPS matrices;
decode retains BF16 weights and the default policy remains `shared`.

| Model | Batch | MPS-prefill Garnet TTFT, ms | vLLM TTFT, ms | Native process RSS, GiB |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 113.91 | 49.54 | 3.00 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 67.54 | 34.56 | 2.38 |
| Qwen3-0.6B | 2 | 181.87 | 64.93 | 3.11 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 84.99 | 37.74 | 2.39 |

Warm TTFT falls 4.1–25.4% against the visible-cache shared-weight profile,
at roughly twice its native RSS. Decode does not improve. First-use conversion
cost remains in the first request's timing. Neither comparator target is met;
the three-model resident-memory scenario has not been validated with this
optional policy.

An opt-in serialized-operation diagnostic (`GARNET_METAL_PROFILE_OPS=1`)
passed the default matrix test and retained the exact 8-token native text
output. It deliberately submits and waits after each operation, so its timings
are not serving performance or a comparator result. Raw commands, operation
timings and aggregation are in `mps-prefill/op-profile-20261012` and
`mps-prefill/op-profile-build-checks-20261012`.

The diagnostic points to both projection and KV-update cost. Across this
22-token prefill plus 7 subsequent decode steps, prefill matrix operations
sum to 71.70 ms and chunk KV updates to 32.80 ms. Decode matrix operations
sum to 75.89 ms, single-token KV updates to 35.11 ms, and 8 vocabulary-head
projections to 23.89 ms. These isolated sums include different submission
overhead/cache behavior; they guide the next experiment rather than replacing
the normal end-to-end timers. The current chunk KV kernel scans all input
tokens separately for every output cache element, even for contiguous slots.

## Direct slots, aligned copies and private state-buffer handoff

The direct-slot path checks the host controls for a contiguous active prefix
before replacing the per-element chunk scan with direct lookup. Gaps,
overwrites and paused/resumed rows retain the original scan. All 11 checks
and 48 native/vLLM requests passed. A close native ABBA comparison then
confirmed text-model warm TTFT reductions of 9.3% (batch 1) and 8.1% (batch 2),
with unchanged decode speed. Coder reductions were 1.6% and 0.4%.
`direct-slots/` saves those commands, libraries, exact outputs, controls and
the tested source snapshot; these native ABBA gains are not vLLM-relative gains.

The next build adds bit-preserving float4 cache copies where alignment and
uint-index bounds permit, with the scalar fallback retained. Resident sessions
now exchange independently written output buffers with their private state
instead of copying every layer's entire state back. Storage aliases move
together; input/constant storage and overlapping output parts retain the copy
path. The private session state remains protected by the graph/session locks.

All 11 validation jobs passed, including paused rows, resets, chunk/profile
replacement, shared-graph session isolation and independent hidden snapshots.
The mixed image pipeline and pretrained layer/head references passed too.
Another 48 matched native/vLLM requests produced exact outputs and clean
shutdown. `state-handoff/` retains the complete evidence.

| Model | Batch | Garnet tokens/s per row | vLLM tokens/s per row | Garnet TTFT, ms | vLLM TTFT, ms |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 56.94 | 74.52 | 105.69 | 49.74 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 72.69 | 92.18 | 74.26 | 34.79 |
| Qwen3-0.6B | 2 | 51.88 | 75.85 | 188.86 | 70.82 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 70.44 | 92.25 | 112.33 | 37.99 |

Against the preceding direct-slot build, native decode improves about 42%
and 84% for text batch 1/2, and 7% and 16% for coder. Memory remains close to
the shared-weight profile. These are meaningful native improvements, but
Garnet still trails vLLM-Metal and every 2x/half-TTFT flag remains false.

## Remaining investigation

The native CLI cannot import NumPy in this SDK build. The actual preprocessing
probe failed before producing any patches; its raw failure is in
`direct-slots/xlang-image-preprocessing-probe-20261012`. A common CPU image
frontend feeding resident native/MLX workers is still needed to establish a
fair image-to-result timing boundary. The runtime/SDK were not modified.

The common frontend and persistent-worker comparison has subsequently executed;
see the matched VL section below. The failed direct-import probe remains intact.

Real full-head query/score samples from all three checkpoints are saved in
`state-handoff/head-bound-feasibility-20261012`. The basic whole-row Cauchy
norm bound prunes no rows in the sampled queries, even when supplied the
true maximum score. No pruning kernel or pruning speed advantage is claimed.
The approximate CPU normalization and 0.1% margin make this feasibility data,
not production proof. Further projection/attention optimization and the
matched VL comparison remain open. The full goal is active and unmet.

## Matched fresh-frame VL comparison

`vl-frames/vllm-matched-vl-file-events-20261012` executed 24 actual requests:
two fresh processes per runtime/batch, each with one first-use and two warm
frames. Every process exited cleanly. Both runtimes received identical resized
RGB pixels and the same 79 prompt IDs/54 visual tokens. Native patch bytes
match the canonical fixture exactly, in batches 1 and 2. Source preparation,
reset and worker transport are included in the parent monotonic first/done
timers; compiler/filesystem caches were not flushed. All learned native
operators execute in Garnet. The common CPU frontend uses the isolated
comparator Python for both runtimes, without importing an inference framework.

| Batch | Garnet tokens/s per row | vLLM tokens/s per row | Garnet TTFT, ms | vLLM TTFT, ms | Garnet total, ms | vLLM total, ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 43.47 | 48.59 | 698.28 | 200.47 | 1319.33 | 756.18 |
| 2 | 37.62 | 47.32 | 1186.70 | 327.12 | 1904.38 | 982.92 |

Batch 1 produced exactly the same completed 28-token caption in all 12
requests. Batch 2 native requests produced that same 28-token caption in
both rows, while comparator requests generated a different 32-token caption
and reached the common output cap. Thus batch-2 exact-output/completion checks
fail and its timing is a matched-input measurement with that quality/length
limitation. Neither profile meets either target. Internal native FP32 versus
comparator BF16 arithmetic is disclosed; identical checkpoints are not a claim
of bit-identical internal model execution.

Common source-image preparation costs about 50–54 ms per warm request. Native
batch-1 vision alone costs about 79–82 ms; most remaining TTFT is model prefill.
Each native frame is re-encoded and each comparator row/frame has a unique
encoder-cache identity. The first/done event protocol polls at 1 ms. Diagnostics
are read after the observed done event, so larger comparator logs do not
inflate its response timer. Per-worker times, all captions/token IDs, process
snapshots and input hashes remain available in `frame_summary.json`.

Two earlier attempts are retained: the system-Python frontend lacked Pillow,
and the native stdout ready marker was buffered. The second worker was stopped
through its verified owned control file before any request inference; the
interruption record and terminal failure are preserved. The successful run
uses the isolated frontend and event files. No SDK/runtime was changed.

The VL measurement gap is now closed. Its optimization gap and batch-2 output
discrepancy remain open, alongside the text/coder speed targets. The full goal
remains active and unmet.

## Rejected VL MPS-prefill and decode-geometry experiments

The optional MPS-prefill policy executed 24 fresh-frame requests in
`vl-mps-roofline/vllm-matched-vl-mps-prefill-20261012`. Batch-1 native TTFT
was 789.36 ms versus 698.28 ms with shared BF16 prefill; batch 2 was
1195.74 ms versus 1186.70 ms. Decode did not improve. The same batch-2
comparator caption/output-cap discrepancy persisted. The profile is not
selected for this VL workload; its raw results remain preserved.

`single-column/` retains a decode geometry experiment with one output column
per SIMD group and compile-time batch specialization. All 11 checks passed,
as did 48 native ABBA requests with exact outputs. Decode gains were only
1.0–2.7%, while TTFT increased 2.5–8.8%. The kernel was reverted.
`rejected_kernel.diff` records the implementation; these are native-control
results, not a vLLM advantage.

## Streaming constraint and exact packing feasibility

Apple advertises 120 GB/s for the M4 Mac mini:
[official specifications](https://support.apple.com/en-gb/121555).
The exact checkpoint headers identify 1,191,968,768 BF16 projection/head bytes
for text, 987,922,432 for coder and 1,503,789,056 for Qwen3.5 VL. Reading every
such value once per decode step gives optimistic streaming ceilings of
100.67, 121.47 and 79.80 steps/s, before other traffic or arithmetic. The 2x
comparator targets exceed these ceilings. This is a conditional streaming
roofline, not a universal algorithmic lower bound: caching, exact pruning or
lossless packing can change the assumed bytes read. Header inventories and
the calculation are saved in `vl-mps-roofline/decode-streaming-roofline-20261012`.

`probe_lossless_bf16_codec.py` implements a CPU-only exact-bit prototype:
8 sign/mantissa bits, 3 exponent tags, full exponent escapes and 64-bit
metadata per 256-value block. It round-trips all 65,536 bit patterns, including
signed zeros, subnormals, infinities and NaN payloads, plus non-aligned tails.
Every sampled BF16 matrix block from the real local Qwen3.5 checkpoint also
reconstructs exactly. The weighted sampled storage ratio is 1.3971x
(11.4521 bits/value, roughly 28.4% fewer bytes). This is not quantization.
The report is `vl-mps-roofline/lossless-bf16-codec-report.json`.

At the CPU-prototype stage, no GPU codec or packed-model speed gain had been
demonstrated. That prototype established representation correctness only.
The subsequent GPU checks are recorded below; full-model numerical and
performance validation remains required before a codec could count toward
the goal. All 2x/half-TTFT targets remain unmet.

### Subsequent exact GPU packing measurements

`lossless-gpu/` now closes the GPU representation-feasibility question for
four experimental codecs/schedules. All three checkpoints' complete heads
and first-layer down projections reconstructed exactly on the GPU; all 48
isolated projection comparisons were bit-exact. All four full-head profiles
were slower than the unchanged raw BF16 kernel, despite 22–27% fewer stored
bytes, and are rejected for integration. A small Qwen3 projection improved
in isolation; no full-model gain or generated-output validation is claimed.
The earlier CPU-only paragraph describes the state before these experiments.
Exact commands, source snapshots, hashes, timing windows and limitations are
retained in `lossless-gpu/README.md`. Production native code was not changed;
the matched throughput and TTFT targets remain unmet.

### Partial output-head bounds on actual native queries

`partial-head/` captures actual native normalized head inputs and full GPU
logits from all three models in batches 1/2, including real-image VL inference.
All six complete output sequences match the saved native baselines. Every
tested partial Cauchy bound covers every actual GPU logit across 72 sampled
row/step queries; the earlier approximate CPU-normalization limitation is
removed. Offline fixed-seed estimates suggest 23–36% fewer head weight reads
at 32-value block granularity, before bound/gather/dispatch and batch-sharing
costs. At this stage no pruned GPU kernel or full-model speed gain had been
demonstrated. Exact sources, failed setup attempts, commands, full-logit hashes and
limitations are saved in `partial-head/README.md`.

### Actual GPU partial-head measurements

`pruned-head-gpu/` executed four layouts/cadences, including actual seed
projection, GPU norm/bound preparation, shared batch reads and the native
argmax. All 204 head checks passed: raw and retained logits are bit-exact,
pruned candidates cannot beat the observed seed, and selected tokens match
the complete native head. The checks include 312 row/state instances and
mixed-state batch pairs. They do not constitute accelerated full-model output
or serving validation.

The best text batch-1 component ratio was 1.076x; most other profiles were
near break-even or slower. Lower weight-element visits did not yield a broad
advantage once bound costs were timed. The profiles are rejected for broad
integration. No production source was changed; full-model throughput and
TTFT remain at the previously measured values, with all targets unmet.
Commands, source snapshots, timings and limitations are in
`pruned-head-gpu/README.md`.

### Validated wide prefill tiles

`prefill-tiles/` records five exact-value tile experiments, 840 bit-exact checks,
all 11 native validations, and 144 actual full-model requests across native
controls and updated vLLM comparisons. The selected wide tiles reduce warm
TTFT by 8.9% for text batch 2, 10.3% for coder batch 2 and 4.5% for real-image
VL batch 2 in close native ABBA tests; batch-1 changes are smaller. All native
control outputs remain exact and complete. Decode speed is effectively
unchanged. The candidate is retained in `build-prefill-wide`; the prior
`build-state-handoff` remains intact as its control.

The updated matched vLLM runs still fail every 2x/half-TTFT target. The VL
batch-2 different/capped comparator caption also persists. The full objective
remains active and unmet. `prefill-tiles/README.md` provides the current selected
implementation, complete metrics, precision limits, source snapshots and
first-use/warm accounting.

### Rejected whole-decoder lossless packing

The subsequent `../../20261013/decoder-codec/` evidence tested exact packing on
29 real projection matrices, then an actual native decoder implementation.
All 58 matrix checks, all 11 native validations and all 72 native-control
requests passed with exact outputs. The isolated 1.66–1.93x projection ratios
did not transfer to full inference: text gained about 4–5% decode, coder
batch 1 about 3%, while text/coder TTFT, initialization and memory regressed.
VL decode slowed about 2%. The implementation and CMake option were reverted.
`build-prefill-wide` remains the selected native candidate; all full-goal
targets remain unmet. The rejected source, commands and complete results are
retained in the later report.

### Rejected compute encoder reuse

`../../20261013/encoder-reuse/` records a serial shared-encoder prototype, all
11 existing validations, a new custom-compute abort/recovery test, and 72
actual native ABBA requests. All outputs matched, but decode/TTFT changes were
under about 0.5%. CPU encoding saved roughly 0.08 ms against 13–18 ms of GPU
work/wait per decode step. The reuse implementation was reverted; the focused
transaction test is retained and passed on the selected wide-prefill build.
GPU execution and data movement remain the larger targets. The full objective
remains active and unmet.

### Validated vector state slices

`../../20261013/vector-slices/` records serialized operation attribution,
all 11 native checks, 192 independent bitwise slice checks, 72 native ABBA
requests and 72 fresh matched vLLM-Metal requests. `build-vector-slice` is now
selected. Aligned state slices copy four floats per thread, with bounds/layout
guards and the scalar fallback. Actual image TTFT falls 11.3%/19.6% and decode
gains 4.4%/10.8% versus the wide-prefill control at batch 1/2; text/coder decode
is unchanged. All native control outputs match exactly. The updated matched
image batch-1 result is 45.67 versus 48.06 tokens/s, with TTFT 614.39 versus
204.99 ms. Image batch 2 retains its different/capped comparator caption.
All six profiles still fail both targets. Same BF16 weights and the native
FP32 versus comparator BF16 internal arithmetic limits remain explicit.
The objective remains active and unmet.

## Comparator teardown compatibility

The preceding 48-request suite in
`baseline/vllm-matched-native-syntax-baseline-20261012` generated identical
outputs but its vLLM workers crashed during shutdown. Its derived summary
flags `clean_backend_shutdown=false`; a parent exit code of zero does not
make that suite clean. Raw failures remain intact.

The private comparator patch skips Torch accelerator cache cleanup for
platforms whose Torch device type is CPU. The Metal plugin uses that metadata
for compatibility while inference runs on MLX. The original vLLM cleanup
checked only its OOT platform enum and entered the Torch accelerator cleanup
path incorrectly. No inference, scheduler or numerical code was changed.
`baseline/vllm-cleanup-compatibility-20261012` contains the exact one-condition
diff, original/patched SHA-256 and provenance; the second full suite proves
clean shutdown. `patch_vllm_metal_cleanup.py` verifies the official wheel's
original source hash and atomically patches only this isolated installation.

`setup/` retains completed setup attempts and the verified wheel-transfer job.
These are installation/network diagnostics, excluded from all inference timing.
The cached-URL setup was interrupted deliberately after verified local wheels
became available; its separate `interruption.json` records that action. A retry
with an incomplete UV copy failed the size/digest guard before installation.

`baseline/` retains the later setup, download, compatibility and benchmark
attempts, including interrupted transfers and early guard/parser failures.
The successful offline installation resolved 161 packages. Verified public
wheels, requirements and installed versions are recorded in the manifests.
All setup/download time is excluded from inference measurements.

The runner files have now passed actual native and vLLM engine execution. See
`test2026/metal/VLLM_COMPARISON.md` for matching, accounting and precision limits.
The GPT-OSS and Pi checkouts were checked clean and were not edited.
