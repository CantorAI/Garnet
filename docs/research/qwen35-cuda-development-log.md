# Qwen3.5 CUDA implementation and optimization log

## Objective and acceptance criteria — 2026-10-22

Implement and validate Qwen3.5-2B and Qwen3.5-4B on Garnet CUDA. Preserve Metal
behavior and Qwen3-VL-2B regression coverage. Compare with LM Studio/llama.cpp
and CUDA vLLM, one model and one engine at a time on the same hardware.

User performance target: at least **2x faster**, primarily full-request latency
at most half each matched baseline, while preserving answer quality. Decode
throughput is a separate metric. Match checkpoint/precision, actual token work,
images/resolution, prompt/template, thinking mode and cache controls. Report
unmatched conditions instead of presenting them as equivalent comparisons.

User memory requirement: avoid the previous high-memory implementation. Aim
for model-owned memory close to the matched llama.cpp baseline. Prefer one
immutable copy of weights, BF16 storage and bounded reusable buffers. No exact
accepted GiB ceiling was specified. Report speed/memory tradeoffs explicitly.

User also requires batch profiles. Start with batch 1/2/4/8 as capacity permits,
with explicit input/visual/output/context limits and matching baseline profiles.
Record complete-request latency, aggregate output tokens/s, per-request decode,
correctness, peak/resident allocations and any OOM admission rejection. Keep
interactive latency and batch aggregate throughput claims separate. Profile
transitions must release prior engines/state; do not benchmark co-resident
models, engines or batch profiles.

### Evidence rules

For each method, append its hypothesis, exact change and source/build identity,
control/candidate settings, correctness gate, warmup and measured samples,
latency/TTFT/decode rates, memory boundaries, result, and retain/reject decision.
Preserve failed attempts. Link raw evidence; do not replace unfavorable results
with only the best runs. Separate compiler/component tests from full inference.
Allocation counters and whole-GPU samples do not alone establish process peaks.

Workspace evidence: `D:/CantorAI/work/qwen35-cuda/`. Prior regression baseline:
`D:/CantorAI/work/lmstudio-comparison/REPORT.md` and raw JSON/CSV files beside it.
Generated weights, engines, installers and private images remain outside Git.

## 001 — Audit existing implementation

- Starting Garnet checkout: `c66161dbfdfdbe98cb22a4921492ed8c2c4fe770`.
- Existing `xModel/qwen35/vl_4b/model.py` imports MetalDecoder and
  MetalImageEncoder. Its README and serving optimizations target Metal/MPS.
- TensorRT has no Qwen3.5 hybrid-attention lowering at this starting revision.
  CUDA support must be implemented and verified; the existing package is not
  proof of CUDA execution. No 2B package existed.
- Qwen3.5-2B official config confirms 24 decoder layers, hidden width 2048,
  8 full-attention query heads/2 KV heads, 16 linear key/value heads, and a
  24-layer 1024-wide vision encoder. Model source: `Qwen/Qwen3.5-2B`.
- Host: Windows, two RTX 4080 16 GiB GPUs, driver 617.14. Benchmarks use only
  device 0. WSL Ubuntu can access the same CUDA devices, allowing CUDA vLLM.
- Decision: implement backend support over the shared Qwen3.5 tensor programs;
  do not label a framework fallback or Metal result as native Garnet CUDA.

## 002 — Prior Qwen3-VL-2B regression baseline (not a Qwen3.5 result)

- Same checkpoint converted locally to BF16 GGUF; four 320x192 image families,
  60 visual/76 prompt tokens, exactly 32 generated tokens; eight warmups and
  twenty measured requests per engine. Native Garnet versus local LM SDK.
- Median caller latency: Garnet 292.40 ms, LM Studio 291.50 ms. Engine-reported
  decode: 113.96 versus 131.15 tokens/s. **2x target not achieved.**
- Whole-GPU sampled peak: 12,886 versus 7,141 MiB. Garnet capture includes
  compilation and background graphics; generation remained around 12.3 GiB.
  This is not isolated resident/process/shared-memory accounting.
- Checkpoint accounting estimates 8.327 GiB for duplicated text-engine weights
  plus vision, including FP32-expanded embedding. Runtime buffers remain
  unassigned. Estimated KV arrays are only 14 MiB for this shape.
- LM full-prompt cache reuse invalidated initial repeated-image TTFT results.
  Final paired corner-pixel variants prevented whole-image/prompt hits; LM
  still reused a common 10-token text prefix. Timing boundaries differ.
- Tested Garnet binary was from October 7, not rebuilt from the starting
  checkout. Keep it as historical evidence, not current-source proof.
- Evidence: the prior report and its `summary.json`, request JSON, and GPU CSV.

## 003 — 2B contract and allocation diagnostics (in progress)

- Added a 2B model contract and Metal serving routing. Architecture validation
  imports no GPU backend and rejects wrong grouped heads, attention layout and
  vision output width before graph allocation.
- Validation: four new 2B contract tests and existing 4B contract test passed
  with `Garnet/.venv/Scripts/python.exe`. This validates contracts only.
- Added opt-in `GARNET_TRT_LOG_ENGINE_MEMORY=1` allocation checkpoints before/
  after engine deserialization, refit and each context creation. Include engine
  path, CUDA device used/total bytes, context memory upper bound and pool count.
  No synchronization or diagnostic output is introduced with logging disabled.
- Hypothesis: duplicate weights, FP32 promotion and execution-context buffers
  explain the high footprint. Context pools can grow to eight slots; actual
  growth and memory effects must be measured, not assumed.
- Build attempt using the old standalone build cache failed at configure:
  its architecture selection excluded all devices admitted by GPT-OSS kernels.
  No benchmark resulted. A dedicated SM89 build is being configured in
  `work/qwen35-cuda/build`, without the unrelated GPT-OSS extension.
- Official 2B checkpoint download completed after disabling the stalled Xet
  transport; local safetensors file size 4,548,221,488 bytes. Download logs are
  retained. WSL vLLM setup uses a separate environment; system Python and the
  existing TensorRT-LLM environment are unchanged.
- Started primitive TensorRT Qwen3.5 lowering: BF16 projection constants,
  explicit FP32 recurrent state/reductions, stable gates and inactive-row
  masking. Compilation, numerical parity, full attention, vision and serving
  remain pending. No CUDA support/performance claim is made yet.
- Status: **in progress; no accepted CUDA optimization yet**.

## 004 — Native CUDA graph boundary and operator correctness

### Method

Added the CUDA `compiled_graph` implementation over Garnet's captured tensor
graph and TensorRT lowering. Added Qwen3.5 RMSNorm/MLP, explicit recurrent
linear attention, grouped gated full attention with partial interleaved MRoPE,
and the still-image tower (position interpolation, vision RoPE, attention,
LayerNorm and merger). Large projection constants stay in checkpoint BF16;
state, norms and reductions are FP32. Request state is explicit. The initial
graph boundary copies outputs to host for existing state orchestration; this
is a correctness foundation and **not an accepted final performance design**.
Resident state/session orchestration and chunked prefill remain required.

### Failed attempts and corrections

1. The dedicated VS2022 build configured after adding Cargo to PATH, then
   failed because XLang3 uses `regex_constants::multiline`, absent from that
   installed MSVC standard library. Switched to existing VS18/MSVC14.51;
   source was not patched to remove regex behavior. SM89/CUDA13.2 build passed.
2. Full-attention FP32 batch4 exceeded the strict reference tolerance, e.g.
   -0.02162255 versus -0.02157698. Disabled TensorRT's default TF32 only for
   this strict CUDA graph path; other existing model paths keep their policy.
   Retained evidence: `attention-b4.log`.
3. Initial BF16 fixture references performed their projections in FP32. Added
   a PyTorch reference with explicit BF16 input/output projection rounding,
   FP32 state/norms/reductions, and a cross-check against the independent NumPy
   algorithm for FP32 cases. Retained `*-pre-reference-fix.log` files.
4. A remaining BF16 discrepancy was isolated to TensorRT's matrix projection.
   Identity projection proved input casting matched; ordinary QKV projection
   differed from both CPU and GPU PyTorch by up to 0.00048828125. This isolates
   a precision difference but does not prove its internal tactic mechanism.
   Retained `projection-debug.log`, `projection-debug-trace.json`, and
   `linear-failure-trace.json`; no tolerance was loosened to accept the result.
5. Added a projection plugin using cuBLAS with explicit FP32 accumulation,
   BF16 inputs/weights/output, and one shared device-scoped handle across
   plugins. No full weight conversion to FP32. The handle's enqueue state is
   mutex-protected and scratch is disabled across context streams. Its owner
   is shared by live plugins and released when the last owner retires.

### Component result (before final scratch-policy recheck)

- Linear: 8 cases (FP32/BF16 × batch1/2/4/8), 3 state steps each: 24 passes.
- Full attention: same cases/steps: 24 passes.
- Vision: FP32 weights, batch1/2/4/8, square and nonsquare grids, 2 inputs each:
  8 passes. BF16/pretrained vision validation is still pending.
- All 56 checks used absolute tolerance 2e-5 plus relative tolerance 2e-4.
  Maximum observed absolute error across these fixtures was below 1e-6.
- Explicit completion markers were checked, not just shell exit status.
  Native test runner now records case starts/failures and completion counts.
- These are small independent fixtures; they do **not** prove pretrained
  token trajectories, real-image answer quality, end-to-end CUDA support,
  speedup, or full-model memory savings.
- Evidence: `linear-parity.log`, `attention-parity.log`, `vision-parity.log`,
  fixture JSON/checkpoints, `build-current.log`, `identity.json` under workspace.
- Source tests and fixture generators: `test2026/cuda/`. Batch workload plan:
  `test2026/cuda/qwen35_benchmark_profiles.json` (not an admission-tested serving
  declaration). Build identity is a working-tree hash set, not the base commit.
- Decision: retain as correctness groundwork; performance acceptance pending.

Follow-up: the final shared-handle scratch policy passed six additional BF16
step checks (linear/full attention, batch8) with the strict thresholds and
completion marker. Evidence: `scratch-policy-parity.log`. No large FP32 weight
copy or per-projection cuBLAS handle was introduced.

Final component rerun on the bounded-static-shape build: 24 linear checks
(max1.49e-8),24 full-attention checks(max2.38e-7),8 FP32 vision checks
(max8.95e-7),all with completion markers. Six additional FP32/BF16 attention
checks passed with partial rotary width1/4 of the head dimension and untouched
tails(max2.38e-7). `component-summary.json` records all68 checks including the
six scratch-policy regressions. The fixture generators and runner are included
in the source identity hashes. These remain component—not model—results.

### Method 005: resident CUDA decoder and first pretrained text smoke

Implemented reusable device outputs, GPU embedding/tied BF16 head sharing,
GPU state and hidden-buffer copies, native JPEG preprocessing, and a direct
CUDA serving module for 2B/4B. Prefill currently loops token steps; this is
correctness groundwork and still requires optimization and public API routing.
Independent I/O parity and a two-layer recurrent/full-attention decoder fixture
passed with explicit completion markers (`io-parity.log`, `decoder-parity.log`).
The decoder fixture includes ragged active rows and reset/replay.

Official Qwen3.5-2B text-only smoke completed successfully on GPU0:
`pretrained-2b-text.json` and `.log`, marker
`QWEN35_PRETRAINED_CUDA_COMPLETE`. Load/compile 59.51 seconds; 22 prompt tokens,
8 generated tokens, 0.468 seconds full request, 0.408 seconds TTFT. Output:
"I can generate text, analyze information,". Diagnostic logging was enabled;
this is a single smoke run, not a benchmark or quality acceptance. Device-used
snapshot was 5,347,213,312 bytes; shared embedding/head table 1,017,118,720 bytes.
The device snapshot is not process-isolated peak memory and excludes vision.
No speedup or final memory claim follows from this measurement.

User reiterated batch profiles: preserve separate batch1/2/4/8 correctness,
shape bounds and admission, latency, aggregate throughput, and steady/peak
memory results. Run model/engine/profile cohorts sequentially, unload before
switching, and do not count batching throughput as 2x single-request latency.
`test2026/cuda/qwen35_benchmark_profiles.json` is the workload plan; full-model
batch2/4/8 validation and realistic image profiles remain pending.

### Method 006: pretrained image and ragged batch validation; precision disclosure

2B native real-image run completed (frame_0.jpg, grid1x12x20, 60 visual tokens,
85 prompt tokens, 16 output tokens). First-use request19.56s includes18.57s
vision compilation and must not be used as a warm benchmark. Evidence:
`pretrained-2b-image.json`/`.log`. Native patches exported diagnostically outside
measurement (`native-image-patches.json`, `export-patches.log`).

Independent Transformers5.13/PyTorch2.11 eager BF16 on those exact prompt IDs
and patches shared9 tokens then added "wooden". Transformers with explicit
BF16 projection input/output and FP32 accumulation/residual/state boundaries
matched all16 Garnet tokens (`reference-2b-image-mixed.json`). This proves only
this small trajectory. CUDA result metadata now explicitly records arithmetic;
BF16 weights do not imply full-BF16 intermediate parity. Reference mixed mode
uses FP32 model parameter storage for diagnostics and is not a memory/perf
baseline. All reference scripts and native exports are retained.

Batch2 profile (capacity384, maxinput256/output128, same65536 pixel budget)
completed ragged85/79-token image prompts twice,16 outputs per row, exact
reset/replay. Second request0.873s includes0.721s TTFT, diagnostic logging and
CIM polling, so this is not accepted performance evidence. No vision compile
on second request. Batch2 first row differs from batch1. Independent mixed
Transformers batched/chunk-prefill reference matches12/16 and16/16 tokens;
serial-prefill reference matches16/16 and6/16. Row1 run alone matches16/16.
Shape/padding/serial-vs-chunk numerical behavior still needs quantitative
teacher-forced validation; full batch correctness is NOT declared passed.
Evidence: `pretrained-2b-image-b2*`, `reference-2b-image-b2*`.

Added diagnostic WDDM process-memory sampler. Raw adapter identity rows retained
in `pretrained-2b-image-b2-memory.jsonl`; summary does not aggregate adapters.
Polling began during compilation; maxima are sampled observations, not full
process-lifetime peaks or isolated steady inference memory. The TensorRT logs
and I/O-owned buffer counts are separate scopes. Native I/O parity rerun passed
including newly added row and stack/reorder checks.

Official4B checkpoint download completed at pinned HuggingFace revision
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a (provenance inside checkpoint directory).
2B and4B contract tests passed. Existing4B Metal factory behavior preserved;
imports made lazy so CUDA does not import Metal implementation unnecessarily.
4B native text smoke is the next validation; no4B performance/quality claim yet.

4B follow-up: native text smoke completed successfully with22 prompt/16 output
tokens; Transformers eager BF16 independently matched all16 generated tokens.
Evidence: `pretrained-4b-text.json`/`.log`, `reference-4b-text.json`/`.log`, explicit
completion markers. One text trajectory is not full model validation. Native
text load/compile83.71s, first request0.889s, no matched benchmark conclusion.
Device-used snapshot10,277,617,664 bytes is whole-device, not process allocation.
WDDM raw sampled process rows retained in `pretrained-4b-text-memory.jsonl`.
A separate4B real-image/reset run started only after both text processes exited.

### Method 007: pretrained 4B image/reset and independent serial reference

4B native real-image run completed twice with exact reset/replay token equality:
`pretrained-4b-image.json`/`.log`,85 prompt tokens including60 visual tokens,
16 output tokens. Warm diagnostic request1.508s with zero vision compilation.
This is still a smoke (not a matched benchmark); serial prefill is slow.
Raw process adapter samples: `pretrained-4b-image-memory.jsonl`, sampling began
late during load. Load/request timestamps are emitted by the runner for future
phase correlation. Whole-device snapshot11,011,620,864 bytes is NOT process VRAM.

Standard eager BF16 Transformers and mixed/chunk-prefill references diverged
after6 output tokens; the mixed **serial-prefill** independent Transformers
reference matched all16 tokens (`reference-4b-image-mixed-serial.json`/`.log`).
The matching serial reference uses official Transformers attention/recurrent
architecture, explicit same projection/state arithmetic, native-exported input
patches and recorded prompt IDs. This validates only this short trajectory,
not a broad quality suite. Shape/prefill numerical effects remain material.

Improved the mixed reference to retain large matrices and tied table as BF16,
converting only small norm/conv parameters to FP32 and embedding outputs to
FP32. This avoids a full FP32 4B weight copy that would exceed16GiB. Earlier
2B all-FP32-parameter mixed reference measurements remain retained and must not
be treated as memory baselines. Reference diagnostics are not speed baselines.

Added `summarize_qwen35_allocations.py`: preserves per-engine stage snapshots,
observed whole-device deserialize/refit/context deltas, and context estimates
with explicit scope. It parsed26 engines for2B image and33 for4B text. Memory
acceptance still requires isolated steady/peak comparisons at matched work.
Runtime now bounds requested image grids by pixel profile and reports actual
per-row visual-token counts (text-only requests after image use must report0).
No model engine remains loaded after these tests.

### Method 008: public CUDA serving integration and batch4/8 image profiles

Connected Qwen3.5-2B/4B to native `serve_model`, `serve_status_json`,
`list_loaded_models_json`, `infer_json` and `stop_serving`. Existing instance
map/lifecycle is reused; Metal and existing CUDA model routes are preserved.
Requested root/profile changes on a loaded Qwen3.5 model require explicit stop;
output limits are rejected rather than clamped. Explicit lists serve the loaded
batch; no silent request replication. Status discloses profile, batch and mixed
precision. Python runtime is resolved beside the supplied xModel tree, with a
source-checkout fallback. Public batch8 test validates load/status/list/idempotent
load, conflicts, oversized output and batch mismatch rejection, two real-image
requests with reset/replay, and stop/unloaded status. Completion marker present:
`public-2b-b8.log`/`.json`. Build succeeded in `build-current.log`.

2B real-image batch4 smoke also completed twice (ragged85/79-token rows), exact
reset/replay; `pretrained-2b-image-b4.json`/`.log`. Both batch4 and batch8 generate
16 tokens per row. These tests establish executable bounded profiles and replay,
not broad numeric/quality acceptance. Independent ragged batch reference and
full-capacity admission remain pending. Single short trajectory success is not
an arbitrary256-input/128-output boundary proof.

Fixed JPEG pixel-budget semantics: `prepare_image` now supplies1024 minimum
(merged32x32 patch) and the requested maximum, rather than equal min/max that
could upsample small already-resized images beyond budget. Runtime validates
minimum budget and rejects grids above maximum. Large source images in prior
smokes keep the same12x20 grid. New binary built after the fix. A dedicated
small-image check remains pending after the active engine exits.

Added native Garnet and WSL vLLM real-image benchmark harnesses. Garnet2B batch1
cohort completed28 requests (8 warmups,20 measured), identical prompt/output
limits,60 visual tokens, precompiled vision. Median full API wall834.26ms,
TTFT569.16ms, decode117.66tok/s. Actual outputs include31 and32 tokens, so a
comparison must inspect output-work equality. Raw `benchmark-garnet-2b-b1.*`.
Startup emits unsupported optional CPython hashlib-algorithm warnings; SHA256
request hashes and all requests complete. Timing excludes load; warnings are
retained, not suppressed. This is an initial baseline, not speed acceptance.

vLLM2B batch1 cohort then started only after Garnet process exit. It uses
BF16 model, FP32 recurrent-cache types, explicit96MiB KV/cache budget,384 context,
max1 sequence,32 output limit, min1024/max65536 pixels, prefix cache off and
processor cache0. Arithmetic/JPEG preprocessing differences remain disclosed;
actual prompt IDs and outputs must be audited before speed ratios. Reserved
cache configuration is separate from process memory. Frontend torch allocator
counters do not cover the engine worker and are not total vLLM VRAM evidence.
LM Studio2B original BF16 text/mmproj GGUFs imported by copy; no LM engine loaded
concurrently. CLI import/load help was inspected locally. Full baseline and
memory comparisons are still pending.

### Method 009: sequential three-engine initial baseline and memory RPC diagnostics

All primary2B batch1 common-variant cohorts completed28 requests with explicit
completion markers. Inputs are byte-identical320x192 JPEG variants with a
per-round corner pixel change,4 image families,2 warmup/5 measured rounds.
Hashes/transformation provenance: `inputs/variants/manifest.json`. All engines
use original BF16 source/derived BF16 GGUF weights, nonthinking greedy sampling,
32 output limit. Garnet/vLLM prompt IDs match on all28 requests,85 tokens each
including60 visual tokens. vLLM reports0 prefix-cache hits. LM Studio native
REST reports85 input tokens and0 reasoning output tokens. Its runtime logs show
60 image tokens each request; short text-only prefix reuse remains unquantified.

20-request medians: Garnet821.63ms wall /560.17ms TTFT /119.20tok/s decode;
vLLM243.75ms /39.25ms /152.99tok/s; LM Studio297.98ms /45.71ms /136.66tok/s.
Some vLLM outputs reach EOS earlier (23..32 sampled tokens). On13 measured
cases with equal sampled-token work across all three engines, wall medians are
Garnet820.60ms, vLLM246.45ms, LM Studio298.81ms. These are initial baselines,
NOT speed acceptance. Garnet is slower, with serial prefill the largest gap.
Raw artifacts, limitations and generated statistics are in
`initial-batch1-comparison.json` and `INITIAL_BASELINE.md`. TTFT boundaries differ
by engine; LM actual prompt IDs unavailable, requested384 context rounds to512,
JPEG implementation/intermediate arithmetic differs. Full matching remains
qualified by these limits. Native result metadata now explicitly records
sampled tokens including EOS, stop token/reason, and decode steps; retained
older batch1 sampled counts are inferred from its EOS-only early-stop behavior.

vLLM startup failed first with system nvcc12.0 (unknown --compress-mode=size).
Pointing CUDA_HOME at the isolated environment's13.4 compiler then failed due
mismatched13.0 runtime headers. Pinned nvidia-cuda-nvcc/crt/nvvm13.0.88 alongside
runtime13.0.96 in the isolated venv; no system toolkit changed. Successful
cohorts use CUDA_HOME explicitly. Failed logs are retained in
`benchmark-vllm-2b-b1.log`, `*-cuda13.log`; compiler install log retained. Native
model MTP weights are present in the checkpoint, but all these baselines disable
speculation. No MTP speed claim follows from presence of weights.

LM Studio older SDK/manual-template attempt failed the work gate:83 prompt
tokens, reasoning output. Excluded; `benchmark-lmstudio-2b-b1-variants.*` retains
failure. Official native REST schema reasoning="off",store=false solved it;
`benchmark-lmstudio-2b-b1-rest-variants.*` is successful. SDK1.5 configuration
mapping does not expose the newer enableThinking prediction key. Official
schema references: https://lmstudio.ai/docs/developer/rest/chat and
https://github.com/lmstudio-ai/lmstudio-js/blob/main/packages/lms-kv-config/src/schema.ts .
All engines were unloaded/exited before starting the next engine. Imported LM
model CLI key is `qwen3.5-2b`; requested publisher/repo path was not a CLI key.

Separate vLLM diagnostic run uses a named worker extension and scalar RPC to
measure the engine worker allocator, resetting peaks after warmup. Direct
callable RPC initially failed serialization (`memory-vllm-2b-b1-variants.log`);
replaced by named `qwen35_vllm_memory.MemoryExtension` and method string, retaining
the default serialization policy. Successful12-request diagnostic evidence:
`memory-vllm-2b-b1-named-rpc.json`/`.log`. Worker steady allocated4,605,155,328B,
reserved4,729,077,760B; post-warmup peak allocated4,614,088,704B. Scope is the
worker torch allocator, excluding possible CUDA driver/library allocations.
Whole-device snapshot6,207,045,632B includes desktop activity and is separate.
This scope differs from Garnet WDDM counters; do not rank cross-engine process
VRAM from those unlike measurements. Comparable isolated steady/peak memory
acceptance remains pending.

### Method 010: unified bounded CUDA chunk, no second decoder engine set

Implemented opt-in prefillChunk1/4/8/16 profile (default1). A single static chunk
engine per layer serves both prefill and padded single-token decode. Linear
attention input/output projections and MLPs span the chunk; convolution/GDN
state evolves token order. Full attention is initially serial within each
chunk, with vectorized layer norms/MLP. This is an intermediate implementation,
not final speed acceptance. Metal TE programs/factories remain unchanged.
GPU token scatter and last-active extraction preserve ragged/padded rows and
explicit caller-owned states. No CPU hidden/state boundary was introduced.

Factored linear-step recurrence into shared primitive core and reran all24
linear regression checks (8 cases,3 steps), all pass. Chunk4 and8 synthetic
2-layer PyTorch/NumPy reference tests pass13 strict hidden/state checks each,
max absolute error1.49e-7. Explicit markers checked: `chunk4-parity.log`,
`chunk8-parity.log`, `linear-after-chunk.log`; I/O chunk scatter/last-active
checks pass in `io-chunk-parity.log`. Final chunk4 test also fills actual cache
through its final valid slot, rejects overflow without mutating hidden/state or
poisoning request; `chunk4-capacity-parity.log`.

Added opt-in CUDA graph enablement to compiled_graph.run_device, retaining
existing fallback behavior but exposing capture/replay/failure counters. Added
byte accounting to the existing optional TRTSyncAllocator (enabled only with
GARNET_TRT_SYNC_ALLOCATOR=1), per-graph stats, and retained decoder/model ownership
reports. Untracked allocators return null byte counts, never false zero values.
Unit fixture verified actual capture/replay and0 failures, same predictions and
states; `chunk4-graph-parity.log`. Build passed after fixing a missing JSON
include in the new stats code; `build-chunk-graph.log`. Counts/bytes exclude
external allocations and are not process peaks.

### Method 011: chunk4 pretrained correctness, timing and allocation evidence

Initial2B original-image smoke completed twice, same reset/replay tokens.
Warm16-output request425.54ms:228.16ms prefill,186.18ms decode. Decode regressed
with padded chunks. This was a smoke, not a speed claim. Evidence:
`pretrained-2b-image-chunk4.*`.

Graph-off/on diagnostic then ran sequentially on one loaded2B model. Graph-on
median of2 measured repeats558.33ms vs618.87ms graph-off at32 outputs. Captures25,
replays5166 and0 failures at diagnostic end; tokens identical across modes and
all repeats. Retained owned bytes4,716,914,200. Independent Transformers mixed
chunk-policy reference matched all32 output tokens using exact recorded prompt
IDs and native variant0 patches (`reference-2b-chunk4.*`). This reference keeps
full-attention execution serial within a chunk and pads BF16 GDN/MLP projections
for single-token calls, preserving the candidate's arithmetic shapes. It does
not prove full-BF16/broad quality equivalence or arbitrary batch/capacity support.

Full common-image cohort completed28 requests (8 warmups/20 measured), same
four input families and settings as initial baselines. Median wall554.12ms,
prefill202.80ms, TTFT210.52ms, decode90.40tok/s.13 equal32-sampled-token cases:
chunk4/graph553.02ms; old serial820.60ms; vLLM246.45ms; LM Studio298.81ms. Still
slower than competitors. Decision: retain explicit experimental profile, not a
general default; longer outputs may lose the prefill gain.2x target NOT achieved.

Allocation snapshots were stable across all20 measured requests. Active BF16
checkpoint payload4,426,488,512B (MTP excluded); owned model4,716,914,200B is1.0656x
payload. Text-layer weights2,746,532,544B; text engines/final norm/context TRT
runtime allocations2,958,843,588B. One shared embedding/head table1,017,118,720B.
These values support no second full text-weight copy. Totals include retained
request/engine output tensors and tracked TRT allocations; they exclude possible
driver/library allocations, graph executable overhead, transient requests and
builder workspace. Do not call this isolated process peak or directly rank it
against unlike counters. Per-engine peaks must not be summed as process peak.

Evidence/reproduction: `CHUNK4_CANDIDATE.md`, `chunk4-graph-comparison.json`,
`benchmark-garnet-2b-b1-chunk4-graph.*`, `chunk4-graph-probe.*`,
`weight-ownership-2b.json`, and source summarizers. All GPU runs remained serial.
4B, optimized batch2/4/8 competitive cohorts, broader quality/capacity, matched
process-memory and existing3-VL/Metal regressions remain pending. Next performance
work must address inefficient padded decode and remaining recurrent/full-attn
execution; do not relabel the serial-relative gain as the competitor2x goal.

### Method 012: rejected TensorRT conditional and fused delta prototype

Tried one-engine chunk decode with a single-token IIfConditional branch. Native
build succeeded (`build-single-branch.log`), but the fixture engine build failed:
TensorRT Error9 PLUGIN_V2 operation not supported within this graph, then Error10
no implementation. Source snapshots and binary hash are retained under
`work/qwen35-cuda/failed-single-branch/`; raw failure is in
`chunk4-single-branch-parity.log`. Removed the unsupported branch from source;
no conditional profile is admitted. Rebuilt successfully in
`build-fused-delta.log`.

Added an opt-in fusedDelta chunk kernel: convolution plus ordered gated-delta
recurrence and gated RMS normalization, using FP32 state and existing BF16
projections. Active holes skip recurrence and preserve state. Plugin weights
are ordinary refittable engine inputs, no additional persistent weight copy.
Existing primitive path stays default and Metal code is unchanged. Small
independent fixture passed13 strict checks at1.49e-7 max absolute error,
including actual CUDA graph captures/replays, reset and transactional capacity
rejection (`chunk4-fused-parity.log`). This does not establish pretrained-model
quality, full-size state correctness, or performance. Real2B graph-off/on probe
started sequentially; inspect its live session before any other GPU run.

Prepared configurable independent fixture for batch1/2/4/8, recurrent dimension
4/128 and grouped value-head ratios1/2. Larger fixture not yet executed.
Progress latency chart source and PNG/SVG saved; plotting dependencies installed
in the existing Python venv with original NumPy1.26.4 restored (contourpy1.3.2).

Method012 follow-up: all larger independent fixtures completed successfully,
13 checks each with unchanged abs2e-5+rel2e-4 tolerance. Batch8/dim128/ratio2
max3.58e-7, batch1/dim128/ratio1 max1.19e-7, batch4/dim4/ratio2 max2.38e-7.
Each includes graph captures/replays with0 failures, active holes, reset,
padded single-token decode and actual final-slot/overflow checks. Logs:
`decoder-fixture-b8-d128-fused-parity.log`,
`decoder-fixture-b1-d128-fused-parity.log`,
`decoder-fixture-b4-fused-parity.log`. These are synthetic two-layer fixtures,
not official full-model batch admission or full-model quality acceptance.

Real2B graph-off/on diagnostic completed8 requests. Graph-on measured median
513.93ms; decode98.70tok/s. All8 generated32-token sequences match the prior
chunk4 candidate's tested sequence (which independently matched Transformers
chunk-policy reference32/32). Retained owned bytes4,619,395,224 vs previous
4,716,914,200, same accounting exclusions. Evidence:
`chunk4-fused-graph-probe.json`/`.log`. Diagnostic repeats do not substitute
for full common-image cohort. Both model contract suites passed5 tests.
Full28-request fused candidate cohort started after all fixture processes exited;
observe its live process before another GPU run. Current identity is retained
in `fused-delta-identity.json`. No2x claim or default-profile promotion.

### Method013: full fused-delta candidate cohort and remaining quality limits

Full28-request2B batch1 cohort completed8 warmups/20 measured sequentially.
Median API wall511.85ms, prefill190.49ms, TTFT198.67ms, decode98.65tok/s.
The12 equal32-sampled-token cases shared across allfive profiles give medians:
fused513.08ms, priorchunk553.01ms, originalserial821.34ms, LM299.12ms,
vLLM246.49ms. Improvement7.2% vs previous candidate,37.5% vsoriginal; still
slower than both competitors and2x target NOT achieved. The paired subset
changed from13 to12; allreported comparison rows use the same12 requests.

Retained owned model stable4,619,395,224B across20 measured requests, down
97,518,976B; active-payloadratio1.04358. TextTRT/runtime2,861,324,612B,
25contexts/25captures/36,287replays/0failures. Excludes external/driver,
graph executable, transient and builder memory; not process peak.

Full-cohort tokens match prior candidate25/28, measured19/20. Differences:
frame_10200 at trials0/1/5; trial5 fused exits after24visible tokens while
prior generated32. Rawoutputs preserved; do not treat an earlier stop as a
speedup or infer broad quality from one exact32-token reference trajectory.
Candidate remains opt-in. Need larger quantitative quality/reference suite.

Additional chunk8/16 fused fixtures passed atbatch8/dim128, max3.58e-7,
actualgraphs/reset/capacitychecks. Overall sixfixtures78 strictchecks passed.
Logs `chunk8-b8-d128-fused-parity.log`, `chunk16-b8-d128-fused-parity.log`.
Full report `FUSED_DELTA_CANDIDATE.md`, summary
`fused-chunk4-graph-comparison.json`, raw
`benchmark-garnet-2b-b1-fused-chunk4-graph.json`/`.log`. ProgressPNG/SVG updated
using this12-case subset; priorchart retained asprogress-latency-chunk4.*.
Normalized TRTbuilder line endings to originalLF to remove unrelated diffnoise;
no semantic change. gitdiff--check andPython compilation pass. Both model
contract suites passed5tests. AllGPUprocesses exited; noengine left running.
Next: remaining full-attention/efficient-decode optimization, quantitative
quality,4B andfullbatchprofiles, matchedmemoryisolation and3-VL/Metalregressions.

### Method014:4B fused profile and independent-reference divergence

Official4B fusedDelta/chunk4/graph smoke completed twice with identical32-token
outputs after reset. Warm request1.088s, prefill425.43ms, decode47.38tok/s;
these two requests are diagnostic, not a competitive cohort. Retained owned
model9,552,128,856B stable in both, active payload9,078,538,752B.32decoderlayers
plusfinalnorm have actual graph capture/replay with0failures. Evidence:
`pretrained-4b-image-fused-chunk4.json`/`.log`, `weight-ownership-4b.json`.
Same-token-ID/native-patch independent mixed chunk-policy Transformers reference
completed but matched only14/32 prefix tokens; `reference-4b-fused-chunk4.*`.
Do not label this profile quality-validated or use its timing as acceptance.

Added optional diagnostic native normalized-hidden trace (extra norm calls/CPU
copies explicitly invalidate performance timing), and reference teacher-forced
hidden/logit comparisons on a recorded trajectory. Added optional explicitstop
handling for single-row reference and selectablecaseindex. Existing default
reference behavior preserved. Native primitive4B chunk4 trace now running to
isolate fusion from chunk/reference policy; observe its session before another
GPUengine. LMREST harness now accepts explicit2B/4B modelname/identifier while
preserving previous2B defaults. Python sources compile.

### Method015: quantitative4B diagnostics and accumulation-contract correction

Primitive4B chunk4 diagnostic produced the same32tokens as fused onvariant0.
Fixed-trajectory independent reference matched29/32 deterministictop1 and31/32
reference tied-maximum sets; reconstructed nativehead matched32/32 from native
normalizedhidden. MaxhiddenrelativeL2=.044806, mincosine=.998996. Native fused
vsprimitive normalizedhidden relativeL2=.008186,maxabs=.31138, despite identical
32tokens. This is evidence of numerical differences, not exact model parity.

Exported native prefill embeddings/RoPE positions and repeated teacher forcing
with the same native embeddings. Positions matched exactly. Independent vision
embedding relativeL2=.049269,maxabs1.3828125. With nativeinputs and PyTorch
BF16 reduced reductions disabled, top1 IDs matched30/32; all32selectedtokens
were in the reference's tied-maximum set; maxhiddenrelativeL2=.038484,
mincosine=.999259. Native-input mode bypasses independentvision and must not be
used to assert full-model quality. Broader quality and numerical acceptance
remain outstanding. Summaries `quality-4b-primitive.json`,
`quality-4b-fused-native-input.json`, script
`summarize_qwen35_teacher_reference.py`; raw teacherreference and native traces
are retained. The first trace comparison failed because native tensor.tolist
is flat; fixed reshape to the independently known reference shape and reran,
keeping initialfailure and retrylogs.

Audited cuBLAS precision contract: Compute32F alone does not explicitly forbid
reductions in lower output precision. NVIDIA documentation describes
CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION as forcing accumulator-type
reductions. Added this mathmode when creating our shared private projection
context; no extra weights or memory copies. Reference flag --strict-reductions
sets PyTorch allow_bf16_reduced_precision_reduction=False (its default isTrue).
Sources: https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html and
https://docs.pytorch.org/docs/stable/notes/cuda . Existing BF16 projection
arithmetic claims are now explicitly enforced; prior runs remain historical
Compute32F/default-policy evidence, not proof of that stricter contract.

Build passed (`build-strict-reductions.log`). Reran all8linearcomponentcases
(24steps), passed; batch8/dim128 fused chunk4 strictfixture13checks passed,
max3.58e-7, actualgraphs/reset/capacity checks intact. Logs
`linear-strict-reductions.log`, `chunk4-b8-strict-reductions.log`.
Current-binary full-modelquality and competitivecohorts must be rerun before
acceptance. Preserved previousidentity aspre-strict-reduction-identity.json;
existingreports explicitly mark historicalprecisionpolicy. AllGPUtests exited.
Prepared and started CPU4B BF16GGUF conversion with pinned llama.cpp
74ade52741203e5c8f81eaf06a96cb1cfe15f2a3 usingconvert-4b.cmd; inspect live
conversion before using files. No GPUengine is running during conversion.

### Method016:current4B cohort and BF16GGUF provenance

Current strict-reduction4B fusedchunk4/graph cohort completed28requests,
8warmups/20measured. Median APIwall1249.63ms, prefill487.42ms,
decode41.19tok/s. Retainedownedbytes stable9,552,128,856B. Allrequests explicitly
record projection_reduced_precision_reduction=false. Variant0 output32tokens
matches the pre-fix smoke, but that alone does not prove general quality or
bit-equivalent hiddenstates. Raw `benchmark-garnet-4b-b1-fused-chunk4-strict.*`.
No competitor2x claim; broadquality and isolatedpeak memory remain pending.

Saved `inspect_qwen35_gguf.py` after the earlier shellone-liner quoting failure.
Initialinspection deliberately failed the too-broad all-large-tensors-BF16 rule:
mmproj v.patch_embd.weight isF32. Checked pinned converter source, not an
arbitrary exception: conversion/base.py explicitly promotes patch embeddings
and visionposition tables; qwen3vl.py splits temporalConv3D into2Conv2Dweights.
Revised inspection requires exactly the3expectednames andbytes: patch tensors
3,145,728Beach andpositiontable9,437,184B, allF32. Otherlarge tensors mustBF16.
These are exactBF16-to-F32 promotions, not low-bit quantization. No claim of
identical runtime arithmetic acrossengines. Initialfailure retained in
inspect-4b-gguf.log; retrypassed. conversion-4b-provenance.json records pinned
HFrevision, convertercommit plushelpers/sourcehashes, GGUFtensorstorage and
fullfileSHA256. Conversion/GGUFinspection ran afterGarnetGPUprocess exited.

Generalized LMREST modelname/identifier and summaryscript to4B, preserving2B
defaults. vLLM4B commonimage cohort started only afterinspectioncompleted;
128MiB explicitcachebudget, BF16weights, FP32recurrentcache, no prefix/mmcache,
maxlen384, batch1, nonspeculative. Liveexecsession46823,
benchmark-vllm-4b-b1-variants.log. Poll thathandle beforeLMStudio or anotherGPU
engine. LM4Bfiles notyetcopied/loaded. Thefullscope remainsopen.

### Method017:4B competitor cohorts complete and audited

AfterGarnet exited, vLLM4B completed28requests (8warmups/20measured), with all
20measured requests32sampled tokens. Medianwall547.40ms, reportedTTFT45.51ms,
decode62.25tok/s. All28 reportzero cachedtokens and exactpromptIDequality with
Garnet. WSLparent/enginePID29834/29994 absent afterexit beforeLMstarted.

Copied bothverified4BGGUFfiles toLMmodeldirectory andverifieddestinationSHA256
againstprovenance. LMmodelkey/path checkedviaCLI. Emptyloadedmodels confirmed
beforeloading; requestedcontext384, parallel1, gpumax, noDraftMTP. LMRESTcohort
completed28requests,20measuredall32outputs/85inputs/0reasoningtokens. Medians
wall665.91ms, TTFT93.71ms, reporteddecode57.92tok/s. RuntimeCUDA2.41.0; actual
loadedcontext512 despite384request. Runtime28promptevalrecords each85tokens;
no separateimageeval/cachehitrecord in thisrun, so do notclaim itsvisioncache
is provablydisabled. Freshper-roundJPEGs/storefalse/requestwork stillchecked.
LoggerPID/starttime checkedbeforestop; modelunloaded, ps=[] anddaemonshutdown.

All20measuredcases match32sampledtokens acrossengines. Garnet1249.63ms vs
LM665.91ms/vLLM547.40ms: slower1.88x/2.28x;2x goalnotachieved. Retainedowned
Garnet9.55GB stable; isolatedprocesspeak/comparablememory stillpending. Summary
initial-4b-batch1-comparison.json, reportINITIAL_4B_BASELINE.md, currentmodelchart
progress-models.png/svg. Summaryscript correctedconditionalEOSlimitations:
new4Bexplicitstopcounts andall20equalwork are reported accurately; historical
2Bdefaults preserved. Rawconfig/log/result files and conversionprovenance kept.

Next,current2Bstrict-reductionGarnet28requestcohort started sequentially after
allcompetitorenginesexited. Liveexecsession recorded in GOAL.md. No4Bacceptance
oroverallcompletionclaim; quality,batches,regressions,memoryand2xspeed remain.

### Method018:current2B precision-policy cohort and consolidated checkpoint

Current2B fusedchunk4 strict-reductionGarnet cohort completed28requests,
8warmups/20measured. All28generatedsequences match the pre-correctionfusedrun.
Fullcohort medianwall592.43ms, TTFT230.10ms, decode84.88tok/s, retainedowned
model4,619,395,224B stable.12equal32-sampled-token cases sharedwithbaselines:
Garnet595.14ms, LM299.12ms, vLLM246.49ms.2x target123.25ms unmet. Runtime
explicitlyrecordsreducedprecisionreductionsfalse. Sourceweights/settings and
baselineversionsunchanged; reused the completed2BvLLM/LMcohorts, not concurrent
runs. Source/binaryidentities retained. Timingincrease versuspre-fix512ms is
observed, not an isolatedcausalestimate of the flag. Noqualityclaim from output
replay alone; broaderqualityremainsopen.

Consolidated current2B/4B reportCURRENT_BATCH1_STATUS.md and chart
progress-models.png/svg generatedfromthetwo current summaryJSONfiles. Both
panelsnowuse currentGarnetpolicy, with matchedcasecountsandtargets explicit.
Historicalcharts retained. Generalizedsummary nowrecords matchedsubset token
counts andscopedretainedownedmemory, conditionallydescribesEOSwork and explicit
stopcounts to avoid incorrectlyapplying old2Blimitations to new4B.

AllGPUcohortsfinished; no nativeengineprocess remains, LMmodelunloaded/daemon
down, WSLvLLMworkers exited beforeLM. Pythoncompilation/diffwhitespacechecks
pass. Stillpending: quality, isolatedmemory, full-modelbatch2/4/8competitive
profiles,3-VL/Metalregressions and2xperformance. Next memorymapping should use
WindowsCUDA driver cuDeviceGetLuid (in cuda.h; not cudaDeviceGetLuid runtimeAPI)
to associateNVMLGPU0PCIID withWDDMper-processcounterLUID, preservingadapter/PID
rows andsampled-vs-truepeak distinctions. Do notsumunresolvedadapters orcompare
unlikeowned/frontend/processcounters. No goalcompletionclaim.

### Method019: adapter-resolved native process memory diagnostics

Added capture_windows_gpu_memory.py. NVIDIA GPU0 PCI00000000:01:00.0 matched
to CUDA driver cuDeviceGetLuid via pinned cuda.h API: LUID0/0x00011dec,node_mask1.
Collector preserves raw WDDM PID/adapter/node/duplicate-instance rows; missing
rows are unobserved, not zero. Initial preflight rejected valid #1 suffix;
added explicit suffix parsing without aggregation. Mapping-only failure and
retry retained. Native summary permits one nonzero row plus zero duplicates;
multiple nonzero instances stay ambiguous. Holds/polling are diagnostic only.

Garnet2B/4B ran sequentially with strict reductions/fused chunk4/graphs,batch1,
capacity384. Warm8s holds before/after inference have3 enclosed samples each,
min=median=max.2B dedicated4,961,357,824B/shared81,788,928B;4B dedicated
9,927,450,624B/shared81,788,928B. After model.close with process alive, dedicated
270,012,416B/274,206,720B respectively; shared unchanged. Largest sampled
load/build dedicated5,244,469,248B/10,141,356,032B, not true peaks. Retained owned
4,619,395,224/9,552,128,856B as before. All raw sessions/counters/logs and
memory-garnet-{2b,4b}-b1-summary.json retained. Native and collector exit0;
sessions47877/91098 terminal. No direct comparison to older whole-device12.5GB
or screenshots from another machine.

resolve_lmstudio_gpu_worker.py requires one >=2GB dedicated counter process on
matched adapter under controlled llmster ancestry, not an arbitrary largestPID.
Verified llama-server.exe51692/56968 direct children of llmster56584/30408.
LM2B/4B sequential,noMTP,gpumax,1slot,requested384/actual512. Models loaded before
collector, so loading peak unavailable.2B dedicated5,041,065,984B/shared90,177,536B;
4B dedicated9,759,694,848B/shared90,177,536B.3 samples before/2-3 after, all stable.
No ambiguous nonzero rows. Workers exit on unload; absent closed rows are not
reported zero. Model unload/daemon shutdown and sessions43399/33345 finished0.
Context/precision qualifications remain;1.6-1.7%difference is not a general
memory winner. Report ISOLATED_MEMORY.md, chart progress-memory.png/svg and
phase runners/summarizer retained. Query intervals wholly inside holds exclude
1s at both boundaries. No primary timing cohort ran while polling.

### Method020: WSL worker allocator and unresolved VM counter instances

Added diagnostic-only --memory-phase-output to vLLM harness, requiring
--memory-diagnostic and >=3rounds.8warmup/4additional requests,8s warm holds,
worker peak reset after warmup; primary default path unchanged. Windows
collector watches host wsl.exe lifetime and8s after exit, retaining all adapter
rows.2B session56742 and4B26927 finished0; guest workers31634/32175 absent after
exit. WorkerRPC:2B allocated4,605,155,328B,reserved4,729,077,760B,postwarm peak
allocated4,614,088,704B.4B allocated9,340,961,792B,reserved9,414,115,328B,
postwarm peak allocated9,357,178,368B. These are isolated Torch allocator
counters; whole-device mem_get_info remains separate. No cross-scope winner.

Host counter changes correlate with persistent PID40872 vmwp, not LinuxPID.
Multiple positive instances sharePID/LUID: one changes456,634,368B to
5,059,928,064B for2B or9,751,293,952B for4B; others include24,576B and456,634,368B
plus zero. Initial summary refused PID collapse (empty candidate assertion);
failure log retained. Revised summarize_wsl_memory_probe.py retains individual
instances, marks native_equivalent_process_metric_available=false and
multiple_nonzero_vm_instances_ambiguous=true. No summation or validated guest
baseline-subtraction metric. Instance suffix identity/hierarchy and full VM
identity association remain unverified. After guest exit observed instance
returns456,634,368B. Raw baseline/WDDM/phases/config/requests/logs, failure/retry
and summaries retained under memory-vllm-{2b,4b}-windows-*. VM values are not
added to the native chart. Worker allocator statistics remain separately usable.

All six memory sessions sequential and terminal. No GPU engine running; LM
daemons shut down. Full scope open:2x speed unmet, full-model batch2/4/8,
broad quality,3-VL/Metal regressions. Final common-capacity cohorts should use512
to remove384/512qualification after optimizations. Batch1 memory near native
baseline is measured progress, not overall acceptance.

### Method 021: official 2B batch decoder diagnostics

Native profiles 2/4/8 were run sequentially with strict BF16 projection
reductions, FP32 state, chunk4 and CUDA graphs. Raw batch-validation-2b-b*.json
and batch-reference-2b-b*-*.json are retained. Uniform and ragged prompt reset
replays passed for all profiles; mixed-image row isolation also passed for
4/8. Batch2 mixed-image coverage remains pending. Slot counts are exactly
prompt + sampled - 1. Graph capture/replay succeeded without failures.

Independent Transformers teacher-forced decoder comparisons use exported
native embeddings and mRoPE positions, bypassing independent vision. Local
attention masks are explicitly supplied to cached GDN chunk forwards because
the installed HF mask helper otherwise drops remaining left padding.
Padded GDN/MLP projections mirror chunk4 geometry; full attention remains
serial. Of 1,280 decisions, 1,279 exact IDs match, and all 1,280 native choices
are in the reference maximum set (one tie). All native hidden/head
reconstructions match. Retained owned bytes are stable at 4,792,640,744,
5,140,336,008 and 5,834,543,816 for batch2/4/8 respectively. This metric is
separate from process WDDM dedicated memory.

Diagnostic copies invalidate performance timing. These cases do not prove
broad/free-running quality, full official-model cache capacity, or comparative
batch speed. 4B batch validation is the next sequential GPU work. Updated
latency, isolated-memory and batch charts retain those scope distinctions.

4B native batch2 subsequently exited0 with all six cases passing and stable
owned10,004,095,912B. Independent references and batch4/8 continue through
work/qwen35-cuda/run-4b-batch-suite.ps1, which checks each exit and refuses
overwriting existing evidence. Pending stages are not acceptance claims.

4B batch2 references complete:188/192 exact IDs and maximum-set matches,
192/192 native-head reconstructions. Four nonmaximum choices all occur at
step14 on the unchanged long-prompt image row: native264 vs reference1330,
rank2, reference top1 gap0.125, hidden relativeL2~0.00804. This is an open
decoder arithmetic discrepancy, not a reference tie or accepted parity.
Summaries now retain every nonmaximum decision and explicit match flags.
Batch4/8 collection continues sequentially to establish geometry dependence.

### Method 022 candidate: vectorize full-attention chunk projections

Source qwen35_lowering.h now separates attentionProjected (F32 norm/RoPE/
cache/attention/gate) from attentionStep projection. fullDecoderChunk performs
Q/K/V/O projection once over [batch,tokens,hidden], retaining serial causal
cache updates and per-token attention. This removes repeated projection
dispatches/weight reads for static chunk padding. Default chunk1 retains one
token geometry. This is provisional: no speed or correctness acceptance yet.

build-full-projection-object.cmd compiled trt_builder.cpp.obj successfully;
DLL was deliberately not relinked while session45122 validates the previous
binary. batch-validation-identity.json describes the pre-candidate source and
binary. Current source and runtime DLL therefore differ until explicit build.
Do not relabel prior batch evidence as vectorized-projection evidence.

CPU-generated attention-real-fixtures and decoder-fixture-b8-real-attention
add full-attention head_dim256, rotary64, mRoPE[11,11,10], theta1e7 and
16:4 query/KV heads. Hidden width remains a synthetic8; these fixtures cover
real attention geometry, not full-model projection dimensions. The decoder
fixture also uses GDN dim128. They await GPU execution after old suite exits.
Existing fixture generation defaults are unchanged.

Old-binary sequential suite session45122 endedexit1 at4Bbatch8. Batch4
reference377/384 exact/max-set matches,384/384 head reconstruction. Batch8
first full-attention layer build rejected tactic requesting67,111,712B against
64MiB (67,108,864B) workspace cap; buildSerializedNetwork failed. No completed
batch8 result. This is not proof of physical16GB VRAM exhaustion. Preserve
batch-validation-4b-b8.log and inspect configured workspace before retry.
progress-batch-status.png includes this failure and unresolved4B parity gaps.

Candidate DLL built successfully; oldDLL SHA256
6effa0196b264373ede7115d71f513196aa40aa16a7a5ba8af1367fd44ab01c6
archived in pre-full-projection/. Explicit per-graph builder-workspace setter
accepts1MiB..1GiB before build, reports its limit in stats. Only batch8 full
attention selects128MiB; other graphs retain64MiB. This addresses the measured
64MiB cap without asserting any physical memory allocation or successful4B8.

Candidate batch8-real-attention chunk fixture failed the existing first hidden
check, maxabs0.00787353515625. Tolerances not changed. Batch4 same geometry
passes13checks at maxabs7.15e-7 on both archived and candidate DLL, including
actual capacity32 fill/reject and graph replay. Batch8 historical comparison
pending through probe_qwen35_legacy_chunk.py (only omits new workspace API for
capacity32 fixture; leaves graph math, expected tensors and tolerances intact).
Do not infer the batch8 failure is caused by vectorization until that baseline
comparison. Raw failed and passing logs retained.

reference_qwen35_batch.py adds explicit --full-projection-chunk opt-in:
whole-chunk self-attention with padded Q/K/V/O GEMM geometry. Old serial
reference default remains for historical evidence. New runtime result metadata
states chunk_batched layout. Current2B batch1 candidate timing cohort is
diagnostic until correctness acceptance, with all prior evidence preserved.

Candidate2B cohort exited0,28requests/20measured. Equal32sample subset12:
candidate555.051ms vs priorstrict595.135ms (1.072x); LM299.124/vLLM246.490ms.
All measured candidate median555.128ms, TTFT211.925ms, decode90.443tok/s,
retainedowned4,610,578,584B stable. Exact prior token sequences26/28; changed
trial1/frame0 andtrial5/frame10200. No2xclaim or quality acceptance. Comparison
script records hashes and unequal-work exclusions; external cohorts reused,
not contemporaneous. Dedicated diagnostic runner now traces only these two
changed inputs in one model session, preserving the benchmark cohort.

Historical batch8-real-attention fixture comparison completedexit1 with the
same first-hidden maxabs0.00787353515625 as candidate. Thus this synthetic
numeric discrepancy predates vectorization; neither build passes that check.
No tolerance change and no blanket batch8 acceptance. Candidate/oldbatch4
bothpassed13checksmax7.15e-7. Raw test-old-projection-b8-real-attention.log
and candidate failure log retained for further localization.

Changed-trajectory diagnostic session54775 exited0, both cases reproduced the
original candidate token IDs. Independent reference session8548 exited0 with
--full-projection-chunk:64/64 exact IDs, maximum-set choices and native-head
reconstructions. This is decoder-only evidence with fixed native vision inputs,
not broad quality acceptance. Full-projection candidate summary now includes
these reference results and historical-batch8 failure reproduction evidence.

Next isolated GPU job repeats the failed official4Bbatch8 profile on candidate
DLL with128MiB workspace. Output batch-validation-4b-b8-full-projection.* is
separate from the original batch8 failed log; no result claimed before exit.

### Method 023: isolate the official batch8 full-attention build

128MiB retry session15558 failedexit1; candidate tactic requested134,226,048B,
above134,217,728B cap. No reference was launched on incomplete native output.
probe_qwen35_full_layer_build.py builds only official layer3, batch8/chunk4/
capacity384, avoiding repeated whole-model loads for workspace experiments.
At256MiB it builds in7.235s, retained TRT allocator419,704,704B. This proves
one layer compiles; it does not prove full residency, inference or peak memory.
Selected256MiB for batch8 full layers only; raw128 failure retained and full
model rerun uses separate *b8-full-projection-w256.* paths.

Synthetic numerical localization now has independent per-layer CPU hidden
values in a newly generated diagnostic fixture. Existing failed fixture and
tolerances stay intact. probe_qwen35_fixture_layers.py records per-layer native
chunk hidden errors; queued for execution after the official batch8 job.
It is a host-copy diagnostic, not timing evidence. Candidate workspace256
source revision must be distinguished from the measured batch1 candidate
(batch1 remains64MiB, same compiledDLL).

Official4B batch8 workspace256 session86961 exited0: all6 uniform/ragged/
mixed-image reset cases passed, graph capture/replay succeeded, slots exact.
Retained owned model12,527,637,896B stable. Partial-process WDDM capture11000
also exited0:96 samples with positive target rows, sampled dedicated maximum
12,926,402,560B; shared range79,691,776..81,788,928B. Capture began during
loading, no guarded warm holds; not a full loading peak or idle baseline.
Small shared counters alone do not prove every allocation stayed resident.

Sequential reference+probe session83284 exited0. Native-input decoder reference
uniform248/256 exact/max-set; ragged248 exact,252 max-set; mixed254 exact/max-set.
Totals750/768 exact,754/768 max-set,768/768 native-head reconstructions. Four
exact-ID differences are ties; fourteen nonmaximum rank2 choices remain open.
Candidate summary retains all discrepant decisions and source/layout scope.
Do not combine old serial-projection batch2/4 and candidatebatch8 into one
validated current matrix or infer comparative batch speed from diagnostics.

Per-layer synthetic probe completed64records: linear layer0 hidden matches CPU
exactly for every tested row/step. Full layer1 first notable error at row5/
step3 (active), maxabs0.00787353515625. Same failed final-hidden check occurs on
archived and candidate DLL. This localizes away from the recurrent layer for
this fixture; attention/projection/MLP stage and rounding cause still unproven.
No tolerance changes. New4B batch1 candidate cohort launched next, preserving
all historical timings and explicit experimental status.

4B batch1 candidate session64381 exited0:28requests,20measured,all20 equal32
sample work. Median1072.4445ms vs priorstrict1249.633ms (1.1652x speed,14.18%
latency reduction), LM665.905/vLLM547.399ms. TTFT422.822ms,decode47.775tok/s,
retainedowned9,528,724,312B stable. Exactpriorsequences26/28. Separate changed
trajectory native trace session33218 started; candidate remains experimental.
External timing cohorts reused;384/512context and arithmetic limits remain.

### Method 024: Metal regression prerequisites and sharded reference reader

Read-only SSH inspection through shawns-air-ssh verified Shawns-Air Darwin,
AppleM5/24GB, clean Garnet checkout637ec954b246ed1fb79d2ec3988b458a555f2f63,
and /Users/shawn/CantorAI/out/models/Qwen3.5-4B with two indexed shards.
No remote model job launched alongside CUDA benchmarking. Existing native
build found at out/build/macos-arm64-release/bin; other experiment builds
exist, so filenames alone do not establish the correct baseline binary.
Runtime/source identity and baseline tests still must be verified before a
preservation claim. Remote4B factory is Metal-only; local changes preserve
Metal constructor arguments and add lazy imports plus CUDA factories.

Existing test_qwen35_extended_prefill_reference.py required model.safetensors,
so it cannot read that actual sharded4B checkpoint. Added independent NumPy
qwen35_weight_reference.py reader for single-file and index-mapped shards;
the test passes a directory to native build for the indexed case. Selected
norm tensors from2B(single) and4B(sharded),layers0/3/23, compare exactly with
Safetensors reader (6checks). Raw metal-reference-sharded-reader-check.log.
No Metal execution, ABI or full-model quality acceptance follows from this
CPU reader check; remote tests and current-source regression remain pending.

4B changed native trajectories session33218 exited0 and reproduced both
original candidate outputs. Independent reference62627 exited0:64/64 exact,
max-set and native-head reconstructions. Teacher-forced native-input decoder
scope only; max hidden relativeL2~0.02545, broad quality still open.

Metal pretrained single-step test had the same single-file reader assumption;
updated it to the shared sharded reader. Copied test/helper/independent NumPy
references and graph runners to isolated remote out/tests/
qwen35-cuda-preservation-20261022 (remote checkout remains clean).
System python lacked NumPy; existing out/deps/garnet-metal-frontend/bin/python3
has it. No installation or replacement of the remote environment was needed.

Metal component baseline session95896 exited0. Official4B linear layer0 and
full-attention layer3, batches1/2,4 stateful steps each, including inactive-row
masking, pass independent FP32 reference checks. Native backend reports
Metal/MPS AppleM5;resident_kv_inplace;attention_prefix;body_shape. BinarySHA256
libgarnet b5f4908ebbcd4d7804dfc757af8b383358563b8f65542f6670fbcea23e01e69c,
xlang3 786efe136fe9d6ff9c573f15ef3b684b447889e0ac4fa3de7fc68a623187ff5b.
Raw local metal-pretrained-4b-component-baseline.log retained. This validates
existing kernels on selected pretrained layers, not the current Python factory,
current-source native rebuild, vision or end-to-end4B Metal inference. Native
Metal CMake uses metal_main.cpp, not the CUDA garnet.cpp entry path.

### Method 025: same-binary Metal factory preservation smoke

Original remote runtime/factory and copied current Windows runtime/4B factory
ran in separate processes, strictly sequential, on the existing M5 binary and
same4B indexed checkpoint/image320x192. Two8-token requests per process have
identical generated IDs, input counts and image grid; reset/replay exact.
Raw metal-factory-{baseline,current}.{json,log} plus preservation summary saved.
First probe failed before model load because embedded runtime lacks resource;
second compact-try syntax was rejected. Both remote failure logs retained.
Optional resource measurement now reports null when unavailable; no falseRSS.
Baseline-v3 session64277 and current76457 exited0. Isolated remote directory
used, checkout unchanged. No comparative timing claim from short smoke.

Remote native sources differ from Windows base in Metal CMake, kernels,
compiled graph, sessions and entry code. Full difference list retained in
metal-remote-native-source-differences.txt. This proves current Python factory
compatibility on that existing binary for one input, not all local native
sources or broad/end-to-end model quality. Fresh native source/build parity
remains open; do not overwrite newer remote native work to force equivalence.

Next sequential CUDA job is legacy Qwen3-VL-2B current-DLL public serve/infer
regression:512input/240patch/48KVpages/65536pixels,16outputlimit,2requests,
dedicated cache path. Raw qwen3-vl-current-regression.* retained independently
from older October7 baseline. Result unclaimed until terminal.

### Method 026: legacy regression geometry and future matched context

Qwen3-VL legacy regression58410 exited1 after model-ready108.744s. Inference
reported compiled_frontend_failed: image patch grid does not match compiled
vision profile. No generated output and no pass claim. Hardcoded240patch
profile did not match this JPEG's legacy preprocessing. Also audited legacy
infer_json schema: it exposes text/output_tokens/visual_tokens, not raw
generated_token_ids. Corrected runner asserts decoded text/count replay and
16output tokens; derives patchCount and expected visual count from actual
native qwen_vl_prepare_request before model load. This does not resize/alter
the input to force the old profile or treat different model preprocessors as
equivalent. New dedicated matched-grid cache/output retains original failure.

Future matched-capacity cohorts now have explicit512 support: Garnet accepts
GARNET_BENCHMARK_CONTEXT=512 (input384+output128); vLLM --context-length512.
Historical default384 and raw cohorts remain unchanged. New version2 profile
manifest is marked planned/unmeasured and keeps batch1/2/4/8,32request limit,
2warm/5measured rounds and sequential engine residency. Python compile check
passes. These configuration edits do not constitute matched512 measurements.

Matched-grid legacy session90965 exited0: actual native resize192x320 to
224x352, grid[1,14,22],308patches/77visual tokens. Model-ready104.607s;
two16-token public serving responses pass status/count and decoded-text
reset/replay checks. Raw qwen3-vl-matched-grid-regression.{json,log} retained.
This is current-DLL single-image legacy serving regression evidence, not a
raw-token parity test (API does not expose IDs), independent quality reference,
or timing comparison with Qwen3.5. No remaining model process after exit.

### Method 027: matched512 batch workload and LM group harness

Earlier per-round variants suffice for batch1, but batch2/4 reuse image families
across groups and batch8 repeats them within a group. To avoid accidental image
cache reuse, a new dataset has7rounds*4groups*8rows=224 unique decoded RGB
images, each320x192. Source frames are bilinear resized, a distinct8x8 corner
tile is added, JPEG95 written. Both JPEG and decoded RGB hashes plus original
source hashes retained. Generator asserts decoded hashes are unique. These
are controlled variants, not unchanged original frames; old data stays intact.

Garnet/vLLM optional matched-batch-inputs path selection uses shared helper;
all raw rows now record per-input hashes and group index. Defaults preserve
historical paths/context384. New matched profiles use actual capacity512,
request limit32; use same group firstB rows for eachB1/2/4/8. Actual raw hashes
must be compared between engines before any matched-results claim.

New LM Studio batch harness synchronizes B HTTP requests to one loaded model,
records group completion and individual request latency, reasoningoff/temp0/
top_k1/repeat1,32limit. It checks loaded context512 and85inputtokens. Loaded
parallel-slot/cache behavior remains explicitly unverified until runtime-log
audit; HTTP overlap is not proof of GPU batching. No batch speed results yet.
Syntax checks pass. Dataset224images generated under inputs/matched512-batches.
First matched512 Garnet2Bbatch1 cohort starts sequentially; prior cohorts are
not relabeled or overwritten. Current Qwen3.5 candidate correctness limits
remain, including batch-reference nonmaximum choices and synthetic stress case.

Matched512 2Bbatch1 all three engine cohorts complete sequentially: Garnet66236,
vLLM3539, LM78860 all exited0 with28requests/20measured. WSL frontend36293/
worker36454 absent before LM load. LM loaded with context512/parallel1/GPUmax/
noDraftMTP, nativeREST reasoningoff. Logger PID/start-time checked before stop;
model unloaded and daemon shut down. Raw input hashes equal all28, G/vLLM
prompt IDs equal; LM input counts85 and reasoning0. Equal32sample subset16:
G553.278ms,LM336.646ms,V286.767ms. Aggregate measured sampled throughput
57.350/94.339/110.304tok/s respectively; not decode-only throughput or a2x win.

LM runtime logs retained; actual model worker33872, version2.41.0. Observed
prompt-eval lines show85tokens. Logger began after loading, so slot/load/cache
configuration audit is incomplete. No explicit image/prefix cache setting
exposed/verified in these logs; unique decoded inputs mitigate image reuse,
but do not establish every internal cache policy. Summary keeps auditfalse.
SDK context512verified; LM raw prompt IDs unavailable. No broad quality claim.
Next sequential job is Garnet2Bbatch2 context512 with same workload manifest.

### Method 028: matched512 batch2 and loading-time slot evidence

Garnet16821, vLLM4774 and LM15380 batch2 cohorts all exited0 sequentially,
28groups/56requests,20measuredgroups. Raw hashes equal, G/V promptIDs equal.
Equal32sample groups14: G618.050ms,LM415.810ms,V339.544ms group completion.
Aggregate measured sampled throughput G101.191/LM153.572/V183.764tok/s;
not decode-only throughput or per-row latency. All speculative paths off by
requested config; current arithmetic and quality qualifications retained.

LM logger started before load. CLI context512/parallel2/GPUmax/noDraftMTP;
runtime worker12924 logs initializing n_slots=2,n_ctx_slot=512,kv_unified=true.
audit_qwen35_lmstudio_runtime.py accepts the actual slot-specific field names;
initial audit lacked these names and stayedfalse until raw messages inspected.
Frozen load-runtime snapshot and SHA preserve capacity evidence before timing.
Batch>1 LM harness now refuses timing without verified slot/context audit.
SDK metadata remains recorded separately. Full runtime capture shows slots0/1
and56 prompt-eval events of85tokens, all expected work observed. No explicit
cache-disabled setting verified; summary distinguishes work evidence from
configuration proof. Batch1 audit similarly confirms28 full85-token events,
but loading-time slot/context fields were not captured there.

LMmodel unloaded, daemon down, logger stopped after PID/start-time verification.
vLLMworker37357 absent before LM load. New matched512 latency chart includes
only completed2B batch1/2 cohorts and pending/quality/cache qualifications.
Next isolated GPU job44178 is Garnet2Bbatch4; no result claimed before exit.

### Method 029: sequential external runner and measured batch allocation growth

44178Garnet2Bbatch4 exited0,28groups. External runner27814 subsequently
completed V then LM, exited0. It checks native/guest workers between engines,
refuses overwriting cohorts, gates LM timing on frozen load-slot/context audit,
records stage journal, verifies loggerPID/start-time, unloads/shuts down on
exit. Existing manual protocol is now reproducible as a single per-profile
script rather than interleaved commands. Raw request/config/log/audit files
remain separate; stage labels alone are not acceptance evidence.

Batch4 equal32output groups8: G698.829ms,LM554.144ms,V348.987ms; aggregate
measured samplethroughput181.622/230.059/355.825tok/s respectively. Inputhashes
equal, G/V promptIDs equal. LMload actual4slots512each; timedslots0/1/2/3,
112 full85token evaluations observed. Cache-disabled setting still unverified.
No2x or broadquality acceptance. All batch4 engine processes exited before
the next profile. Current2B batch4 native retainedowned5,230,953,864B stable.

New allocation summary uses only completed matched512 Garnet cohorts. 2B
batch1/2/4 ownedbytes4,642,042,008 /4,837,946,600 /5,230,953,864. RelativeB1,
B4 full-attention engine allocations+306,487,296B,linear engines+79,054,848B,
state+99,680,256B,engineoutputs+102,064,128B. Engine counters include weights
and runtime buffers, not pure workspace. They identify where storage grows;
not true processpeak/residency or bandwidth attribution. No per-engine peaks
summed. This supports targeting full-attention repeated-KV/intermediate storage
next, alongside redundant state/output representations, after baseline matrix.

Remaining matched matrix runner launched sequential2Bbatch8 then4B1/2/4/8,
Garnet thenvLLM thenLM each. Stop on firstfailed command; preserve evidence,
never restart from a poll timeout. While matrix runs, keep runtime math and
DLL unchanged so all these baseline profiles share one implementation.

### Method 030: shared-cache capacity correction and batch8 failure

Master62084 stoppedexit1 at2Bbatch8 LM first group, HTTP500. G/V cohorts
completed; no4B job started. LMruntime worker logs show attention ubatch/KV
slot allocation failures while image tokens enter, followed by context-size
errors. It had n_slots8,n_ctx_slot512,kv_unified=true. Earlier audits mistook
slot ceiling for independent per-row cache reservation. Therefore Method028/
029 LM512 capacity equivalence claims are superseded: those shared-pool runs
retain equal-input/output-work evidence but are provisional capacity matches.
Original failed/runtime/config/daemon-down logs remain. Cleanup succeeded.

Official LM Studio schema exposes llm.load.useUnifiedKvCache and parallel
session fields: https://github.com/lmstudio-ai/lmstudio-js/blob/main/packages/lms-kv-config/src/schema.ts
Installed Python typed SDK lacks these fields. New isolated loader adapts
only its config encoder to add documented server KV fields, without changing
installed packages. It requests useUnifiedKvCache=false,numParallelSessions=B,
draftMtp=false, GPUmax and flashAttention=true. Initial hypothesis context4096
forB8 yielded actual slot4096, so no timing was performed there. Correct SDK
context512 yields runtime n_slots8,n_ctx_slot512,kv_unified=false. SDK loader
requires HTTPserver running; one startup attempt failed server-off and was
retained, then actual server readiness verified before retry. FlashAttention
also became explicit; success is not a single-variable causal KV experiment.

Auditor now records unified flag and independent-slot-capacity gate. LM harness
refuses B>1 shared pools and preserves HTTPerror bodies per row on future
failures. Frozen load snapshots preserve proof before benchmark. Corrected
batch8 session75411 exited0,28groups,224 full85token prompt evaluations across
slots0..7. New partitioned output suffix keeps old failure/config artifacts.
Equal32output groups12: G1014.594ms,LM1099.914ms,V445.208ms. Aggregate sampled
throughput251.651/229.042/595.433tok/s, not per-request speed. Explicit cache-off
setting still unverified; unique images/full prompt evaluations are work proof.

Corrected runner35360 reran only LM2B1/2/4 with non-unified512slots, reusing
unchanged validG/V raw cohorts. All corrected LM jobs succeeded,slot/capacity
and full85token prompt work verified. Equal-work medians G/LM/V ms:
B1(16groups)553.278/298.856/286.767;
B2(14)618.050/393.662/339.544;
B4(8)698.829/611.737/348.987.
Use *-partitioned-summary.json for capacity comparisons; older shared-cache
results remain historical/provisional. Current chart selects partitioned
results when present. Runner then advances4Bbatch1/2/4/8, strictlyG/V/LM;
currently4Bbatch1 loading. Original failed master remains terminal, not retried.

### Method 031 prototype: direct-KV FP32 attention core (unwired)

New qwen35_attention_core.cuh/.cu proposes compact scores, softmax and gated
context with direct query-to-KV head mapping, avoiding repeated KV tensors.
Only active chunk positions attend; value accumulation stops at visible slot.
Old KV copied and active chunk keys written before causal attention; caller
must validate unique active integer slots, bounds and nonaliasing. BF16 large
projections, RMS/RoPE and output projection stay outside this FP32 core.
It is not registered, linked, compiled or serving-enabled in these baselines.
No correctness/performance claim. Independent CPU reference fixture generator
written for real head ratios/dim256, padding/all-inactive rows,ragged slots and
final cache slot; generation/GPUtests queued until primary matrix finishes.
Do not rebuild DLL/change current runtime math while corrected35360 runs.

### Method 032: 4Bbatch1 baseline and standalone attention-core harness

Corrected35360 remains live. It completed4Bbatch1 G/V/partitionedLM sequentially,
28requests/20measured. Equal32sample subset19: G978.061ms,LM672.345ms,
V535.769ms. Hashes all equal, G/VpromptIDs equal. LM independent capacity512
verified and28 full85token prompt-eval events observed. Actual cache-disabled
setting and broad quality remain unverified; no2x claim. Runner advanced to
Garnet4Bbatch2, keeping the same DLL/runtime math. 4B2/4/8 unclaimed.

2B allocation summary now includes completed batch8: retainedownedbytes
4,642,042,008 /4,837,946,600 /5,230,953,864 /6,015,791,816 for1/2/4/8.
Stable after warmup within each recorded cohort, not process residency/peak.
Historical shared-pool LM comparisons remain superseded by partitioned data.

New standalone C++ attention-core harness reads independent binary CPU fixtures,
compares FP32 context at2e-5+2e-4*abs(reference), requires bit-exact outputKV and
unmodified input tensors, checks padded buffer canaries, NaN-poisons output/
scratch before runs, captures CUDA graph and checks3 bit-exact replays against
direct execution. Rejected host shape must leave output unchanged. Planned
seven fixture profiles cover1/2/4/8 batches,T1/4/16,D32/64/128/256,actual4:1
head:KV mapping, all-inactive row,ragged slots,final cache index,capacity33
partial key tile and8192 maximum cache capacity. Independent CPU oracle uses
serial writes and visible-prefix PyTorch attention, no native code imports.

build-attention-core-probe.cmd compiles only standalone prototype driver/core,
not the servingDLL. Neither fixture generation, NVCC build nor GPU test has
been run yet; defer until35360terminal to avoid interference with benchmarks.
This is written validation infrastructure, not passing test/speed evidence.
No TensorRT registration or lowering integration exists yet.

### Method 033: matrix stop, prototype parity/sanitizers, targeted workspace retry

35360 terminalexit1 at4Bbatch4/context512 first full-attention build. Tactic
requested67,119,232B vs64MiB (67,108,864B), not proven physicalVRAM exhaustion.
4Bbatch2 had completed all3engines:19equal32groups, G1224.203ms,LM776.227ms,
V626.507ms; hashes/GVpromptIDs and LM independent capacity/full prompt work
pass. Original4B4 failure log retained and copied to workspace64-failure.log.

No model workers remained after35360exit. Generated7 independent attention-core
fixtures and built standalone probe (servingDLL unchanged). Direct/3graph
replay parity, bit-exact KV, input immutability and canaries passed all cases,
maxcontextabs4.47035e-8. Initialmemcheck zero errors, but racecheck exited7
with56 reported hazards in normalize reduction. Matching outputs did not prove
race freedom. Archived pre-fix executable SHA256
6a72c2bb6bb4d174052c4eb64b2843e48ca940a4fb1b82dec50b3492eeabb01c
and CUDA source plus failure logs. Fix separates partial sums from broadcast
storage and restricts partial reads to the reducing warp. Rebuilt standalone
parity unchanged, racecheck0hazards,memcheck0errors,synccheck0errors. No model
speedup or pretrained quality claim, no TRT registration/lowering integration.
Replaced INFINITY macro with typed CUDA constant after compiler overflow
warnings; remaining SDK codepage-comment warnings are in build logs.

Workspace probe extended with explicit batch/capacity arguments. Official4B
layer3,batch4,chunk4,capacity512 builds at128MiB in7.058s, TRTlive350,903,680B.
This is one-layer evidence only. Runtime policy selects128MiB solely for full
4B(hidden2560),batch4,capacity512,chunk>=4; prior completed profiles keep their
limits, operations and precision. NoDLLrebuild. Fresh full retry uses original
outputJSON basename (none existed from failedload) and separate workspace128
log, retaining oldfailure. New resume19835 runs4B4 then8G/V/partitionedLM
sequentially, stoponfailure. Do not integrate prototype while this baseline
runner is live. Working comparison report generated from completed artifacts,
with pending profiles and acceptance gaps explicit; it is not a final report.

### Method 034: complete baseline matrix and opt-in direct-KV attention

Resume19835 exited0 after sequential Garnet/vLLM/partitioned-LM profiles4B4
and4B8. Equal32-token groups:16 and15. Median group millisecondsG/LM/V:
4B4=1531.660/1113.579/565.075;4B8=2223.663/1351.577/669.510.
All eight2B/4B batch1/2/4/8 capacity-matched cohorts are complete. Input hashes,
G/V prompt IDs and independent512-token LM slots/full prompt evaluations pass;
all-cache-off settings and broad quality remain unverified. No competing model
workers remained before attention testing. BaselineDLL archived in existing
work directory as garnet-matched512-baseline.dll, SHA256
1d66919b0085f569bada20635ecde9bfb5c1dae967cc0cb29f4e17305d9493c9.
4B retained model ownershipB1=9,612,618,584B,B8=13,198,763,400B. This is not
process peak/residency. B8-B1=3,586,144,816B includes full-engine+1,902,800,896,
linear-engine+451,780,608,state+609,222,656 and packed-output+618,469,376B.

New TensorRT wrapper passes all7 independent core fixtures after actual engine
serialization/deserialization, invalid/truncated config rejection, clone
namespace/serialization, reference tolerance, exactKV/input guards and3graph
replays. Maxabs4.47035e-8; TensorRT memcheck0errors. Initial sanitizer command
used wrong toolkit/bin path and did not launch; corrected executable is under
CUDA13.2/compute-sanitizer. Logs test-attention-trt-parity/memcheck.log.
The standalone sanitized kernel remains the same. Source and test wrapper
checkpoint recorded in attention-trt-unwired-baseline-complete-identity.json.

Read prior GPT-OSS research summary OPT0001/0009(resident/batched controls),
0027(sharedKV),0025(avoid duplicatedweight storage), and Metal handoff plus
WS-M5-76/77. Historical handoff instructions are source context, not current
authorization to switch host/model/goal. Old Metal outcomes do not predictCUDA.
Important lesson: distinguish local kernels/CPU overhead/GPU spans/full requests
and retain rejections. Current hypothesis is directKV head mapping + one core
per chunk removes repeated KV expansion, serial per-token attention graphs
and inactive attention work, reducing launch/storage cost. It does NOT yet
prove the dominant bottleneck or promise2x. Existing weights/BF16 projection
rounding/FP32 residual and state remain unchanged; attention reduction order
changes and requires validation. No speculation or caching added.

Prospective opt-in fusedAttention profile defaultsFalse. Norm/RoPE computed for
whole static chunk then core; same BF16Q/K/V/O projections and packed states.
Validate existing independent decoder fixtures without loosening tolerances,
then official2B/4B stateful/teacher-forced checks and matched workloads. Compare
memory as well as latency. Do not promote defaults from component tests. Reuse
existing source/build/work folders, keep each experiment's distinct filenames.

### Method 035: integrated attention measurements, numerical limits, projection screen

Opt-in attention build38674bfeec97bcf8b1f4225a4a5906f1157e6ef3d0e7b8b9b7898c5b76c1248a.
Batch4 real-head fixture passed13 strict checks/max7.15e-7, graph replay and
capacity rejection. Batch8 FAILED padded-step hidden bound max0.000244140625;
old primitive path previously failed0.00787353515625. These are different
failures, no tolerance relaxation or batch8 correctness acceptance.

Official2B batch1 matched512 candidate completed28 groups,22/28 output sequences
equal to baseline. Four-way equal32-token subset16groups has median control
553.27775ms,candidate456.73190ms,LM298.85550ms,V286.76657ms:1.21138x speed,
17.4498% lower latency against owncontrol, NOT2x competitor goal. Stable owned
4,554,105,240B vs4,642,042,008B (87,936,768B lower), not processpeak/residency.
Summaries preserve candidate suffix and baseline files; historical competitor
cohorts reused, not fresh paired interleaved engine trials. Source identity
fused-attention-first-cohort-identity.json. Changed-case path resolver updated
to actual matched trial/group/row image paths, preserving historical default.

All6 changed native cases reproduced exactly, then six separate independent
Transformers decoder references ran sequentially after native exit:192teacher
decisions,185 deterministic top1/189 maximum-set/192 reconstructed-head matches.
Three non-tied differences remain; maxhiddenrelativeL2=.024789473. References
consume native image embeddings, not independent vision. No broad quality or
default promotion. Raw fused-attention-2b-reference-case0..5.json/log retained.

Nsight2025.6.3 node-level graph trace captures only third repeated request after
two warmups via cudaProfilerStart/Stop. Compiler/load excluded; no model job ran
concurrently. Instrumented results not speed evidence. GPU activity span466.237ms,
union423.850ms; kernels28,251 sum417.377ms. Top two BF16WMMA families298.489ms
and headGEMV48.810ms, about83.2% of summed kernel time. ActualGPUcopy time6.239ms
vs124.509ms summed cudaMemcpyAsync API time; blocking/wait overlap prevents
equating API time with transfer cost. ProfilerStart724.220ms setup retained,
excluded from model timing interpretation. Read-only SQLite/CSV/interval sum,
union/span analyses saved. This does not measure bandwidth/occupancy or prove
isolated causality, but supports investigating projections before host controls.

SyntheticcuBLAS workspace screen: same BF16 inputs/weights/output and FP32
accumulation/no reduced reductions;3warmups/20event samples per shape/setting,
ascending then descending workspace sizes0/16KiB/64KiB/256KiB/1MiB/8MiB. One
engine/profile at a time, no model residency. Initial compile missing<string>
failed, log preserved; corrected build passed. 16KiB lowered representative
2B gate/up4x2048x6144 median25.77->19.30us;4B4x2560x9216 33.79->28.98us.
8MiB regressed4B32x2560x9216 37.98->69.63us, so never globally select large
scratch. 1MiB changed58/65536 outputBF16 values on2B32x6144x2048 (max.000488281),
both orders;16KiB changednone on sampled inputs. Not a modelquality proof or
projected whole-model gain. Files bf16-workspace-probe.jsonl and
bf16-workspace-sizes.jsonl, executable and buildlogs retained in samefolder.

Prospective next projection candidate: explicit default-OFF16KiB per-invocation
TRT workspace, serialized with plugin, retained by engine context, not a single
mutable handle-global shared scratch. Existing head/default calls remainzero.
Keep strict FP32 contract. Validate clone/serialization/graph/decoder fixtures
and memory plus matched model runs before selecting it. Do not rebuild or edit
consumed runtime while4B8attention benchmark13626 is live.

### Method 036: large-batch attention memory and bounded projection workspace

13626 terminal0:4B8attention complete28groups. Four-way equal32subset14groups,
Gcontrol2223.1872ms/candidate1357.09805/LM1349.9252/V667.57869ms. Candidate
39.0% lower own latency (1.6382x), still slower than references. Only13/28
complete output sequences match priorG; no correctness promotion. Retained
modelownership13,198,763,400->11,320,320,392B (1,878,443,008B lower).
Saved controlDLL as garnet-fused-attention-control.dll with Method035 hash.

Separate memory session83000 terminal0,ownednativePID40880, same attentionDLL,
4B8/context512/chunk4. Adapter verified PCI->CUDA LUID,node0. Four enclosed
samples per steady hold:dedicated11,703,750,656B/shared81,788,928B before and
after inference. Sampled lifetime dedicated maximum11,716,333,568B,shared
81,788,928B,total committed11,798,122,496B. Five enclosed after-close samples:
dedicated272,105,472/shared81,788,928. No ambiguous nonzero duplicate instances.
These are sampled WDDM process counters, not true lifetime peak or physical
residency proof; no matched LM/vLLM large-batch memory comparison yet. Memory
polling/holds invalidate latency. Source/session/counter/owner/summary files
memory-attention-4b-b8-512-* retained; same existing folders. Powershell counter
subprocess uses CREATE_NO_WINDOW; native owned helper launched Hidden.

Projection16KiB option implemented as BF16 plugin version2; old version1 retains
zero serialization/workspace. Version2 serializeduint32=16384, clone namespace
and workspace retained, creator rejects null/truncated/unsupported payloads.
Scratch is supplied per invocation byTRT and bound after cublasSetStream, so
concurrent context streams do not share mutable handle-global scratch. FP32
accumulation/reduced-reduction policy unchanged. Explicit profile property
projectionWorkspaceBytes defaults0 and admits0/16384. No shared-head change.
Initial full build failed misplaced keywordInt scope; preserved log then moved
setter to actual unary lowering scope. Corrected DLL
0ca86db7997a399b79da635ab2890065c674f51fa49305df6ddaafb5daad9825.
Host plugin-contract probe initially used wrong registry namespace; failure
log retained, corrected contract passes both versions. Existing batch4 realhead
chunk4 strict fixture/graph/capacity passes13checks/max7.15e-7 with16KiB; synthetic
ownedbytes unchanged7,954,756. Whole-model memory/performance remain unmeasured.

Independent batch4 realhead chunk16 attention fixture also passes same13checks
and bounds. Hypothesis:85prompt tokens require6 chunks instead of22, reducing
repeated weight passes; padded decode16 could offset savings. Do not predict
whole-request improvement from this arithmetic. Controller75638 nowsequentially
runs workspace16KiB graph-off/on exact-replay diagnostic, full2B1 C4/ws16KiB
cohort, then separate2B1 C16/ws0 cohort. Stop on failure; no runtime/DLL edits
or model jobs while active. New source identity capture after controller launch
must be identified as post-launch, not a prospective frozen closure.

### Method 037: chunk/workspace cohorts and MTP feasibility, not speculation acceptance

75638 terminal0. Workspace16KiB2B1 C4 cohort442.8509ms on16equal32groups,
vs attention456.7319ms (3.04% lower), unchanged28/28 attention outputs and
retained4,554,105,240B. Graph-off/on probe exact8request outputs and actual
captures/replays pass. C16/ws0 candidate353.2359ms vs original553.27775ms,
36.16% lower own latency;20/28 sequences match attention C4,19/28 original.
Retained4,578,699,912B. Engine-specific medians, not additive: TTFT66.42ms,
prefill58.16ms,decode286.57ms vs C4attention174.50/166.03/282.22ms. Fewer prompt
passes helped here; decode slightly regressed. Original baselines retained.
New trajectories and broad quality remain unaccepted, target143.38ms unmet.
Synthetic repeated single-matrix workspace timing does not reproduce the full
model's cache behavior; use actual cohort outcomes rather than extrapolation.

Read installed vLLM0.32 qwen3_5_mtp.py and Qwen3NextModel target final norm,
plus Transformers5.13 attention/RMSNorm source and actual MTP checkpoint15tensor
shapes. Standalone reference uses MTP pre-fc norms, concat(embed,hidden), BF16fc,
one full decoder layer, MTP final norm/shared head, FP32 residual/attention with
BF16 projections and reduced precision reductions disabled. No vLLM serving
engine substituted into Garnet. Weights121,656,320B currently unused by serving.

Native target trace38881 terminal0 on C16 baseline case8:116 complete input/
pre-final-norm hidden rows, including85prompt+31decode inputs; exact same32
recorded outputs. Host copies invalidate timing. Offline independent PyTorch
MTP study ran alone afterward, full shifted prompt and causal cache, no future
target hidden rows supplied to earlier proposals.24/31 one-step proposals match
teacher, target-head reconstruction32/32. Restricted full-GEMM logits using
previous target distribution top8192 retains23/31, same as32768/65536; smaller
K32/128/512/2048 gives10/16/19/21. These are one teacher-forced trajectory and
different full-sequence GEMM geometry, not rollout acceptance, actual selected
weight projection, speculative inference or performance. Original reference
source saved before adding wide shortlist sizes; both outputs retained.

Next native method, before implementation: add a standalone default-unreferenced
MTP chunk using existing BF16 projection/full-attention primitives and explicit
KV state. Borrow the existing tied table, no duplicate shared-head copy. Validate
independent synthetic state/norm/projection/causality and real traced MTP inputs
before production integration. Existing non-spec serving/protocol stays intact.
Subsequent speculative serving would require target verification for every
proposed token, bounded per-prefix GDN/conv state selection, corrected draftKV,
transactional rejection/capacity/reset, EOS/output-limit semantics and exact
target trajectory checks. Shortlist affects proposals only; it must never decide
an unverified output. Any future speculative cohort is a separately labeled
algorithm/configuration, with comparator-MTP capability/baselines audited; do not
silently mix it with the current speculation-OFF baseline acceptance.

### Method 038: standalone native MTP, preserved numerical failures and conditional drafts

Added CUDAMTPProgram, currently unreferenced by serving. MTP pre-fc normalization,
concat(embed,normalized-target), BF16fc, existing full decoder chunk atmtp.layers.0,
MTP final norm and explicit KV state. Caller supplies/owns buffers and shared
tied head. No duplicate1GB table. First componentDLL
ac5ceb454e51a821b074dd92d51b03605bbf214d1ef6f4526a3f5b35298a2a36.
Seven independent CPU-generated profiles use fixed seed258315, BF16 weights,
real2B/4B head geometry, batches1/2/4/8,T4/16,ragged/allinactive controls and
ordered causal cache. Inputs/expected states in flat files under existing root.

First strict test passed cases0/1 then failed case2(batch4,T4) hiddenmax
.00042170286. A reciprocal-factor RMS hypothesis gave exactly the same failure;
rejectedDLL/source/log preserved, code reverted, archived divide-RMS controlDLL
restored. Fixture/reference/tolerances unchanged. Collection mode subsequently
completed all structural/graph/lifetime checks but EXITED1 as required: strict
numerical cases2/4/6 fail, eighteen observations across three execution modes,
max.0029655695. Four other cases pass. Includes one4B-geometry value-cache
difference. All input and unwritten-cache bytes, graph captures/replays and
borrowed-head lifetime checks completed. No claim the full numerical suite passes.

Actual2B MTP teacher study on115shifted rows with CPU-normalized target inputs:
native31/31 proposal IDs equal independent full-sequence reference;24/31 equal
teacher next tokens. Actual graph1capture/6replays, runtime-live126,861,060B,
packed output2,228,224B; caller's shared table1,017,118,720B. These are components,
not process peak. Fresh repeat exported complete learned KV/hidden snapshots.
Conditional native drafts reset each start to a teacher-correct KV prefix and
feed previous normalized MTP output into later draft steps. Depth1/3/7 implies
18/12/11 hypothetical target passes on this one32-token trajectory. Omitted
target verification/commit/corrected-draft/EOS/limit costs and overlapping starts
make these ineligible for speed or rollout acceptance. Actual spec serving absent.

### Method 039: prerequisites for a separately gated target verifier

Native multi-token versus padded serial tests, same profiles/arithmetic, all
bitwise hidden rows and finalstates equal for batches4/8 and chunks4/16 using
existing real-head decoder fixtures. This is self-equivalence, not independent
oracle correctness or pretrained proof. Raw chunk-self-equivalence-* files saved.

Added writable-contiguous GPU clone/copy with exact dtype/shape/byte validation,
partial-overlap rejection and ownership fences. Added explicit predict_tokens
head API accepting flattened2D or3D rows through128; legacy predict retains8row
rank2 limit and Model still admits only batches1/2/4/8. Separate single-token
and verification input/output buffers avoid per-call resizing between those
modes; one table shared. Counters aggregate both buffer pools and break them out,
close releasesboth. CurrentDLL
ce9a401727ff794725f2189280f35a7bfb620c65eaa8ed1fd99fe484c0c0a7aa.
Independent CPU BF16-head fixture passes rows1/8/9/16/32/128 and8x16 cube,
legacy/expanded bounds with unchanged cache on rejection, signed-zero snapshot,
independent storage, self-copy, shape rejection and complete buffer close.

Prospective next implementation: separate CUDATargetVerifier over an existing
decoder, default unconnected. One small global-normalization graph, bounded
multi-token head, reusable GPU state backups, transactional rollback and explicit
accepted-prefix commit. Initial partial commit may restore/recompute the prefix;
this is a correctness prototype, not a speed claim. Test every accepted length,
inactive rows, restored state bytes, output IDs, capacity/invalid admission and
borrowed lifetime against serial native states before any serving integration.
Later prefix snapshots can replace recomputation under the same tests. Keep
numerical/oracle limits and speculation-OFF baselines visible and unchanged.

### Method 040: target transactions, actual verified MTP session and gated frontend

Standalone target verifier added with reusable GPU state backups, one small
packed-normalization graph and bounded verification head. Full commits reuse
computed states; partial commits restore/recompute the accepted prefix. Initial
startup failed an embedded nested-generator scope lookup(active); failure log
preserved and validation rewritten as plain loops. No numerical gate changed.
All6prefix patterns (0..4 accepted plus ragged rows), predictedIDs, bitwise target
state, complete rollback buffers, invalid token/control/commit/capacity rejection,
final actual cache slot and borrowed-lifetime tests pass B4/B8,C4/C16. Helper
close rolls back pending work. Real2B(C16) and4B(C4) case8 teacher-forced target
probes match all32 saved/serial tokens and every final target state byte. Images
and prepared features reused only in these explicitly untimed diagnostics.

Experimental CUDASpeculativeSession then added, initially outside frontend:
shifted MTP prompt cache excludes each row's last prompt until target prediction
exists, handles ragged lengths, then stays one shifted input ahead. Every emitted
draft is target-checked; target commit excludes EOS/final-limit input. Rejected
prefixes restore/recompute. MTP speculative KV tail is overwritten with corrected
accepted target hidden rows and next pending token; no cross-request cache reuse.
One caller-owned tied table/head. Current target snapshot implementation is a
correctness path with known extra work, not a claimed optimized final design.

Six synthetic profiles pass24cases EACH: actual MTP proposals plus explicitly
test-only oracle proposals exercise all-accept branches; limits1/2/7/8, selected
early/later EOS, ragged/inactive rows, exact tokens/sample/stop counts and every
final target state byte. ProfilesB1C16D3,B2C16D3,B4C4D3,B8C16D3,B1C16D7,
B8C16D7. Oracle proposals are never connected to runtime/benchmark and are not
performance evidence. Real2B and4B case8 then each passed TWO fresh-image actual
MTP requests and one fresh serial control: same32 tokens, sample/stop data and all
final target states byte-for-byte. Target forwards2B20(12verify+8recompute),
4B16(11verify+5recompute), vs31serial. This does not prove a speedup. Warmed
single diagnostic2B368ms,4B834ms are ineligible for primary comparison;2B is
not faster than its353ms non-spec candidate. State/reference/quality gaps remain.

Cleanup now restores the original decoder class descriptor rather than leaving
a self-referencing bound-method override; preserves any pre-existing instance
override. Helper/session close releases owned buffers and borrowed references.
B8C16D3 repeated24syntheticcases pass after cleanup.

Default-OFF profile speculativeMTP=0/1/3/7 now wired into native Model, validated
before expensive loading, depth must fit chunk. Request reset clears target and
draft state, image encoding still fresh. Responses explicitly record MTP/depth/
target-check/restore-recompute method and separate target-call metrics. Logical
decode_steps remains sampled steps afterfirst. Memory totals include MTP engines,
buffers and target backups; table/head counted once in parent IO. Close order is
session, vision, decoder with cleanup in finally blocks. Existing mode0 retained.
Summaries label MTP-versus-saved-OFF as DIFFERENT algorithm settings and never
claim overall goal acceptance. First full2B1 C16D3 cohort70641 nowLIVE, sequential
and unique-image matched workload. Pre-launch identity includes controller and
sources in verified-mtp-frontend-before-cohort-identity.json. Do not edit consumed
runtime/binary while live; standalone unwired source preparation is allowed.

Next proposal-only head hypothesis: selected vocabulary rows via existing tied
BF16 table, no second table or gathered persistent weight copy. Independent
FP64-dot error-bound tests, input/guard/invalid-index/graph tests and component
timing must precede frontend connection. It cannot choose an emitted token;
target verification stays authoritative. Do not extrapolate one-step shortlist
accuracy or warm component timing into full-request2x success.

### Method041 — reject initial full MTP performance candidate; proposal head prototype

Controller70641 completed exit0. The28-request native frontend2B B1 C16 D3
cohort used actual checkpoint MTP proposals and authoritative target verification.
All28 token sequences equal the saved C16 speculation-off control. On the same
16 four-way equal32 groups, latency increased353.2359→384.22405ms (+8.77%).
Retained ownership increased4,578,699,912→4,745,831,440B (+167,131,528).
Median target forwards20 comprise12 verification plus8 partial-prefix
recomputations. This is a rejected speed candidate and remains default off.
Saved LM/vLLM runs have speculation off; algorithm settings differ. Neither
correctness parity with native control nor fewer target calls proves2x speed.

Raw: matched512-garnet-2b-b1-verified-mtp-d3.json; matched512-2b-b1-verified-mtp-d3-candidate.json.
Next isolated hypothesis: reduce MTP full-vocabulary projection using indexed
rows of the existing tied BF16 table, followed by target verification. No second
table, no gathered persistent weights. Three unwired source files prepared;
standalone independent FP64 numerical bounds, invalid/alias/input/guard checks,
graph replay and CUDA sanitizer checks precede any connection. Removing prefix
recomputation requires a separate state-correctness implementation and evidence.
All full-goal acceptance gaps remain open.

### Method042 — indexed proposal head, isolated proof then native validation

New CUDA head reads selected rows directly from the one existing BF16 tied
table; FP32 query is rounded BF16, accumulated FP32, output rounded BF16.
It is proposal-only. Invalid/nonfinite scores never win; ties select lowest
global token ID. Independent fixed-seed574991 FP64 fixtures cover5profiles
(H8/2048/2560,B1/2/8,K32/512/8192), plus invalid indices/all-invalid/NaN,
immutable inputs, output guards, host rejection/alias, and3bit-exact graph
replays. Error bound fixed in advance: gamma_(ceil(H/256)+8) times absolute
product sum plus BF16 rounding. Max bound fraction0.9831; all cases pass.
Initial standalone build lacked /Zc:preprocessor required by CUDA13.2 CUB;
failure retained, build flags corrected to match repository flags.

Candidate selection is a stable descending GPU radix sort of the previously
computed target logits; equal scores retain increasing vocabulary IDs.
Nonfinite candidates become-1. Scratch reused sequentially across batch rows,
not a second weight table. Independent CPU stable sort, reordered source rows,
ties/nonfinite/invalid/short-scratch/alias/immutability/guard/graph checks pass.
Combined head+selector memcheck0errors, racecheck0hazards, synccheck0errors.
Warm cached synthetic component timings are explicitly ineligible as model
speed evidence. Original passing source snapshots and logs retained.

Native IO adds head_candidates and predict_indexed with separate reusable
scratch/index/score/ID/value ownership, included in decoder/model totals.
API6CPU-oracle cases B1/B8,K1/16/64 pass, ordinary and token target-logit
caches remain independent, rejected shapes do not resize/mutate outputs,
close clears owned fields and rejects further prediction. Existing expanded
head/copy/legacy-limit regression passes. DLL768f02f1da624b7c0b62221f69d191033a2509348cb90c9ac2236c0bcee06c5a.

Seven synthetic sessions passed24cases each: indexedK16 at B1/B2/B4/B8,
C4/C16 and D3/D7, plus B8C16D3 K0 regression. Every emitted proposal remains
target-verified; oracle_test_only cases exercise acceptance branches but are
never performance/runtime proposals. Full stop/limit/sample/slot/final-state
comparisons pass. Fresh-image pretrained2B then4B probes99350 nowLIVE,
sequential, K8192/D3, same saved real-image case8 and32-token controls.
Do not edit consumed runtime/DLL until terminal. No speed claim yet.

Method042 results:99350TERMINAL0. Fresh-image2B/4B indexedD3 K8192 probes both
passed two requests plus serial controls: saved32outputs/counts/stops and every
final target state bit-exact. The2B one-case diagnostic target forwards17=11verify
+6recompute, warmed273ms; this diagnostic does not establish cohort performance.
56274TERMINAL0 full2B1 C16 D3 K8192 cohort:28/28same C16control sequences,
16four-way equal32groups median337.1631ms vs353.2359control, LM298.8555,
vLLM286.766573. Own latency4.55%lower; still17.57%slower thanvLLM and2xUNMET.
Retainedowned4,751,876,636B vs4,578,699,912control (+173,176,724).
Raw matched512-garnet-2b-b1-verified-mtp-d3-indexed8192.json, candidate summary,
prelaunchidentity and progress-speculative-2b-b1.png/svg retained. DefaultOFF.

### Method043 — small recurrent replay for partial prefix commits (validation in progress)

Hypothesis: retain BF16-projection results convertedFP32 in each linear target
layer packed output. On partial acceptance rerun only its existing ordered
convolution/delta core from the original backup, skipping QKV/gate/MLP matrix
projections. Attention restores unaccepted KV suffixes from existing backups;
accepted target hidden rows are masked and last-hidden chosen at the prefix.
No per-token copy of the large recurrent states; helper graphs load only small
conv/dt/A/norm tensors, not a second set of projection weights. All extra graph
outputs/allocations/spans and retained parent payloads are counted. DefaultOFF.

Unwired CUDA mask/KV-restore helpers passed24independent exact bit-pattern cases
acrossB1/B8,T4/T16,ragged/zero/final-slot prefixes, including signed zero/NaN
payloads, guards, rejection/partial-alias and3graph replays. memcheck0errors,
racecheck0hazards,synccheck0errors. These helpers do not prove model commits.
Native replay lowering and CUDADeltaReplayVerifier now compiled; original
recompute path remains available. Controller35391nowLIVE:unchanged transactional
verifier testsB4/B8,C4/C16, followed by seven24-case speculative session profiles.
Do not edit consumed runtime/DLL or run another GPU workload while active.
Fresh pretrained probes and full benchmarks require these tests to pass first.

35391TERMINAL1 was a harness fixture-selection error: the selected legacy tiny
fixture has attentionD8, outside the fused attention kernel's admittedD32multiple.
The existing real-attentionD256 fixtures used in Method039 were intended. Both
the failed log and original controller are retained. Corrected controller43577
TERMINAL0: all4transactional verifier profiles B4/B8 C4/C16 passed six prefix
patterns and rollback/invalid/capacity/close checks. Seven speculative session
profiles passed24cases EACH, covering B1/2/4/8 C4/C16 D3/D7 indexedK16 plus
B8K0. Emitted IDs/counts/stops/slots and all tested target state bytes match
ordinary native decoding. No old numerical bound was changed. Native DLL
c6978546a88484330ab5a10eb0cad2c4975b12f722ddd9cc641d28f41531852b.
57207nowLIVE fresh-image pretrained2Bthen4B D3K8192 delta replay probes, each
two actual MTP requests plus serial native control. Keep runtime/binary fixed.

57207TERMINAL0: both real2B and4B passed two fresh-image MTP requests and serial
controls with identical32sampled IDs/finish/stop/counts and all final target
state bytes. Full target forwards2B11/4B12, no full prefix recomputations;
the small recurrent replays are counted separately (2B6). Warmed one-case2B
diagnostic244ms vs prior indexed273ms, not matched speed evidence. Full2B1C16
D3 K8192 replay cohort now running; prelaunch identity and raw evidence are
retained separately. Earlier independent numerical/vision/quality gaps remain.

31045TERMINAL0 full2B1 C16D3 K8192 recurrent replay cohort completed28requests,
28/28same C16non-spec control sequences. Same16four-way equal32groups:
Garnet273.45905ms vs C16control353.2359,LM298.8555,vLLM286.766573. Own latency
22.58%lower,4.64%lower than savedvLLM and8.50%lower than savedLM; speculation
differs from their OFF settings. This does NOTmeet2x target143.3833ms. Retained
ownership4,828,113,188B, not process peak/residency. No full prefix recompute;
recurrent replay counts separate. Raw candidate/cohort/identity retained.
Depth7 replay cohort10157nowLIVE, same fixed runtime/binary/workload, strictly
sequential. No consumed runtime/DLL edits until terminal. Standalone unwired
external BF16 projection-weight pool test source prepared only; not implemented
in native IO/lowering yet. It would let separate prefill/decode profiles share
one physical matrix set, avoiding the earlier duplicated-engine weight problem.

10157TERMINAL0 depth7 full replay cohort:313.8058ms vs depth3's273.45905ms,
same28/28control sequences; retainedowned4,830,148,900B. Reject depth7 as the
best speed profile on this workload; keep depth3 candidate. No default or2x
promotion. Deeper one-case teacher rollout predictions did not translate into
better full-request latency. Charts/report retain both results.

### Method044 — external BF16 projection pool (standalone prototype, not serving)

Header audit identifies186target projection matrices totaling2,745,434,112B
for2B;248matrices totaling7,138,181,120B for4B. These are checkpoint payloads,
not GPU measurements. Shapes/dtypes/offset byte ranges and complete rank2
coverage checked. The tied table, vision, small norm/conv and MTP are separate.
Raw external-projection-weight-plan.json retains names and header hashes.

New native IO pool lazily loads a named target/MTP BF16 matrix once, returning
a readonly GPU view whose retained owner keeps the allocation alive. Embedding
and head are excluded from this API to avoid a second tied table. Pool bytes
are counted once in parent IO/decoder ownership. Native Qwen lowering accepts
an explicit external_weights tensor dictionary, validates each checkpoint name,
BF16 dtype, declared input and shape, rejects unused/conflicting bindings, then
uses those inputs instead of embedding corresponding constants in the engine.
Default graphs still use original constants. Graph stats report borrowed weight
inputs/bytes separately from graph-owned allocations. This path is unwired in
serving; it is intended to permit different prefill/decode chunk sizes sharing
one physical matrix set. Native build17802 in progress; standalone tests next.

17802TERMINAL0. Standalone component11877 passed three MLP graph sizesM1/4/16
sharing one readonly pool; exact native constant-versus-external outputs and
graph replay. Additional lifetime39731 passed explicit rejection reasons for
wrong shape/dtype/missing/unused bindings, rejected writes, repeated lookup
without pool growth, graph-close independence and retained views still usable
after pool close. Pool768B for this tiny three-matrix fixture; not full-model
GPU memory proof. It is a native equivalence/ownership test, not independent
mathematical or speed evidence. Raw external-weights-{component,lifetime}.json.

Pooled CUDADecoder now supports explicit BF16 matrix symbols and shared IO/state
as internal opt-ins, default off. Four sequential constant-versus-pooled tests
B4/B8 C4/C16 all match every tested output ID, hidden and state byte. Borrowed
graph weight bytes sum to exactly one caller pool for each decoder. Attention
and fused-delta admission extended to128 for proposed wide prefills; old
profiles unchanged. Four independently generated FP32 wide attention cases
T32/64/128,B1/2/8,D32/256 pass the unchanged attention bound, maximum4.47e-8,
exact KV/guards/input/graph checks and all three CUDA sanitizers0errors/hazards.
55488TERMINAL0. DLL5b99f97a9fc0faf5f024fbc7f37a587bba0f2003cd2d96bab3dd590bcc53b9d1.

### Method045 — hybrid prefill/decode profiles, still unwired

CUDAHybridDecoder owns two bounded graph sets, one readonly projection pool,
one table, and the exact same state/last-hidden tensors. Parent memory subtracts
only shared tensor/IO ownership; all graph buffers and small constants remain
counted. It is not connected to serving yet. First cross-profile test46939
TERMINAL1; diagnostic40706TERMINAL1 retains the unchanged bit-exact criterion.
On B1 prefill64/decode4 versus a narrow-only prefill4/decode4 control, output IDs,
slots, last-hidden, first-layer convolution/recurrent state and value cache
match exactly; only attention key cache differs, max7.152557e-7 (8,928values).
This is a measured floating-point profile difference, not an accepted bitwise
pass. The underlying cause is not yet established; do not silently relax the
test or claim equivalent state bytes across those different prefill shapes.
Need isolate state-sharing correctness against an independent explicit-copy
two-profile control using the SAME matrix shapes, then pretrained/quality
analysis. Earlier cross-profile failure remains separate and unresolved.

37785 initially passed identical-shape state/output comparisons but stopped on
the graph replay-count assertion after only two requests. The wide final norm
runs once/request; the corrected suite uses three requests to exercise warm,
capture and replay. Original log retained.45536TERMINAL0 all6explicit-copy
controls B1/B4/B8 P64/P128 D4 passed every compared output/state byte, shared
tensor identities, pool counts, graph replay, capacity fill/rejection and reset.
This isolates state-sharing correctness at identical arithmetic shapes. It does
not erase the prior wide-versus-narrow key-cache rounding difference.

The MTP session now observes wide target chunks using a small extra final-norm
graph, then feeds their causal shifted teacher inputs to its ONE existing small
MTP engine in slices. No second MTP weight set. Extra norm graph allocations are
counted.24419TERMINAL0:8profiles24cases EACH pass with67-token ragged prompts,
P64/P128,decode4,B1/2/4/8,indexedK16/full-headK0 and the old B8C16 regression.
Tokens/stops/sample counts/slots and all final target states match each profile's
ordinary native target. Oracle_test_only remains confined to tests.

Opt-in frontend sharedProjectionWeights/decodeChunk now connected, validated
before loading.42096 stopped TERMINAL1 on the4B saved-sequence assertion.2B
passed two fresh-image MTP requests plus serial/saved32-token/all-state tests.
4B MTP matched its OWN new-profile serial output and every target state byte,
but diverged from the older narrow profile at token26: old continuation
"suggesting a scene of daily life", new "surrounded by lush greenery visible".
The assertion remains unchanged and failure retained.4B quality/drift analysis
is OPEN; no default promotion or goal acceptance. Warmed single-case diagnostics
2B218ms,4B409ms are NOT primary comparisons. Model+session retained2B
4,979,255,900B,4B10,295,909,148B; not process peak. Actual target pool counts
match header payloads2,745,434,112/7,138,181,120B, shared across both graph sets.
Full2B1 P64D4 cohort8773nowLIVE; keep consumed runtime/binary fixed and do not
run GPU tests or compiler jobs during its requests. Prelaunch identity retained.

### Method046 — cold projection layout/algorithm screen, unwired

Standalone quick screen completed before8773 began. BF16 input/output, FP32
accumulation; Lt candidates restricted to INPUT_16BF/ACCUMULATOR_32F and no
output-type reduction. Same row-major weights versus transposed storage, classic
GemmEx and Lt heuristics, zero/16KiB/8MiB workspace limits. Fixed seed61983;
256MiB eviction sweep before each cold sample versus reported64MiB deviceL2.
This is not hardware-counter proof of cache residency. Independent FP64 fixed
error bounds checked128output samples/candidate; full BF16 outputs compared to
classic TN. No math-bound failures in the quick2B gate(M4/M16,K2048,N6144).
Classic TN cold61.44/60.42us, transposed NN60.42/59.39us; fastest eligible Lt
56.32/57.34us. Warm cache timings differ substantially. No runtime integration
or model-speed prediction follows from this one-matrix screen. Wider shapes
and current full-request profiling should determine the next optimization.

8773TERMINAL0: full2B1 P64D4 MTP3K8192 replay cohort28requests. Same16equal32
four-way groups:255.0285ms vs older C16Garnet353.2359, savedLM298.8555 and
vLLM286.766573. Retainedowned4,979,255,900B.21/28same C16outputsequences;
seven changed trajectories need independent quality analysis. This is not a
2x or quality/default promotion. Native same-profile speculative verification
was proven on selected real probes and synthetic suites, not all independent
model/vision quality. Different speculation settings from saved competitors.
73922nowLIVE Nsight diagnostic of warmed2B1 P64D4 MTP3K8192 replay. Two warm
requests, capture third via profiler API; same image repeated purposefully for
diagnosis, fresh reset/encoding each time. Trace times cannot rank engines.
No runtime/DLL edits or other GPU workloads while live. Profiling source and
controller hashes recorded in hybrid-before-nsys-identity.json.

### Method047 — packed whole-decoder state and launch experiment

73922 TERMINAL0. Current hybrid Nsight trace records 10,240 kernels (159.302ms
union), 3,021 copies (4.069ms summed) and 451 memsets. All GPU activity union
163.739ms within a 237.655ms span. GPU idle gaps total 73.916ms; runtime API
interval union overlaps 41.536ms of those gaps, with 32.380ms outside API calls.
These intersections do not establish causation or recoverable time. Instrumented
single-request timings are diagnostic and cannot rank engines. Detailed summary
and handoff evidence: profile-hybrid-2b-b1-p64-d4-{summary,handoffs}.json.
GDN recurrence is 22.072ms summed, 342 calls; resource dump shows REG40,
STACK512, LOCAL0. No measured stack traffic or occupancy claim follows.

58804 TERMINAL0 built new packed-view/concat lowerings and an opaque FP32
materialization barrier between decoder layers. DLL SHA256
686b2294f1ae012edf1db0c6e6316a69010080cec3929a654eb0f3fae9753574.
CUDAPackedDecoderProgram uses one canonical state input, one whole-decoder
graph and the existing single readonly BF16 projection pool. It is UNWIRED to
serving and has no speed or quality acceptance. Original per-layer/default paths
remain available. Projection precision and FP32 layer boundaries are preserved.
First B4/C4 padded single-token component test completed 12 byte-exact hidden,
state, ID and slot records; packed-program-b4-c4.json/log. The originating tool
result was truncated and did not preserve its process exit status; subsequent
inspection found its completion marker/artifact and no live xlang3 process.
Expanded ragged/full-chunk suite 10103 now LIVE, sequential B1/2/4/8 C4,
B4/8 C16, B1/8 C64, each three resets for graph warm/capture/replay.
run-packed-program-validation.ps1 records explicit per-profile exit statuses.
Keep consumed runtime/DLL/test sources fixed while live. Whole-state prefix
commit, pretrained probes, physical memory, all-profile speed and independent
quality remain OPEN. Current best full cohort remains 255.0285ms; 2x UNMET.

10103 TERMINAL0: all eight packed-program profiles PASS, 12 records each.
Inputs include full chunks, padded single steps and ragged/empty rows; equality
is exact for every hidden/state byte and token/slot. These are two-layer
synthetic checkpoint controls, not independent pretrained-quality acceptance.
64360 TERMINAL0: packed-prefix build adds bounded KV prefix selection lowering
(select authoritative verified cache only in accepted span; restore backup
elsewhere), plus overlapping-range rejection in the FP32 barrier copy.
Experimental CUDAPackedDecoder owns one canonical state and both profile
programs. CUDAPackedDeltaReplayVerifier uses one graph for all small delta-core
replays and KV selection, avoiding rerunning target matrix projections on partial
commit. CUDATargetVerifier gains an explicit canonical snapshot hook; previous
per-layer fallback remains. CUDASpeculativeSession dispatches the packed helper
only for an explicit packed decoder. No frontend integration/default change.
Expanded transactional verifier and 24-case speculative session suites queued
sequentially by run-packed-prefix-validation.ps1; identity captured before run.

76847 TERMINAL0: all four transactional packed verifier profiles PASS (B4/8,
C4/16), six accepted-prefix patterns plus invalid token/control/count, complete
capacity, reset and close rollback checks. Seven packed session profiles each
PASS24cases: B1/2/4/8 P64D4, B1/8 P128D4, B8 P64D16. Native MTP and explicit
test-only oracle branch coverage are distinguished in raw records. Exact target
state/IDs/slots and EOS/length behavior agree with their own serial decoder.
DLL SHA256 a44cfc84a17d8bddf5a977a32d44025b25656b8dbe393c26d5ac1268eaa9483a.
An explicit default-off packedDecoderProgram CUDA profile is now connected;
requires the sharedProjectionWeights profile and both fused cores. Actual
profile and packed commit method are reported. Metal factory behavior unchanged.
Before pretrained use, run-packed-parent-validation.ps1 launches a separate
scalar cache-span selection oracle (signed zeros/input immutability/graph replay)
and five same-shape per-layer-hybrid versus whole-program parent controls.
No speed/quality/default acceptance; all prior gaps and2x target remain OPEN.

54858 TERMINAL0: independent scalar KV selection oracle all four shapes PASS,
six span patterns each, signed zeros and input bytes preserved. Five whole
parent controls PASS against same-profile per-layer hybrid: B1/2/4/8 P64D4,
B8 P128D4, three requests and18state/slot/token records each, reset/capacity
rejection and close/pool-release checks. All graphs capture/replay without
failure. Scripts and explicit exit artifacts retained beside original controls.
Pretrained2B P64D4 packed probe now launching, checkpoint/image/token controls
unchanged. Identity packed-before-pretrained-identity.json. One model/engine at
a time; no other GPU tests or compiler jobs and keep consumed runtime fixed.

52274 TERMINAL0 pretrained2B: two fresh speculative requests plus serial PASS,
all32IDs match saved case8 and every canonical target-state byte agrees. Warm
diagnostic220.78ms is not a timing cohort; retained model+session5,070,430,496B.
52299 TERMINAL1 in postprocessing ONLY: native28request cohort TERMINAL0;
summary rejected the new explicit packed commit label. Original failure log
preserved. Summarizer now validates packed_delta_replay_kv_selection only when
the explicit packed profile is true and still requires zero full recomputes;
repaired CPU-only summary succeeds. No benchmark rerun or evidence replacement.
Same16equal32groups: packed250.04055ms vs hybrid255.0285ms, LM298.8555ms,
vLLM286.766573ms. All28packed output sequences equal same-profile hybrid.
Retained ownership5,070,430,496B vs4,979,255,900B: +91,174,596B for~2%gain.
Tradeoff unresolved, option remains defaultOFF; quality and2x143.3833ms UNMET.
Updated progress-speculative-2b-b1 chart and comparison report.

### Method048 — standalone register-tiled GDN experiment, unwired

The profile's recurrence STACK512 motivates a bounded128x128-state prototype.
Four warps partition K and32value columns per block; carry propagation keeps
the original sequential K accumulation order with explicit FP32 multiply/add.
Q/K normalization and final RMS use the same128-thread reductions in separate
kernels. This increases blocks and workspace/launches; performance is unknown.
No runtime changes. Standalone clean build completed, native-equivalence and
independent FP64 fixed2e-5+2e-4 bounds/guard/input/replay screen now launching.
Tiled sources/test/build script live in existing folders; do not run another
GPU workload while active. Component timings cannot establish model speed.

Standalone delta-tiled-probe TERMINAL0 all9profiles: native output/history/state
bytes exact; independent FP64 max abs7.93e-8 within fixed2e-5+2e-4 bounds;
guards/input immutability and three CUDA graph replays pass. Resource metadata
original512localBytes/42regs versus tiled0localBytes/64regs. Warm component
examples2B B1T4 38.15us→32.00us, B1T64 398.80→386.00us, B1T128
861.80→674.82us;4B B1T64 479.34→326.35us. No full-model gain inferred.
73896 TERMINAL0 memcheck/racecheck/synccheck all9profiles zero findings.
Instrumented sanitizer timings are explicitly invalid for performance.

### Method049 — small-row BF16 projection bandwidth screen, unwired

Prepared because current trace attributes most summed GPU time to matrix
projections. Warp-per-weight-row kernels reuse weights across1/4/8input rows;
BF16 inputs/output, explicit FP32 FMA/reduction and no TF32. These reductions
can differ from cuBLAS, so full output drift is counted and128fixed FP64
FP32-accumulation/BF16-rounding samples are checked per candidate. No runtime
integration or precision/default acceptance. Clean standalone build; screen
launching alone on GPU0 with256MiB eviction before cold samples and immediate
warm samples. Eviction is not hardware-counter proof of cache residency.

Method049 TERMINAL0 all8shapes/3methods passed128FP64 samples per method;
small BF16 differences from classic counted, not hidden. Cold2B gate65.54us
classic vs59.39us custom; down62.46 vs62.53/63.49us;4B gate104.35 vs
107.52/108.54us and down107.52 vs111.62/112.13us (custom slower). Full
vocabulary1/4rows ~1.55/1.61ms unchanged. Warm cache results differ markedly.
Reject runtime integration on this screen: no consistent cold benefit, plus
new reduction-order quality obligations. Sources/raw evidence retained.
Current packed Nsight diagnostic now launching alone, two warm requests and
third capture. Identity packed-before-nsys-identity.json; keep all consumed
runtime/DLL fixed, no compiler or other GPU workloads until terminal.

57434 TERMINAL0 packed Nsight diagnostic: kernels11,665, summed171.357ms,
union171.303ms; allGPUunion174.921ms/span233.840ms. Earlier hybrid union
163.739/span237.655ms. Fewer copies1683versus3021 but more kernels/GPUwork;
packed orchestration trades CPU/copy overhead for extra graph work. Instrumented
single-image traces are diagnostic and cannot rank engines or isolate causes.
This supports keeping packed defaultOFF; its measured2%gain costs91MB.

### Method050 — inline request controls and direct embedding chunks

Existing CUDATensorIO::Update issues a host-to-device copy and stream sync per
small control array. New explicit update_request copies bounded CPU scalars as
CUDA launch parameters to three existing GPU outputs under one write lease.
embedding_chunk gathers the one BF16 table directly into caller-owned padded
chunk storage, eliminating temporary ID/embedding tensors and copy steps.
No projection, recurrence, norm or arithmetic order changes. Old paths remain.
44713 native build TERMINAL0, DLL8b0a83722bed6466b605a696bae7cab9c5c142c80997bc5f2a756562881c76b4.
16batch/chunk component profiles PASS exact legacy controls/gather bytes,
signed zeros, padding, CPU argument lifetime and invalid-input immutability.
Resource dump: WriteRequest20regs/stack0/local0; GatherChunk26regs/stack0/local0.
Default-off residentControls profile connects both hybrid and packed decoders,
verification and MTP controls/step/correction gathers. Wide visual prompt shift
keeps its existing GPU feature copies. Full model/quality/speed unaccepted.
73686 LIVE sequential integration: parent controls, transactional verifiers and
eight24-case speculative profiles. Identity and checked source/native archive
resident-controls-before-validation-source-runtime.zip retained. No other GPU
or compiler jobs; keep consumed runtime/tests fixed until terminal.

73686 TERMINAL0: all16sequential integration jobs PASS. Four same-profile
parent controls compare old IO/hybrid against resident hybrid/packed atB1/B8;
fourB8C4/C16 transactional verifier profiles pass six prefix patterns,
rollback/invalid/count/capacity/close checks. EightB1/2/4/8 P64D4 session profiles
each pass24native-MTP/test-only-oracle EOS/limit/ragged acceptance cases, exact
target state/slot/output agreement. Both layouts tested; old paths unchanged.
Three sanitizer checks of the independent16-profile native IO control are now
launching sequentially. Next pretrained hybrid resident probe, then matched
cohort, to measure actual gain and retained ownership. No speed acceptance yet.

93017 TERMINAL0: native IO16-profile suite under memcheck, racecheck and
synccheck allPASS, zero findings. Native byte-exact scalar/gather controls and
all previous integration gates retained unchanged. Pretrained2B HYBRID resident
probe now launching alone. Full source/runtime/controller archive and identity
resident-controls-before-pretrained-* retained. No other GPU/compiler activity
or consumed runtime edits until terminal. Speed/quality acceptance still OPEN.

74414 TERMINAL0: pretrained2B hybrid resident, two fresh MTP requests plus
ordinary serial control PASS. All32IDs match saved case8 and every target-state
byte agrees. Warm diagnostic229.89ms versus prior hybrid217.61ms is not a speed
gain/cohort; retained ownership after replay5,005,911,644B versus prior
4,979,255,900B. New inline IO owns no extra persistent GPU tensors; investigate
runtime graph allocator differences rather than assuming memory unchanged.
15291 LIVE full28request2B1P64D4 resident cohort. Identity plus checked complete
source/controller/native archive resident-controls-before-cohort-* retained.
Keep consumed runtime/DLL fixed and no other GPU or compiler jobs. Primary
latency/ownership judgment waits for complete same-input/equal-output results.

15291 TERMINAL0 full28request resident cohort. Same16equal32groups239.1925ms
versus old hybrid255.0285ms, saved vLLM286.766573/LM298.8555ms. All28output
sequences identical. Retained5,010,354,268B versus4,979,255,900B (+31,098,368B).
Allocator audit localizes growth entirely to MTP runtime contexts: old1context
126,123,780B versus resident8contexts157,222,148B. Probe had7contexts and
152,779,524B. Target decoder graph live allocations are unchanged. Removing
blocking metadata copies exposed pool retirement policy during queued prefill;
the new IO itself owns no additional persistent tensors.2×143.3833ms UNMET.

### Method051 — explicit serial CUDA graph reuse, validation pending

Default-off enable_serial_graph_reuse API added to CUDACompiledGraph. A pool
slot can be queued before GPU retirement only for an instantiated graph with
identical bindings, empty static-shape signature, no provider owner and the
same stream (per-thread sentinel also requires identical CPU thread). Warm
enqueue retires before capture to prevent duplicate warmup contexts. Dynamic
profiles, changed pointers, provider ownership and profiler/graph-disabled
paths retain conservative retirement. Static input dtype/shape checks added
for callers opting in. No serving/MTP integration yet.
NVIDIA documents sequential ordering of launches of the same executable graph:
https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/04-special-topics/cuda-graphs.html#using-graph-apis
This supports captured replay ordering, not overlapping enqueueV3 context use.
56162 first native build and60772 guarded build TERMINAL0. Standalone real-TRT
pool contract probe compiling: warmup, pending same-stream replay, changed
bindings, changed stream, dynamic signature, provider owner and cross-thread
per-thread sentinel. No speed/memory/correctness acceptance until tests pass.

Initial standalone pool probe native PASS. Its first memcheck run91117 returned
test assertion "Different bindings reused pending graph", zero CUDA errors:
fixed37ms spin had completed under instrumentation, so the pool legally reused
the retired slot. Failure/source preserved as serial-graph-pool-timing-sensitive-*
and serial-graph-pool-memcheck-timing-assertion.*. The corrected test uses explicit
host-function gates and precreated contexts, proving work remains pending during
each check. It also tests warmup retirement with a counter-driven gate release.
Gated native PASS, then12473 TERMINAL0 all3sanitizers PASS/zero findings,
including cross-thread per-thread-stream rejection. No bound or production
retirement assertion was weakened.
Explicit default-off serialGraphReuse serving/profile flag now enables reuse
only for the MTP program, before its first request. Requires residentControls
and CUDA graph profiles. All default paths remain unchanged. Ten24-case native
MTP session profiles now launching sequentially, requiring exact serial state/
IDs/EOS/limits and oneMTPcontext with positive serial reuses, zero graph failures.
Identity plus checked source/runtime archive serial-graph-before-mtp-validation-*
retained. Freeze consumed runtime/native/tests; no other GPU/compiler jobs.

17900 TERMINAL0 all10session profiles PASS24cases EACH. Both hybrid/packed,
B1/2/4/8 P64D4 indexed16 plusB8P128D16 full vocabulary. Every MTPgraph retains
exactly1context, graph failures0, and observed pending reuses116..383per suite;
native serial IDs/states/slots/EOS/limits agree exactly. Guarded DLL SHA256
a1782278b16c5a5da58d3e4a91feb4ecf04d85c0d566f878014f05ba9c961af9.
Pretrained2B hybrid resident+serial probe now launching alone. Checked source/
runtime archive and identity serial-graph-before-pretrained-* retained. Full
model memory/latency still unaccepted,2× unmet; no other GPU/compiler workloads
or consumed runtime edits until terminal.

### Method052 — fixed128 GDN state register specialization, unwired

Prepared as a separate standalone variant while pretrained serial-context probe
compiles; no consumed runtime/DLL changes. The earlier tiled variant adds control
and raw-core workspace. This alternative keeps the original128-thread block,
reductions and sequential K order, with fixed/unrolled128-state loops intended
to eliminate dynamic local-array access. Original convolution workspace and
output shapes remain. Not built, tested or integrated yet; no resource/speed
claim. Existing standalone delta test now accepts optional register128 while
preserving default tiled128 behavior and fixed independent FP64 bounds.

27445 TERMINAL0 real2B hybrid resident+serial: two fresh MTP requests and
ordinary serial match saved32IDs and every target-state byte. MTPcontexts1
both requests; pending reuses19then40. Retained ownership4,979,255,900B both,
restoring the pre-resident footprint. Warm single-case diagnostic209.20ms is
not a cohort. Full28request resident+serial cohort now launching alone; checked
source/runtime archive serial-graph-before-cohort-* retained. Freeze consumed
runtime/DLL; no other GPU/compiler jobs until terminal.
28511 standalone register128 build/screen TERMINAL0 BEFORE this cohort launch.
All9profiles byte-exact to original, independent fixed FP64 bounds/guards/input
immutability and three graph replays PASS. Raw delta-register128-probe.jsonl
retained. Native integration and sanitizers remain pending; no full-model claim.
Installed vLLM source audit separately confirms Qwen3_5MTP class/config support
in this vLLM0.32.0 environment; this is capability evidence only, not a working
MTP comparator run. Saved LM/vLLM cohorts still speculationOFF.

42001 TERMINAL0 full resident+serial cohort. Same16equal32groups239.44085ms
versus resident-only239.1925ms (no latency gain), all28sequences identical.
Retained4,979,255,900B versus5,010,354,268B:31,098,368B saved. OneMTPcontext
and positive serial reuses checked in every raw row. This is a memory fix;
2×143.3833ms and independent quality remain unmet. Chart updated.
34849 TERMINAL0 second fixed128 specialization variant with volatile shared Q/K:
all9profiles nativebitexact/FP64/guard/input/replay PASS. Resource168regs,
stack/local0 versus first variant255regs/local144B, original42regs/local512B.
No extra convolution workspace. B1T4 37.53→15.00us; full raw timings retained
delta-register128-volatile-probe.jsonl. First variant source/exe snapshots saved.
This variant is still UNWIRED; three sanitizers now launching with timing loops
disabled (all native/independent/guard/input/3replay checks preserved).
Trace audit separately sums packed ConcConc kernels9.693ms versus0 for hybrid;
flat variadic concatenation is a hypothesis, NOT implemented or a speed claim.

67143 TERMINAL0 register128_volatile standalone all9profiles under memcheck,
racecheck/synccheck PASS0findings with timing loops disabled. Numerical/native
comparison, guards, immutable inputs and3graph replays remain unchanged.
76716 TERMINAL0 native build integrates a distinct GarnetQwen35DeltaRegister128
plugin type (old type/version/payload preserved), original convolution workspace
and explicit default-off registerDelta128 profile. Target and prefix-replay
cores select the same implementation. First native B4C4 whole-program control
9935 PASS12records exactly against the original per-layer kernel/decoder.

### Method053 — flat variadic packed concatenation, validation pending

Packed trace's ConcConc kernels sum9.693ms versus0 in hybrid. Added a bounded
qwen35_concat_parts unary lowering with explicit tensor-list dependencies and
one TensorRT concatenation, replacing the optional chained outer packs. Explicit
default-off flatPackedConcat profile requires packedDecoderProgram. Includes
target state/payload and prefix-state joins; ordering/precision unchanged.
Native build76716 includes this independent feature, DLL SHA256
0820541ad5a81ab449011a8656ab70c48a027d7b0fd842afc39f148f06cba5a9.
9935 second native B4C4 flat-only PASS12byteexactrecords. Both initial features
tested independently; no speed/memory/model-quality acceptance yet.
54699 LIVE expanded validation: independent CPU concatenation oracle up to129
parts,5combined packed profiles,5same-shape parent controls,4transactional
verifiers and8x24-case speculative profiles. Checked archive/identity
register-flat-before-validation-* retained. No consumed runtime/test/DLL edits
or other GPU/compiler jobs until terminal. All previous failures retained.

54699 TERMINAL0 all expanded register/flat tests PASS: CPU concat oracle4shapes
through129parts,5combined programs throughC128,5same-shape old/native parent
controls (separate register-only, flat-only and combined),4B8C4/C16 transactional
verifiers,8x24-case MTP sessions. Exact target state/IDs/slots/EOS/limits preserved.
Native production resource dump differs from standalone: registerState128Kernel
REG255/STACK40/LOCAL0 versus standalone168/STACK0/LOCAL0; old native kernel
REG40/STACK512/LOCAL0. No full-model speed inference from standalone numbers.
Pretrained2B hybrid resident+serial+register-only probe now launching alone;
checked archive/identity register128-before-pretrained-* retained. Next compare
full final-state hashes to the prior same-profile old-kernel probe, then cohort.
Flat-only pretrained/cohort should be separate before interpreting combinations.
Keep consumed runtime/DLL fixed, no other GPU/compiler jobs until terminal.

60721 TERMINAL0 real2B register-only on hybrid resident+serial: two fresh MTP
requests and serial PASS saved32IDs and all target state bytes. Cross-probe
comparison to the prior same-profile old kernel matches all49final-state
SHA256 fields; pretrained-register128-state-control.json. Warm diagnostic
193.71ms, retained4,979,255,900B.1314 TERMINAL0 full28cohort: same16equal32
groups218.06535ms versus239.44085ms; all28sequences identical, retained
4,979,255,900B unchanged. SavedLM298.8555/vLLM286.766573, speculationOFF.
9%full-request reduction, not a2× or independent quality acceptance.
69692 LIVE flat-only packed pretrained probe, with register, resident IO and
serial reuse disabled to isolate concatenation versus original packed250.04055ms.
Identity/archive flat-concat-before-pretrained-* retained. Keep consumed runtime
fixed, no other GPU/compiler jobs until terminal.

69692 TERMINAL0 real2B packed flat-only, old register/IO/context options OFF:
two fresh MTP requests and serial PASS saved32IDs/allstates. Cross-probe final
last-hidden and canonical-state hashes match old chained packing exactly;
pretrained-flat-concat-state-control.json. Retained4,886,611,232B versus old
5,070,430,496B (-183,819,264B). Warm diagnostic212.85ms is not a timing cohort.
Flat-only full28cohort now launching alone, source/runtime archive and identity
flat-concat-before-cohort-* retained. Keep consumed runtime fixed/no other GPU
or compiler jobs until terminal. No full-request speed/2× acceptance yet.
Prepared vLLM MTP benchmark options/controller from installed source audit
(method=mtp, num_speculative_tokens); not executed. Defaults preserve old OFF
cohorts/cache budgets. A separate smoke/cohort must validate actual capability,
greedy outputs and cache capacity before any speculation-matched claim.

69503 TERMINAL0 flat-only28cohort: same16equal32groups241.2396ms versus old
chained packed250.04055ms; all28sequences identical. Retained4,886,611,232B
versus5,070,430,496B (-183,819,264B). Target graph runtime accounts for the
savings; session ownership238,298,804B is unchanged.3.5%latency reduction,
defaultOFF, independent quality and2×target still unmet. Chart updated.
Combined packed+flat+register+resident+serial pretrained probe now launching
alone after independent feature controls. Identity and checked archive
combined-before-pretrained-* retained. No consumed runtime edits or other GPU/
compiler activity until terminal. Next full combined cohort, then real larger
batches/4B and matched physical memory; vLLM MTP controller prepared, not run.

### Combined packed/flat + register + resident/serial cohort (2026-10-23)
Controller session22744 terminal0. Archived combined-before-cohort identity/source/runtime before execution. Real2B probe exited0; canonical and last-hidden SHA256 match flat-only, own serial and saved32 IDs match. Full28-request cohort: 211.0379 ms vs flat-only241.2396 ms, 16 equal32 groups, all28 sequences identical. Retained ownership4,886,611,232 B unchanged from flat-only (183,819,264 B below chained packed). Saved LM Studio298.8555 ms and vLLM286.766573 ms have MTP OFF. No algorithm equivalence, independent quality or physical peak acceptance. Target143.3833 ms remains unmet; options default OFF. Regenerated report and chart; move to actual4B/batches2/4/8 and isolated memory.

### Correction: audit actual LM Studio drafting, not requested flags (2026-10-23)
Saved LM2B batches1/2/4/8 have MTP-context creation and28/56/112/224 draft-acceptance timing events (1005/2008/4008/8037 generated draft tokens). Earlier blanket LM Studio MTP-OFF descriptions, including previous entries above, are WRONG and superseded here. The loader requested false; the cause of actual drafting is unresolved. No drafting events appear in4B saved logs: absence is UNKNOWN, not proof of OFF. All raw timing data retained. Added identifier-attributed, SHA256-linked parser and saved-comparator-speculation-audit.json. Derived candidate/summary labels corrected with previous versions preserved, plus correction manifest. vLLM saved options omit speculative_config. Exact cross-engine proposal settings remain unmatched. No goal or quality acceptance.
4B B1 combined cohort session93944 terminal0:353.8429ms vs original978.0611, LM672.345, vLLM535.769319;19equal32groups,23/28 same sequences, owned10,245,517,856 B. Changed trajectories remain quality-open. NOWLIVE2B B2 combined cohort session97275; archive combined-batch-before-*; freeze runtime/native/consumed scripts until terminal.

2B B2 combined cohort session97275 terminal0:250.91165ms,14equal32 groups,20/28same output sequences vsold. Retained ownership grew5,195,303,204→5,297,935,140B at trial4/group2. Exact counter evidence combined-2b-b2-allocation-growth.json: wide packed target graph contexts1→2, +102,631,936B; MTP graph remains1. Current serialGraphReuse only enables MTP graph, so this is a target-prefill context-pool growth finding. Do not hide growth or assert constant ownership. Report updated to ranges. Need bounded target graph reuse validation after current matrix, no runtime edits while live.
GGUF metadata audit (CPU-only session15193 terminal0):2B nextn_predict_layers1 and nextn weights exist;4B GGUF lacks both. This explains capability asymmetry but does not prove how LM flags resolved. Preserve provenance; any4B MTP competitor needs an explicitly new converted artifact/runtime capability check.
NOWLIVE2B B4 combined cohort45566, same archived runtime, sequential.

### Combined matrix completed; large-batch regression retained (2026-10-23)
All eight combined profiles completed serially.2B B1/B2/B4/B8 latencies211.0379/250.91165/327.74765/468.50130 ms on each profile's common equal32 subset.4B B1/B2/B4/B8:353.84290/444.97300/587.84380/1750.21180 ms.4B B8 has long-tail measured requests above6seconds. No2x or quality acceptance.2B B4/B8 same older sequences7/28,6/28;4B B2/B4/B8 17/28,17/28,13/28; changes remain open.
4B B8 timing cohort36275 terminal0: prefill target context grew1→2 on request2, retained16,479,981,628 B vs short memory diagnostic15,534,163,516 B with1context. Preserve long run; do not substitute short diagnostic as its memory usage.
Isolated memory75959 terminal0: steadyGarnet dedicated15,806,144,512/shared251,658,240 B; largest observed dedicated15,814,533,120 B, not true peak. Model close dedicated110,956,544 B, shared251,658,240 B retained. Matched LM memory40843 terminal0: steady dedicated10,284,007,424/shared100,663,296 B. All6JPEG hash groups equal, independent512slots and8parallel slots runtime-verified. LM loading not sampled, after-unload process exited so no enclosed close samples. MTP configuration differs;4B GGUF deliberately converted with --no-mtp, raw checkpoint has MTP. Native target+MTP state ownership not equivalent to Windows counters.
### Target graph serial reuse (implementation, not accepted yet)
Default-OFF serialTargetGraphReuse for packed/resident/CUDA graph profile enables existing same-stream stable-binding graph replay for targetP/D, final norm, verification norm and packed prefix replay. Existing serialGraphReuse still controls MTP. No new arithmetic or native DLL change. Motivation: bound target-prefill context duplication observed at2B B2 and4B B8. NOWLIVE validation47481,6parents/4verifiers/8MTP profiles; archive target-serial-before-validation-*. No speed/memory improvement claimed before measurement.

### Capture lifetime gate failed; target reuse not accepted (2026-10-23)
47481 terminal0: all18 target-serial integration profiles pass exact states/IDs/rollback and bound contexts1 (6parents,4verifiers,8MTPprofiles×24cases).87967 controller terminal1, native memcheck exit99 with68,169 reported errors; racecheck/synccheck not started. Repeated use-before-allocation reports in TensorRT Myelin fused kernel; original unmodified log preserved. TargetSerial OFF same24casecontrol48756 nativeexit99 with7,052reports, same kernel and allocation10,633,216B; counts are not causal evidence and execution aborts.
Target-only isolation3967 terminal0 controller, noMTP/noReplay: g0/s0 nativeexit0, repeats/states stable andmemcheck0; g1/s0 exit99 (41,952reports),g1/s1 exit99 (50,758reports), both die in firstprefill. New option remains defaultOFF; no speed/memory claim. This localizes to capture but does not identify a bug or prove a sanitizer false positive.
Installed ComputeSanitizer2026.1.1 build37663202. Primary references consulted: NVIDIA ComputeSanitizer manual https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html describes stream-ordered use-before-allocation detection; release notes https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html record prior false-positive fixes, which do NOT resolve this case. Related June2026 primary reproducer with NVIDIA tracking response https://forums.developer.nvidia.com/t/compute-sanitizer-and-cuda-graph-false-positives/373484 is analogous only, not acceptance evidence.
Prepared defaultOFF native diagnostic GARNET_TRT_REUSE_STATIC_BINDINGS: skip unchanged static-input/output address setters by querying actual context addresses, still enforceexactstaticshape/type and retainGPUaccessleases. Optional GARNET_TRT_ALLOC_TRACE logs synchronous allocator pointers/bytes. NOTBUILT/NOTTESTED yet; no claims. Native build waits until current vLLM job exits.
NOWLIVE vLLM2B B1 MTP3 smoke53273, default96MiBcacheperrow, fourrequests diagnostic only. Archive capture-controls-after-vllm-before-* records sources after isolation (unchanged during controls) and before competitor execution.

### vLLM MTP admission and full cohort (2026-10-23)
2B B1 MTP3 smoke53273 failed at96MiBcache/request (nativeexit1);43874 also failed at112MiB;36122 terminal0 at128MiB, fourrequests matching input hashes/promptIDs,2/4same sequences vsoldvLLM. Full28cohort25314 terminal0 at128MiB.15common equal32 groups includingoldvLLM serial: Garnet211.03570 ms, vLLM MTP203.273821, vLLM serial286.736432, LM302.209. Formal four-way candidate/control(newGarnet/flatcontrol/vLLMMTP/LM) has17equal32groups, same median values here; distinguish these subsets. vLLMMTP17/28same oldserial sequences; independentqualityopen.128MiBsmallest successfulTESTED budget, not provenminimum; actualcapacity512 unchanged. vLLM runtime resolvesQwen3_5MTP and captures speculator. FP32conv/statecache requested, runtime selectsTritonGDNspecpath becausefusedCUDArequiresBF16conv. NativeBF16conv-cache andotherMTPbatch/model profiles remainpending. Latest2xthreshold101.63691ms;GarnetcurrentlyslowerthanMTPvLLM. CPUmatched summaries support explicit vLLM suffix, preservingoriginalcohorts; report/charts updated.
### Capture lifetime isolation continued (2026-10-23)
99730 terminal0 build+eight independentplainCUDAcontrols: sync/async allocation, created/PTstreams, with/withoutprelaunchwait, allnative0/memcheck0. Native DLL5aa1d6ac18992ab0f0044a0e6393ac870a47448c63f56665ec6b4c6f53232d9e.96786 terminal0controller, actualcacheBindingdiagnostic g0/s0 exit0,g1/s0 99,g1/s1 99. Redundantsetter hypothesisREJECTED asfix. All rawlogsretained.
30413 terminal0controller: contextsetTemporaryStorageAllocator(existing synchronousallocator) g0/s0 exit0,g1/s0 99,g1/s1 99; noadditionalallocatorcallbacks comparedruntime-onlytrace. NativeDLLf7749296a897042ae7fe74583cb07d15ee1f96a4aa2b087f1639c7dd277d6434. TemporaryAPIoverrideREJECTED asfix; failing10,633,216B blocknotintrackedruntimeallocator. Do notimplementcached temporaryallocator onthisunsupportedhypothesis.
81372 terminal0 build+fouradditionalplainCUDAcontrols withmallocAsync/memset/kernel/copy/free INSIDE capture: all0/memcheck0, includingPTstream. NativeDLL3b686435fd41fd8cd831c3427cb4fe06cbefef0a26ea4668481648e43e08d4f6 addsdiagnosticgraphnodetraces,auxcounter,anddefaultOFFzeroAux flag. Defaultg0/s0 memcheck0 andall3trials byteidentical tooldnative. Defaultg1/s0 andg1/s1 memcheck99. Defaultdecodergraphsactualauxstreams[0,2,0] fordecode,prefill,finalnorm. Capturedprefillgraph52nodes, NOalloc/free/childnodes;dumpcapture-garnet-cuda-graph-71355333737800-2.dot andmatchingerrorlogsretained. Failingbufferallocatedoutsidecapturedgraph. No sanitizerwaiver andnocausalclaimyet.
NOWLIVE71726 singlechangedbuilderoption maxAuxStreams0; all3target-onlycapturecontrols -trace-aux0. Otherdiagnostics(staticbindingreuse,syncTempAllocator)OFF. Native/sourcefrozenuntilterminal. Needaux0outcome→exactstatecomparisons→fullMTPsanitizers→real4B8cohort+memory ifclean.2xqualitymemorygatesremainopen.

### Zero-auxiliary-stream route and instrumentation scope (2026-10-23)
71726 terminal0: all3target-onlymemcheckcontrols PASS0, states/IDs/slots byteequal oldnocapturebaseline. Aux[0,0,0]; onlyserialtarget1boundscontexts[1,1,1], otherwiseP64canstillgrow2. zero-aux-capture-control-comparison.json records exactcomparison.40997:18numerical/lifetimeprofilesPASS0; fullMTPB8memchecknative0, all24casesPASS, zeroerrors. Fullracecheck interrupted forcost, targetPID34392 verified owned; exit-1 andinterruptionJSONpreserved, notpass. ScopedGarnetracecheck24064alsointerrupted forexpensive/redundantcoverage, noerrorsdisplayedbutINCOMPLETE; syncchecknotrun.
CORRECTION to scope wording: zero-aux-native-device-code-preservation.json proves currentnative .nv_fatb sectionbyteidentical toolder08205native (SHA8367538762faca0bb914017bef4da7fb6d9f712b6e921603754d9efe9eaa2cc7,1,111,552B). CPUregistrationsectiondiffers. Prior delta race/sync checks usedSTANDALONE -O3component binaries, whose resources differ fromnativeCMakebuild; doNOTclaimexact-DLLrace/synccheckcoverage. Nativecompiledkernelsunchangeddoesnotreplaceunperformedintegrationchecks. Originalblanketstatement superseded. Numerical/fullmemcheckgatespass; newbenchmarkisexperimentalwithquality/race/sync/2xacceptanceOPEN.
Nextsingle-stream+targetserial4B8cohort andisolatedmemoryrunaresequential, native3b6864... archivedbeforelaunch. No cachesettingorprecisionrelaxation; stalehistoricalcohortsretained. Planforremainingmemory(aftermeasurement): considerdeferredpackedstatecommit toavoidfullcanonicalbackup. Current4B8backup/headfield696,992,384B withcanonicalstate696,254,464B; thisisapendingdesign, NOTimplementedorclaimedassaved. ComponentsJSONexactlysums15,534,163,516B.

### Bounded-context measurement and exact-DLL component checks (2026-10-23)
57933 terminal0: real4B batch8 targetserial+zeroAux28-request cohort and isolated
memory ran sequentially.15common equal32groups:1129.03750ms versus oldcombined
1763.29130ms, savedvLLM665.647265/LM1348.2737ms;28/28sameoutputsequences.
Retained15,534,163,516B constant,945,818,112B below oldlongcohort with2contexts.
NewisolatedWDDMdedicated15,975,682,048/shared81,788,928B. The oldshortmemory
diagnostic alreadyhad1context andsameownedbytes; itscounterdistribution differs,
so no largephysicalresidencygain is claimed from context reuse alone.
33706 terminal0: exactnativeDLL C4/B8 two-layer packed/per-layer12recordcomponent
passes Garnet-filter racecheck and synccheck withzero findings. This is distinct
from standalone probes and doesNOTreplace interruptedfullMTPintegration checks.

### Deferred canonical-state commit (2026-10-23)
Implemented default-OFF deferPackedStateCommit for packed target+prefix replay.
Verifier borrows canonical state unchanged until final acceptance, retaining only
small request backups. Exclusive owner/thread guards block parent mutation while
pending; rollback restores request buffers. Final-write failure poisons until reset
because completion is unknown without a full backup. No native math/precision change.
25832 terminal0:4verifier+8MTP profiles pass original state/ID/stop/limit checks;
new tests cover pending reset/step/close rejection, unsupported verifier rejection,
recoverable read failures and post-write failure poisoning/reset. Unfiltered B8
verifier and full24-case MTP memchecks both exit0, zeroerrors. NativeDLL remains
3b686435fd41fd8cd831c3427cb4fe06cbefef0a26ea4668481648e43e08d4f6.
53866 terminal0: same real4B8 full28cohort, then isolated memory, archived
deferred-state-before-real-4b8 identity/source/runtime.15common equal32groups:
805.40880ms vs1129.03750ms same-profile control (-28.66%),28/28same sequences.
Owned14,837,909,052B vs15,534,163,516B, exactly696,254,464B removed.
Warmdedicated15,279,423,488/shared81,788,928B; largestobserveddedicated
15,287,812,096B is sampled, nottruepeak. Dedicateddrop696,258,560B; counters
andretainedownership are separate metrics. LMdedicated10,284,007,424B remains
farbelow. vLLM665.647265ms andLM1348.2737ms; threshold332.8236325ms UNMET.
Independent quality andotherfullprofileacceptance remain open; no default change.
NOWLIVE74899 one warmed4B8 Nsightdiagnostic, repeatedsameimage perrow, notcohort
timing/quality; freeze consumed source/native, no competing GPU/compiler. New
run-combined-nsys controller and profile flags recorded in prelaunch archive.

### Current4B8 trace and native prefix-commit implementation (2026-10-23)
74899 terminal0 diagnostic: one warmed request, same real image repeated in8rows,
fresh encoding/reset. GPU kernel sum1339.104911ms, activityunion1453.137850ms,
span2763.029824ms. Trace overhead is substantial; primary805ms remains separate.
Top summed families: WMMA32 projection318.83ms, convolution171.17ms, Myelin
packed moves139.89ms, indexedDot124.87ms, registerState110.14ms. These are
instrumented families, not causal predictions, transfer estimates or speedups.
Actual projection implementation is cuBLAS GemmEx, not a homemade WMMA kernel.
80576 terminal0 sequential vLLM4B8 MTP3/BF16conv/FP32recurrent cache128MiBrow:
smoke4requests then full28cohort. Runtime explicitly reports GDN spec kernel cuda.
On14common equal32groups G823.80430ms/control1131.57290ms/VMTP850.555313ms/
LM1347.86690ms. SavedserialvLLM is faster; do not replace it with the slower
MTP profile to claim a win. New intermediates differ; quality remains open.

Native prefix method implementation precedes this journal entry; no prospective
pre-code design timestamp is claimed. Motivation is measured1,459,170,304B of
prefix replay output+TRT runtime on4B8, beyond its small request buffers.
New default-OFF cuda_packed_prefix API validates complete ordered packed layout,
loads small conv/dt/A_log parameters, and runs3state-only kernels: convolution,
128-wide recurrent prefix replay and accepted KV-slot copy. Canonical state is
read/written in place after recoverable head work. Q/output normalization and
gated hidden reconstruction are omitted because this operation only commits state;
original target hidden/output still determines all predictions. FP32 state-update
order is retained; exact numerical validation must establish this, not inference.
No input/output aliasing is introduced into TensorRT; this is a separate CUDA op.
Native build80594 terminal0, DLL763cd3ce40e74688ad18fd936e7464ae551a9497ca699e24848fc89e2b758227.
62276 validation currently running: first4verifiers pass C4/C16 B4/B8 with exact
ordinary native states and new invalid-span/layout/final-write fault controls.
MTPbatchprofiles progressing; sanitizer and pretrained/memory/timing evidence
still pending. Do not claim1.46GB saved or performance from this implementation.

### Native prefix validation, pretrained proof and first cohort (2026-10-23)
62276 terminal0: all12 native prefix profiles pass exact ordinary-target states,
outputs, stops/limits, ownership, invalid-span/layout admission and fault recovery.
15425 terminal0: verifier B8 and full24-case MTP B8 unfiltered memcheck zeroerrors;
new3 packedCommit kernels scoped racecheck0hazards and synccheck0errors inside
fullMTPsession. This does not claim full-vendor/old-kernel integration race/sync.
Native resource inspection retained: convolution42regs/stack80, recurrence255regs/
stack152/shared540, KV28regs/stack64, LOCAL0. No register/occupancy speed claim.
66164 terminal0 real4B8: two production native-MTP requests then separately loaded
ordinary non-MTP target, never concurrent. Same recorded sequences and ordinary
outputs; full last-hidden20,480words and canonical174,063,616words are finite and
have matching SHA256 in bounded column-chunk order. Hashing is diagnostic, not
timing; ordinary control explicitly asserts speculationdisabled. Native helper
owns28,318,144B (workspace25,165,824; parameters3,151,872; layouts448).
5683 full28 latency cohort completed native0, isolated memory still running:
15common equal32groups727.84200ms vs deferred805.40880ms (-9.63%),28/28sameIDs.
Retained13,407,056,828B versus14,837,909,052B (-1,430,852,224B). SavedvLLMserial
665.647265ms andLM1348.2737ms; 2x332.8236msUNMET. DefaultOFF, qualityopen.

### Prospective shared output-storage design (before source edits)
Recorded work/qwen35-cuda/shared-output-arena-design.json before implementation.
SDK X::Tensor::View shares actual storage/provider ownership; tensor_info.byte_size
describes remaining storage span, not a logical view footprint. Proposal: allocate
one maximum output pool and bind stable exact-shape contiguous views to sequential
P64/D4 graphs before their first runs/captures. Canonical state and all graph inputs
remain separate. Pending verifier ownership already rejects prefill/reset. Reject
input/output overlaps and rebinding, validate engine output descriptors, and count
the pool once with explicit external-output accounting. Estimated4B8 duplication
701,497,344B is a design estimate, not measured. No source edit or result yet.

### Native-prefix physical counters and shared-output validation (2026-10-23)
5683 terminal0: isolated native-prefix4B8 warmdedicated13,834,473,472/shared
81,788,928B; largestobserveddedicated13,842,862,080B, samplednottruepeak.
Owned13,407,056,828B. LMwarmdedicated10,284,007,424B remainslower; memoryand2x
acceptanceOPEN. Updatedownershipcomponents sumexactly; nativehelper28,318,144B.

Shared-output design then implemented defaultOFF sharePackedOutputStorage. New
set_output_storage validates the built engine's exact static output shape/type,
current device, writable contiguous storage, sufficient span and256-byte alignment
(installedNvInferRuntime.h requirement). Binding is accepted only once before
firstexecution. run_device rejects input/output storage overlap before enqueue;
host run rejects this explicit device-only mode. SDK Tensor.View retains actual
storage/provider ownership. TensorInfo.byte_size is a remaining storage span, so
stats now separate logical output footprint, external reachable span and owned
output bytes. Decoder counts one max-sized pool plus privately owned head output.
Nativeprefix input validation admits sufficient spans for true SDK views.
88874 build terminal0, DLL4420266ece5baae383fd8c0f0bb9c58a8a77b70ee58aaa5046d6e9ec54e1893a.
PE .nv_fatb exactly matches prior763cd3native:1,189,376B,
SHA78eb0a537951f9aa599aaf7bbcf8571dee8fd5ef06cca951ea98a578f6376e4c.
CPUregistrationmetadata differs; this does not prove new host lifetime behavior.

Firstgenericfixture exit1 beforeengineexecution: omitted requiredfusiongraph
boundary, expectedgraph error. Prior source/archive/log retained. Only harness
boundary corrected; numerical/lifecycle assertions unchanged. Corrected generic
storage fixture PASS: two shapes/fiveinterleavings/capture/replay, exactprivate
control, invalidshape/CPU/symbolic/overlap/rebinding/hostrun rejection, caller
owner-drop and closure of other graph/IO owners preserve remaining views.
67223 terminal0 all12 shared-output verifier/MTP profiles PASS exactstates/IDs,
all24cases persession and transactionfault/guard tests. Every target program
reports borrowed outputstorage/owned0 and decoder poolcountpositive exactlyonce.
Initialsanitizercontroller failed argumentparsing before targetlaunch: PowerShell
flattened single prepended job array into strings; retained output-arena-s-b8-t
log/exit1. Controller now declares explicit job tuples and validates each row.
21898 NOWLIVE correctedsanitizers: genericstorage and B8verifier unfiltered
memcheckPASS0/zeroerrors; full24-caseMTPmemcheck pending. No performance/memory
gainclaimed for arena yet. Freezeconsumed source/native, nootherGPU/compiler.

### Completed shared-output and2B controls (2026-10-23)
21898 terminal0 genericstorage/verifier/full24caseMTP unfilteredmemcheck allzero.
7564 terminal0 real4B8 fullfinitehashprobe then28cohort thenisolatedmemory.
All174,063,616canonicalwords+20,480lasthiddenwords finite, hashes and outputs
matchtwoMTPrequests andseparately loadedordinary target. Recordedoutputs match.
727.64920ms vs727.84200ms,15common equal32groups,28/28sameIDs: no speedgain.
Owned12,705,559,484B saves701,497,344B exactly. Warmded13,161,283,584/shared
81,788,928B; largestobservedded13,169,672,192B, nottruepeak. Physicalcounter
drop673,189,888B differsfromownedreduction; keepmetricsseparate, noresidencyclaim.
Addedall-baselines common-subsetsummary:14groups G741.09425/control736.19025,
LM1347.86690,Vserial667.5786915,VMTPBF16conv850.555313ms. SerialfasteronSAME
subset,threshold333.7893458msUNMET. This subset differsfromfour-way15groups.

28926 terminal0 RDC2B1 currentnative+arena pretrained/28cohort/isolatedmemory.
Fullfinite-statehashes/ordinaryoutputs/recordedIDsPASS.17common equal32groups
199.96260ms vsoldcombined211.04010ms,VMTP203.273821/LM302.209;28/28sameIDs.
Owned4,749,211,752B; warmded5,078,798,336/shared81,788,928B. No matched-current
LM2Bphysicaldiagnostic yet.2x101.63691msUNMET. Thesecombine severalmemoryoptions,
so their11mschange is not a one-feature causal claim.
Derivedcomparison initiallyselectedcandidateasitself whenmissingintermediate
controlfiles leftanunchangedtag. Rawtimingsintact; uncorrectedsummarybackedup,
recomputedagainstoldcombined andall28actualoutputsequencecomparisonsmatch.
Controller now walks strictly changed predecessor tags and summaryrejectsself
comparison. InitialauditlistedtwolegacyJSONschemaswithbothsuffixfieldsmissing;
theywerefalsepositives(None==None), notselfcontrols. Presence-awareauditfinds
no remainingexplicitselfcomparison. No historicalrawdata changed.

### CUDA linking-mode resource audit and scoped build experiment
Standalone168versusnative255registerdifference investigated beforeanother
kernelrewrite. First50862 CPU-onlycompilemirroredFLAGS/definitionsandplaced
native+templatebodiesinsameTU: nativeandvolatiletemplate168regs/stack0.
InspectionthenfoundCMakeCUDAcompilerRULEadds-rdc=true beyondFLAGS.53125 sameTU
withoneadditional-rdc=true: both255regs beforefinaldevicelink. Objectresources
andSHA256sources/objects retainedregister-template-audit-comparison.json.
This isolates a resourceeffect, not amodelspeedup. NVCC-O specifieshostoptimization;
deviceoptimizationimplicitwithout-G. Missingexplicit-O3 wasNOTadiagnosis;
primarydocs https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/ consulted.
Prospectiveqwen35-whole-program-design.json recordedbeforeproductionedits.
DefaultOFF GARNET_QWEN35_WHOLE_PROGRAM_CUDA createsnonseparableobjecttargetonly
for selfcontainedqwen35_delta.cu+qwen35_packed_commit.cu; allotherCUDAkeepRDC.
No mathematical/precision/fast-math changes. RuntimebuildmarkerfromactualCUDA
unit andcachepolicy saved; benchmarklabels mustmatchactualDLL.
43139 buildterminal0. DLLf34b5894897220f48938d86c81282fdfea2222e29b80302b6d374bc5fa228788.
ActualDLLtarget/prefixrecurrences168regs/stack0/local0; prefixconvolution64regs/
stack0.88543 all12originalC4verifier/MTPprofilesPASS exactstates/IDs/faultguards;
actualmarkerTrueinall12.34860 unfilteredstorage/verifier/full24caseMTPmemchecks
zeroerrors.7340 NOWLIVE exactDLLtwo-layerC4B8 program/controlrace+sync followed
bynewprefixkernelsscopedrace+sync infullMTP. Numericalandresourceproofdoesnot
establishmodelperformance. Fullpretrainedcross-RDChashes/cohorts/memorypending.
WiderD8/MTP7gatesprepared; verifierwidth8 extendscounts0..8/mixedcounts while
defaultwidth4 andalloldbounds remainunchanged. Notexecutedyet. No2xacceptance.


2026-10-23 continuation: session54165 completed native0 for whole-program 2B B1 pretrained probe, full28 cohort and isolated memory. Cross-RDC full finite state/output checks all PASS;28/28 saved sequences unchanged. Same17-group subset: whole-program227.03210ms versus RDC199.96260ms, vLLM MTP203.273821ms and LM302.209ms. This is a13.5% latency regression; lower registers are not a demonstrated speedup. Owned memory unchanged4,749,211,752B. Keep option experimental/defaultOFF; no promotion. Session28081 now runs 4B B8 whole-program pretrained probe, full28 cohort and isolated memory strictly sequentially. Sources/runtime archived in source-runtime-whole-program-4b8-pretrained-cohort-memory.zip with verified identity. Consumed source/native/controllers frozen. Goal ACTIVE;2x unmet. No confirmation dialogs needed or automated.

2026-10-23 prepared next experiment (not executed): D8/MTP7 uses existing admitted runtime switches and native state kernels. Added run-wide8-{cohort,pretrained,sanitizers}.ps1 and probe_qwen35_wide8_pretrained.py in existing source/work directories; PowerShell/Python syntax passed. Numerical controller covers four verifier and eight MTP batch/prefill profiles; sanitizers must pass before real pretrained state probe and full28 cohort. No source precision/tolerance changes. Ordinary D8 target is separately loaded; saved D4 output equality is reported independently. Current session28081 remains exclusive GPU owner. New harness is not evidence of performance or quality success.
Research consulted: docs/research/metal-method-evidence/index.json embedded original reports for combined-graph-prefill and combined-prefill-reset, because relative result directories are absent from this checkout. They document rejected numerical profiles and end-to-end tradeoffs; no Metal component gain is used as a CUDA speed prediction.
Prompt packing candidate identified: runtime/cuda_serving.py currently creates one copied row/visual tensor per prompt token then batches with another per-token copy. A prospective contiguous [T,B,H] prompt buffer could assemble each row using pitched copies and expose contiguous [B,H] SDK views, preserving exact text/visual values and padded zeros. Not implemented; lifetime, overlap, device, logical-span validation and real-state equality are required. No image/prompt/output caching.

2026-10-23 diagnostic correction: session28081 was INTERRUPTED_INCOMPLETE after the 4B B8 probe stayed CPU-active >13min without a completed-request marker. Stopped only owned parent48724 first, then child52116; retained original log and explicit exit-1. No 4B NoRDC cohort/memory result was produced or accepted. This does not establish a native kernel failure or prove the exact cause of the delay.
Prospective finite-f32-bytes-design.json recorded before edits. Added bounded diagnostic-only CUDATensorIO.finite_f32_bytes: contiguous writable CUDA FP32, <=1048576 logical words/4MiB, READ lease, synchronized D2H, reject all NaN/Inf, owned exact little-endian bytes. Probe retains original column chunk order, full coverage and SHA256; legacy branch remains available. No inference path uses this helper. New DLL6d657c13097263db6837fcdd91041519854fb72b4f5b869a188a65bc15f47f30; PE CUDA fatbin exactly matches F34B5894 predecessor (1230848B SHA1393e7d5e2d7d53524728b407010853a2c2810cc5205be666f07be3939f565bf). Host-only diagnostic addition; device equality is not a complete host correctness proof.
Build85881 exit0. Session91859 normal validation and unfiltered memcheck exit0, all10 checks and zero errors: exact finite IEEE bits including signed zero/subnormals/maxfinite, old SHA equivalence, all nonfinite rejection, CPU/symbolic/oversize rejection, logical storage view + retained owner, repeated reads, maximum4MiB copy. Future probes have explicit load/infer/hash markers.
Session71927 LIVE: new-byte 2B B1 full-state probe + cross old RDC and same NoRDC legacy hashes; then 4B B8 full-state probe + old RDC hashes; then 4B B8 full28 cohort and isolated memory. Parent ErrorActionPreference=Stop explicitly gates dependent work. Exact source/runtime archive finite-f32-bytes-before-real-source-runtime.zip verified, now includes xlang3_runtime.dll hash and binary. Existing runtime DLL predates both compared builds; no runtime-upgrade explanation established. Freeze consumed sources/native/controllers; no compiler or competing model.

2026-10-23 native finite-byte real2B B1 probe PASS0, full ordinary/repeated finite-state hashes and outputs exact versus BOTH saved RDC and same-policy NoRDC legacy-method probes. Evidence finite-f32-bytes-2b1-cross-method-control.json, whole-program-fastbytes-pretrained-2b-b1-cross-rdc-state-control.json. Current71927 continues 4B B8 probe then cohort/memory; consumed files unchanged. Redirected stdout phase markers appear buffered; absence of a marker does not identify a slow phase. Future wide8 probe now writes separate per-stage progress JSON, avoiding that ambiguity.
4B B8 existing RDC/shared-output measured phase medians across20 warm groups (not additive): input61.771ms, prefill180.898ms, decode472.857ms, fullwall727.379ms. Target verification13calls, draft26calls, correction13calls median. Wider windows remain experimental, no forecast of gain.
Prepared only (not compiled/run): standalone cublasLt layout diagnostic gains explicit decode rows4/8/16/32/64 and prefill rows64/128/256/512, across existing2B/4B gate/up/down matrix shapes. Historical quick/full grids unchanged. Existing fixed FP64 bounds, BF16 I/O/FP32 accumulation eligibility and compute-type reduction controls unchanged. This unconsumed future diagnostic source is independent of the frozen model jobs. No compiler or competing GPU diagnostic during71927. Need archive old probe executable before eventual rebuild; isolated matrix timings cannot establish full-model gain or cache residency.


2026-10-23 continuation: completed NoRDC 4B B8 with finite-byte probe/full28/isolated memory (71927 exit0). Full ordinary and saved RDC state/output hashes exact. 15 equal32 groups716.79650ms versus RDC727.64920ms, a small unrepeated difference; owned12,705,559,484B unchanged. All-baselines common14 remains far above2x target333.789ms. NoRDC2B regressed13.5%; no promotion. Restored RDC DLLb6e239537e1a2710d755b86644f5c3c28989a935c9f1905ed1f466e720091fee. Fatbin differs from oldRDC; complete SASS listing matches only after explicit generated source identifier normalization (rdc-finite-bytes-sass-preservation.json); not full binary/host equality.
Wide D8/MTP7:92048 numerical12 profiles and unfiltered verifier/session memchecks pass; prefix-only race/sync pass.26168 terminal0 real probes then full28 cohorts sequential.2B B1 ordinary-state probe pass;17 common groups205.90170ms versus D4 199.96260;24/28 same saved sequences.4B B8 ordinary-state probe pass;14 common groups833.61020ms versus D4 741.09425;14/28 same saved sequences. Owned grows to4,759,163,224B and12,841,941,820B respectively. Reject both speed candidates; changed cases need independent ordinary controls if pursued. One-case state preservation is not broad quality proof.
Isolated cuBLASLt36-shape screen1893 exit0: current-layout/bounded-workspace median shape ratio1.0401,max1.383; any-layout median1.096,max1.394. Heterogeneous isolated shapes are not model speed estimates; no production integration. Conditional WHILE/child graph prototype51022 exit0:84 bounded-counter cases and unfiltered memcheck/racecheck/synccheck zero findings. No model weights/acceptance algorithm/performance claim. Initial incompatible CUDA API signature compile failure retained.
NEXT existing static-binding opt-in with aux0, prospective static-bindings-aux0-design.json. No runtime math/precision change. Require12 numerical profiles,3 unfiltered memchecks, full finite pretrained state and full28 cohort sequential. No promotion without repeat. Same existing work folder, no permission popup currently. GoalACTIVE;2x and independent quality unresolved.

2026-10-23 12:00 PDT:86197 TERMINAL0 static bindings+aux0. All12 numerical profiles and3 unfiltered memchecks PASS/zero errors. Real2B1 full finite state and output hashes exact against separately ordinary and saved RDC. Full28 cohort17 common equal32:198.21250ms versus historical199.96260ms, vLLMMTP203.273821ms;28/28sameIDs;owned4,749,211,752B unchanged. Difference0.9% unrepeated, no promotion/defaultOFF. Need paired repeat before attributing speed gain.
Next prompt packing prospective design recorded before edits. New copy-only cuda_prompt_io.cpp and exact-byte oracle prepared while prior jobs ran but were uncompiled/unconsumed. After86197 exited, added API declaration and defaultOFF packed-decoder serving flag packPrompt. One [T,B,H] buffer, 1..2 pitched row copies per batch item and SDK [B,H] views replace per-token allocation/copy/stack. All descriptors, runtime/device/shape/logical bytes/range and256MiB cap validated before allocation. Source owners use grouped SDK leases; views retain backing storage. No weight, precision or arithmetic change. Require build/device comparison, component bytes/ownership/invalid gates+unfiltered sanitizers, unchangedC4 numerical profiles+memchecks, real full finite state,28 cohort. No performance claim yet.

2026-10-23 12:03 PDT: Prompt component28checks plain/mem/race/sync PASS0 (copy helper correct). Source review found serving bug BEFORE real-model timings: vision.encode_patches returns reused graph output, so retaining all batch rows before packing would replace earlier row images. Stopped owned chain parent55996 then current child55332;35669 exit-1 INTERRUPTED_INCOMPLETE, explicit metadata preserved. First PID stop attempt saw child changed and refused without killing anything; refreshed exact owned tree then stopped safely. Partial C4 profile logs retained, not all12 pass. No real probe/cohort accepted. Fix snapshots vision rows for batch>1 with existing io.clone before next encode; batch1 uses live output immediately. Added production _prepare test with deliberately reused fake vision buffer, deterministic distinct images/text and independent expected bytes, B1/2/4/8 mixed/ragged. New C4 evidence prefix promptpack-build-fixed avoids overwriting original partial logs. Native DLL unchangedd31, no compile needed for this Python lifetime fix.

2026-10-23 12:09 PDT:19903 terminal0 fixed production _prepare lifetime test8cases (B1/2/4/8 legacy/packed, mixed images and ragged text, deliberately reused fake encoder output) plain+unfiltered mem/race/sync PASS0/zero findings. Native helper28check suite remains PASS. Fixed source/runtime archived prompt-pack-fixed-before-gates-source-runtime.zip, DLLd31ed044a39ac9157f2f8c19754cf6602759d2aad574d99d778091b0ce8a6931.33755 LIVE sequential fixed C4 numerical12profiles PASS;3unfilteredmemchecks in progress, then2B1 and4B8 full finite real-state probes/28cohorts. No confirmed prompt-packing timing gain yet. Freeze consumed runtime/native/tests/controllers.
Prepared only: window2-mtp1-design.json and D2/MTP1 numerical/sanitizer/pretrained/cohort harnesses. Runtime still rejects packed width2 until current jobs terminal; no math/precision change planned, original default4 retained. Smaller window may reduce recursive draft and unused target slots but increases window/prefill calls; no forecast. Prepared run-packed-graph-nsys.ps1 and whole-graph interval parser, not run. Current profiler source now reads native-prefix/shared-output/packed-prompt flags. Use graph trace with CUDA event tracing disabled based on local Nsight CLI; internal graph gaps remain hidden and no recoverable-time inference. Input-only packing cannot2x2B:20warm phase medians7.148ms input/30.575prefill/160.854decode/fullwall198.854 (not additive or comparator subset).4B8 input61.771/prefill180.898/decode472.857/full727.379.
Prepared current-DLL Qwen3-VL two-request16-token legacy smoke controller with same existing regression-cache folder; not run. Need archive identity before each future model/profiler. Main chart now uses D4 reference memory profiles, excludes unselected NoRDC/D8/static;2B1/4B8 include latest memory methods, other profiles older. User shown chart at12:07;2xstillunmet. No permission popup blocked any work.

2026-10-23 12:16 PDT:33755 TERMINAL0. Fixed C4 all12numerical profiles PASS,52 recorded actual RDC markersFalse,3unfilteredmemchecks0errors. Full finite ordinary/repeated target state and output signatures exact against savedD4 for BOTH2B1 and4B8; serving lifetime snapshots preserve distinct batch images.2B1 full28/17common199.03010ms versus199.96260 savedcontrol,28/28sameIDs,owned4,749,211,752B; small difference unpromoted.4B8 full28/15common698.51660ms versus727.64920 savedcontrol, vLLMserial665.647265, LM1348.2737;28/28sameIDs,owned12,705,559,484B unchanged.20warm phase medians input40.4204ms vsold61.771;prefill181.1048;decode472.7068;fullwall700.4319 (not additive or comparator subset). About4% savedcontrol difference, not paired/repeated; no promotion or2xclaim. Need isolated memory and same-build paired comparison.
Native d31 device fatbin unchanged versusb6. Found preexisting native IO view-span distinction in code review: Zero uses remainingstoragebytes and Clone/CopyInto compare whole accessible spans, despite SDK views having smaller logical shapes. New production token views are consumed as sources by WriteChunkToken, so current real-state results are preserved; general writable-view API needs logical-footprint fix/tests before promotion. Prospective prompt-view-io-logical-span-design.json, not implemented yet; currentmodel/native frozen for memory/profiler.
User asked chart again; shown D4 reference chart, explicitly older2/4/8 profiles vs latest2B1/4B8. Asked meaning of retained ownership: explained tracked GPU allocations kept alive, sharedbufferscountonce, excludes some temporaries/driver and isnotpeakVRAM. Renamed mainchart memorytitles to GPU allocations kept by Garnet, regenerated and visually checked. Mainchart stillreference200/728 because new4% candidate awaitsrepeat.

LIVE continuation checkpoint 2026-10-23 12:20 PDT: session51830 owns GPU chain.2B1 packed-input isolated memory COMPLETE (model/monitor exit0), warm dedicated5,076,701,184B/shared81,788,928B, versus prior5,078,798,336B dedicated (2MiB difference, no meaningful memory gain claim).4B8 modelPID44324 now loading in same chain; after its isolated memory, pending run-packed-graph-nsys.ps1 -Size2b -Batch1 current D4/native/arena, packPrompt OFF. Do not edit consumed runtime/native/memory/profile source/controllers until51830 terminal. No compiler or other model. CurrentDLL d31; verified source archive prompt-pack-before-memory-source-runtime.zip.
Future test_cuda_logical_views.py prepared/unconsumed, syntax only; do NOT count it as passed. After51830, first preserve failure demonstration against old native view behavior if needed, then implement logical-footprint Zero/Clone/CopyInto fix per prospective design, add neighboring-view tests/unfiltered sanitizers, archive/build samefolder and verify CUDAfatbin unchanged. Current serving reads token views only, fullstate tests preserved; generic writable-view issue prevents promotion untilfixed. D2/MTP1 controllers/probe and design prepared, runtime admission UNCHANGED/rejects2. Need profile findings before choosing next speed method. Fresh same-build paired repeats required for new4B4% timing beforepromotion. CurrentQwen3VL legacy regression controller prepared and must run on eventual finalDLL; freshMetal/currentotherbatch/teacherquality/baselineexpansion stillopen. GoalACTIVE2xUNMET.
User latest asks meaning of retained ownership; answered tracked GPU allocations Garnet keeps alive, sharedbuffersonce, excludes some drivers/temporaries/peak. Mainchart title renamed and visually checked. Ongoing memory/profiler work remains authorized; no confirmation popup encountered.

2026-10-23 12:26 PDT research correction: current authoritative source DOES repeat full-head work on partial packed-prefix commit. cuda_packed_replay.py:137 calls inherited _read; cuda_verify.py:50-54 re-normalizes and calls predict_tokens. Same traced2B case has11verify+6partial+1initial=18heads;17CUTLASS M4 kernels27.020ms and1GEMV M1 agree. Earlier summary saying no redundant head was WRONG and superseded. Prospectively recorded reuse-verified-head-design.json BEFORE any implementation. Keep normalization refresh and existing fault hook; reuse only accepted predictions, zero-active-prefix fallback, candidate-row exact oracle with same packed shape, actual head counters, all unchangedquality/memcheck gates. Do not edit runtime while96532 executes logical-view/currentQwen3VL regression chain. D2experiment deferred until duplicate work addressed.

2026-10-23 12:31 PDT: Previous goal turn classified PROGRESS (new input packing path/fullstate+28cohort evidence);51830 now TERMINAL0. Isolated packed-input memory2B1 dedicated5,076,701,184/shared81,788,928B;4B8 dedicated13,154,992,128/shared81,788,928B. Only2MiB/6MiB differences from prior memory runs, no meaningful reduction claim. All6 image-hash groups exact across savedreference/newpacked/LM4B8; prompt-pack-isolated-memory-input-control.json.
Whole-graph trace finished successfully, source/profile archived before execution.88graphlaunches sum110.234ms,792uncapturedkernels39.308ms,680copies1.056ms,259memsets.205ms; intervalunion150.804/span244.092/externallyvisibleidle93.288ms.1219eventrecords,8511queries. Sameinput/IDs/requestcounters as savedcase8, but historicalcohort176.2214ms vsinstrumented244.3529ms. Trace timings not primary, no recoverabletime/causality/speedforecast; wholegraphintervals hide internal gaps. Full-head17CUTLASS M4(27.020ms)+1GEMV M1 correlates exact1initial+11verify+6partialheads; confirmedrepeatedpartial_read, prospective reuse-verified-head design recorded.
Logical-view old-DLL failure reproduced beforepatch, exactsame newtest source: zero middle token also zeroedfollowingtoken(view_index2), exit1 retained as expectedfailureNOTpass. NewhostLogicalBytes function fixesZero/Clone/CopyInto logicalcontiguousfootprints, validates backing spans and retainspartialoverlap checks/providerleases. Buildexit0 DLL371953ad1770e1170e1a9202203529fa51e4b56f921b25a8b01d04cd6d5b4b81; CUDAfatbinbyteidenticald31.34100 terminal0,8viewcases plain+unfilteredmem/race/sync allzero findings; neighboringview/source/close/disjointalias behavior checked. Nativefixed source/runtime archive logical-views-fixed-before-gates-source-runtime.zip.
96532 LIVE exclusiveGPU chain: currentDLL C4 numerical12PASS and3unfilteredmemchecksPASS0; numerical/sanitizerexitsnowrecordbinarySHA and dependentstageverifiesmatch. Real2B1then4B8 finite-state probes crosspriorpackedstate hashes, thencurrentQwen3-VL legacy smoke withreusedcache. Allconsumedsource/controllersfrozenuntil96532terminal. No newheadreuse code yet. Prepared run-paired-packed-study.ps1 +summarize_qwen35_packed_study.py, notexecuted;ABBA forpackingorABCCBA old/packed/packed+head reuse, all20warmgroups/all28signatures, samebinary, fullrawlogs, derivedpostfirstsamplethroughputlabel. Needsactualheadreuseadmission/gatesbeforeCvariant. Sameexistingfolder;goalACTIVE2xUNMET.

LIVE 2026-10-23 12:34 PDT:96532 remains authoritative live handle. New371 DLL all12 C4 numerical and3unfilteredmemchecks PASS;2B1 currentDLL full finite state and output crossprior packed build PASS0.4B8 currentDLL probe now running, then same-chain run-current-qwen3-vl-regression.ps1. DO NOT mutate consumed Model/cuda_speculative/cuda_verify/cuda_packed_replay/test/probe/legacy/controllers until96532 terminal. Currentsource/default headreuse NOTimplemented. Main report regenerated12:33, primarycombined table now D4 references (NoRDC remains separate trial); added isolatedmemory/trace/viewfailure+fix evidence/prospectiveheadreuse sections.
Next concrete speed change afterchain: add defaultOFF reuseVerifiedHead for native packed replay. Keep normalizer/head_input refresh on partial commit; skip only duplicate predict_tokens when safe accepted predictions remain. Guard fallback forany n==0 withsubmitted>0. Keep self._read hook invoked (original injected fault tests must stillrollback); phaseflagresetfinally. Cached tokenHeadOutput accepted rows remain authoritative; MTPheadCandidates selectsaccepted n-1 ororiginalinactivezero row. Newnumerical oracle: capture cached candidateIDs forpositiveacceptedrows, then fresh full _read withsamepackedshape, comparecandidateIDs exactly (testonly; avoidcross-M GEMM oracle). Newmetrics request-reset head_evaluations/reuses, actualrows prove headprojection removed. Normalized maskedhidden stillrecomputed exactly forMTPcorrection. No tolerance/precision/kernels/weights changes. Need12opt-inverifier/session profiles+allrollback/stop/fault assertions, unfilteredmemchecks,fullfinite2B1/4B8probes,full28 andpairedABCCBA beforepromotion. Earlier no-redundant-head assumptionwrong; currentcode/traceproofalreadylogged/designsaved. D2 experiment deferred.
Prepared paired study nowhas fresh-processA1B1C1C2B2A2 for baseline(native/arena),packing,packing+headreuse; all20warmgroups/all28 input/output/stop signatures, sameDLLgate, sourcehashlinks andderivedpostfirstsamplethroughput(scopehost/stop included). It is NOTRUN, requiresfutureheadflag+probe. run-paired-packed-study.ps1 andsummarize_qwen35_packed_study.py syntaxpassed, addtofutureidentity/archivebeforelaunch. GoalACTIVE2xUNMET;allotherbatch/baseline/quality/Metalrequirements retained.

2026-10-23 12:45 PDT continuation checkpoint:96532 TERMINAL1 only because legacyharness usedfileinsteadmodeldirectory. BOTH371 logical-view2B1/4B8 pretrained fullfinite targetstate and outputs crosspriord31packedbuild PASS0. Legacy sourcediagnostic nowprints/savesfullstartup and stillrequiresreadyTrue; controllercorrectedrootdirectory, newfixed-rootartifactpreservesfailedoriginal.38943 TERMINAL0 correctedcurrent371 Qwen3-VL smoke:readyTrue,2matching16tokenrequests,77visualtokenseach, sameexistingcachefolder. Readytime110s enginebuild/load; not performancecomparison. qwen3-vl-startup-harness-correction.json retainscause/evidence; no native/kernel change.
WhileQwen3-only job ran, unconsumed Qwen3.5 Pythoncode gained defaultOFF reuseVerifiedHead. No secondengine/compile ran. CUDA_verify _read alwaysrefreshesnorm+head_input, skips repeatedpredict_tokens onlyundertransient_reuse_accepted_head; validatescachedpredictionslength, trackshead_evaluations/reuses. PackedReplayrequiresnativeprefixforflag, partialcommitsetsreusewhenallcountspositiveororiginalsubmitted0; activezero-prefixforcesfullread, andfinallyclearsphaseflag. Existing self._readfault hook remainscalled. SpecSessionresetclearscounter/phase perrequest;resultmetricsreportactualtargethead_evaluations/reuses. ModelguardsnativepackedMTPprofile andpropagatestoDecoderbeforeVerifierconstruct; benchmarkflagfromenv +counterasserts. Ordinaryprobeexplicitlydisablesheadreuse.
ExistingtargetVerifier testsaddoptionalargv14flag(defaultFalse), preservealloriginalstate/ID/rollback/faultchecks; assertheadcall/reusecountersandphasecleanup, compareacceptedTopK IDs to freshfull_readwithSAMEpackedshape (testonly). Sessionoptionalargv18flag(defaultFalse), assertresetcounterszero, retainall24cases andrequireactualcachehits. Source/math/tolerances/kernels/weightsunchanged. NativeDLLstill371;headfeaturePythononly. Sourcearchive reuse-head-before-gates-source-runtime.zip captured BEFORE92583.
92583 NOWLIVE authoritativehandle:all12reuse-head numericalprofilesPASS0 (4verifiersB4/8C4/16 +8sessionsB1/2/4/8P32/64D4MTP3). FirstB1P32session24caseshas11actualheadreuses/19evaluationsacrosstestcases. VerifierunfilteredmemcheckPASS0;full24caseB8sessionmemcheckthenprefix-scopedrace/syncpending. Freezeconsumedruntime,xModel,test,validation/sanitizercontrollersandnativeuntil92583terminal. CurrentcodehasNOheadreuseperformanceclaimyet.
PreparedUNCONSUMEDfuture run-reuse-head-pretrained.ps1 requirescurrentDLL +4runtimeSHAexactagainstgatearchiveandsanitizerexits; sourceguardcheckedCPU andmatches. It invokesnativeprefixprobe argv9headflagTrue, originalsamepackTrue control logical-views-pretrained-$size-b$batch-promptpack, --verified-head-transition permitsONLYheadFalse→True, ordinarycontrolheadFalse. Probeassertsactualhead_evaluations==targetverifications andhead_reuses==prefixreplays>0. Newrun-hybrid-mtp-cohort -ReuseHead -PackPrompt uses...-arena-promptpack-headreuse, strips-headreusefirsttocontrololdpackcohort;2B1mustpassVllmSuffix -mtp3-cache128. Needarchivefuturecontroller/sourcebeforecoldmodels. Afterheadgates→real2B1then4B8probes/full28 sequential;thenfreshsamebuildABCCBA studyforold/packed/packed+head (all20warmgroups/all28signatures), no promotionbeforequalityandrepeats.
CMakepackagedmodels currentlycopyxModel assets but standaloneCUDApackageruntimewithoutsourcecheckout isUNVERIFIED; nativeServeModel hasexplicit sourcefallback. Do notauditasdone; freshMetal/currentotherbatch/baselinecoverageandindependentteacherqualitystillopen. D2/MTP1onlyprepared/NOTadmitted;deferredinfavorofconfirmedduplicatework. GoalACTIVE2xUNMET. Reportregeneratedwithmemory/trace/viewfix/currentQwen3smoke andnumericalheadprogress.

2026-10-23 13:00 PDT continuation: previous user-question turn supplied terminology only (NO PROGRESS); revalidated work via missing84513 handle plus terminal cohort artifacts and no active model processes.92583 completed all12 head-reuse numerical profiles and unfiltered verifier/full24caseB8 session memchecks0; prefix-scoped race/sync0 (not full integration proof).84513 completed both real full finite state/output probes and full28 cohorts.2B1 head reuse186.13990ms versus savedpacking199.03010, vLLMMTP203.273821, LM302.209;17common groups,28/28same outputs,owned4,749,211,752B unchanged.4B8 head reuse678.37740ms versus packing698.51660,vLLMserial665.647265,LM1348.2737;15common groups,28/28same,owned12,705,559,484B unchanged. These single trials remain unpromoted;2xUNMET.
Current371 native/source/controllers archived in paired-head-before-study-identity.json + verified source-runtime.zip BEFOREtimings. Session35420 NOWLIVE: sixfresh-process4B8 study A1B1C1C2B2A2, thensame2B1, strictlysequential. A=baseline nativeprefix/arenaD4, B=packPrompt, C=packing+headreuse; nativeDLLidentical,20warmgroups/all28signatures,allrawlogs. Do not mutate consumed native/runtime/xModel/benchmark/studysummary/controller until parentterminal. No compiler/hash sweep/othermodel duringtiming. Report-only edits allowed. No permission popup encountered;goalACTIVEfullscope/independentquality/otherbatches/Metal/comparatorcoverage remainopen.
