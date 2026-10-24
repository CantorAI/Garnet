# Qwen3.8-27B on one RTX 5090: implementation task

Status: pinned architecture/quantization audit, native SM120 build, NVFP4/FP8
projection/plugin tests, real layer0 MLP parity, real GDN/full-attention decoder-layer
parity at4K capacity, and native W4A4 head checks have passed. Both references have
passed text/image smoke; SGLang's initial eager profile now has raw streaming and
concurrency measurements. The34 CPU tests pass. Full64-layer native text smoke
also reproduces the reference token IDs. Four native VLM color/spatial answers
are semantically correct; strict reference token parity is2/4 due to capitalization.
Broader vision/precision validation,
serving integration, tuned matched comparisons and the2x target remain incomplete.
A direct-call native text pilot measured37.3 decode tokens/s and1060ms first-token
time over eight32-token requests. This excludes HTTP and is not a matched2x result.
The full BF16-projection vision tower executes but has failed both the F32 and
mixed-projection CPU comparison limits. A separate full27-layer F32-projection
structural diagnostic passed with max error0.000576 and relative L2 error2.874e-6.
Keep serving admission closed and retain the BF16/strict-token failures. Broader
quality, real streaming, fused prefill and matched comparisons are the next gates.

Audit date: 2026-10-23. The preserved source baseline was
`5b0dc7ff8e5c1a77807138db310605bcd12aa390` plus inherited local edits, captured
separately at2355fc4b5. Work continues on qwen38-5090; command ledgers retain exact
commit/patch manifests. Do not reset the original checkout or infer tested source
identity from a parent commit alone. Research method/failures are in the development
log; no matched full-model throughput or successful2x comparison is claimed.

## Objective and acceptance

Implement native Windows Garnet inference for `Qwen/Qwen3.8-27B` on one RTX
5090 32GB. Prefer NVIDIA's mixed NVFP4/FP8 checkpoint on Blackwell. Compare a
separately identified INT4 W4A16 implementation and Qwen3.5-4B. Start with batch
1 and 4096 total context capacity, including image tokens and output reserve.
Target >=2x the faster of vLLM and SGLang for both decode tokens/s and aggregate
output tokens/s at each matched concurrency. Report TTFT independently; seek
half TTFT as an additional optimization objective. No successful-target claim
until both independent baselines have valid matching results. Retain failures,
OOMs, quality regressions, and below-target results.

## Verified checkpoints

Pinned configs, revisions, URLs and SHA-256 values are in
`test2026/cuda/qwen38_5090/evidence/checkpoints.json`.

* Qwen/Qwen3.8-27B: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
* nvidia/Qwen3.8-27B-NVFP4: `482ca0f3832238542f8f5295dde86b5f22711d80`.
* Qwen/Qwen3.5-4B: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
* INT4 candidate pearsonkyle/Qwen3.8-27B-GPTQ-W4A16:
  `4e2c2d8f592d5641beefa6a75357f43cf3756392`.

The INT4 candidate config uses compressed-tensors pack-quantized, symmetric
group128 INT4 weights and unquantized activations (W4A16). Actual index/header
inspection proves all496 text projections, including linear attention, are packed;
lm_head, embedding, vision and MTP remain unquantized. The earlier config-only
inference about ignored linear attention was incorrect. Installed compressed-tensors
0.18.0 passes the CPU packing fixture: eight offset-binary nibbles per int32,
low lane first, decoded value nibble minus8. Actual-weight numerical checks,
image inference, SM120 kernel dispatch and Garnet compatibility remain unverified. Keep it a candidate until
these pass. Do not compare quantization recipes as if identical.

The 27B config uses `Qwen3_5ForConditionalGeneration`, model_type `qwen3_5`.
It has 64 dense decoder layers: 48 Gated DeltaNet and 16 full-attention layers.
Hidden width 5120; MLP width 17408; Q/KV heads 24/4, head dimension 256;
linear QK/V heads 16/48 with dimension 128. Embeddings are untied.
Vision differs from 4B: depth 27, hidden width 1152, intermediate width 4304,
16 heads, output width 5120. Preserve patch16, merge2, temporal-patch2 and
multimodal RoPE behavior. Architectural reuse does not prove checkpoint parity.

The NVIDIA config declares ModelOpt MIXED_PRECISION: NVFP4 block16 MLP/head,
FP8 attention projections, and an ignored MTP set. Its card recommends Linux,
lists both engines and Blackwell, and reports testing on GB300, not RTX 5090.
The card says ModelOpt 0.48.0; the pinned config producer reports
0.47.0.dev80+g913f5e224. Preserve that discrepancy in provenance.
Do not relabel this checkpoint all-NVFP4, MXFP4, GPTQ, or W4A16.

## Current Garnet constraints

* `runtime/cuda_serving.py:Model` admits only Qwen3.5-2B and 4B. Its 8192
  capacity ceiling allows a 4K profile but does not validate the new model.
* `xModel/qwen35/vl_4b/model.py` validates 4B-specific decoder and vision shapes.
  Add a dedicated 27B contract and factory; never alias to the 4B factory.
* Shared external projections require BF16 matrices in
  `xModel/qwen35/vl_0_8b/cuda_decoder.py` and
  `src/backends/tensorrt/cuda_projection_io.cpp`.
* `trt_builder.cpp` has captured-weight `int4_fp16` block64 symmetric
  quantization/dequantization. It is not a ModelOpt loader, calibrated GPTQ/AWQ,
  or evidence of packed shared-projection support. No NVFP4 implementation was
  located in the audited serving/projection path. GPT-OSS MXFP4 is unrelated.
* Existing Garnet/vLLM Qwen3.5 benchmark scripts constrain context to 384/512
  and model names to 2B/4B. Do not merely change labels on their old results.
* CMake defaults to SM80/86/89. Explicitly configure SM120 on RTX5090.
* Reuse current Qwen3.5 CUDA lowering and state kernels below the backend-neutral
  tensor-program boundary. No Python/PyTorch model fallback presented as Garnet.

## Target machine and safe setup

SSH target `CD-RTX5090` / `192.168.193.113` reports Windows hostname
`zsdx-5090`, one RTX5090, 32607 MiB, driver 617.14, CUDA toolkit12.8.
`D:\CantorAI\Garnet` exists but contains deployment assets (`out`, `xModel/qwen3`),
not a buildable Git checkout. `C:\CantorAI` contains deployment output too.
Initial checks found no WSL installation, Visual Studio build tools or CMake
at their standard locations. PATH alone is not a full installed-software audit.

Preserve these deployments. Stage current source in a fresh sibling study
directory, with adjacent current XLang3 and separately versioned TensorRT SDK.
Record per-file hashes including dirty files and dependencies. Build native
Windows Release with SM120. Preserve a control executable/DLL before changes.
Use WSL only for Linux-only reference dependencies as needed; user authorized
installation. Do not reboot automatically. Never save SSH credentials in code,
reports, Git config, model manifests, or the source archive.

## Ordered implementation

1. Environment: inventory compiler, CMake/Ninja, CUDA, TensorRT and XLang3;
   resolve SM120 compiler/runtime compatibility. Pin exact versions and build
   commands. Compile current Garnet unchanged, then run tensor capture/native
   API, Qwen3.5 serving/state/vision correctness checks on the 5090.
2. Checkpoint loader: download only pinned metadata first. Inspect safetensors
   headers/index, naming, tensor shapes, tied/untied embeddings, scale formats,
   FP8 exclusions and total actual bytes. Make unsupported formats fail before
   allocation. Integrate dedicated 27B factory, serving admission and lifecycle.
3. Correctness first: tiny layer/subgraph numerical tests against independent
   Transformers references, including full/linear attention, conv/state reset,
   multimodal positions, vision merger, logits and greedy tokens. Use one layer
   at a time or an offloaded reference if BF16 model does not fit. No BF16 full
   model load assumption on 32GB. Exercise real images, text, repeated requests,
   cancellation, EOS and context overflow. Keep MTP/speculation disabled initially.
4. Quantization: implement packed NVFP4 block scales, per-tensor scales and FP8
   projections compatible with the pinned ModelOpt format and actual SM120
   kernels. Keep stable accumulation/state precision; document unquantized
   vision/norm/embedding tensors. Separately implement/admit INT4 W4A16 with
   declared group size, zero-point convention, scales and calibration dataset.
   Existing block64 symmetric path can be a separate experimental arm, never
   labeled calibrated GPTQ/AWQ. Verify kernel dispatch does not expand weights
   into a persistent full BF16 copy. Pin any external INT4 checkpoint only after
   config, tensor-layout and image-path inspection; the pinned candidate above
   is not admitted yet.
5. Fit test: load B1 context4096, reserve256 output tokens. Track build/load and
   inference VRAM separately. Warm text, then actual image. Fail explicitly on
   OOM; don't silently offload, split across GPUs, lower context, or reduce image
   pixels. Scale concurrency1/2/4/8 only as memory allows, recording failures.
6. Reference engines: use isolated pinned environments; run vLLM then SGLang
   sequentially with TP1, context4096, max sequences1 initially, the same local
   checkpoint, image budget and deterministic generation. The NVIDIA card's
   TP4/262K command is not suitable for this study. Verify help/flags and actual
   selected ModelOpt/FP8/NVFP4 kernels on the installed releases. Start 4B BF16
   baselines before the 27B quantized ones. Cache and recurrent-state dtypes
   must be recorded; separate unmatched precision arms from matched comparisons.
7. Optimization experiments after valid control: profile dispatch/host gaps,
   projection kernels, decode memory bandwidth, vision/prefill and state copies.
   Test packed-weight reuse, fused gate/up and attention projections, CUDA graph
   replay, in-place KV/state updates and chunked prefill independently. Benchmark
   each candidate against the preserved current control in alternating order.
   Add speculation/MTP only as a separately quality-validated experiment after
   serial decode parity; apply equivalent baseline optimizations where supported.

## Measurement contract

Separate text and image cohorts. Use fixed UTF-8 prompts and hashed image bytes;
record decoded dimensions, image grid, processor revision, chat template hash,
thinking mode (disabled initially), rendered prompt IDs and visual-token counts.
Use short, ~1K and near-3840-token input cases, always input+output<=4096.
Use matching semantic prompts for cross-model quality; exact token-count matches
are required for speed ratios. Set temperature0, seed0, output cap256, and
record actual generated tokens/EOS. A length-forced throughput arm is separate.

For each engine/model/precision/cohort: one cold run, >=3 warmups excluded,
>=20 measured requests (repeat full cohort >=3 times). Report median/p95 and
sample counts. Disable prefix, image/processor and answer caches for primary
tests; execute vision afresh. Report cached scenarios separately. Establish
single-request measurements before concurrency sweep1/2/4/8. Concurrency is
simultaneous clients; static model batch is a distinct field. Count queuing.

* End-to-end TTFT: monotonic submission timestamp to first generated token;
  includes tokenization, image decode/vision and queueing. Also record internal
  GPU prefill/vision and first visible text latency separately, especially if
  reasoning parsers buffer tokens. SSE chunks are not tokens.
* Per-request decode tok/s: (generated_tokens-1)/(last_token_time-first_token_time).
  Fewer than2 tokens => undefined, never infinity or a fabricated value.
* Throughput: total actual output tokens / cohort wall seconds from first
  submission to final completion. Also report requests/s, errors and OOMs.
* VRAM: poll whole-device used memory with timestamps and UUID; preserve idle
  baseline, absolute peak and incremental peak. Record sampling interval and
  that sampling can miss spikes. Add framework allocator high-water and native
  allocation instrumentation. Report compile/load peak separately. Windows WDDM
  process memory may be N/A; never substitute host RSS or reserved cache as
  measured GPU VRAM. No unrelated GPU work during timed runs.

Save raw per-request timings, token IDs/output, prompt/image hashes, failures,
logs, GPU temperature/clocks/power, driver, build flags, commit+dirty-file hashes,
quantization/calibration provenance, engine versions and launch commands.
No benchmark result files are pre-populated by the preparation task.

At4K, the contract checker estimates canonical FP32 full-attention KV at512 MiB
and recurrent matrices at144 MiB per request, plus convolution history. These
are shape calculations, not measured VRAM; duplicate buffers, weights, graphs,
vision and workspace can dominate. Do not infer fit from 27B*0.5 bytes.

## Completed safe changes and tests

Added offline checkpoint-contract preflight and pinned config evidence under
`test2026/cuda/qwen38_5090`. It diagnoses architecture/cache sizes and always
reports 27B runtime integration pending. It does not register unsupported model
names, mutate existing serving code, launch GPU workloads or invent results.
Added metric arithmetic and a two-engine 2x gate. Its test records are synthetic
unit fixtures, never benchmark output. The caller still must establish workload,
precision, hardware and quality equivalence before using that gate.

Run from Garnet using a Python3 interpreter:

```text
python -m unittest discover -s test2026/cuda/qwen38_5090 -p test_*.py -v
python test2026/cuda/qwen38_5090/preflight.py test2026/cuda/qwen38_5090/evidence/base-config.json
```

## Primary evidence

* https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json
* https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/blob/482ca0f3832238542f8f5295dde86b5f22711d80/config.json
* https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4
* https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json
* https://docs.vllm.ai/en/latest/features/quantization/modelopt/
* https://docs.sglang.io/docs/advanced_features/quantization

Published engine support is a candidate baseline, not evidence that a particular
Windows/WSL package and consumer SM120 GPU have passed this workload.

## Current native-build path

The optional CMake wrapper in `test2026/cuda/qwen38_5090` accepts preserved
XLang3 Release binaries and a matching public SDK. From a VS x64 developer
shell, configure it with `XLANG3_RUNTIME_DIR`, `XLANG3_ROOT`,
`GARNET_TENSORRT_ROOT`, `CMAKE_CUDA_COMPILER` and an explicit architecture.
Use89 for the recorded local RTX4080 correctness build; use120 for the target
RTX5090 build, which has passed native build, component correctness and the full64-layer text smoke. Full VLM executes with four semantic color/spatial answers, but strict parity and broad quality remain incomplete. Build target `garnet`. No VM sources
are modified by this wrapper. Record executable/DLL/import-library and SDK
hashes, as well as the complete compiler and dependency versions.

`windows-sdk-requirements.txt` pins the whole native CUDA/TensorRT package set.
Install it as one set with `--no-deps`; unconstrained transitive dependencies
were observed selecting newer CUDA components. The study uses one SDK folder
under the fixed remote root. Do not use the mixed-version first attempt as a
validated performance environment.


## Current integration checkpoint and next gates

Native SM120 packed NVFP4 and FP8 projection/plugin checks passed, including GPU
activation quantization, serialized engines and actual CUDA graph replay. The
embedded-weight/refit fix preserves BF16 behavior. Actual layer0 MLP matches5120
CPU-oracle outputs exactly. Actual decoder layers0 (GDN) and3 (full attention)
passed two stateful tokens with all hidden values and sampled states checked.
Grouped-query matrix multiplication removed explicit sixfold K/V expansion for
the verified27B geometry and allowed the full-attention graph to build at64MiB;
the old graph had failed at both64/128MiB. Native output-head W4A4 mode passed
synthetic batches1/3/8 and11 actual sparse checkpoint rows, while preserving the
separate W4A16 mode. Owned-buffer accounting includes untied heads/native workspace.

Mixed plans embed compressed weights after TensorRT rejected stripped opaque
INT8 constants. Do not retain separate full-model prefill/decode plans until
shared packed bindings or measured memory admission proves they fit. FP4 projection
rows are bounded to1024; use explicit tiling or validate a larger bound for wide
prefill. Initial strict CUDA factories are batch1/capacity32..4096; general serving
admission remains closed. Do not silently truncate context or lower the comparison
capacity to fit larger batches.

1. Full64-layer decoder smoke passed with4096 capacity, the saved reference26-token
   prompt and exact[19,248046] output. This is not broad quality or a benchmark.
   The checkpoint JPEG preprocessing gate passed; four matched JPEG VLM cases
   give4/4 semantic colors and3/4 exact token sequences on the completed fused16
   run (earlier serial run2/4). Preserve the strict failure and broaden quality
   checks. Real serial6/fused4/fused16 layer, state, inactive mask and replay gates
   pass with the corrected native-order FP8 CPU oracle. The exploratory fused16
   direct-call text pilot measured99.8ms TTFT and36.5 decode tokens/s; it excludes
   HTTP and used32 output tokens, so it is not matched to the new reference run.
2. Establish text/image quality beyond basic smoke and record the precision
   differences. Current Garnet state/cache arithmetic is F32; initial references
   use BF16 KV/conv and F32 SSM. Close this matching gate before any2x claim.
3. Native HTTP B1 validation and128-token pilots completed. Original per-layer
   text TTFT104.85ms/decode36.10tokens/s; packed full64 text123.04ms/34.68tokens/s.
   Packed lowers sampled peak from27.81GB to24.51GB but does not improve speed.
   Both retain4/4 native VLM-ID/prompt-ID gates and3/4 exact SGLang sequences;
   broad quality/precision remain open. Image pilots are exploratory: the old
   single-pixel JPEG nonce collided after compression. Rerun all engines with the
   tested binary-grid nonce and precomputed request bodies. The bridge streams
   actual native tokens, has63 CPU study checks, and remains a single queued slot;
   it does not establish batch2/4/8. Validate native concurrency and actual4K
   admission without duplicating packed weights or silently lowering capacity.
4. Establish tuned vLLM/SGLang profiles, including feasible CUDA graph/kernel
   options. Both initial profiles deliberately used eager startup for feasibility;
   they are not evidence of the fastest attainable reference. A completed
   SGLang full-decode-graph candidate captured1/2/4/8 and replayed during the
   seed3/JPEG/image-first128-token benchmark: text/c1 decode65.1tokens/s and
   TTFT141.9ms; c8 output423.6tokens/s text and351.1tokens/s images. Preserve
   early image EOS and actual lengths. Autotuning/prefill graphs remain disabled
   in that candidate; broader reference tuning is still required. Compiler/linker
   failures and SGLang's first memory exhaustion are retained in the research log.
5. INT4 is now partially integrated through a distinct symmetric G128 W4A16
   loader, direct packed CUDA kernel and serialized zero-workspace plugin.
   Generic TensorRT INT4-DQ graphs failed in three retained configurations;
   the native route passed an actual projection and stateful layers0/3 against
   independent CPU oracles, including graph replay. The guarded factory requires
   all496 text projections to remain packed and preserves the BF16 head/sidecars
   through an explicit index. Full64-layer INT4 VLM, chunk16 quality, memory,
   HTTP timing and the downloaded Qwen3.5-4B comparison remain unexecuted.
   Do not label the mixed NVIDIA checkpoint as INT4 or promote component gates
   to full-model support. The opt-in MTP worker and16-sequence greedy gate are
   implemented but have no GPU correctness/performance result yet.
6. Collect repeated text/image TTFT, decode, aggregate throughput and memory with
   fixed4K total capacity and matched request hashes, quality, cache behavior,
   output lengths and concurrency. Both the initial32-token/eight-request SGLang
   data and the new128-token graph candidate are exploratory until matched native
   quality/arithmetic/transport gates close. Preserve repeated longer runs,
   distributions and raw failures. The NVML
   sampler requested20ms but measured about31ms typical gaps and one7.787s gap;
   investigate timing/observer repeatability and report sampled peaks honestly.
7. Optimize from measured bottlenecks, then test against twice the faster valid
   reference for decode and throughput; seek half TTFT separately. Publish unmet
   targets as measured. No automatic success follows from primitive correctness,
   server process duration, a reduced workload or an untuned reference.


## Current execution dependencies (2026-10-23 continuation)

The vLLM graph attempt failed at final FlashInfer SM120 linking, not benchmarking.
Existing CUDA-library aliases were repaired and the retained objects successfully
linked with the original WSL driver-library path. No post-repair full-server result
exists. Preserve the original failure/source snapshot, use a fresh attempt label,
record resolved cache capacity, and require readiness/VLM/token protocol gates
before the corrected grid workload. SGLang's graph pilot above remains a candidate,
not a fastest-attainable claim; old image cache uniqueness is unproven.

User authorized stopping Cantor to release GPU memory; verified Cantor/WorldSense
processes were stopped with identities retained. After4B exited, GPU use was464MiB.
A later Cantor desktop restart was observed; recheck actual memory before each engine.
Keep the fixed study directory. Do not manufacture results, lower4K capacity, change
arithmetic or omit a stronger reference to fit. The linker-repaired vLLM graph retry
is now live under separate logs/output/metadata with source1e19927bc. Do not restart
it for quiet JIT output or run another GPU model concurrently. Next, run strict
MTP/self-greedy gates; run full INT4 and4B
correctness before timing; repeat native/SGLang/vLLM matched grid cohorts and actual
concurrency admission; compare the faster valid reference with raw-event replay.

Attention scheduling optimization is a separate, unwired experiment. The first
per-query compact core passes all seven CPU/KV/graph gates and improves short-prefix
component timing, but regresses full prefixes. It is not a serving default or a
model speedup. The output-tiled variant also regresses full prefixes. A split-key candidate
passes the seven numerical/KV/graph profiles and improves tested4K-capacity warm
component timings with extra scratch memory. It remains unwired, with model-level
qualification pending. Retain all variants/failures and raw CUDA-event samples.
Do not extrapolate a component ratio to HTTP TTFT, decode, throughput or the2x goal.

The goal remains active. Completion requires the requested full-model comparisons,
quality/precision/cache/transport matching, repeated metrics, actual concurrency
and honest evidence for any claimed speedup; the2x objective is currently unmet.


The attention experiments now include a split-key candidate. All seven component
qualification profiles pass; warm-core4K cases improve versus legacy with extra
scratch memory. It remains unwired and requires opt-in plugin/model validation.
Small-capacity regression is retained. See P067 and verified attention-split evidence;
no component result establishes the requested2x end-to-end reference comparison.


4B current status: initial full32-layer/vision load and text4 passed; image failed
because default min_pixels1024 produced14x14 while preloaded grid was16x16.
Initial failure archive is preserved and verified. Native image-only probe proves
explicit min=max65536 gives16x16. Source1e19927bc adds opt-in imageMinPixels and
separate retry output paths. The corrected image/repeat/3968-input qualification
subsequently passed (see current checkpoint below). Model token callbacks passed
actual GPU smoke;4B HTTP transport remains GPU-unqualified. Do not infer4B speed
or broad quality from these checks.

User explicitly requested replacing local D:/CantorAI/Garnet with our source.
Original dirty/untracked changes are retained in a pinned Git backup ref and
D:/CantorAI/backups/garnet-before-qwen38-5090 replacement record. Primary checkout
now uses a detached study commit, preserving qwen38-5090 ownership in the dedicated
worktree.9992 tracked files matched byte-for-byte immediately after replacement.
Ignored environments/build products were preserved, so source replacement alone
is not proof those old binaries were rebuilt. Remote study DLL identity remains
separately pinned and verified. Do not silently use original installed binaries.


## Current execution checkpoint (P096)

Completed corrected-grid vLLM candidate cf140177a4f24bf3ad845ad32f8cb4c0: actual
decode graphs1/2/4/8, four JPEG token/color gates, eight measured cohorts and raw
event replay verified. B1 text/image decode both about68.5tok/s; text/image TTFT
152.4/166.5ms. Output-head tuning coverage remains incomplete; MTP reference modes,
repetition, full4K concurrent admission and native/reference precision/quality
equivalence remain required before a fastest-valid-reference or2x claim.

Corrected full4B native text/image/repeat/token-event/3968-input smoke now passes.
It generated78 sampled outputs on the long input before EOS. Independent4B
reference/quality and real HTTP/concurrency measurements remain pending.

Native MTP3 exact16-sequence self-greedy gate plus actual HTTP pilot passed;
text/image decode72.1/71.7tok/s and TTFT111.2/269.9ms. B1 request bytes and counts
match vLLM, but native speed is only about1.05x and image TTFT is worse. MTP7
passes the same gates but regresses to69.0/64.0tok/s; preserve this negative result.
Neither measured profile establishes2x or real native GPU batched concurrency.

Optional resident vision input now passes six full raw output equality pairs,
actual graph replay0->4 and wrong-shape rejection, using1.5MiB extra input storage.
It is allocation staging, with fresh image preprocessing on every call. No
end-to-end TTFT benefit has yet been measured. Initial B2 packed backend qualification failed full B1-ID equality, with first
divergences at text token0 and image token8; preserve the strict gate and raw IDs.
Fixed-B2 prefill isolation now passes ten captures/eight full-row checks, and
head-only replay of identical normalized vectors agrees for B1/B2 and native/
dequantized heads. Both raw evidence archives were independently verified.
Fresh B1 retain0 configuration control is live on sourcef8587f5ee, isolating the
saved MTP3 target's retain1 setting before assigning an upstream numerical cause.
Original cross-B1 full-ID gate remains failed; no actual native GPU batch serving
or concurrency throughput is qualified.

Next GPU sequence, preserving serial GPU ownership and raw evidence:
1. Finish/archive the same-configuration B1 diagnostic; use actual hidden/ID
   differences to choose the next batch correctness investigation. Do not weaken
   the original gate or treat within-B2 repeat equality as B1 equivalence.
2. Qualify opt-in MTP3 draftT4 against unchanged targetT16 using all16 complete
   self-greedy sequences, retained VLM checks and actual stream/count gates, then
   benchmark the corrected workload only on success. Default draftT16 stays.
3. Separately repeat MTP3 image/text timing with component-qualified resident
   vision input. Keep this separate from draftT4 initially to attribute effects.
4. Establish feasible stronger reference MTP modes. vLLM MTP3 launcher and a
   same-engine ordinary/speculative16-sequence full-ID capture/check tool are
   prepared; actual mixed-checkpoint runtime/quality remains pending. Repeat
   corrected-grid SGLang ordinary and tuned feasible MTP before final comparison.
5. After batch correctness, implement the real native batch worker, qualify
   HTTP/early-EOS/ragged/state isolation, then measure actual C2/4/8 where memory
   permits. CPU grouped-transport fixtures alone do not establish GPU throughput.
6. Complete full INT4 model/precision/VLM/HTTP and4B reference/quality/HTTP runs,
   repeated measured cohorts, actual long-context/concurrent admission and memory
   accounting. Preserve all failed candidates and negative results.

Actual worker depth/chunk/vision profile can now be required before native MTP
quality gates/timing. Keep source on the dedicated branch and pin runtime DLL
separately.2x remains unmet; no claimed final fastest-reference baseline yet.


P091 update: fresh B1retain0 matches all four saved B1 sequences/repeats, so retained-output setting alone does not explain B2 mismatch. Truncated prefix1 matches B1/B2 byteexact; prefix4 differs. Isolate prefixes2/3 next, preserving full-model failed gate. Opt-in MTP3draftT4 passed all16 self-greedy/VLM/stream gates but measured71.3text/71.9image tok/s, no clear throughput gain versus defaultT16; negative result archived. Resident vision end-to-end, stronger reference MTP/tuning, actual native batching, full INT4/4B comparisons and repeated final evidence remain required.2x unmet.

P096 update: complete chunk/state evidence confirms tiny first-layer hidden differences despite exact first-layer GDN states; later states amplify differences. Substitute-input norm/MLP and residual-barrier experiments are exact B1/B2 and do not reproduce the real-layer issue. No barrier/default fix is justified. All three raw archives now locally integrity/array verified. Actual layer-intermediate isolation remains needed for batch admission; stronger referenceMTP, resident vision end-to-end and full INT4/4B/final repeated comparisons stay in scope.2x unmet.


## Structural TTFT candidate: shared compressed weights across static profiles

Original constraint: the legacy mixed packed factory embeds weights and requires one
shared prefill/decode chunk profile. At T16,41-token text uses3 target calls and
104-token image uses7. This is a call-count fact, not a measured attribution of
latency. The native NVFP4 lowering itself notes external packed bindings are
needed to share compressed constants across prefill/decode engines. The earlier
external weight bindings admitted BF16 only, which cannot safely stand in for this
mixed checkpoint within the32GB target budget.

Implementation sequence (opt-in code now present; qualification remains staged):
1. Add an IO-owned readonly packed matrix/scale pool retaining exact checkpoint
   bytes and validated logical dimensions/scalars; repack NVFP4 block scales once.
   Keep FP8 and NVFP4 recipes separate and reject BF16 expansion/fallback.
2. Add guarded external compressed weight bindings to TensorRT lowering. Validate
   declared input types, physical shapes, calibration metadata and single-owner
   lifetime. Record external compressed bytes separately from embedded weights
   and existing externalBF16 statistics; do not double-count shared storage.
3. Build static wider-prefill and T4-decode programs against one pool/canonical
   state/head, initially B1/4096 with bounded prefillT128. Reuse existing MTP wide
   prefill normalization/slicing rather than adding a second draft weight copy.
4. Qualify exact raw packed storage and same-profile component outputs versus
   embedded lowering for both recipes; prove two engines share GPU allocations,
   close/lifetime rules and no silent precision substitution. Then qualify real
   prompt/state/full-token/VLM behavior across profiles before any performance.
5. Only after these gates, repeat matched text/image HTTP TTFT/decode/VRAM against
   the original T16 candidate and valid tuned references. Preserve any numerical
   or performance regression. Wider prefill does not establish a2x model speedup.

This is a backend optimization candidate motivated by concrete current-source
constraints; it neither replaces unresolved B2 correctness work nor shrinks the
remaining reference, INT4,4B,quality,concurrency and repetition requirements.

Current status, 2026-10-24: exact packed/tiled bytes and retained-owner lifetime
passed for actual NVFP4 and FP8 matrices. Ten synthetic plugin cases and the
actual MLP at T4/T128 passed embedded-versus-external full-output checks. Both
static full64-layer shared engines now construct with 400 matrices in one pool.
Twelve untimed full-model runs passed repeat determinism and four VLM semantic
smokes. Text/image long IDs differ from the old T16 profile; this is recorded,
and full numerical equivalence is not established by those smokes.

Full same-profile T128 arithmetic now matches embedded lowering across every
packed hidden/state chunk for text/image and three repeats. MTP3/draft4 matches
its own target on20pairs, including16full128-token sequences. Six pinned retained
shared-profile goldens and real SSE/ID checks passed. The old T16 VLM gate rejected
bottom-blue because old output was blue and shared output Blue; preserve that
rejection and explicit versioned-golden policy, not a silent relaxation.

First matched shared B1 pilot: text/image TTFT71.27/105.08ms, decode69.96/71.89tok/s,
aggregate67.10/66.15tok/s. Raw SSE and sampled memory were independently replayed.
Relative to the faster supplied ordinary reference, decode1.022x/1.049x and
TTFT0.468x/0.672x. Text meets half ordinary-reference TTFT in this one short-prompt
trial; image and2xdecode/concurrency do not. No fastest-tuned or broad-quality claim.

Head-cache follow-up passed strict correctness and exercised reuse in all timed
requests, but its separate single HTTP pilot was slower:67.65/69.56tok/s and
TTFT82.74/112.51ms. No benefit demonstrated; default stays off and causal repeat
controls remain necessary. Instrumented stages point to target verification as
the main cost; nested graph/state/head profiling is running before kernel changes.

Next: profile decode/verification costs and improve actual decode and native
batching, not just first-token delay. Preserve all regressions and repeat trials.
Separate construction memory from serving memory. Continue reference MTP/tuning and
precision/kernel audits, including native F32 KV storage versus reference BF16
KV, broader quality, actual long-context admission, INT4/4B measurements, native
GPU concurrency and repeated trials. Full per-layer INT4 now passes12repeat/VLM
smoke runs with496native G128 projections; HTTP/reference measurements and the
optional whole-graph comparison remain pending. Installed-source audit identifies
a small merged96-row INT4 projection risk for SGLang; confirm actual startup
before declaring support or failure. The overall2xdecode/concurrency and half-TTFT objectives remain
unproven; no goal completion is claimed.


### Joined NVFP4 MLP implementation plan (implemented and bounded full-model qualified)

The raw calibration audit verifies all64 gate/up pairs have identical input_scale and weight_scale_2. Original output width17408 is divisible by128, so concatenating packed row matrices and corresponding128-row scale tiles is a plausible lossless storage transformation. This is an implementation hypothesis, not measured kernel compatibility or numerical equivalence.

1. Add an explicit default-off joined-MLP study option. Validate every original scalar, dtype, shape and tile boundary before allocation; reject unmatched metadata. Preserve the existing400-matrix path.
2. Load one34816x5120 NVFP4 gate/up matrix per layer directly into a readonly pool, with correct concatenated tiled scales. Avoid retaining duplicate original gate/up GPU buffers. A complete joined candidate should contain128 NVFP4 and208 FP8 projection matrices (336 total), versus192+208=400 original; expose actual counts and owned bytes. Synthetic joined metadata must reference both pinned originals and exact calibration provenance.
3. Lower one existing NVFP4 GEMM followed by correctly ordered gate/up slices, SiLU and product. Preserve every downstream rounding/materialization boundary initially. Both T4 andT128 graphs must borrow the same joined pool; bound bindings and output storage exactly. Do not concatenate the full weight tensors on every inference.
4. Before performance, qualify original-versus-joined packed/scales, standalone actual MLP atT4/T128, then full text/image teacher sequences and normalized/state comparisons, repeats and all VLM gates. Joined output shape can change cuBLASLt tactics, so identical input/scales alone do not prove identical arithmetic. Archive any failure and retain default OFF.
5. Only after correctness, measure cold-weight component timings (avoid a single matrix fitting entirely in L2), then repeat matched unprofiled HTTP pilots and concurrency1/2/4/8 where qualified. Record algorithms, actual profile, per-request outputs, GPU telemetry and sampled VRAM. Compare against faster matched reference; retain negative results. Layer-count or kernel-launch reduction is not an end-to-end speedup claim.

Current first-use eager profiler has cold initialization artifacts and generic plugin names; it does not establish a hot-kernel bottleneck. A warm diagnostic trigger or external profiler remains useful before promoting this candidate. Packed INT4 correctness has completed and independently matched all twelve per-layer token/vector goldens. Joined pool/binder and explicit MLP component are now GPU qualified: pinned packed/tiled-byte oracle, ownership/lifetime assertions and six full raw T4/T128 output vectors matching the original MLP. These use deterministic synthetic inputs and actual layer0 weights. This describes the earlier component stage. Subsequent explicit joined full-program bindings, linear/full-attention lowering, bounded complete state/token gates and HTTP pilot are complete; see the current result below. The original path remains available and joined mode is default OFF. Full remaining scope includes reference tuning/MTP, native batching correctness, INT4/reference benchmarks, Qwen3.5-4B comparisons, long-context and broad quality gates. 2x target remains unachieved.


### Joined MLP current result and next performance step

Complete bounded full-model state/token/normalized gates and20 MTP self-target pairs passed; matched B1 pilot produced72.259 text/74.157 image decode tok/s,TTFT64.719/103.715 ms,aggregate69.184/67.902 tok/s. This is about3.3%/3.2% above the prior original-MLP pilot,not a controlled stable gain or2x result. Native concurrency has not been qualified. Keep the option explicit/default OFF. Evidence and raw event verification are in research-log P143.

Next compare original/joined on the same qualified DLL with repeated ABBA trials and optional GPU clock/thermal telemetry; then measure supported cuBLASLt algorithms/workspaces on real shapes with streaming/cold-weight behavior. Do not turn a hot-L2 microbenchmark into a full-model claim or assume arbitrary FP4 layouts are supported. Correctness gates precede each full-model performance candidate. Resume the remaining INT4/4B/reference/concurrency/long-context/quality requirements as stated above;the target stays2x the faster feasible matched reference.


### Current active comparisons and recent negative kernel experiment

The standalone operand-order NVFP4 experiment passed structured synthetic BF16 sum/scale/transpose oracles on RTX5090, including projection dimensions at B1/4/128. ABBA component medians showed only about4–5% gate/up improvement at B1, no down improvement, and T128 regressions. Do not promote this candidate or infer model tok/s; actual checkpoint/general numerical coverage and end-to-end gains are unproven. Raw archive and independent CPU verification are recorded in P145.

INT4 whole-packed correctness is complete (all12 retained per-layer full token/vector goldens). The explicit B1/T16/cap4096 HTTP worker was deployed from5194b73b7 with qualified2d902723 DLL retained. That server/workflow later completed the12 exact HTTP cases,2 SSE gates and measured pilot; both are terminal, as recorded below. Never restart a live run solely on an observation timeout. Fixed folders: results/native-int4-worker, native-int4-http-pilot and corresponding ledgers. Source was frozen while the worker lived. Its terminal evidence was independently verified in P152.

Pinned INT4 vLLM/SGLang launch candidates preserve the existing4K/seq8/cache/dtype policies and pass shell syntax checks. Actual startup resolved vLLM INT4 to MarlinLinearKernel with padding; stock SGLang0.5.22 failed its96-wide repack against tile64 and yielded no admitted timing. Preserve this specific launch failure; it is not a universal SGLang support claim. After terminal native completion, identity-guarded shutdown and evidence verification, test one reference engine at a time, preserve startup failures and check actual resolved quantization before quality/timing. No different precision/checkpoint substitution. Then complete4B serving/reference comparisons, repeated same-binary original/joined trials, tuned MTP reference investigation, native batch correctness/concurrency, long-context and broader quality. The requested2x faster-baseline decode/concurrency target remains unmet.


### Latest INT4 measurement status

The whole-packed INT4 B1/T16/cap4096 pilot is terminal and independently verified after12 exact HTTP goldens and2 SSE gates. Text/image decode4.4556/4.4495 tok/s; TTFT697.010/1641.609ms; aggregate4.3815/4.2406tok/s; sampled peak21786648576/21792940032bytes. This target-only compatibility path is much slower than the measured mixed NVFP4 MTP3 configuration; configurations differ in quantization and speculation, and no precision-only causal gain is claimed. Raw evidence/source/metric verification recordedP152. The ordinary vLLM INT4 pilot and later cache-reset exploratory pilot are terminal and archived; see latest qualification limits below. Keep SGLang INT4,4B/reference, repeated original/joined, native batching/concurrency and broader/full-context requirements intact. Do not substitute this cross-precision comparison for the2x faster-reference target.


### vLLM INT4 reference update and cache gate

The first ordinary vLLM INT4 trial completed4 VLM and16 full128-token quality captures plus all8 B1/2/4/8 text/image cohorts. Raw-event verification records about77.37/77.33tok/s B1 text/image decode and535.72/501.22tok/s C8 aggregate throughput. Current native INT4 implementation is much slower. All16 prompt-ID pairs match; only5 full output-ID pairs match, so broad/cross-engine quality equivalence remains unproven. Nonzero multimodal cache hits after reused quality fixtures disqualify matched image TTFT from this first trial. Retain it with explicit cache flag; verify installed reset routes, clear BOTH processor and encoder caches while idle, and repeat with a new preserved result path and cache evidence. SGLang INT4 original load terminated9 at its96-wide/tile64 Marlin repack; startup evidence is independently verified, with no timings admitted. Remaining optimization/comparison scope and2x target are unchanged. SeeP153 for complete raw numbers/hashes/limitations.


### Current cache-reset qualification and fixed NVFP4 candidate gates

The first strict vLLM INT4 cache-reset workflow stopped before timing because previous-process full128-token captures matched15/16, although all16 prompts/semantic requests matched. A further same-live-process repeat matched16/16 exactly. A separately labelled exploratory pilot then completed8 cohorts after both processor and encoder reset acknowledgements; raw counters show62 queries and zero new hits, process unchanged. All559 archive file hashes,64 raw measured SSEs,8 sampled VRAM cohorts and current C1 texts were independently verified. Text/image C1 TTFT86.3078/124.8349ms,decode77.4505/77.4228tok/s; C8 aggregate531.6427/490.3604tok/s. Retain unexplained cross-process drift; do not label this2x-eligible or broadly quality-equivalent. Earlier cached image TTFT stays excluded. Both servers are terminal;GPU39MiB observed after shutdown. Research logP159.

Repeated synthetic NVFP4 screening identifies a fixed down-projection candidate: N5120/K17408, algorithm70,tile20,stages37,cluster5,inner0,swizzle0,custom0,splitK-2,reduction2,64MiB workspace allowance. Component gains~1.26(B1/B4) and1.22(B128) are best-of-many synthetic screening, not end-model gains. Candidate remains unpromoted. Next gate uses one actual pinned full down-projection and signed synthetic inputs with full baseline/candidate raw BF16 outputs and a bounded independent decoded CPU oracle. Do not assume exact production activation coverage.

For runtime integration, preserve the legacy plugin/workspace contract. An opt-in candidate must serialize its workspace/selection policy, retain old-engine deserialization with the legacy policy and prevent environment changes from changing a cached engine's scratch-buffer requirement. Test clone/serialize/reload, fixed configuration matching, supported batches1/4/128, missing-candidate/error behavior and all retained full-model target/MTP/HTTP gates before timing. Keep the original path available for comparison. Measure full-model performance after correctness; do not extrapolate the component ratio.4B serving/reference, native batching/concurrency, repeated/full4K-context and broader quality remain open;2x faster-reference target remains unmet.

### Current runtime candidate qualification: failed prefill option excluded

The default-off rebuilt native DLL e82279af/source5d6759336 passed12 target/vector goldens and both complete option-off original/joined T128 state controls against the retained teacher. The all-batch opt-in v2/policy1 candidate FAILED image state equality:21 chunks and307084 retained hidden values differed, max absolute15.375, affecting active rows44..103. Same-binary controls isolate this to candidate configuration; internal cause and quality impact remain unproven. It is not admitted for VLM serving or end-to-end benchmarking. Failure/control archives and independent verification are recorded inP164; no new model-speed claim.

Next test a separately serialized decode-only policy that preserves baselineT128 prefill; retain v1/v2 engine semantics, add a new version/tag and rerun workspace/clone/deserialization plus the SAME full-model state/target/MTP/HTTP gates. Decode correctness and speed are unproven. This changes the implementation candidate, not the required full VLM/4K/concurrency/4B/reference/quality scope or2x target.
### V3 decode-only implementation status

V3/tag2 now preserves baselineT128 prefill and routes only qualified1/4 activation rows to the fixed candidate. V1 and historical failedV2 retain their serialized semantics; contracts/component output checks and native build passed on5090, source754cfa424/DLLba67dc9d. The originalC2660 test compile failure and corrected run are retained and independently verified (P165). Full same-binary state/target/vector qualification completed and was independently verified. Subsequent HTTP20 target/MTP pairs match each other and six retained goldens, but five full outputs differ from the prior qualified20 set and repeat exactly on the same live worker. A new same-DLL option-off HTTP20 control restores all20 prior full outputs. V3 option-on is therefore rejected for serving/timing; no SSE or benchmark was admitted. Both HTTP workers are terminal and all raw failure/control evidence independently verified (P166/P167). P164 failure remains valid, and no kernel ratio is a model gain. Full VLM/4K/concurrency/4B/reference/repeat/quality scope and2x target remain unchanged.

### Next optimization decision: warm full-model CUDA evidence

The installed Nsight Systems2024.6.2 passed a bounded actual CUDA trace smoke on the loaded runtime:5 normalization kernels, CUDA API/graph records, graph capture/replay and exact producer output checks; raw trace/SQLite and all13 archive file digests independently verified. Cold module-loading times are not serving bottleneck evidence. The first interactive warm-mtp-cuda-trace-attempt1 passed12 warm cases but exited0xC0000005 at collection start, completing zero captured cases; raw SQLite has no CUDA kernel/API data and the failure archive is independently verified. The startup-instrumented attempt also exited0xC0000005 after12 exact warm-up cases and before either capture case; its partial CUDA trace has no valid diagnostic markers and is not a warm ranking. Both failure archives are independently verified. The identical unprofiled14-case control completed0 and was independently verified: all12 warm and2 follow-up full outputs match, same probe bytes, same DLL and launcher differing only output directory. The failure did not reproduce in this one control; internal cause and universal stability remain unproven. A side-by-side signed NVIDIA2026.5.1 profiler extraction into the fixed study tools folder is now running; official release notes explicitly support loadedCUDA13.4. No driver/runtime/application-source replacement or restart. Revalidate actual tool identity and full trace coverage; a version upgrade alone admits no profiling or speed result. Unique35892/35908byte HtoD markers must independently delimit the warm window; full startup statistics contain cold/build work and are not a hot ranking. It must pass twelve warm exact text/image/VLM cases before collection, then two full128-token text/image cases inside a bounded window; launch/start/stop/stats and controller have distinct raw ledgers. Verify actual warm CUDA rows, full token IDs and linked source/binary evidence before choosing the next kernel/host-control optimization. Instrumented timing is diagnostic and cannot replace uninstrumented matched HTTP benchmarks.

The implementation priority after that trace is an evidence-selected change to the dominant target/MTP/head/state/control work, with full original correctness gates and repeated same-binary A/B performance trials. For native concurrency, return to actual layer-intermediate B1/B2 diagnosis; synthetic norm/MLP/residual substitutes did not reproduce the real discrepancy. A serial B1 queue is only an offered-load service diagnostic and never substitutes for true GPU batching or the2x concurrency requirement. Complete Qwen3.5-4B matched HTTP/reference measurements and long-context/broad-quality coverage, preserve INT4/reference/cache limitations, and tune feasible reference/MTP configurations. The objective remains at least2x the faster matched feasible baseline in decode and true concurrency throughput; half TTFT is additional. No candidate or chart is promoted from a component ratio or weaker substitute gate.


### Current4B comparison and profiler decision

Native Qwen3.5-4B BF16 target-only B1/T16/cap4096 HTTP pilot is complete and independently verified (P171): text/image decode85.8324/85.7470tok/s,TTFT63.2354/119.1346ms,aggregate82.6198/79.6275tok/s,sampled adapter usage11335159808/11337256960bytes. Gates include four retained old smoke captures,four VLM semantics,16 full128 repeats,two real SSE gates; all16 measured streams and worker IDs replay exactly. Same semantic requests/images and full prompt IDs as the prior27B pilot are verified; architectures,precision,speculation,chunk profiles and binaries differ. This descriptive4B comparison does not establish model-quality parity, causal model-size/precision gain or the27B2x objective. Native4B vLLM/SGLang references, repeated confidence, GPU concurrency and broad/full-context checks remain open.

Old Nsight2024 full traces failed0xC0000005 after12 warm cases, while the identical unprofiled14-case control passed. The newer signed Nsight2026.5.1 Windows payload is now extracted side-by-side into fixed tools/n26 after preserving MSI/path/guard failures. Bounded actual CUDA smoke and all49 evidence hashes/SQLite records are independently verified. A new original full-model trace handle warm-mtp-cuda-trace-2026-attempt1 is live on the5090, app754cfa424/DLLba67dc9d/candidate0 frozen. Same producer as unprofiled control;12 warm plus2 capture full-output gates and unique ordered HtoD markers precede any hot ranking. The full trace is not yet a success or a serving-speed result. After it closes, select the next optimization from verified hot operations, rerun original correctness gates and refresh current27B HTTP same-binary baseline/candidate with repeated trials. Keep native batching diagnosis,4B references, full-context/broad quality and fastest feasible tuned reference requirements intact.2x decode and trueGPUconcurrency remain unmet.

P172: full2026 warm CUDA trace independently verified (24-file archive6d8662718341ffb9e5580ff41a263458aae0e7780e9577a596ce211d87f81dec). See development log for scoped kernel/API ranks. Next correctness experiment is existing resident_controls=True (inline request updates/chunk gathers), first transport48trials then14 full-model cases, unchanged754/ba67/defaultNVFP4. No speed/default admission before full exact-output and actual SSE/HTTP gates; uninstrumented same-binary A/B required. Entire2x and trueGPUbatching target remains unchanged.
