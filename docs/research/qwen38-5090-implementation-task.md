# Qwen3.8-27B on one RTX 5090: implementation task

Status: pinned architecture/quantization audit, native SM120 build, NVFP4/FP8
projection/plugin tests, real layer0 MLP parity, real GDN/full-attention decoder-layer
parity at4K capacity, and native W4A4 head checks have passed. Both references have
passed text/image smoke; SGLang's initial eager profile now has raw streaming and
concurrency measurements. The32 CPU tests pass. Full64-layer native text smoke
also reproduces the reference token IDs. Vision validation,
serving integration, tuned matched comparisons and the2x target remain incomplete.
No Garnet full-model timing or successful2x comparison is claimed.

Audit date: 2026-10-23. The preserved source baseline was
`5b0dc7ff8e5c1a77807138db310605bcd12aa390` plus inherited local edits, captured
separately at2355fc4b5. Work continues on qwen38-5090; command ledgers retain exact
commit/patch manifests. Do not reset the original checkout or infer tested source
identity from a parent commit alone. Research method/failures are in the development
log; no full-model throughput or successful2x comparison is claimed.

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
RTX5090 build, which has passed native build, component correctness and the full64-layer text smoke. Full VLM execution remains pending. Build target `garnet`. No VM sources
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
   Next validate image processing
   and vision plus full VLM generation. The reference224x224 PNG smoke produced64
   image tokens; inspect actual resize/normalization rather than assuming49.
2. Establish text/image quality beyond basic smoke and record the precision
   differences. Current Garnet state/cache arithmetic is F32; initial references
   use BF16 KV/conv and F32 SSM. Close this matching gate before any2x claim.
3. Add streaming/count accounting to Garnet's serving path and validate concurrency
   profiles1/2/4/8 with real memory admission. Retain compressed weights shared
   across profiles; inspect fused GDN/attention and resident control paths already
   present in current Garnet before introducing another execution stack.
4. Establish tuned vLLM/SGLang profiles, including feasible CUDA graph/kernel
   options. Both initial profiles deliberately used eager startup for feasibility;
   they are not evidence of the fastest attainable reference. Compiler/linker
   failures and SGLang's first memory exhaustion are retained in the research log.
5. Complete the separate INT4 W4A16 loader/kernel compatibility gate and4B comparison.
   Do not label the mixed NVIDIA checkpoint as an INT4 baseline.
6. Collect repeated text/image TTFT, decode, aggregate throughput and memory with
   fixed4K total capacity and matched request hashes, quality, cache behavior,
   output lengths and concurrency. The current32-token/eight-request SGLang data is
   exploratory. Preserve longer runs, distributions and raw failures. The NVML
   sampler requested20ms but measured about31ms typical gaps and one7.787s gap;
   investigate timing/observer repeatability and report sampled peaks honestly.
7. Optimize from measured bottlenecks, then test against twice the faster valid
   reference for decode and throughput; seek half TTFT separately. Publish unmet
   targets as measured. No automatic success follows from primitive correctness,
   server process duration, a reduced workload or an untuned reference.
